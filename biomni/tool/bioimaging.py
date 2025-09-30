import logging
import os
import zipfile

import matplotlib
import requests

matplotlib.use("Agg")  # Use non-interactive backend
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.serialization
from nnunet.inference.predict import predict_from_folder

# Apply safe globals for torch serialization
torch.serialization.add_safe_globals([tuple, list, dict, set, int, float, str, bytes, bytearray])
torch.serialization.add_safe_globals([complex, slice, range])
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# SEGMENTATION CLASS
# ============================================================================


class SegmentationTool:
    """
    A comprehensive tool for medical image segmentation using nnUNet.
    Handles BRATS dataset processing, modality splitting, and segmentation visualization.
    """

    def __init__(self):
        """Initialize the SegmentationTool."""
        self.supported_formats = [".nii", ".nii.gz"]
        logger.info("SegmentationTool initialized")

    def split_modalities(self, input_file, output_dir, case_name="BRAT"):
        """
        Split a 4D NIfTI file into separate modality files for nnUNet
        Args:
            input_file: Path to the 4D NIfTI file
            output_dir: Directory to save the split files
            case_name: Base name for the case (default: BRAT)
        Returns:
            output_dir: Path to directory containing split modality files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load the 4D image
        print(f"Loading {input_file}...")
        img = nib.load(input_file)
        data = img.get_fdata()

        print(f"Image shape: {data.shape}")
        print("Expected shape: (X, Y, Z, 4) for 4 modalities")

        if len(data.shape) != 4:
            raise ValueError(f"Expected 4D image, got {len(data.shape)}D")

        if data.shape[3] != 4:
            raise ValueError(f"Expected 4 modalities, got {data.shape[3]}")

        # Split into separate files
        modalities = ["FLAIR", "T1w", "t1gd", "T2w"]

        for i, modality in enumerate(modalities):
            # Extract the modality data
            modality_data = data[:, :, :, i]

            # Create a new NIfTI image with the same header but 3D data
            modality_img = nib.Nifti1Image(modality_data, img.affine, img.header)

            # Save with the expected naming convention
            output_file = os.path.join(output_dir, f"{case_name}_{i:04d}.nii.gz")
            nib.save(modality_img, output_file)

            print(f"Saved {modality} modality to {output_file}")
            print(f"  Shape: {modality_data.shape}, Data type: {modality_data.dtype}")

        print(f"\nAll modalities saved to {output_dir}")
        return output_dir

    def prepare_input_for_nnunet(self, input_path, output_dir, case_name="BRAT"):
        """
        Prepare input data for nnUNet by handling both 4D and pre-split modality files
        Args:
            input_path: Path to input file or directory
            output_dir: Directory to save prepared files
            case_name: Base name for the case (default: BRAT)
        Returns:
            prepared_dir: Path to directory with nnUNet-ready files
        """
        os.makedirs(output_dir, exist_ok=True)

        if os.path.isfile(input_path):
            # Single file - check if it's 4D
            if input_path.endswith((".nii", ".nii.gz")):
                try:
                    img = nib.load(input_path)
                    if len(img.shape) == 4 and img.shape[3] == 4:
                        print("4D NIfTI file detected, splitting modalities...")
                        return self.split_modalities(input_path, output_dir, case_name)
                    else:
                        print("Single 3D file detected, copying to output directory...")
                        # Copy single file with proper naming
                        output_file = os.path.join(output_dir, f"{case_name}_0000.nii.gz")
                        import shutil

                        shutil.copy2(input_path, output_file)
                        return output_dir
                except Exception as e:
                    print(f"Error reading file {input_path}: {e}")
                    raise
        elif os.path.isdir(input_path):
            # Directory - check if it already has split modalities
            files = [f for f in os.listdir(input_path) if f.endswith((".nii", ".nii.gz"))]

            if any(f.endswith("_0000.nii.gz") for f in files):
                print("Directory already contains split modality files, using as-is...")
                # Copy existing files to output directory
                for f in files:
                    if f.endswith((".nii", ".nii.gz")):
                        import shutil

                        shutil.copy2(os.path.join(input_path, f), os.path.join(output_dir, f))
                return output_dir
            else:
                # Check if there's a 4D file to split
                for f in files:
                    if f.endswith((".nii", ".nii.gz")):
                        try:
                            img = nib.load(os.path.join(input_path, f))
                            if len(img.shape) == 4 and img.shape[3] == 4:
                                print(f"4D NIfTI file {f} detected, splitting modalities...")
                                return self.split_modalities(os.path.join(input_path, f), output_dir, case_name)
                        except Exception as e:
                            logging.debug("Skipping file %s during 4D check: %s", f, e)
                            continue

                print("No 4D files found, copying existing files...")
                # Copy existing files to output directory
                for f in files:
                    if f.endswith((".nii", ".nii.gz")):
                        import shutil

                        shutil.copy2(os.path.join(input_path, f), os.path.join(output_dir, f))
                return output_dir

        raise ValueError(f"Input path {input_path} is neither a valid file nor directory")

    def setup_nnunet_environment(self, results_folder=None, raw_data_base=None, preprocessed=None):
        """
        Setup nnU-Net environment variables according to official documentation
        Args:
            results_folder: Path to nnUNet results folder (default: ~/nnUNet_results)
            raw_data_base: Path to raw data base (default: ~/nnUNet_raw_data_base)
            preprocessed: Path to preprocessed data (default: ~/nnUNet_preprocessed)
        """
        # Set nnUNet environment variables as per official documentation
        if results_folder:
            os.environ["nnUNet_RESULTS_FOLDER"] = os.path.expanduser(results_folder)
        elif "nnUNet_RESULTS_FOLDER" not in os.environ:
            os.environ["nnUNet_RESULTS_FOLDER"] = os.path.expanduser("~/nnUNet_results")

        if raw_data_base:
            os.environ["nnUNet_raw_data_base"] = os.path.expanduser(raw_data_base)
        elif "nnUNet_raw_data_base" not in os.environ:
            os.environ["nnUNet_raw_data_base"] = os.path.expanduser("~/nnUNet_raw_data_base")

        if preprocessed:
            os.environ["nnUNet_preprocessed"] = os.path.expanduser(preprocessed)
        elif "nnUNet_preprocessed" not in os.environ:
            os.environ["nnUNet_preprocessed"] = os.path.expanduser("~/nnUNet_preprocessed")

        # Create directories if they don't exist
        for path in [
            os.environ["nnUNet_RESULTS_FOLDER"],
            os.environ["nnUNet_raw_data_base"],
            os.environ["nnUNet_preprocessed"],
        ]:
            os.makedirs(path, exist_ok=True)

        print("nnU-Net environment variables set:")
        print(f"  nnUNet_RESULTS_FOLDER: {os.environ['nnUNet_RESULTS_FOLDER']}")
        print(f"  nnUNet_raw_data_base: {os.environ['nnUNet_raw_data_base']}")
        print(f"  nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")

    def _download_model_with_browser_headers(self, url, output_path):
        """
        Download model with browser-like headers to bypass Zenodo's anti-bot protection
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        logger.info(f"Downloading model from: {url}")

        try:
            response = requests.get(url, headers=headers, stream=True, timeout=300)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Download completed: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Download failed: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False

    def _download_and_extract_model(self, task_id, model_type="3d_fullres"):
        """
        Download and extract nnUNet model with browser-like headers - extracts directly to nnUNet directory
        """
        results_folder = os.environ.get("nnUNet_RESULTS_FOLDER", "~/nnUNet_results")
        results_folder = os.path.expanduser(results_folder)

        # Create the nnUNet directory
        nnunet_dir = os.path.join(results_folder, "nnUNet")
        os.makedirs(nnunet_dir, exist_ok=True)

        # Check if model already exists
        task_dir = os.path.join(nnunet_dir, model_type, task_id)
        if os.path.exists(task_dir):
            # Check if it has the expected structure
            plans_file = os.path.join(task_dir, "nnUNetTrainerV2__nnUNetPlansv2.1")
            if os.path.exists(plans_file):
                logger.info(f"Model already exists for task '{task_id}' with {model_type}")
                return True

        # Download URL for the task
        download_url = f"https://zenodo.org/record/4003545/files/{task_id}.zip?download=1"

        # Create temporary file for download
        temp_zip = os.path.join(nnunet_dir, f"{task_id}_temp.zip")

        # Download with browser headers
        if self._download_model_with_browser_headers(download_url, temp_zip):
            try:
                logger.info(f"Extracting {temp_zip} to {nnunet_dir}")
                with zipfile.ZipFile(temp_zip, "r") as zip_ref:
                    zip_ref.extractall(nnunet_dir)

                # Remove temporary zip file
                os.remove(temp_zip)

                # Verify extraction worked
                if os.path.exists(task_dir):
                    logger.info(f"Model successfully downloaded and extracted for task '{task_id}'")
                    logger.info(f"Model directory: {task_dir}")
                    return True
                else:
                    logger.error(f"Task directory not found after extraction: {task_dir}")
                    return False

            except Exception as e:
                logger.error(f"Extraction failed: {e}")
                if os.path.exists(temp_zip):
                    os.remove(temp_zip)
                return False
        else:
            return False

    def segment_with_nn_unet(
        self,
        image_path,
        output_dir,
        task_id,
        model_type="3d_fullres",
        folds=None,
        use_tta=False,
        num_threads=1,
        mixed_precision=True,
        verbose=True,
        auto_prepare_input=True,
        results_folder=None,
        auto_download=True,
    ):
        """
        Segment images using nnUNet with proper environment setup and automatic model downloading
        Args:
            image_path: Path to input image file or directory
            output_dir: Directory to save segmentation results
            task_id: Task identifier (e.g., 'Task001_BrainTumour')
            model_type: Model type (default: '3d_fullres')
            folds: Model folds to use (default: [0, 1, 2, 3, 4])
            use_tta: Use test time augmentation (default: False)
            num_threads: Number of threads for preprocessing (default: 1)
            mixed_precision: Use mixed precision (default: True)
            verbose: Verbose logging (default: True)
            auto_prepare_input: Automatically prepare input for nnUNet (default: True)
            results_folder: Path to nnUNet results folder (default: None, will use environment variable or default)
            auto_download: Automatically download missing models (default: True)
        """
        if folds is None:
            folds = [0, 1, 2, 3, 4]
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

        # Setup nnUNet environment first
        self.setup_nnunet_environment(results_folder=results_folder)

        # Prepare input data if requested
        if auto_prepare_input:
            temp_input_dir = os.path.join(output_dir, "temp_input")
            prepared_input_dir = self.prepare_input_for_nnunet(image_path, temp_input_dir)
            image_path = prepared_input_dir
            logging.info(f"Input prepared for nnUNet: {image_path}")

        logging.info("Verifying NIfTI input files...")

        def verify_nifti_input(image_path):
            if os.path.isfile(image_path):
                nib.load(image_path)
            else:
                for file in os.listdir(image_path):
                    if file.endswith(".nii") or file.endswith(".nii.gz"):
                        nib.load(os.path.join(image_path, file))

        verify_nifti_input(image_path)

        # Get or set up the results folder
        results_folder = os.environ.get("nnUNet_RESULTS_FOLDER")
        if not results_folder:
            # Try common locations
            common_paths = [
                "./models/nnUNet",
                "~/nnUNet_results",
                "~/biomni_models/nnUNet",
            ]

            for base_path in common_paths:
                expanded_path = os.path.expanduser(base_path)
                if os.path.exists(expanded_path):
                    results_folder = expanded_path
                    break

            # If still no folder found, create one
            if not results_folder:
                results_folder = os.path.expanduser("~/nnUNet_results")
                os.makedirs(results_folder, exist_ok=True)
                logging.info(f"Created new results folder: {results_folder}")

        os.environ["RESULTS_FOLDER"] = results_folder
        os.environ["nnUNet_RESULTS_FOLDER"] = results_folder
        logging.info(f"Set RESULTS_FOLDER environment variable to: {results_folder}")

        # Construct the expected model path - using the correct structure
        model_folder = os.path.join(results_folder, "nnUNet", model_type, task_id, "nnUNetTrainerV2__nnUNetPlansv2.1")

        logging.info(f"Looking for model at: {model_folder}")

        # Check if model files actually exist (not just the directory)
        def check_model_files_exist(model_folder, folds):
            """Check if actual model weight files exist for the specified folds"""
            if not os.path.exists(model_folder):
                return False

            # Check for plans.pkl and other required files
            required_files = ["plans.pkl"]
            for req_file in required_files:
                if not os.path.exists(os.path.join(model_folder, req_file)):
                    return False

            # Check if at least one fold has model files
            for fold in folds:
                fold_dir = os.path.join(model_folder, f"fold_{fold}")
                if os.path.exists(fold_dir):
                    # Look for model files (.model, .model.pkl, etc.)
                    model_files = [f for f in os.listdir(fold_dir) if f.endswith((".model", ".model.pkl"))]
                    if model_files:
                        return True
            return False

        # Check if model exists, download if not
        if not check_model_files_exist(model_folder, folds):
            if auto_download:
                logging.info(f"Model weights for {task_id} not found. Downloading...")
                # Ensure the results folder structure exists
                os.makedirs(os.path.dirname(model_folder), exist_ok=True)

                # Download the model using our custom download function
                if self._download_and_extract_model(task_id, model_type):
                    logging.info(f"Downloaded pretrained model for {task_id} successfully.")
                else:
                    raise RuntimeError(f"Failed to download model for {task_id}")

                # Verify the download worked
                if not check_model_files_exist(model_folder, folds):
                    raise RuntimeError(
                        f"Model download completed but files not found at expected location: {model_folder}"
                    )
            else:
                # Ask user for permission to download
                user_input = (
                    input(f"Model weights for {task_id} not found. Do you want to download them? (y/n): ")
                    .strip()
                    .lower()
                )
                if user_input == "y":
                    # Download the model using our custom download function
                    if self._download_and_extract_model(task_id, model_type):
                        logging.info(f"Downloaded pretrained model for {task_id} successfully.")
                    else:
                        raise RuntimeError(f"Failed to download model for {task_id}")
                else:
                    raise RuntimeError("Model weights not found and download declined by user.")

        # Double-check that we now have the model
        if not check_model_files_exist(model_folder, folds):
            raise RuntimeError(f"Model files still not found at {model_folder} after download attempt")

        logging.info(f"Using model: {model_folder}")

        # Patch torch.load for compatibility
        original_torch_load = torch.load

        def patched_torch_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        # Run the segmentation
        torch.load = patched_torch_load
        try:
            predict_from_folder(
                model=model_folder,
                input_folder=image_path,
                output_folder=output_dir,
                folds=folds,
                save_npz=False,
                num_threads_preprocessing=num_threads,
                num_threads_nifti_save=num_threads,
                mixed_precision=mixed_precision,
                lowres_segmentations=None,
                part_id=0,
                num_parts=1,
                tta=use_tta,
            )
        finally:
            # Restore original torch.load behavior
            torch.load = original_torch_load

        # Clean up temporary input directory if it was created
        if auto_prepare_input and os.path.exists(temp_input_dir):
            import shutil

            shutil.rmtree(temp_input_dir)
            logging.info("Cleaned up temporary input directory")

        logging.info(f"Segmentation outputs stored at: {output_dir}")
        return output_dir

    def create_segmentation_visualization(self, original_mri, segmentation, output_dir="./visualization_output"):
        """
        Create and save visualization of segmentation results using nilearn
        Args:
            original_mri: Path to original MRI file
            segmentation: Path to segmentation file
            output_dir: Directory to save visualization images
        Returns:
            list: List of saved image file paths
        """
        try:
            # Import nilearn here to avoid dependency issues
            from nilearn import plotting

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Check if files exist
            if not os.path.exists(original_mri):
                raise FileNotFoundError(f"Original MRI file not found: {original_mri}")

            if not os.path.exists(segmentation):
                raise FileNotFoundError(f"Segmentation file not found: {segmentation}")

            print("✅ Files found, creating visualizations...")
            saved_files = []

            # Create and save the main overlay plot
            display = plotting.plot_roi(
                segmentation,
                bg_img=original_mri,
                cmap="Set1",
                alpha=0.6,
                title="Segmentation Overlay",
            )

            # Save the main overlay plot
            output_file = os.path.join(output_dir, "segmentation_overlay.png")
            display.savefig(output_file, dpi=150, bbox_inches="tight")
            saved_files.append(output_file)
            print(f"✅ Main overlay saved to: {output_file}")
            display.close()

            # Create additional views and save them
            # Axial view
            display_axial = plotting.plot_roi(
                segmentation,
                bg_img=original_mri,
                cmap="Set1",
                alpha=0.6,
                title="Segmentation Overlay - Axial View",
                display_mode="z",
            )
            axial_file = os.path.join(output_dir, "segmentation_axial.png")
            display_axial.savefig(axial_file, dpi=150, bbox_inches="tight")
            saved_files.append(axial_file)
            print(f"✅ Axial view saved to: {axial_file}")
            display_axial.close()

            # Sagittal view
            display_sagittal = plotting.plot_roi(
                segmentation,
                bg_img=original_mri,
                cmap="Set1",
                alpha=0.6,
                title="Segmentation Overlay - Sagittal View",
                display_mode="x",
            )
            sagittal_file = os.path.join(output_dir, "segmentation_sagittal.png")
            display_sagittal.savefig(sagittal_file, dpi=150, bbox_inches="tight")
            saved_files.append(sagittal_file)
            print(f"✅ Sagittal view saved to: {sagittal_file}")
            display_sagittal.close()

            # Coronal view
            display_coronal = plotting.plot_roi(
                segmentation,
                bg_img=original_mri,
                cmap="Set1",
                alpha=0.6,
                title="Segmentation Overlay - Coronal View",
                display_mode="y",
            )
            coronal_file = os.path.join(output_dir, "segmentation_coronal.png")
            display_coronal.savefig(coronal_file, dpi=150, bbox_inches="tight")
            saved_files.append(coronal_file)
            print(f"✅ Coronal view saved to: {coronal_file}")
            display_coronal.close()

            print(f"\n✅ All visualizations saved to: {output_dir}")
            print("Files created:")
            for file_path in saved_files:
                print(f"  - {os.path.basename(file_path)}")

            return saved_files

        except ImportError:
            print(" nilearn not available. Install with: pip install nilearn")
            return []
        except Exception as e:
            print(f" Visualization failed: {e}")
            import traceback

            traceback.print_exc()
            return []


# ============================================================================
# IMAGE REGISTRATION CLASS
# ============================================================================


class ImageRegistrationTool:
    """
    A comprehensive tool for medical image registration using SimpleITK.
    Supports rigid, affine, and deformable registration with preprocessing and visualization.
    """

    def __init__(self):
        """Initialize the ImageRegistrationTool."""
        self.supported_formats = [".nii", ".nii.gz", ".nrrd", ".mha", ".mhd"]
        logger.info("ImageRegistrationTool initialized")

    def load_image(self, image_path: str) -> sitk.Image:
        """
        Load a medical image using SimpleITK.

        Args:
            image_path: Path to the image file

        Returns:
            SimpleITK Image object
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info(f"Loading image: {image_path}")
        try:
            image = sitk.ReadImage(image_path)
            logger.info(f"Successfully loaded image with size: {image.GetSize()}")
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise

    def save_image(self, image: sitk.Image, output_path: str) -> None:
        """
        Save a SimpleITK image to file.

        Args:
            image: SimpleITK Image object
            output_path: Path to save the image (must include filename and extension)
        """
        # Validate output path
        if os.path.isdir(output_path):
            raise ValueError(
                f"Output path '{output_path}' is a directory. Please provide a file path with extension (e.g., '.nii.gz', '.png', '.jpg')"
            )

        if not os.path.splitext(output_path)[1]:
            raise ValueError(
                f"Output path '{output_path}' has no file extension. Please add an extension (e.g., '.nii.gz', '.png', '.jpg')"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Saving image to: {output_path}")
        try:
            sitk.WriteImage(image, output_path)
            logger.info("Image saved successfully")
        except Exception as e:
            logger.error(f"Failed to save image to {output_path}: {e}")
            raise

    def preprocess_image(self, image: sitk.Image, denoise: bool = True, normalize: bool = True) -> sitk.Image:
        """
        Preprocess an image with denoising and normalization.

        Args:
            image: Input SimpleITK image
            denoise: Whether to apply denoising
            normalize: Whether to apply normalization

        Returns:
            Preprocessed SimpleITK image
        """
        logger.info("Preprocessing image...")
        processed_image = sitk.Image(image)

        if denoise:
            # Apply Gaussian smoothing for denoising
            processed_image = sitk.SmoothingRecursiveGaussian(processed_image, sigma=1.0)
            logger.info("Applied denoising")

        if normalize:
            # Normalize to [0, 1] range
            min_max_filter = sitk.MinimumMaximumImageFilter()
            min_max_filter.Execute(processed_image)
            min_val = min_max_filter.GetMinimum()
            max_val = min_max_filter.GetMaximum()

            if max_val > min_val:
                processed_image = sitk.IntensityWindowing(
                    processed_image, windowMinimum=min_val, windowMaximum=max_val, outputMinimum=0.0, outputMaximum=1.0
                )
                logger.info("Applied normalization")

        return processed_image

    def create_rigid_transform(
        self, fixed_image: sitk.Image, moving_image: sitk.Image, initial_transform: sitk.Transform | None = None
    ) -> sitk.Transform:
        """
        Create a rigid transform for image registration.

        Args:
            fixed_image: Reference (fixed) image
            moving_image: Image to be registered
            initial_transform: Optional initial transform

        Returns:
            Rigid transform object
        """
        logger.info("Creating rigid transform...")

        if initial_transform is None:
            # Create identity transform
            transform = sitk.Euler3DTransform()
        else:
            transform = initial_transform

        return transform

    def create_affine_transform(
        self, fixed_image: sitk.Image, moving_image: sitk.Image, initial_transform: sitk.Transform | None = None
    ) -> sitk.Transform:
        """
        Create an affine transform for image registration.

        Args:
            fixed_image: Reference (fixed) image
            moving_image: Image to be registered
            initial_transform: Optional initial transform

        Returns:
            Affine transform object
        """
        logger.info("Creating affine transform...")

        if initial_transform is None:
            # Create identity transform
            transform = sitk.AffineTransform(3)
        else:
            transform = initial_transform

        return transform

    def create_deformable_transform(
        self, fixed_image: sitk.Image, moving_image: sitk.Image, number_of_control_points: int = 4
    ) -> sitk.Transform:
        """
        Create a deformable (B-spline) transform for image registration.

        Args:
            fixed_image: Reference (fixed) image
            moving_image: Image to be registered
            number_of_control_points: Number of B-spline control points per dimension

        Returns:
            Deformable transform object
        """
        logger.info("Creating deformable transform...")

        # Create B-spline transform
        transform = sitk.BSplineTransformInitializer(fixed_image, [number_of_control_points] * 3, order=3)

        return transform

    def setup_registration_method(
        self,
        transform: sitk.Transform,
        metric: str = "mutual_information",
        optimizer: str = "gradient_descent",
        learning_rate: float = 0.01,
        number_of_iterations: int = 100,
        gradient_convergence_tolerance: float = 1e-6,
    ) -> sitk.ImageRegistrationMethod:
        """
        Setup the registration method with specified parameters.

        Args:
            transform: Transform object
            metric: Similarity metric name
            optimizer: Optimizer name
            learning_rate: Learning rate for gradient descent
            number_of_iterations: Maximum number of iterations
            gradient_convergence_tolerance: Convergence tolerance

        Returns:
            Configured registration method
        """
        logger.info(f"Setting up registration method: {metric} metric, {optimizer} optimizer")

        registration_method = sitk.ImageRegistrationMethod()

        # Set similarity metric
        if metric == "mutual_information":
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        elif metric == "mean_squares":
            registration_method.SetMetricAsMeanSquares()
        elif metric == "correlation":
            registration_method.SetMetricAsCorrelation()
        elif metric == "normalized_correlation":
            registration_method.SetMetricAsNormalizedCorrelation()
        else:
            logger.warning(f"Unknown metric {metric}, using mutual information")
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

        # Set optimizer
        if optimizer == "gradient_descent":
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=learning_rate,
                numberOfIterations=number_of_iterations,
                convergenceMinimumValue=gradient_convergence_tolerance,
            )
        elif optimizer == "lbfgsb":
            registration_method.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=gradient_convergence_tolerance, numberOfIterations=number_of_iterations
            )
        elif optimizer == "powell":
            registration_method.SetOptimizerAsPowell(numberOfIterations=number_of_iterations, maximumLineIterations=20)
        elif optimizer == "amoeba":
            registration_method.SetOptimizerAsAmoeba(
                numberOfIterations=number_of_iterations, parametersConvergenceTolerance=gradient_convergence_tolerance
            )
        else:
            logger.warning(f"Unknown optimizer {optimizer}, using gradient descent")
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=learning_rate,
                numberOfIterations=number_of_iterations,
                convergenceMinimumValue=gradient_convergence_tolerance,
            )

        # Set interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Set initial transform
        registration_method.SetInitialTransform(transform, inPlace=False)

        return registration_method

    def register_images(
        self,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        transform: sitk.Transform,
        registration_method: sitk.ImageRegistrationMethod,
    ) -> tuple[sitk.Transform, sitk.Image]:
        """
        Perform image registration.

        Args:
            fixed_image: Reference (fixed) image
            moving_image: Image to be registered
            transform: Transform object
            registration_method: Configured registration method

        Returns:
            Tuple of (final_transform, registered_image)
        """
        logger.info("Starting image registration...")

        # Add iteration callback
        def command_iteration():
            logger.info(
                f"Iteration: {registration_method.GetOptimizerIteration()}, "
                f"Metric value: {registration_method.GetMetricValue():.6f}"
            )

        registration_method.AddCommand(sitk.sitkIterationEvent, command_iteration)

        try:
            # Execute registration
            final_transform = registration_method.Execute(fixed_image, moving_image)

            # Apply transform to moving image
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_image)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(final_transform)

            registered_image = resampler.Execute(moving_image)

            logger.info("Registration completed successfully")
            return final_transform, registered_image

        except Exception as e:
            logger.error(f"Registration failed: {e}")
            raise

    def calculate_similarity_metrics(self, image1: sitk.Image, image2: sitk.Image) -> dict[str, float]:
        """
        Calculate similarity metrics between two images.

        Args:
            image1: First image
            image2: Second image

        Returns:
            Dictionary of similarity metrics
        """
        logger.info("Calculating similarity metrics...")
        metrics = {}

        try:
            # Convert to numpy arrays
            array1 = sitk.GetArrayFromImage(image1)
            array2 = sitk.GetArrayFromImage(image2)

            # Flatten arrays
            flat1 = array1.flatten()
            flat2 = array2.flatten()

            # Remove invalid values
            valid_mask = np.isfinite(flat1) & np.isfinite(flat2)
            flat1 = flat1[valid_mask]
            flat2 = flat2[valid_mask]

            if len(flat1) == 0:
                logger.warning("No valid pixels for similarity calculation")
                return {
                    "mutual_information": 0.0,
                    "mean_squares": 0.0,
                    "correlation": 0.0,
                    "normalized_correlation": 0.0,
                }

            # Mean Squared Error
            mse = np.mean((flat1 - flat2) ** 2)
            metrics["mean_squares"] = -mse  # Negative because we want to maximize

            # Pearson Correlation
            if np.std(flat1) > 0 and np.std(flat2) > 0:
                correlation = np.corrcoef(flat1, flat2)[0, 1]
                metrics["correlation"] = correlation if not np.isnan(correlation) else 0.0
            else:
                metrics["correlation"] = 0.0

            # Normalized Cross Correlation
            if np.std(flat1) > 0 and np.std(flat2) > 0:
                ncc = np.corrcoef(flat1, flat2)[0, 1]
                metrics["normalized_correlation"] = ncc if not np.isnan(ncc) else 0.0
            else:
                metrics["normalized_correlation"] = 0.0

            # Mutual Information (simplified calculation)
            try:
                # Create 2D histogram
                hist_2d, x_edges, y_edges = np.histogram2d(flat1, flat2, bins=50)
                hist_2d = hist_2d + 1e-10  # Add small value to avoid log(0)

                # Normalize histogram
                pxy = hist_2d / np.sum(hist_2d)
                px = np.sum(pxy, axis=1)
                py = np.sum(pxy, axis=0)

                # Calculate mutual information
                mi = 0.0
                for i in range(len(px)):
                    for j in range(len(py)):
                        if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                            mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))

                metrics["mutual_information"] = mi

            except Exception as e:
                logger.warning(f"Failed to calculate mutual information: {e}")
                metrics["mutual_information"] = 0.0

        except Exception as e:
            logger.warning(f"Failed to calculate similarity metrics: {e}")
            metrics = {
                "mutual_information": 0.0,
                "mean_squares": 0.0,
                "correlation": 0.0,
                "normalized_correlation": 0.0,
            }

        logger.info("Similarity metrics calculated")
        return metrics


# ============================================================================
# CONVENIENCE FUNCTIONS FOR BIOMNI INTEGRATION
# ============================================================================


# Segmentation convenience functions
def split_modalities(input_file, output_dir, case_name="BRAT"):
    """Convenience function for splitting modalities"""
    tool = SegmentationTool()
    return tool.split_modalities(input_file, output_dir, case_name)


def prepare_input_for_nnunet(input_path, output_dir, case_name="BRAT"):
    """Convenience function for preparing nnUNet input"""
    tool = SegmentationTool()
    return tool.prepare_input_for_nnunet(input_path, output_dir, case_name)


def segment_with_nn_unet(
    image_path,
    output_dir,
    task_id,
    model_type="3d_fullres",
    folds=None,
    use_tta=False,
    num_threads=1,
    mixed_precision=True,
    verbose=True,
    auto_prepare_input=True,
    results_folder=None,
):
    """Convenience function for nnUNet segmentation"""
    tool = SegmentationTool()
    return tool.segment_with_nn_unet(
        image_path,
        output_dir,
        task_id,
        model_type,
        folds,
        use_tta,
        num_threads,
        mixed_precision,
        verbose,
        auto_prepare_input,
        results_folder,
    )


def create_segmentation_visualization(original_mri, segmentation, output_dir="./visualization_output"):
    """Convenience function for segmentation visualization"""
    tool = SegmentationTool()
    return tool.create_segmentation_visualization(original_mri, segmentation, output_dir)


# Registration convenience functions
def preprocess_image(image_path: str, output_path: str, denoise: bool = True, normalize: bool = True) -> str:
    """
    Standalone image preprocessing function for Biomni integration.

    Args:
        image_path: Path to input image
        output_path: Path to save preprocessed image
        denoise: Whether to apply denoising (default: True)
        normalize: Whether to apply normalization (default: True)

    Returns:
        Path to the saved preprocessed image
    """
    tool = ImageRegistrationTool()
    image = tool.load_image(image_path)
    preprocessed_image = tool.preprocess_image(image, denoise, normalize)
    tool.save_image(preprocessed_image, output_path)
    return output_path


def quick_rigid_registration(
    fixed_image_path: str,
    moving_image_path: str,
    output_dir: str,
    metric: str = "mutual_information",
    optimizer: str = "gradient_descent",
    preprocess: bool = True,
    create_visualizations: bool = True,
    learning_rate: float = 0.01,
    number_of_iterations: int = 100,
    gradient_convergence_tolerance: float = 1e-6,
) -> dict:
    """
    Quick rigid registration function for Biomni integration.

    Args:
        fixed_image_path: Path to reference image
        moving_image_path: Path to image to register
        output_dir: Directory to save results
        metric: Similarity metric
        optimizer: Optimization method
        preprocess: Whether to preprocess images
        create_visualizations: Whether to create visualizations
        learning_rate: Learning rate for optimizer
        number_of_iterations: Maximum iterations
        gradient_convergence_tolerance: Convergence tolerance

    Returns:
        Dictionary with registration results
    """
    tool = ImageRegistrationTool()

    # Load images
    fixed_image = tool.load_image(fixed_image_path)
    moving_image = tool.load_image(moving_image_path)

    # Preprocess if requested
    if preprocess:
        fixed_image = tool.preprocess_image(fixed_image)
        moving_image = tool.preprocess_image(moving_image)

    # Create transform and registration method
    transform = tool.create_rigid_transform(fixed_image, moving_image)
    registration_method = tool.setup_registration_method(
        transform, metric, optimizer, learning_rate, number_of_iterations, gradient_convergence_tolerance
    )

    # Perform registration
    final_transform, registered_image = tool.register_images(fixed_image, moving_image, transform, registration_method)

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save registered image
    registered_path = os.path.join(output_dir, "rigid_registered.nii.gz")
    tool.save_image(registered_image, registered_path)

    # Save transform
    transform_path = os.path.join(output_dir, "rigid_transform.tfm")
    sitk.WriteTransform(final_transform, transform_path)

    # Calculate metrics
    metrics_before = tool.calculate_similarity_metrics(fixed_image, moving_image)
    metrics_after = tool.calculate_similarity_metrics(fixed_image, registered_image)

    results = {
        "registered_image_path": registered_path,
        "transform_path": transform_path,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "registration_type": "rigid",
    }

    return results


def quick_affine_registration(
    fixed_image_path: str,
    moving_image_path: str,
    output_dir: str,
    metric: str = "mutual_information",
    optimizer: str = "gradient_descent",
    preprocess: bool = True,
    create_visualizations: bool = True,
    learning_rate: float = 0.01,
    number_of_iterations: int = 100,
    gradient_convergence_tolerance: float = 1e-6,
) -> dict:
    """
    Quick affine registration function for Biomni integration.

    Args:
        fixed_image_path: Path to reference image
        moving_image_path: Path to image to register
        output_dir: Directory to save results
        metric: Similarity metric
        optimizer: Optimization method
        preprocess: Whether to preprocess images
        create_visualizations: Whether to create visualizations
        learning_rate: Learning rate for optimizer
        number_of_iterations: Maximum iterations
        gradient_convergence_tolerance: Convergence tolerance

    Returns:
        Dictionary with registration results
    """
    tool = ImageRegistrationTool()

    # Load images
    fixed_image = tool.load_image(fixed_image_path)
    moving_image = tool.load_image(moving_image_path)

    # Preprocess if requested
    if preprocess:
        fixed_image = tool.preprocess_image(fixed_image)
        moving_image = tool.preprocess_image(moving_image)

    # Create transform and registration method
    transform = tool.create_affine_transform(fixed_image, moving_image)
    registration_method = tool.setup_registration_method(
        transform, metric, optimizer, learning_rate, number_of_iterations, gradient_convergence_tolerance
    )

    # Perform registration
    final_transform, registered_image = tool.register_images(fixed_image, moving_image, transform, registration_method)

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save registered image
    registered_path = os.path.join(output_dir, "affine_registered.nii.gz")
    tool.save_image(registered_image, registered_path)

    # Save transform
    transform_path = os.path.join(output_dir, "affine_transform.tfm")
    sitk.WriteTransform(final_transform, transform_path)

    # Calculate metrics
    metrics_before = tool.calculate_similarity_metrics(fixed_image, moving_image)
    metrics_after = tool.calculate_similarity_metrics(fixed_image, registered_image)

    results = {
        "registered_image_path": registered_path,
        "transform_path": transform_path,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "registration_type": "affine",
    }

    return results


def quick_deformable_registration(
    fixed_image_path: str,
    moving_image_path: str,
    output_dir: str,
    metric: str = "mutual_information",
    optimizer: str = "gradient_descent",
    preprocess: bool = True,
    create_visualizations: bool = True,
    learning_rate: float = 0.01,
    number_of_iterations: int = 100,
    gradient_convergence_tolerance: float = 1e-6,
    number_of_control_points: int = 4,
) -> dict:
    """
    Quick deformable registration function for Biomni integration.

    Args:
        fixed_image_path: Path to reference image
        moving_image_path: Path to image to register
        output_dir: Directory to save results
        metric: Similarity metric
        optimizer: Optimization method
        preprocess: Whether to preprocess images
        create_visualizations: Whether to create visualizations
        learning_rate: Learning rate for optimizer
        number_of_iterations: Maximum iterations
        gradient_convergence_tolerance: Convergence tolerance
        number_of_control_points: Number of B-spline control points

    Returns:
        Dictionary with registration results
    """
    tool = ImageRegistrationTool()

    # Load images
    fixed_image = tool.load_image(fixed_image_path)
    moving_image = tool.load_image(moving_image_path)

    # Preprocess if requested
    if preprocess:
        fixed_image = tool.preprocess_image(fixed_image)
        moving_image = tool.preprocess_image(moving_image)

    # Create transform and registration method
    transform = tool.create_deformable_transform(fixed_image, moving_image, number_of_control_points)
    registration_method = tool.setup_registration_method(
        transform, metric, optimizer, learning_rate, number_of_iterations, gradient_convergence_tolerance
    )

    # Perform registration
    final_transform, registered_image = tool.register_images(fixed_image, moving_image, transform, registration_method)

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save registered image
    registered_path = os.path.join(output_dir, "deformable_registered.nii.gz")
    tool.save_image(registered_image, registered_path)

    # Save transform
    transform_path = os.path.join(output_dir, "deformable_transform.tfm")
    sitk.WriteTransform(final_transform, transform_path)

    # Calculate metrics
    metrics_before = tool.calculate_similarity_metrics(fixed_image, moving_image)
    metrics_after = tool.calculate_similarity_metrics(fixed_image, registered_image)

    results = {
        "registered_image_path": registered_path,
        "transform_path": transform_path,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "registration_type": "deformable",
    }

    return results


def batch_register_images(
    fixed_image_path: str,
    moving_images_dir: str,
    output_dir: str,
    transform_type: str = "rigid",
    metric: str = "mutual_information",
    optimizer: str = "gradient_descent",
    preprocess: bool = True,
    create_visualizations: bool = True,
    learning_rate: float = 0.01,
    number_of_iterations: int = 100,
    gradient_convergence_tolerance: float = 1e-6,
) -> dict:
    """
    Batch registration of multiple images to a single reference.

    Args:
        fixed_image_path: Path to reference image
        moving_images_dir: Directory containing images to register
        output_dir: Directory to save results
        transform_type: Type of registration ('rigid', 'affine', 'deformable')
        metric: Similarity metric
        optimizer: Optimization method
        preprocess: Whether to preprocess images
        create_visualizations: Whether to create visualizations
        learning_rate: Learning rate for optimizer
        number_of_iterations: Maximum iterations
        gradient_convergence_tolerance: Convergence tolerance

    Returns:
        Dictionary with batch registration results
    """
    logger.info(f"Starting batch {transform_type} registration...")

    # Find all image files in the directory
    image_files = []
    for file in os.listdir(moving_images_dir):
        if any(file.endswith(ext) for ext in [".nii", ".nii.gz", ".nrrd", ".mha", ".mhd"]):
            image_files.append(os.path.join(moving_images_dir, file))

    if not image_files:
        raise ValueError(f"No image files found in {moving_images_dir}")

    logger.info(f"Found {len(image_files)} images to register")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each image
    results = {}
    ImageRegistrationTool()

    for i, moving_image_path in enumerate(image_files):
        logger.info(f"Processing {i + 1}/{len(image_files)}: {os.path.basename(moving_image_path)}")

        # Create individual output directory
        image_name = os.path.splitext(os.path.basename(moving_image_path))[0]
        if image_name.endswith(".nii"):
            image_name = os.path.splitext(image_name)[0]

        individual_output_dir = os.path.join(output_dir, f"registration_{image_name}")

        try:
            if transform_type == "rigid":
                result = quick_rigid_registration(
                    fixed_image_path,
                    moving_image_path,
                    individual_output_dir,
                    metric,
                    optimizer,
                    preprocess,
                    create_visualizations,
                    learning_rate,
                    number_of_iterations,
                    gradient_convergence_tolerance,
                )
            elif transform_type == "affine":
                result = quick_affine_registration(
                    fixed_image_path,
                    moving_image_path,
                    individual_output_dir,
                    metric,
                    optimizer,
                    preprocess,
                    create_visualizations,
                    learning_rate,
                    number_of_iterations,
                    gradient_convergence_tolerance,
                )
            elif transform_type == "deformable":
                result = quick_deformable_registration(
                    fixed_image_path,
                    moving_image_path,
                    individual_output_dir,
                    metric,
                    optimizer,
                    preprocess,
                    create_visualizations,
                    learning_rate,
                    number_of_iterations,
                    gradient_convergence_tolerance,
                )
            else:
                raise ValueError(f"Unknown transform type: {transform_type}")

            results[image_name] = result
            logger.info(f"Successfully registered {image_name}")

        except Exception as e:
            logger.error(f"Failed to register {image_name}: {e}")
            results[image_name] = {"error": str(e)}

    logger.info(f"Batch registration completed. Processed {len(image_files)} images.")
    return results


def calculate_similarity_metrics(image1_path: str, image2_path: str) -> dict[str, float]:
    """
    Calculate similarity metrics between two images.

    Args:
        image1_path: Path to first image
        image2_path: Path to second image

    Returns:
        Dictionary of similarity metrics
    """
    tool = ImageRegistrationTool()
    image1 = tool.load_image(image1_path)
    image2 = tool.load_image(image2_path)
    return tool.calculate_similarity_metrics(image1, image2)
