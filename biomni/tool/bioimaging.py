import logging
import os
import subprocess

import nibabel as nib
import numpy as np
import torch
import torch.serialization
from nnunet.inference.predict import predict_from_folder

# Apply safe globals for torch serialization
torch.serialization.add_safe_globals([tuple, list, dict, set, int, float, str, bytes, bytearray])
torch.serialization.add_safe_globals([complex, slice, range])
torch.serialization.add_safe_globals([np.core.multiarray.scalar])


def split_modalities(input_file, output_dir, case_name="BRAT"):
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


def prepare_input_for_nnunet(input_path, output_dir, case_name="BRAT"):
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
                    return split_modalities(input_path, output_dir, case_name)
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
                            return split_modalities(os.path.join(input_path, f), output_dir, case_name)
                    except Exception as e:  # ← FIX: no bare except
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


def setup_nnunet_environment(results_folder=None, raw_data_base=None, preprocessed=None):
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
    """
    Segment images using nnUNet with proper environment setup
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
    """
    if folds is None:
        folds = [0, 1, 2, 3, 4]
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Setup nnUNet environment first
    setup_nnunet_environment(results_folder=results_folder)

    # Prepare input data if requested
    if auto_prepare_input:
        temp_input_dir = os.path.join(output_dir, "temp_input")
        prepared_input_dir = prepare_input_for_nnunet(image_path, temp_input_dir)
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

    # Determine model directory using nnUNet environment variables
    model_folder = None

    # Use the nnUNet_RESULTS_FOLDER environment variable
    results_folder = os.environ.get("nnUNet_RESULTS_FOLDER")
    if results_folder:
        model_path = os.path.join(results_folder, model_type, task_id, "nnUNetTrainer__nnUNetPlansv2.1")
        if os.path.exists(model_path):
            model_folder = model_path
            logging.info(f"Using model from nnUNet_RESULTS_FOLDER: {model_folder}")

    # If not found, try to find it in common locations
    if not model_folder:
        common_paths = [
            "./models/nnUNet",
            "~/nnUNet_results",
            "~/biomni_models/nnUNet",
        ]

        for base_path in common_paths:
            expanded_path = os.path.expanduser(base_path)
            model_path = expanded_path
            # model_path = os.path.join(expanded_path, model_type, task_id, "nnUNetTrainer__nnUNetPlansv2.1")
            if os.path.exists(model_path):
                model_folder = model_path
                logging.info(f"Using model from common path: {model_folder}")
                # Update environment variable to match found path
                os.environ["nnUNet_RESULTS_FOLDER"] = os.path.dirname(os.path.dirname(os.path.dirname(model_folder)))
                break

    # If still no model found, ask user for path
    if model_folder is None:
        logging.warning("No model found in common locations. Please specify the results folder path.")
        user_results_folder = input("Please enter the path to your nnUNet results folder: ").strip()
        if user_results_folder:
            os.environ["nnUNet_RESULTS_FOLDER"] = os.path.expanduser(user_results_folder)
            model_path = os.path.join(user_results_folder, model_type, task_id, "nnUNetTrainer__nnUNetPlansv2.1")
            if os.path.exists(model_path):
                model_folder = model_path
                logging.info(f"Using model from user-specified path: {model_folder}")
            else:
                raise RuntimeError(f"Model not found at {model_path}")
        else:
            raise RuntimeError("No results folder specified and no default models found")

    # Check if model weights exist
    if not os.path.exists(model_folder):
        user_input = (
            input(f"Model weights for {task_id} not found. Do you want to download them? (y/n): ").strip().lower()
        )
        if user_input == "y":
            try:
                subprocess.run(f"nnUNet_download_pretrained_model {task_id}", shell=True, check=True)
                logging.info(f"Downloaded pretrained model for {task_id} successfully.")
            except subprocess.CalledProcessError as e:
                # FIX: re-raise with context (ruff B904)
                raise RuntimeError(f"Failed to download pretrained model: {e}") from e
        else:
            raise RuntimeError("Model weights not found and download declined by user.")

    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    # Temporarily patch torch.load for inference only
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


def create_segmentation_visualization(original_mri, segmentation, output_dir="./visualization_output"):
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