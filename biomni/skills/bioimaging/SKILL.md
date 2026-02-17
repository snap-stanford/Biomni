---
name: bioimaging
description: Process and analyze biological images, including microscopy and medical imaging pipelines.
---

## Tools
- **split_modalities**: Split a 4D NIfTI file into separate modality files for nnUNet processing. Handles BRATS dataset format with FLAIR, T1w, t1gd, and T2w modalities.
- **prepare_input_for_nnunet**: Prepare input data for nnUNet by handling both 4D and pre-split modality files. Automatically detects file format and prepares data accordingly.
- **segment_with_nn_unet**: Segment images using nnUNet with proper environment setup. Supports brain tumor segmentation and other medical image segmentation tasks.
- **create_segmentation_visualization**: Create and save visualization of segmentation results using nilearn. Generates overlay plots and multiple anatomical views.
- **quick_rigid_registration**: Perform rigid image registration between two medical images using SimpleITK. Rigid registration handles translation and rotation only, preserving shape and size. Includes preprocessing, similarity metrics calculation, and visualization generation.
- **quick_affine_registration**: Perform affine image registration between two medical images using SimpleITK. Affine registration handles translation, rotation, scaling, and shearing. More flexible than rigid registration but still preserves parallel lines.
- **quick_deformable_registration**: Perform deformable (B-spline) image registration between two medical images using SimpleITK. Deformable registration allows for local non-linear transformations, handling complex deformations. Most flexible but computationally intensive registration method.
- **batch_register_images**: Perform batch registration of multiple images to a single reference image. Automatically processes all medical image files in a directory and registers them to the fixed reference. Supports rigid, affine, or deformable registration for all images.
- **calculate_similarity_metrics**: Calculate similarity metrics between two medical images. Supports mutual information, mean squared error, correlation, and normalized cross correlation.
- **create_registration_visualization**: Create visualization plots for registration results. Generates comparison plots, difference images, overlays, and metric charts.
