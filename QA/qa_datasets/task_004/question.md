# Task: Cell Counting

Please count the number of cells in the provided cell image.

## Requirements:
- **Use the `segment_cells_with_deep_learning` function** from `biomni.tool.microbiology` to perform cell segmentation
  - This function uses deep learning (Cellpose) for accurate cell detection
  - Use `model_type='cyto'` for cell segmentation
  - The function will automatically create the `segmentation_results/` directory and save the output files
- Analyze the segmentation results and count the total number of cells
- Calculate and report the average cell area (in pixels) from the segmentation masks
- **Important**: The visualization image must be saved as `segmentation_results/outlines_cell.png`
  - The `segment_cells_with_deep_learning` function automatically saves this file when you use `save_dir='segmentation_results'` and the input image is `cell.png`
  - This image shows the cell boundaries/outlines overlaid on the original image