from __future__ import annotations


def predict_protein_disorder_regions(protein_sequence, threshold=0.5, output_file="disorder_prediction_results.csv"):
    """Predicts intrinsically disordered regions (IDRs) in a protein sequence using IUPred2A.

    Parameters
    ----------
    protein_sequence : str
        The amino acid sequence of the protein to analyze
    threshold : float, optional
        The disorder score threshold above which a residue is considered disordered (default: 0.5)
    output_file : str, optional
        Filename to save the per-residue disorder scores (default: "disorder_prediction_results.csv")

    Returns
    -------
    str
        A research log summarizing the prediction process and results

    """
    import csv
    import re

    import requests

    # Clean the input sequence
    protein_sequence = "".join(re.findall(r"[A-Za-z]", protein_sequence))

    # Step 1: Submit the sequence to IUPred2A web server
    url = "https://iupred2a.elte.hu/iupred2a"
    payload = {
        "seq": protein_sequence,
        "iupred2": "long",  # Use IUPred2 long disorder prediction
        "anchor2": "no",  # Don't use ANCHOR2 prediction
    }

    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error accessing IUPred2A server: {str(e)}"

    # Step 2: Parse the results to extract disorder scores
    result_lines = response.text.split("\n")
    scores = []

    for line in result_lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                position = int(parts[0])
                residue = parts[1]
                score = float(parts[2])
                scores.append((position, residue, score))
            except (ValueError, IndexError):
                continue

    if not scores:
        return "No valid prediction data was returned from the server."

    # Step 3: Identify disordered regions
    disordered_regions = []
    current_region = []

    for pos, _, score in scores:
        if score >= threshold:
            if not current_region:
                current_region = [pos]
            elif pos == current_region[-1] + 1:
                current_region.append(pos)
            else:
                if len(current_region) > 1:
                    disordered_regions.append((current_region[0], current_region[-1]))
                current_region = [pos]
        elif current_region and len(current_region) > 1:
            disordered_regions.append((current_region[0], current_region[-1]))
            current_region = []
        elif current_region:
            current_region = []

    # Add the last region if it exists
    if current_region and len(current_region) > 1:
        disordered_regions.append((current_region[0], current_region[-1]))

    # Step 4: Save results to CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Position", "Amino_Acid", "Disorder_Score", "Is_Disordered"])
        for pos, aa, score in scores:
            is_disordered = "Yes" if score >= threshold else "No"
            writer.writerow([pos, aa, score, is_disordered])

    # Step 5: Generate the research log
    total_residues = len(scores)
    disordered_count = sum(1 for _, _, score in scores if score >= threshold)
    disordered_percentage = (disordered_count / total_residues) * 100 if total_residues > 0 else 0

    log = f"""
Intrinsically Disordered Region (IDR) Prediction Research Log:
=============================================================
Analysis performed using IUPred2A algorithm (long disorder mode)
Protein sequence length: {total_residues} amino acids
Disorder threshold: {threshold}

Results Summary:
- {disordered_count} residues ({disordered_percentage:.2f}%) predicted as disordered
- {len(disordered_regions)} distinct disordered regions identified

Disordered Regions:
"""

    if disordered_regions:
        for start, end in disordered_regions:
            length = end - start + 1
            log += f"- Region {start}-{end} (length: {length} residues)\n"
    else:
        log += "- No significant disordered regions found\n"

    log += f"\nDetailed per-residue scores saved to: {output_file}"

    return log


def analyze_cell_morphology_and_cytoskeleton(image_path, output_dir="./results", threshold_method="otsu"):
    """Quantifies cell morphology and cytoskeletal organization from fluorescence microscopy images.

    Parameters
    ----------
    image_path : str
        Path to the fluorescence microscopy image file
    output_dir : str, optional
        Directory to save output files (default: './results')
    threshold_method : str, optional
        Method for cell segmentation ('otsu', 'adaptive', or 'manual') (default: 'otsu')

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import os
    from datetime import datetime

    import cv2
    import numpy as np
    import pandas as pd
    from skimage import exposure, feature, filters, io, measure, morphology
    from skimage.color import rgb2gray

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Start log
    log = f"Cell Morphology and Cytoskeleton Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log += f"Analyzing image: {image_path}\n\n"

    # Load image
    log += "Step 1: Loading and preprocessing image\n"
    try:
        image = io.imread(image_path)
        # Convert to grayscale if RGB
        if len(image.shape) > 2:
            gray_image = rgb2gray(image)
            log += "- Converted RGB image to grayscale\n"
        else:
            gray_image = image

        # Enhance contrast
        gray_image = exposure.equalize_hist(gray_image)
        log += "- Enhanced image contrast\n"
    except Exception as e:
        return f"Error loading image: {str(e)}"

    # Segment cells
    log += "\nStep 2: Segmenting cells from background\n"
    if threshold_method == "otsu":
        thresh = filters.threshold_otsu(gray_image)
        binary = gray_image > thresh
        log += f"- Applied Otsu thresholding (threshold value: {thresh:.4f})\n"
    elif threshold_method == "adaptive":
        binary = filters.threshold_local(gray_image, block_size=35, offset=0.05)
        binary = gray_image > binary
        log += "- Applied adaptive thresholding\n"
    else:  # manual
        thresh = 0.5  # Default value, can be parameterized
        binary = gray_image > thresh
        log += f"- Applied manual thresholding (threshold value: {thresh:.4f})\n"

    # Clean up binary image
    binary = morphology.remove_small_objects(binary, min_size=100)
    binary = morphology.remove_small_holes(binary, area_threshold=100)
    binary = morphology.binary_closing(binary, morphology.disk(3))
    log += "- Applied morphological operations to clean segmentation\n"

    # Label cells
    labeled_cells, num_cells = measure.label(binary, return_num=True)
    log += f"- Identified {num_cells} cell regions\n"

    # Analyze cell properties
    log += "\nStep 3: Analyzing cell morphology\n"
    cell_props = measure.regionprops_table(
        labeled_cells,
        gray_image,
        properties=(
            "area",
            "perimeter",
            "major_axis_length",
            "minor_axis_length",
            "eccentricity",
            "orientation",
            "solidity",
        ),
    )

    # Convert to DataFrame for easier manipulation
    cell_df = pd.DataFrame(cell_props)

    # Calculate additional metrics
    if len(cell_df) > 0:
        cell_df["aspect_ratio"] = cell_df["major_axis_length"] / cell_df["minor_axis_length"]
        cell_df["circularity"] = (4 * np.pi * cell_df["area"]) / (cell_df["perimeter"] ** 2)

        # Summary statistics
        log += f"- Average cell area: {cell_df['area'].mean():.2f} pixels\n"
        log += f"- Average aspect ratio: {cell_df['aspect_ratio'].mean():.2f}\n"
        log += f"- Average circularity: {cell_df['circularity'].mean():.2f}\n"
        log += f"- Average eccentricity: {cell_df['eccentricity'].mean():.2f}\n"
    else:
        log += "- No cells detected for morphological analysis\n"

    # Analyze cytoskeletal organization
    log += "\nStep 4: Analyzing cytoskeletal organization\n"

    # Edge detection to highlight cytoskeletal fibers
    edges = feature.canny(gray_image, sigma=2)

    # Use Hough transform to detect lines (cytoskeletal fibers)
    if np.any(edges):
        lines = cv2.HoughLinesP(
            edges.astype(np.uint8),
            1,
            np.pi / 180,
            threshold=10,
            minLineLength=10,
            maxLineGap=5,
        )

        if lines is not None:
            # Calculate line orientations
            orientations = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:  # Avoid division by zero
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    orientations.append(angle)

            if orientations:
                # Convert to numpy array for calculations
                orientations = np.array(orientations)

                # Calculate alignment metrics
                mean_orientation = np.mean(orientations)
                # Normalize angles to -90 to 90 degrees
                norm_angles = np.mod(orientations + 90, 180) - 90
                std_orientation = np.std(norm_angles)

                # Order parameter (measure of alignment, 1 = perfectly aligned, 0 = random)
                # Convert angles to radians for calculation
                rad_angles = np.radians(norm_angles)
                order_parameter = np.sqrt(np.mean(np.cos(2 * rad_angles)) ** 2 + np.mean(np.sin(2 * rad_angles)) ** 2)

                log += f"- Detected {len(orientations)} cytoskeletal fibers\n"
                log += f"- Mean fiber orientation: {mean_orientation:.2f} degrees\n"
                log += f"- Standard deviation of orientation: {std_orientation:.2f} degrees\n"
                log += f"- Order parameter (alignment): {order_parameter:.4f} (0=random, 1=aligned)\n"

                # Add fiber data to dataframe
                fiber_df = pd.DataFrame({"fiber_orientation": orientations})
            else:
                log += "- No fiber orientations could be calculated\n"
                fiber_df = pd.DataFrame()
        else:
            log += "- No cytoskeletal fibers detected\n"
            fiber_df = pd.DataFrame()
    else:
        log += "- No edges detected for cytoskeletal analysis\n"
        fiber_df = pd.DataFrame()

    # Save results
    log += "\nStep 5: Saving results\n"

    # Save cell morphology data
    if len(cell_df) > 0:
        cell_csv_path = os.path.join(output_dir, "cell_morphology_data.csv")
        cell_df.to_csv(cell_csv_path, index=False)
        log += f"- Cell morphology data saved to: {cell_csv_path}\n"

    # Save fiber orientation data
    if len(fiber_df) > 0:
        fiber_csv_path = os.path.join(output_dir, "fiber_orientation_data.csv")
        fiber_df.to_csv(fiber_csv_path, index=False)
        log += f"- Fiber orientation data saved to: {fiber_csv_path}\n"

    # Save segmentation image
    if num_cells > 0:
        segmentation_path = os.path.join(output_dir, "cell_segmentation.png")
        io.imsave(segmentation_path, labeled_cells.astype(np.uint8) * 50)
        log += f"- Cell segmentation image saved to: {segmentation_path}\n"

    # Summary
    log += "\nAnalysis Summary:\n"
    log += f"- Processed image: {image_path}\n"
    log += f"- Detected {num_cells} cells\n"
    if len(cell_df) > 0:
        log += f"- Cell size range: {cell_df['area'].min():.1f} to {cell_df['area'].max():.1f} pixels\n"
        log += f"- Cell shape: average aspect ratio = {cell_df['aspect_ratio'].mean():.2f}\n"
    if "order_parameter" in locals():
        log += f"- Cytoskeletal organization: alignment parameter = {order_parameter:.4f}\n"

    return log


def analyze_tissue_deformation_flow(image_sequence, output_dir="results", pixel_scale=1.0):
    """Quantify tissue deformation and flow dynamics from microscopy image sequence.

    Parameters
    ----------
    image_sequence : list or numpy.ndarray
        Sequence of microscopy images (either a list of file paths or a 3D numpy array [time, height, width])
    output_dir : str, optional
        Directory to save results (default: "results")
    pixel_scale : float, optional
        Physical scale of pixels (e.g., μm/pixel) for proper scaling of metrics (default: 1.0)

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    """
    import os
    from datetime import datetime

    import cv2
    import numpy as np

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load images if paths are provided
    if isinstance(image_sequence[0], str):
        loaded_images = []
        for img_path in image_sequence:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return f"Error: Could not load image {img_path}"
            loaded_images.append(img)
        frames = np.array(loaded_images)
    else:
        frames = image_sequence
        # Convert to grayscale if needed
        if len(frames.shape) > 3:  # Has color channels
            frames = np.array(
                [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.shape[-1] == 3 else frame for frame in frames]
            )

    # Parameters for optical flow
    lk_params = {
        "winSize": (15, 15),
        "maxLevel": 2,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    }

    # Metrics storage
    num_frames = len(frames)
    flow_fields = []
    divergence_maps = []
    curl_maps = []
    strain_maps = []

    # Log initialization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = f"Tissue Deformation and Flow Analysis - {timestamp}\n"
    log += f"Number of frames analyzed: {num_frames}\n"
    log += f"Image dimensions: {frames[0].shape[0]}x{frames[0].shape[1]} pixels\n"
    log += f"Pixel scale: {pixel_scale} units/pixel\n\n"
    log += "Analysis Steps:\n"

    # Create feature points grid (evenly spaced points)
    y, x = np.mgrid[0 : frames[0].shape[0] : 20, 0 : frames[0].shape[1] : 20]
    feature_points = np.stack((x.flatten(), y.flatten()), axis=1).astype(np.float32)

    for i in range(num_frames - 1):
        log += f"Processing frame pair {i} and {i + 1}...\n"

        # Calculate optical flow using Lucas-Kanade
        prev_frame = frames[i]
        next_frame = frames[i + 1]

        # Compute flow for the grid points
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, feature_points, None, **lk_params)

        # Filter valid points
        valid_idx = status.flatten() == 1
        valid_prev_points = feature_points[valid_idx]
        valid_next_points = next_points[valid_idx]

        # Calculate displacement vectors
        displacement = valid_next_points - valid_prev_points
        points = valid_prev_points

        # Interpolate flow field to full image resolution
        flow_field = np.zeros((frames[0].shape[0], frames[0].shape[1], 2), dtype=np.float32)

        # Simple nearest-neighbor interpolation for demonstration
        # In a production system, you might use more sophisticated interpolation
        for j, (x, y) in enumerate(points.astype(int)):
            if 0 <= y < flow_field.shape[0] and 0 <= x < flow_field.shape[1]:
                flow_field[y, x] = displacement[j]

        # Calculate derivatives for deformation analysis
        u = flow_field[:, :, 0]  # x-component of flow
        v = flow_field[:, :, 1]  # y-component of flow

        # Calculate divergence (expansion/contraction)
        # Using central differences for derivatives
        u_x = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3) / (8.0 * pixel_scale)
        v_y = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=3) / (8.0 * pixel_scale)
        divergence = u_x + v_y

        # Calculate curl (rotation)
        u_y = cv2.Sobel(u, cv2.CV_64F, 0, 1, ksize=3) / (8.0 * pixel_scale)
        v_x = cv2.Sobel(v, cv2.CV_64F, 1, 0, ksize=3) / (8.0 * pixel_scale)
        curl = v_x - u_y

        # Calculate strain tensor components
        strain_xx = u_x
        strain_yy = v_y
        strain_xy = 0.5 * (u_y + v_x)

        # Magnitude of strain tensor (Frobenius norm)
        strain_magnitude = np.sqrt(strain_xx**2 + strain_yy**2 + 2 * strain_xy**2)

        # Store results
        flow_fields.append(flow_field)
        divergence_maps.append(divergence)
        curl_maps.append(curl)
        strain_maps.append(strain_magnitude)

        # Visualize flow field
        flow_viz = np.zeros((frames[0].shape[0], frames[0].shape[1], 3), dtype=np.uint8)
        flow_viz[..., 0] = next_frame  # Use next frame as background
        flow_viz[..., 1] = next_frame
        flow_viz[..., 2] = next_frame

        # Draw flow vectors for visualization
        step = 20
        for y in range(0, flow_field.shape[0], step):
            for x in range(0, flow_field.shape[1], step):
                dx, dy = flow_field[y, x]
                if abs(dx) > 0.5 or abs(dy) > 0.5:  # Only draw significant flow
                    cv2.arrowedLine(
                        flow_viz,
                        (x, y),
                        (int(x + dx), int(y + dy)),
                        (0, 255, 0),  # Green
                        1,
                        tipLength=0.3,
                    )

        # Save visualizations
        flow_viz_path = os.path.join(output_dir, f"flow_viz_{i:03d}.png")
        divergence_path = os.path.join(output_dir, f"divergence_{i:03d}.npy")
        curl_path = os.path.join(output_dir, f"curl_{i:03d}.npy")
        strain_path = os.path.join(output_dir, f"strain_{i:03d}.npy")

        cv2.imwrite(flow_viz_path, flow_viz)
        np.save(divergence_path, divergence)
        np.save(curl_path, curl)
        np.save(strain_path, strain_magnitude)

    # Calculate summary statistics
    mean_divergence = np.mean([np.mean(div) for div in divergence_maps])
    max_divergence = np.max([np.max(div) for div in divergence_maps])
    mean_curl = np.mean([np.mean(np.abs(c)) for c in curl_maps])
    mean_strain = np.mean([np.mean(s) for s in strain_maps])

    # Add summary to log
    log += "\nAnalysis Results:\n"
    log += f"Mean tissue divergence: {mean_divergence:.6f} (expansion/contraction rate)\n"
    log += f"Maximum divergence: {max_divergence:.6f}\n"
    log += f"Mean absolute curl: {mean_curl:.6f} (rotation rate)\n"
    log += f"Mean strain magnitude: {mean_strain:.6f} (deformation intensity)\n\n"

    # Save summary data
    summary_data = {
        "mean_divergence": mean_divergence,
        "max_divergence": max_divergence,
        "mean_curl": mean_curl,
        "mean_strain": mean_strain,
    }
    summary_path = os.path.join(output_dir, "deformation_summary.npy")
    np.save(summary_path, summary_data)

    log += "Files Generated:\n"
    log += f"- Flow visualization images: {output_dir}/flow_viz_*.png\n"
    log += f"- Divergence maps: {output_dir}/divergence_*.npy\n"
    log += f"- Curl maps: {output_dir}/curl_*.npy\n"
    log += f"- Strain maps: {output_dir}/strain_*.npy\n"
    log += f"- Summary statistics: {output_dir}/deformation_summary.npy\n"

    return log


# ---------------------------------------------------------------------------
# Internal helpers for FRAP condensate analysis
# ---------------------------------------------------------------------------
import json
import os
from typing import Any

import numpy as np


def _circular_mask(shape: tuple[int, int], cx: float, cy: float, r: float) -> np.ndarray:
    """Boolean mask for a circular ROI centered at (cx, cy) with radius r."""
    Y, X = np.ogrid[: shape[0], : shape[1]]
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r**2


def _soumpasis_recovery(t, F0, Mf, tau_D):
    """Soumpasis 1983 model for 2D diffusion in a uniform circular bleach spot.

    F(t) = F0 + Mf * (1 - F0) * gamma(tau_D, t)
    where gamma(tau_D, t) = exp(-2*tau_D/t) * (I0(2*tau_D/t) + I1(2*tau_D/t))

    `gamma` rises monotonically from 0 at t=0+ to 1 as t -> infinity.
    `tau_D` is the characteristic diffusion time; D = w^2 / (4 * tau_D)
    where w is the bleach radius.
    """
    from scipy.special import ive  # exponentially scaled modified Bessel

    t = np.asarray(t, dtype=float)
    safe_t = np.where(t > 0, t, np.finfo(float).eps)
    arg = 2.0 * tau_D / safe_t
    # ive(n, x) = iv(n, x) * exp(-|x|), so this directly gives exp(-arg)*(I0+I1)
    gamma = ive(0, arg) + ive(1, arg)
    gamma = np.where(t > 0, gamma, 0.0)
    return F0 + Mf * (1.0 - F0) * gamma


def _exponential_recovery(t, F0, Mf, tau):
    """Single-exponential phenomenological recovery."""
    t = np.asarray(t, dtype=float)
    return F0 + Mf * (1.0 - F0) * (1.0 - np.exp(-t / tau))


def _double_exponential_recovery(t, F0, A1, tau1, A2, tau2):
    """Two-component exponential recovery (slow + fast pools).

    A1 + A2 is the effective mobile fraction (A1, A2 each in [0, 1]).
    """
    t = np.asarray(t, dtype=float)
    return F0 + A1 * (1.0 - np.exp(-t / tau1)) + A2 * (1.0 - np.exp(-t / tau2))


def _solve_t_half_soumpasis(tau_D: float) -> float:
    """Numerically solve gamma(tau_D, t) = 0.5 for t."""
    from scipy.optimize import brentq
    from scipy.special import ive

    def f(x):
        arg = 2.0 * tau_D / x
        return ive(0, arg) + ive(1, arg) - 0.5

    try:
        return float(brentq(f, 1e-9 * tau_D, 1e6 * tau_D))
    except Exception:
        # Closed-form approximation: t_half ~= tau_D for Soumpasis 2D.
        return float(tau_D)


def _initial_guesses(t: np.ndarray, F: np.ndarray) -> dict[str, float]:
    """Sensible starting parameters from the data."""
    F0 = float(F[0])
    plateau = float(np.median(F[-max(3, len(F) // 10) :]))
    Mf = max(0.01, min(1.2, (plateau - F0) / max(1e-6, 1.0 - F0)))
    midpoint = (F0 + plateau) / 2.0
    idx_half = int(np.argmin(np.abs(F - midpoint)))
    t_half = max(t[idx_half] - t[0], (t[-1] - t[0]) / 10.0)
    return {"F0": F0, "Mf": Mf, "tau": t_half, "plateau": plateau, "t_half": t_half}


def extract_frap_curve_from_image_stack(
    image_stack_path: str,
    bleach_roi: list,
    reference_roi: list = None,
    background_roi: list = None,
    bleach_frame_index: int = None,
    frame_interval_s: float = 1.0,
    output_folder: str = "./tmp/",
) -> str:
    """Extract a normalized FRAP recovery curve from a TIFF time-series.

    Performs background subtraction (if a background ROI is supplied) and
    reference-ROI photobleaching correction (if a reference ROI is supplied),
    then double-normalizes against the pre-bleach mean intensity in the
    bleach ROI. Writes both the full curve and the post-bleach-only curve
    as CSV.

    Args:
        image_stack_path: Path to a multi-frame TIFF (T, Y, X) or (T, C, Y, X).
            For multi-channel stacks, channel 0 is used.
        bleach_roi: [x_center_px, y_center_px, radius_px] of the bleached spot.
        reference_roi: Optional [x, y, r] for an unbleached reference region
            used to correct ongoing acquisition photobleaching.
        background_roi: Optional [x, y, r] outside any cell, for camera /
            background subtraction.
        bleach_frame_index: Index of the first post-bleach frame. If None,
            auto-detected as the frame with the largest intensity drop.
        frame_interval_s: Time between successive frames, seconds.
        output_folder: Directory for CSV outputs (created if missing).

    Returns:
        Log string describing each step and listing output paths.
    """
    log: list[str] = []
    os.makedirs(output_folder, exist_ok=True)
    log.append(f"Loading image stack from {image_stack_path}")

    try:
        import tifffile
    except ImportError:
        return "ERROR: tifffile is required. Install with `pip install tifffile`."

    try:
        stack = tifffile.imread(image_stack_path)
    except Exception as exc:
        return f"ERROR: failed to read TIFF stack: {exc}"

    if stack.ndim < 3:
        return "ERROR: image stack must have at least 3 dimensions (T, Y, X)."
    if stack.ndim == 4:
        log.append("Multi-channel stack detected; using channel 0.")
        stack = stack[:, 0, :, :]
    if stack.ndim != 3:
        return f"ERROR: unsupported stack shape {stack.shape}; expected (T, Y, X)."

    n_frames, ny, nx = stack.shape
    log.append(f"Stack shape: {n_frames} frames x {ny} x {nx}.")

    cx, cy, r = bleach_roi
    if not (0 <= cx < nx and 0 <= cy < ny and r > 0):
        return f"ERROR: bleach_roi {bleach_roi} out of image bounds {(nx, ny)}."
    bleach_mask = _circular_mask((ny, nx), cx, cy, r)
    log.append(f"Bleach ROI: center=({cx}, {cy}), radius={r} px, area={int(bleach_mask.sum())} px.")

    bleach_intensity = np.array([frame[bleach_mask].mean() for frame in stack], dtype=float)
    raw_bleach = bleach_intensity.copy()

    bg_intensity = None
    if background_roi is not None:
        bg_mask = _circular_mask((ny, nx), background_roi[0], background_roi[1], background_roi[2])
        bg_intensity = np.array([frame[bg_mask].mean() for frame in stack], dtype=float)
        bleach_intensity = bleach_intensity - bg_intensity
        log.append("Applied background subtraction.")

    if reference_roi is not None:
        ref_mask = _circular_mask((ny, nx), reference_roi[0], reference_roi[1], reference_roi[2])
        ref_intensity = np.array([frame[ref_mask].mean() for frame in stack], dtype=float)
        if bg_intensity is not None:
            ref_intensity = ref_intensity - bg_intensity
        ref_safe = np.where(ref_intensity > 0, ref_intensity, np.finfo(float).eps)
        bleach_intensity = bleach_intensity / ref_safe
        log.append("Applied reference-ROI bleach correction (divided by reference).")

    if bleach_frame_index is None:
        diffs = np.diff(bleach_intensity)
        bleach_frame_index = int(np.argmin(diffs)) + 1
        log.append(f"Auto-detected bleach frame index: {bleach_frame_index}.")

    if not (1 <= bleach_frame_index < n_frames):
        return f"ERROR: bleach_frame_index={bleach_frame_index} out of range for {n_frames}-frame stack."

    pre_bleach_mean = float(np.mean(bleach_intensity[:bleach_frame_index]))
    if pre_bleach_mean <= 0:
        return "ERROR: pre-bleach mean intensity is non-positive; check ROIs and background."
    log.append(f"Pre-bleach mean intensity: {pre_bleach_mean:.4f}.")

    intensity_norm = bleach_intensity / pre_bleach_mean
    times = (np.arange(n_frames) - bleach_frame_index) * frame_interval_s

    # Sanity check on the post-bleach drop: should be < 1.0 by a meaningful amount.
    F0 = float(intensity_norm[bleach_frame_index])
    if F0 > 0.95:
        log.append(
            f"WARNING: post-bleach intensity is {F0:.3f} (≥ 0.95). Bleach event may not have been detected correctly."
        )

    import pandas as pd

    full_df = pd.DataFrame(
        {
            "frame": np.arange(n_frames),
            "time_s": times,
            "intensity_raw": raw_bleach,
            "intensity_norm": intensity_norm,
        }
    )
    full_csv = os.path.join(output_folder, "frap_curve_full.csv")
    full_df.to_csv(full_csv, index=False)

    post_df = full_df.iloc[bleach_frame_index:].reset_index(drop=True)
    post_csv = os.path.join(output_folder, "frap_curve_postbleach.csv")
    post_df[["time_s", "intensity_norm"]].to_csv(post_csv, index=False)

    log.append(f"Wrote full curve ({n_frames} frames) to {full_csv}.")
    log.append(
        f"Wrote post-bleach curve ({len(post_df)} frames) to {post_csv}; "
        f"this is the input expected by `fit_frap_recovery_curve`."
    )
    log.append(f"Post-bleach intensity F0 = {F0:.4f} (fraction of pre-bleach).")

    return "\n".join(log)


def fit_frap_recovery_curve(
    curve_csv_path: str,
    bleach_radius_um: float,
    fit_model: str = "soumpasis",
    initial_guess: dict = None,
    output_folder: str = "./tmp/",
) -> str:
    """Fit a normalized FRAP recovery curve and extract biophysical parameters.

    The input CSV must contain columns `time_s` and `intensity_norm`, with
    t=0 at the first post-bleach frame and intensity normalized to the
    pre-bleach mean (which is the output of `extract_frap_curve_from_image_stack`).

    Args:
        curve_csv_path: Path to the post-bleach CSV.
        bleach_radius_um: Radius of the bleached spot in micrometers, used
            to convert tau_D to a diffusion coefficient (Soumpasis only).
        fit_model: One of "soumpasis", "exponential", "double_exponential".
        initial_guess: Optional dict with starting parameters; otherwise
            reasonable values are inferred from the data.
        output_folder: Where to write the fit plot and parameters JSON.

    Returns:
        Log string with the fitted parameters, R^2, RMSE, and output paths.
    """
    log: list[str] = []
    os.makedirs(output_folder, exist_ok=True)

    try:
        import matplotlib
        import pandas as pd
        from scipy.optimize import curve_fit

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return f"ERROR: missing dependency ({exc})."

    log.append(f"Loading FRAP curve from {curve_csv_path}")
    try:
        df = pd.read_csv(curve_csv_path)
    except Exception as exc:
        return f"ERROR: failed to read CSV: {exc}"

    if not {"time_s", "intensity_norm"}.issubset(df.columns):
        return "ERROR: CSV must contain columns 'time_s' and 'intensity_norm'."

    t = df["time_s"].to_numpy(dtype=float)
    F = df["intensity_norm"].to_numpy(dtype=float)
    if len(t) < 5:
        return "ERROR: need at least 5 post-bleach data points to fit."
    if not np.all(np.diff(t) > 0):
        return "ERROR: time_s must be strictly increasing."

    g = _initial_guesses(t, F)
    log.append(
        f"Data F0 ≈ {g['F0']:.3f}, plateau ≈ {g['plateau']:.3f}, "
        f"empirical t-half ≈ {g['t_half']:.2f} s, initial Mf ≈ {g['Mf']:.3f}."
    )

    params: dict[str, Any] = {"model": fit_model, "bleach_radius_um": float(bleach_radius_um)}

    try:
        if fit_model == "soumpasis":
            p0 = (initial_guess or {}).get("p0") or [g["F0"], g["Mf"], g["t_half"]]
            bounds = ([0.0, 0.0, 1e-6], [1.0, 1.5, np.inf])
            popt, pcov = curve_fit(_soumpasis_recovery, t, F, p0=p0, bounds=bounds, maxfev=20000)
            F0_fit, Mf_fit, tau_D_fit = popt
            perr = np.sqrt(np.diag(pcov)) * 1.96
            D_fit = (bleach_radius_um**2) / (4.0 * tau_D_fit)
            t_half = _solve_t_half_soumpasis(tau_D_fit)
            params.update(
                F0=float(F0_fit),
                F0_ci95=float(perr[0]),
                mobile_fraction=float(Mf_fit),
                mobile_fraction_ci95=float(perr[1]),
                tau_D_s=float(tau_D_fit),
                tau_D_ci95_s=float(perr[2]),
                diffusion_coefficient_um2_per_s=float(D_fit),
                t_half_s=float(t_half),
            )
            F_pred = _soumpasis_recovery(t, *popt)

        elif fit_model == "exponential":
            p0 = (initial_guess or {}).get("p0") or [g["F0"], g["Mf"], g["tau"]]
            bounds = ([0.0, 0.0, 1e-6], [1.0, 1.5, np.inf])
            popt, pcov = curve_fit(_exponential_recovery, t, F, p0=p0, bounds=bounds, maxfev=20000)
            F0_fit, Mf_fit, tau_fit = popt
            perr = np.sqrt(np.diag(pcov)) * 1.96
            params.update(
                F0=float(F0_fit),
                F0_ci95=float(perr[0]),
                mobile_fraction=float(Mf_fit),
                mobile_fraction_ci95=float(perr[1]),
                tau_s=float(tau_fit),
                tau_ci95_s=float(perr[2]),
                t_half_s=float(tau_fit * np.log(2.0)),
            )
            F_pred = _exponential_recovery(t, *popt)

        elif fit_model == "double_exponential":
            # Default split: half the mobile fraction in each pool, tau1 fast, tau2 slow.
            p0 = (initial_guess or {}).get("p0") or [
                g["F0"],
                max(g["Mf"] / 2.0, 0.05),
                max(g["tau"] / 3.0, 1e-3),
                max(g["Mf"] / 2.0, 0.05),
                max(g["tau"] * 3.0, 1e-2),
            ]
            bounds = ([0.0, 0.0, 1e-6, 0.0, 1e-6], [1.0, 1.0, np.inf, 1.0, np.inf])
            popt, pcov = curve_fit(_double_exponential_recovery, t, F, p0=p0, bounds=bounds, maxfev=20000)
            F0_fit, A1, tau1, A2, tau2 = popt
            # Enforce tau1 < tau2 for reportability.
            if tau1 > tau2:
                A1, A2 = A2, A1
                tau1, tau2 = tau2, tau1
            perr = np.sqrt(np.diag(pcov)) * 1.96
            params.update(
                F0=float(F0_fit),
                F0_ci95=float(perr[0]),
                amplitude_fast=float(A1),
                tau_fast_s=float(tau1),
                amplitude_slow=float(A2),
                tau_slow_s=float(tau2),
                mobile_fraction=float(A1 + A2),
            )
            F_pred = _double_exponential_recovery(t, F0_fit, A1, tau1, A2, tau2)

        else:
            return (
                f"ERROR: unknown fit_model '{fit_model}'. Choose 'soumpasis', 'exponential', or 'double_exponential'."
            )

    except Exception as exc:
        return f"ERROR: curve_fit failed for model '{fit_model}': {exc}"

    ss_res = float(np.sum((F - F_pred) ** 2))
    ss_tot = float(np.sum((F - np.mean(F)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((F - F_pred) ** 2)))
    params["r_squared"] = float(r_squared)
    params["rmse"] = rmse

    if params.get("mobile_fraction", 0.0) > 1.05:
        log.append(
            f"WARNING: fitted mobile fraction {params['mobile_fraction']:.3f} > 1; "
            "indicates over-correction or unaccounted-for background drift."
        )

    params_path = os.path.join(output_folder, "frap_fit_parameters.json")
    with open(params_path, "w") as fh:
        json.dump(params, fh, indent=2)

    plot_path = os.path.join(output_folder, "frap_fit_plot.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(t, F, label="Data", s=18, color="C0", zorder=2)
    t_dense = np.linspace(t[0], t[-1], 500)
    if fit_model == "soumpasis":
        F_dense = _soumpasis_recovery(t_dense, *popt)
    elif fit_model == "exponential":
        F_dense = _exponential_recovery(t_dense, *popt)
    else:
        F_dense = _double_exponential_recovery(t_dense, *popt)
    ax.plot(t_dense, F_dense, label=f"Fit ({fit_model})", color="C1", linewidth=2, zorder=3)
    title_bits = [
        f"Mf={params.get('mobile_fraction', float('nan')):.2f}",
        f"R²={r_squared:.3f}",
    ]
    if "t_half_s" in params:
        title_bits.insert(1, f"t½={params['t_half_s']:.1f}s")
    if "diffusion_coefficient_um2_per_s" in params:
        title_bits.append(f"D={params['diffusion_coefficient_um2_per_s']:.3f} µm²/s")
    ax.set_title("FRAP fit · " + ", ".join(title_bits))
    ax.set_xlabel("Time post-bleach (s)")
    ax.set_ylabel("Normalized intensity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    log.append(f"Fit model: {fit_model}.")
    if "mobile_fraction" in params:
        log.append(
            f"Mobile fraction: {params['mobile_fraction']:.3f}"
            + (f" (95% CI ± {params['mobile_fraction_ci95']:.3f})" if "mobile_fraction_ci95" in params else "")
        )
    if "t_half_s" in params:
        log.append(f"t-half: {params['t_half_s']:.2f} s.")
    if "diffusion_coefficient_um2_per_s" in params:
        log.append(
            f"Diffusion coefficient: {params['diffusion_coefficient_um2_per_s']:.4f} µm²/s "
            f"(from bleach radius {bleach_radius_um} µm and τ_D {params['tau_D_s']:.3f} s)."
        )
    if fit_model == "double_exponential":
        log.append(f"Fast component: A={params['amplitude_fast']:.3f}, τ={params['tau_fast_s']:.2f} s.")
        log.append(f"Slow component: A={params['amplitude_slow']:.3f}, τ={params['tau_slow_s']:.2f} s.")
    log.append(f"R² = {r_squared:.4f}, RMSE = {rmse:.4f}.")
    log.append(f"Wrote fit parameters to {params_path}.")
    log.append(f"Wrote fit plot to {plot_path}.")

    return "\n".join(log)


def classify_condensate_material_state(
    fit_parameters_json: str,
    output_folder: str = "./tmp/",
    use_llm_interpretation: bool = False,
    llm: str = "claude-sonnet-4-5",
    context_note: str = "",
) -> str:
    """Classify FRAP fit parameters into liquid / gel-like / arrested.

    Heuristic over mobile fraction and recovery half-time. Optionally calls
    an LLM (via biomni.llm.get_llm) to produce a 1-paragraph biological
    interpretation grounded in the classification and any user-supplied
    context (protein name, condition, salt concentration, etc.).

    Args:
        fit_parameters_json: Path to the JSON written by `fit_frap_recovery_curve`.
        output_folder: Where to write the interpretation text.
        use_llm_interpretation: If True, call the configured LLM for a prose
            explanation; if False, return the heuristic verdict only. Default
            False to keep the tool deterministic and CI-safe.
        llm: Model name passed to `biomni.llm.get_llm` when LLM is used.
        context_note: Free-text context (e.g., "FUS in 50 mM KCl after 30 min")
            that the LLM should use when generating the interpretation.

    Returns:
        Log string with classification, the heuristic reasoning, and (if
        requested) the LLM-generated paragraph.
    """
    log: list[str] = []
    os.makedirs(output_folder, exist_ok=True)

    try:
        with open(fit_parameters_json) as fh:
            params = json.load(fh)
    except Exception as exc:
        return f"ERROR: failed to read fit parameters JSON: {exc}"

    Mf = params.get("mobile_fraction")
    t_half = params.get("t_half_s")
    r_squared = params.get("r_squared", 0.0)
    if Mf is None:
        return "ERROR: 'mobile_fraction' missing from fit parameters."

    # Heuristic thresholds (Brangwynne / Hyman lab community conventions).
    # These are flagged in the docstring as rule-of-thumb, not law.
    if Mf >= 0.7 and r_squared >= 0.9 and (t_half is None or t_half < 60):
        verdict = "liquid"
        reasoning = (
            f"Mobile fraction {Mf:.2f} ≥ 0.70 with high-quality fit "
            f"(R²={r_squared:.2f}); recovery is fast and near-complete, "
            "consistent with a liquid-like condensate."
        )
    elif Mf < 0.2:
        verdict = "arrested"
        reasoning = (
            f"Mobile fraction {Mf:.2f} < 0.20; little to no recovery within the "
            "imaging window indicates a glass-like or solid (arrested) state."
        )
    else:
        verdict = "gel-like"
        reasoning = (
            f"Mobile fraction {Mf:.2f} is intermediate"
            + (f" with t-half {t_half:.1f} s" if t_half is not None else "")
            + "; partial recovery is consistent with a gel-like or maturing condensate."
        )

    # If model is double_exponential and the slow component dominates, lean gel-like.
    if params.get("model") == "double_exponential":
        a_slow = params.get("amplitude_slow", 0.0)
        a_fast = params.get("amplitude_fast", 0.0)
        if a_fast + a_slow > 0 and a_slow / (a_fast + a_slow) > 0.6:
            verdict = "gel-like" if verdict == "liquid" else verdict
            reasoning += (
                " Two-component fit shows the slow-pool amplitude dominates "
                f"({a_slow:.2f} vs fast {a_fast:.2f}), suggesting a gel-like component."
            )

    log.append(f"Classification: {verdict}.")
    log.append(reasoning)

    interpretation = ""
    if use_llm_interpretation:
        try:
            from biomni.llm import get_llm  # type: ignore

            client = get_llm(llm)
            prompt = (
                "You are a biophysics expert interpreting a FRAP experiment on a "
                "biomolecular condensate. Given the fit parameters and classification, "
                "produce one short paragraph (3-5 sentences) explaining what the "
                "measurements imply about the condensate's material state, and what "
                "follow-up experiments could test the interpretation. Do not invent "
                "data; only use the values provided.\n\n"
                f"Fit parameters: {json.dumps(params, indent=2)}\n"
                f"Heuristic verdict: {verdict}\n"
                f"Heuristic reasoning: {reasoning}\n"
                f"User context: {context_note or '(none provided)'}\n"
            )
            interpretation = str(client.invoke(prompt))
            log.append("LLM interpretation:")
            log.append(interpretation.strip())
        except Exception as exc:
            log.append(f"WARNING: LLM interpretation skipped ({exc}).")

    out_path = os.path.join(output_folder, "frap_material_state.json")
    with open(out_path, "w") as fh:
        json.dump(
            {
                "classification": verdict,
                "reasoning": reasoning,
                "llm_interpretation": interpretation,
                "input_parameters": params,
            },
            fh,
            indent=2,
        )
    log.append(f"Wrote classification to {out_path}.")

    return "\n".join(log)


def run_frap_analysis_pipeline(
    image_stack_path: str,
    bleach_roi: list,
    bleach_radius_um: float,
    pixel_size_um: float = None,
    reference_roi: list = None,
    background_roi: list = None,
    bleach_frame_index: int = None,
    frame_interval_s: float = 1.0,
    fit_model: str = "soumpasis",
    use_llm_interpretation: bool = False,
    llm: str = "claude-sonnet-4-5",
    context_note: str = "",
    output_folder: str = "./tmp/",
) -> str:
    """End-to-end FRAP analysis: image stack → curve → fit → material-state report.

    Convenience wrapper that calls extract → fit → classify in sequence and
    writes a single Markdown report alongside the intermediate artifacts.

    Args:
        image_stack_path: Path to TIFF time-series.
        bleach_roi: [x_center_px, y_center_px, radius_px] for the bleached spot.
        bleach_radius_um: Bleach radius in micrometers (for diffusion coefficient).
        pixel_size_um: Optional pixel size; not required if `bleach_radius_um`
            is supplied directly. Reserved for a future feature that derives
            the radius from `bleach_roi[2]` * `pixel_size_um`.
        reference_roi, background_roi, bleach_frame_index, frame_interval_s:
            Passed through to `extract_frap_curve_from_image_stack`.
        fit_model: Passed through to `fit_frap_recovery_curve`.
        use_llm_interpretation, llm, context_note: Passed through to
            `classify_condensate_material_state`.
        output_folder: Directory for all outputs.

    Returns:
        Log string for the full pipeline plus the path to a Markdown summary.
    """
    log: list[str] = ["=== FRAP analysis pipeline ==="]
    os.makedirs(output_folder, exist_ok=True)

    extract_log = extract_frap_curve_from_image_stack(
        image_stack_path=image_stack_path,
        bleach_roi=bleach_roi,
        reference_roi=reference_roi,
        background_roi=background_roi,
        bleach_frame_index=bleach_frame_index,
        frame_interval_s=frame_interval_s,
        output_folder=output_folder,
    )
    log.append("\n[1/3] Curve extraction")
    log.append(extract_log)
    if extract_log.startswith("ERROR"):
        return "\n".join(log)

    post_csv = os.path.join(output_folder, "frap_curve_postbleach.csv")
    fit_log = fit_frap_recovery_curve(
        curve_csv_path=post_csv,
        bleach_radius_um=bleach_radius_um,
        fit_model=fit_model,
        output_folder=output_folder,
    )
    log.append("\n[2/3] Curve fitting")
    log.append(fit_log)
    if fit_log.startswith("ERROR"):
        return "\n".join(log)

    fit_json = os.path.join(output_folder, "frap_fit_parameters.json")
    classify_log = classify_condensate_material_state(
        fit_parameters_json=fit_json,
        output_folder=output_folder,
        use_llm_interpretation=use_llm_interpretation,
        llm=llm,
        context_note=context_note,
    )
    log.append("\n[3/3] Material-state classification")
    log.append(classify_log)
    if classify_log.startswith("ERROR"):
        return "\n".join(log)

    # Markdown summary report.
    try:
        with open(fit_json) as fh:
            params = json.load(fh)
        with open(os.path.join(output_folder, "frap_material_state.json")) as fh:
            verdict = json.load(fh)
    except Exception:
        params, verdict = {}, {}

    report_lines = [
        "# FRAP Analysis Report",
        "",
        f"- Image stack: `{image_stack_path}`",
        f"- Bleach ROI (px): {bleach_roi}",
        f"- Bleach radius (µm): {bleach_radius_um}",
        f"- Frame interval (s): {frame_interval_s}",
        f"- Fit model: {fit_model}",
        "",
        "## Fit parameters",
        "```json",
        json.dumps(params, indent=2),
        "```",
        "",
        "## Material-state classification",
        f"**Verdict:** {verdict.get('classification', 'n/a')}",
        "",
        verdict.get("reasoning", ""),
        "",
    ]
    if verdict.get("llm_interpretation"):
        report_lines.extend(["## Biological interpretation", verdict["llm_interpretation"]])
    report_path = os.path.join(output_folder, "frap_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(report_lines))
    log.append(f"\nWrote summary report to {report_path}.")

    return "\n".join(log)
