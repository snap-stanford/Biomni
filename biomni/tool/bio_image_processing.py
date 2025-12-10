from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

_DEFAULT_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
_DEFAULT_BRIGHTNESS_BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 20),
    (20, 50),
    (50, 80),
    (80, 110),
    (110, 140),
    (140, 170),
    (170, 200),
    (200, 256),
)


def analyze_pixel_distribution(image_path: str) -> dict:
    """Analyze western blot or DNA electrophoresis images and return pixel distribution statistics.

    Parameters
    ----------
    image_path : str
        Path to the input grayscale image. Automatically appends .png if no suffix is provided.

    Returns
    -------
    dict
        Summary dictionary containing image shape, intensity statistics, percentiles,
        histogram values, and brightness distribution for predefined buckets.

    """
    import cv2

    _DEFAULT_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    _DEFAULT_BRIGHTNESS_BUCKETS: tuple[tuple[int, int], ...] = (
        (0, 20),
        (20, 50),
        (50, 80),
        (80, 110),
        (110, 140),
        (140, 170),
        (170, 200),
        (200, 256),
    )

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    percentiles = np.percentile(image, _DEFAULT_PERCENTILES).tolist()
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total_pixels = int(image.size)

    brightness_lines = []
    for low, high in _DEFAULT_BRIGHTNESS_BUCKETS:
        count = int(histogram[low:high].sum())
        ratio = round((count / total_pixels) * 100, 2) if total_pixels > 0 else 0.0
        brightness_lines.append(f"Range [{low:>3}, {high:>3}): {count:>8} px ({ratio:5.2f}%)")

    min_intensity = int(image.min())
    max_intensity = int(image.max())
    mean_intensity = round(float(image.mean()), 2)
    std_intensity = round(float(image.std()), 2)

    return {
        "shape": f"({image.shape[0]}, {image.shape[1]})",
        "intensity_stats": {
            "min": min_intensity,
            "max": max_intensity,
            "mean": mean_intensity,
            "std_dev": std_intensity,
        },
        "percentiles_label": "percentiles (1, 5, 10, 25, 50, 75, 90, 95, 99):",
        "percentiles_values": ", ".join(f"{float(p):.1f}" for p in percentiles),
        "pixel_brightness_distribution": brightness_lines,
    }


def find_roi_from_image(
    image_path: str,
    lower_threshold: int,
    upper_threshold: int,
    number_of_bands: int,
    debug: bool = False,
) -> tuple[str, list]:
    """Find the ROIs of the bands from the image which is determined by analyze_pixel_distribution function.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    lower_threshold : int
        Pixel intensities lower than this value are used to make the binary image.
    upper_threshold : int
        Pixel intensities greater than or equal to this value are used to make the binary image.
    number_of_bands : int
        The actual number of bands in the image.
    debug : bool, optional
        If True, draw green contours (hulls) and blue keypoint boxes for debugging.
        Default is True.

    Returns
    -------
    tuple[str, list]
        A tuple containing:
        - str: Absolute path to the saved annotated image
        - list: List of ROI coordinates in (x, y, width, height) format.
        The ROI list can be converted to target_bands for analyze_western_blot:
        annotated_path, rois = find_roi_from_image(...)
        target_bands = [{"name": f"band_{i}", "roi": list(roi)} for i, roi in enumerate(rois)]

    Raises
    ------
    ValueError
        If threshold values are outside the valid range or inconsistent.
    FileNotFoundError
        If the source image cannot be loaded.

    """
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import cv2

    ROI = tuple[int, int, int, int]

    def load_grayscale_image(path: str) -> cv2.Mat:
        """Load a grayscale image from disk."""
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to load image at '{path}'")
        return image

    def build_blob_detector(
        min_threshold: int = 0,
        max_threshold: int = 200,
        min_area: int = 120,
        min_convexity: float = 0.7,
        min_inertia: float = 0.001,
        max_inertia: float = 0.4,
    ) -> cv2.SimpleBlobDetector:
        """Configure and return a SimpleBlobDetector instance."""
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = min_threshold
        params.maxThreshold = max_threshold
        params.filterByArea = True
        params.minArea = min_area
        params.filterByConvexity = True
        params.minConvexity = min_convexity
        params.filterByInertia = True
        params.minInertiaRatio = min_inertia
        params.maxInertiaRatio = max_inertia
        return cv2.SimpleBlobDetector_create(params)

    def detect_blobs(image: cv2.Mat, detector: cv2.SimpleBlobDetector) -> list[cv2.KeyPoint]:
        """Detect blob keypoints in the provided image."""
        keypoints = detector.detect(image)
        print(f"Detected {len(keypoints)} keypoints.")
        for index, keypoint in enumerate(keypoints):
            print(f"[{index}] position={keypoint.pt}, size={keypoint.size}")
        return keypoints

    def find_band_contours(
        binary_mask: cv2.Mat,
        min_area: int = 100,
        use_morphology: bool = True,
    ) -> list[cv2.Mat]:
        """Find band contours from binary mask using morphological operations."""
        processed_mask = binary_mask.copy()

        if use_morphology:
            # 가로(Horizontal) 방향으로 떨어진 덩어리를 잇기 위해 가로가 긴 커널 사용
            # (50, 1)의 50은 두 덩어리 사이의 픽셀 거리보다 커야 합니다.
            kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))

            # OPEN(끊기) 대신 CLOSE(잇기)를 사용하여 빈 공간을 메움
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel_connect, iterations=1)

        # Find ALL contours (not just external) to detect separate bands
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        print(f"Found {len(contours)} total contours")

        # Filter contours by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                filtered_contours.append(contour)

        print(f"Filtered to {len(filtered_contours)} contours with area >= {min_area}")

        return filtered_contours

    def analyze_roi_pixel_distribution(
        image: cv2.Mat,
        roi: ROI,
    ) -> dict:
        """Analyze pixel distribution of an ROI to distinguish between text and bands.

        Parameters
        ----------
        image : cv2.Mat
            Original grayscale image
        roi : ROI
            ROI coordinates (x, y, width, height)

        Returns
        -------
        dict
            Dictionary containing edge_strength, std_dev, and gradient_magnitude

        """
        x, y, w, h = roi

        # Extract ROI region from original image
        roi_region = image[y : y + h, x : x + w]

        if roi_region.size == 0:
            return {"edge_strength": 0.0, "std_dev": 0.0, "gradient_magnitude": 0.0}

        # Calculate standard deviation of pixel intensities
        std_dev = float(np.std(roi_region))

        # Calculate edge strength using Laplacian
        laplacian = cv2.Laplacian(roi_region, cv2.CV_64F)
        edge_strength = float(np.mean(np.abs(laplacian)))

        # Calculate gradient magnitude using Sobel
        sobelx = cv2.Sobel(roi_region, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi_region, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))

        return {
            "edge_strength": edge_strength,
            "std_dev": std_dev,
            "gradient_magnitude": gradient_magnitude,
        }

    def filter_rois_by_pixel_distribution(
        image: cv2.Mat,
        rois: list[ROI],
        hulls: list[cv2.Mat],
        max_edge_strength: float = 10.0,
        max_gradient_magnitude: float = 70.0,
        max_std_dev: float = 50.0,
    ) -> tuple[list[ROI], list[cv2.Mat]]:
        """Filter ROIs to remove text-like regions and keep band-like regions.

        Text regions have very high edge strength, sharp gradients, and high std_dev.
        Band regions have low edge strength, moderate gradients, and moderate std_dev.

        Parameters
        ----------
        image : cv2.Mat
            Original grayscale image
        rois : List[ROI]
            List of ROI coordinates
        hulls : List[cv2.Mat]
            List of corresponding convex hulls
        max_edge_strength : float, optional
            Maximum edge strength for band-like regions (text typically >20)
        max_gradient_magnitude : float, optional
            Maximum gradient magnitude for band-like regions (text typically >100)
        max_std_dev : float, optional
            Maximum standard deviation for band-like regions (text typically >80)

        Returns
        -------
        tuple
            Filtered ROIs and hulls that are band-like

        """
        filtered_rois: list[ROI] = []
        filtered_hulls: list[cv2.Mat] = []

        print("\n=== ROI Pixel Distribution Analysis ===")

        for idx, (roi, hull) in enumerate(zip(rois, hulls, strict=True)):
            analysis = analyze_roi_pixel_distribution(image, roi)

            edge_strength = analysis["edge_strength"]
            gradient_magnitude = analysis["gradient_magnitude"]
            std_dev = analysis["std_dev"]

            # Determine if this ROI is band-like or text-like
            # Text has very high edge strength (>20) and very high gradient (>100)
            # Bands have low edge strength (<10) and moderate gradient (<50)
            # Also check std_dev to filter out high-contrast text regions
            is_band = (
                edge_strength <= max_edge_strength
                and gradient_magnitude <= max_gradient_magnitude
                and std_dev <= max_std_dev
            )

            status = "✓ BAND" if is_band else "✗ TEXT"
            print(f"ROI {idx}: edge={edge_strength:.2f}, grad={gradient_magnitude:.2f}, std={std_dev:.2f} -> {status}")

            if is_band:
                filtered_rois.append(roi)
                filtered_hulls.append(hull)

        print(f"\nFiltered: {len(filtered_rois)}/{len(rois)} ROIs kept as bands")
        print("=" * 40 + "\n")

        return filtered_rois, filtered_hulls

    def compute_rois(
        image: cv2.Mat,
        binary_mask: cv2.Mat,
        keypoints: Iterable[cv2.KeyPoint],
        padding: tuple[int, int] = (5, 5),
        min_contour_area: int = 100,
        filter_by_distribution: bool = True,
    ) -> tuple[list[ROI], list[cv2.Mat]]:
        """Compute global ROIs for each keypoint by matching them to band contours."""
        # Find band contours from binary mask
        band_contours = find_band_contours(binary_mask, min_area=min_contour_area)

        if not band_contours:
            print("No band contours found!")
            return [], []

        auto_rois: list[ROI] = []
        global_hulls: list[cv2.Mat] = []
        matched_contours = set()  # Track which contours have been matched
        used_contours = set()  # Track which contours have already been used to avoid duplicates

        for keypoint in keypoints:
            cx, cy = int(keypoint.pt[0]), int(keypoint.pt[1])
            keypoint_center = (float(cx), float(cy))

            # Find the contour that contains this keypoint
            matched_contour = None
            matched_idx = None
            for idx, contour in enumerate(band_contours):
                # Skip if this contour has already been used
                if idx in used_contours:
                    continue
                # Use pointPolygonTest to check if keypoint center is inside contour
                # Returns positive if inside, negative if outside, zero if on edge
                distance = cv2.pointPolygonTest(contour, keypoint_center, False)
                if distance >= 0:  # Inside or on edge
                    matched_contour = contour
                    matched_idx = idx
                    matched_contours.add(idx)
                    print(f"Keypoint at ({cx}, {cy}) matched to contour {idx}")
                    break

            if matched_contour is None:
                print(f"Warning: Keypoint at ({cx}, {cy}) not matched to any contour")
                continue

            # Mark this contour as used to avoid duplicate ROIs
            used_contours.add(matched_idx)

            # Compute convex hull from the matched contour
            hull = cv2.convexHull(matched_contour)
            if hull is None or len(hull) < 3:
                continue

            # Get bounding rectangle from hull with padding
            pad_x, pad_y = padding
            rx, ry, rw, rh = cv2.boundingRect(hull)

            # Skip if bounding rect is too large (likely covering entire image or invalid)
            max_roi_area_ratio = 0.5  # Maximum 50% of image area
            roi_area = rw * rh
            image_area = image.shape[0] * image.shape[1]
            if roi_area > image_area * max_roi_area_ratio:
                print(f"Warning: ROI too large ({rw}x{rh}), skipping. This may indicate a detection error.")
                continue

            global_x = max(0, rx - pad_x)
            global_y = max(0, ry - pad_y)
            global_w = min(image.shape[1] - global_x, rw + 2 * pad_x)
            global_h = min(image.shape[0] - global_y, rh + 2 * pad_y)

            if global_w <= 0 or global_h <= 0:
                continue

            roi = (global_x, global_y, global_w, global_h)
            auto_rois.append(roi)
            global_hulls.append(hull)

        print(f"Matched {len(matched_contours)} contours to keypoints")

        # Filter ROIs by pixel distribution to remove text-like regions
        if filter_by_distribution and auto_rois:
            auto_rois, global_hulls = filter_rois_by_pixel_distribution(image, auto_rois, global_hulls)

        return auto_rois, global_hulls

    def annotate_keypoints(
        image: cv2.Mat,
        keypoints: Iterable[cv2.KeyPoint],
        rois: Iterable[ROI],
        hulls: Iterable[cv2.Mat] | None = None,
        debug: bool = False,
    ) -> cv2.Mat:
        """Draw ROIs, convex hulls, and index labels on the image.

        Parameters
        ----------
        image : cv2.Mat
            Input grayscale image
        keypoints : Iterable[cv2.KeyPoint]
            Detected keypoints
        rois : Iterable[ROI]
            ROI coordinates to draw
        hulls : Iterable[cv2.Mat] | None, optional
            Convex hulls to draw (only if debug=True)
        debug : bool, optional
            If True, draw green contours (hulls) and blue keypoint boxes

        Returns
        -------
        cv2.Mat
            Annotated image

        """
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw convex hulls in green if debug mode is enabled
        if debug and hulls is not None:
            for hull in hulls:
                cv2.drawContours(output, [hull], -1, (0, 255, 0), 2)  # Green color in BGR

        # Draw ROIs in red (always drawn)
        for roi in rois:
            x, y, w, h = roi
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Draw keypoints in blue and index labels (only if debug mode is enabled)
        if debug:
            for index, keypoint in enumerate(keypoints):
                x, y = keypoint.pt
                size = keypoint.size

                # Draw blue rectangle around keypoint
                # Use size as half-width and half-height for the rectangle
                half_size = int(size / 2)
                pt1 = (int(x) - half_size, int(y) - half_size)
                pt2 = (int(x) + half_size, int(y) + half_size)
                cv2.rectangle(output, pt1, pt2, (255, 0, 0), 2)  # Blue color in BGR

                # Draw index label
                cv2.putText(
                    output,
                    str(index),
                    (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        return output

    def show_rois(rois: Sequence[ROI]) -> None:
        """Print ROI information to stdout."""
        print(f"Detected ROI count: {len(rois)}")
        for index, roi in enumerate(rois):
            print(f"ROI {index}: {roi}")

    if not 0 <= lower_threshold <= 255:
        raise ValueError("lower_threshold must be within [0, 255].")
    if not 0 <= upper_threshold <= 255:
        raise ValueError("upper_threshold must be within [0, 255].")
    if lower_threshold > upper_threshold:
        raise ValueError("lower_threshold cannot be greater than upper_threshold.")

    original_image = load_grayscale_image(image_path)
    mask = cv2.inRange(original_image, lower_threshold, upper_threshold)
    mask = cv2.bitwise_not(mask)
    detector = build_blob_detector()

    # Detect blobs in the mask image
    keypoints = detect_blobs(mask, detector)
    rois, hulls = compute_rois(original_image, mask, keypoints)
    show_rois(rois)

    # Draw ROIs, convex hulls, and keypoints on the mask image
    debug = False
    annotated_mask = annotate_keypoints(mask, keypoints, rois, hulls, debug=debug)
    mask_path = Path(image_path).parent / f"{Path(image_path).stem}_mask.png"
    cv2.imwrite(str(mask_path), annotated_mask)

    # Draw ROIs, convex hulls, and keypoints on the original image
    annotated_image = annotate_keypoints(original_image, keypoints, rois, hulls, debug=debug)
    annotated_image_path = Path(image_path).parent / f"{Path(image_path).stem}_annotated.png"
    cv2.imwrite(str(annotated_image_path), annotated_image)

    if len(rois) != number_of_bands:
        print(f"Warning: Detected {len(rois)} ROIs, but expected {number_of_bands} ROIs.")
        print(
            "Please check the image and try to adjust the thresholds. Or you can manually infer the ROIs from the annotated image."
        )

    return str(annotated_image_path.resolve()), rois

def quantify_bands(
    image_path: str,
    rois: Sequence[Sequence[float]],
    background_width: int = 3,
    back_pos: str = "all",
    back_type: str = "median",
) -> list[float]:
    """Quantify band intensities for the provided ROIs while subtracting local background.

    Args:
        image_path (str): Absolute or relative path to the grayscale source image.
        rois (Sequence[Sequence[float]]): Collection of ``(x, y, width, height)`` ROI tuples.
        background_width (int): Margin, in pixels, used to sample background around each ROI.
        back_pos (str): Background sampling strategy: ``all``, ``top/bottom``, or ``sides``.
        back_type (str): Aggregation method for the sampled background, either ``mean`` or ``median``.

    Returns:
        list[float]: Background-corrected signal intensities for each ROI. Invalid ROIs return ``nan``.

    Raises:
        FileNotFoundError: Raised when the specified image cannot be loaded.
        ValueError: Raised when ``back_pos`` or ``back_type`` are outside the supported options.
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at '{image_path}'.")

    image = cv2.bitwise_not(image)

    height, width = image.shape
    intensities: list[float] = []

    valid_positions = {"all", "top/bottom", "sides"}
    if back_pos not in valid_positions:
        raise ValueError(
            "Invalid 'back_pos' value. Expected one of: 'all', 'top/bottom', 'sides'."
        )

    valid_types = {"mean", "median"}
    if back_type not in valid_types:
        raise ValueError("Invalid 'back_type' value. Expected 'mean' or 'median'.")

    for roi in rois:
        if len(roi) != 4:
            intensities.append(float("nan"))
            continue

        x, y, w, h = map(int, roi)

        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(width, x + w)
        y_end = min(height, y + h)

        if x_start >= x_end or y_start >= y_end:
            intensities.append(0.0)
            continue

        roi_slice = image[y_start:y_end, x_start:x_end]
        area = roi_slice.size
        mean_roi = float(np.mean(roi_slice)) if area else 0.0

        background_samples: list[np.ndarray] = []

        if back_pos == "all":
            x_min = max(0, x - background_width)
            y_min = max(0, y - background_width)
            x_max = min(width, x + w + background_width)
            y_max = min(height, y + h + background_width)

            if y_min < y_start:
                background_samples.append(image[y_min:y_start, x_min:x_max])
            if y_end < y_max:
                background_samples.append(image[y_end:y_max, x_min:x_max])
            if x_min < x_start:
                background_samples.append(image[y_start:y_end, x_min:x_start])
            if x_end < x_max:
                background_samples.append(image[y_start:y_end, x_end:x_max])
        elif back_pos == "top/bottom":
            top_start = max(0, y - background_width)
            if top_start < y_start:
                background_samples.append(image[top_start:y_start, x_start:x_end])

            bottom_end = min(height, y + h + background_width)
            if y_end < bottom_end:
                background_samples.append(image[y_end:bottom_end, x_start:x_end])
        else:  # back_pos == "sides"
            left_start = max(0, x - background_width)
            if left_start < x_start:
                background_samples.append(image[y_start:y_end, left_start:x_start])

            right_end = min(width, x + w + background_width)
            if x_end < right_end:
                background_samples.append(image[y_start:y_end, x_end:right_end])

        if background_samples:
            background_pixels = np.concatenate(
                [sample.ravel() for sample in background_samples if sample.size]
            )
        else:
            background_pixels = np.empty(0, dtype=image.dtype)

        background_value = 0.0
        if background_pixels.size > 0:
            if back_type == "median":
                background_value = float(np.median(background_pixels))
            else:  # back_type == "mean"
                background_value = float(np.mean(background_pixels))

        signal = max(0.0, area * (mean_roi - background_value))
        intensities.append(signal)

    return intensities


def binarize_image(
    image: np.ndarray | str,
    threshold: int,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) and then binarize the image.

    Args:
        image: Input image as numpy array or image file path (str).
        threshold: Threshold value. This value MUST be determined by analyze_pixel_distribution function.
        clip_limit: CLAHE clipLimit parameter (default: 2.0).
        tile_grid_size: CLAHE tileGridSize parameter (default: (8, 8)).

    Returns:
        binary_img: Binary image after applying CLAHE and thresholding.

    Raises:
        FileNotFoundError: Raised when image path is provided but file cannot be found.
        ValueError: Raised when threshold is outside the valid range [0, 255].
    """
    # Load image if path is provided
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image}")
    else:
        img = image.copy()

    # Convert to grayscale if image is color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(gray)

    # Perform binarization
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be within [0, 255].")

    _, binary_img = cv2.threshold(clahe_img, threshold, 255, cv2.THRESH_BINARY)

    return binary_img


def count_cells(
    binary_img: np.ndarray,
    min_distance: int = 10,
    output_path: str | None = None,
) -> tuple[int, np.ndarray, str]:
    """Count cells in a binary image using watershed algorithm.

    This function separates overlapping cells and counts them by:
    1. Applying Distance Transform to create peaks at cell centers
    2. Finding local maxima (peaks) using peak_local_max
    3. Applying Watershed algorithm to separate regions
    4. Counting unique labels from watershed result (excluding background)
    5. Drawing boundaries on the binary image and saving it

    Args:
        binary_img: Binary image where cells are represented as white (255) pixels
            and background as black (0) pixels.
        min_distance: Minimum distance between peaks in pixels. Controls sensitivity
            for separating overlapping cells (default: 10).
        output_path: Path to save the image with cell boundaries drawn. If None, saves to
            "watershed_labels.png" in the current directory (default: None).

    Returns:
        tuple: A tuple containing:
            - count (int): Number of detected cells (counted from watershed labels)
            - labels (np.ndarray): Labeled image where each cell region has a unique label
            - image_path (str): Path to the saved image with boundaries drawn on binary image

    Raises:
        ValueError: If binary_img is not a valid 2D numpy array.
    """
    if not isinstance(binary_img, np.ndarray) or len(binary_img.shape) != 2:
        raise ValueError("binary_img must be a 2D numpy array.")

    # 1. Apply Distance Transform
    # Inner pixels get higher values than outer pixels, creating peak-like structures
    dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)

    # 2. Find local maxima (cell centers) using peak_local_max
    # min_distance controls sensitivity for separating overlapping cells
    coords = peak_local_max(
        dist_transform, min_distance=min_distance, labels=binary_img
    )

    # 3. Apply Watershed algorithm to separate regions
    # Create markers from detected peaks
    mask = np.zeros(dist_transform.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)

    # Apply watershed to create labeled regions
    labels = watershed(-dist_transform, markers, mask=binary_img)

    # 4. Count cells from watershed labels (excluding background label 0)
    unique_labels = np.unique(labels)
    count = len(unique_labels[unique_labels > 0])  # Exclude background

    # 5. Draw boundaries on binary image
    # Convert binary image to 3-channel for drawing colored boundaries
    output_image = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    
    # Draw boundaries for each labeled region
    for label_id in unique_labels:
        if label_id == 0:  # Skip background
            continue
        
        # Create mask for current label
        label_mask = (labels == label_id).astype(np.uint8) * 255
        
        # Find contours of the label region
        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw boundary (black color)
        cv2.drawContours(output_image, contours, -1, (0, 0, 0), 1)

    # Determine output path
    if output_path is None:
        output_path = "watershed_labels.png"
    else:
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

    # Save the image with boundaries
    cv2.imwrite(output_path, output_image)

    return count, labels, output_path