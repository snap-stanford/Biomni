from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

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
    """Analyse western blot or DNA electrophoresis images and return pixel distribution statistics.

    Args:
        image_path (str): Path to the input grayscale image. Automatically appends
            ``.png`` if no suffix is provided.

    Returns:
        dict: Summary dictionary containing image shape, intensity statistics, percentiles,
            histogram values, and brightness distribution for predefined buckets.
    """

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
        brightness_lines.append(
            f"Range [{low:>3}, {high:>3}): {count:>8} px ({ratio:5.2f}%)"
        )

    return {
        "shape": f"({image.shape[0]}, {image.shape[1]})",
        "min_intensity": int(image.min()),
        "max_intensity": int(image.max()),
        "mean_intensity": round(float(image.mean()), 2),
        "std_intensity": round(float(image.std()), 2),
        "percentiles_label": "percentiles (1, 5, 10, 25, 50, 75, 90, 95, 99):",
        "percentiles_values": ", ".join(f"{float(p):.1f}" for p in percentiles),
        "pixel_brightness_distribution": brightness_lines,
    }


def highlight_faint_bands(
    image_path: str,
    lower_threshold: int = 30,
    upper_threshold: int = 200,
) -> str:
    """Apply intensity thresholds to highlight faint bands in western blot or DNA electrophoresis images.

    Args:
        image_path (str): Path to the input grayscale image. Automatically appends
            ``.png`` if no suffix is provided.
        lower_threshold (int, optional): Pixel intensities lower than this value
            are set to 255 after inversion. Defaults to 30.
        upper_threshold (int, optional): Pixel intensities greater than or equal
            to this value are set to 255 after inversion. Defaults to 200.

    Returns:
        str: Absolute path to the saved binary mask image.

    """

    if not 0 <= lower_threshold <= 255:
        raise ValueError("lower_threshold must be within [0, 255].")
    if not 0 <= upper_threshold <= 255:
        raise ValueError("upper_threshold must be within [0, 255].")
    if lower_threshold > upper_threshold:
        raise ValueError("lower_threshold cannot be greater than upper_threshold.")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    mask_path = (
        Path(image_path).parent / f"{Path(image_path).stem}_mask.png"
    )

    mask = cv2.inRange(image, lower_threshold, upper_threshold)
    mask = cv2.bitwise_not(mask)
    cv2.imwrite(str(mask_path), mask)

    return str(mask_path.resolve())


def find_roi_from_image(image_path: str) -> str:
    from typing import Iterable, List, Sequence, Tuple

    import cv2

    ROI = Tuple[int, int, int, int]


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
        min_inertia: float = 0.01,
        max_inertia: float = 0.2,
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


    def detect_blobs(
        image: cv2.Mat, detector: cv2.SimpleBlobDetector
    ) -> List[cv2.KeyPoint]:
        """Detect blob keypoints in the provided image."""
        keypoints = detector.detect(image)
        print(f"Detected {len(keypoints)} keypoints.")
        for index, keypoint in enumerate(keypoints):
            print(f"[{index}] position={keypoint.pt}, size={keypoint.size}")
        return keypoints


    def adaptive_threshold_roi(
        image: cv2.Mat,
        search_window: Tuple[int, int],
        origin: Tuple[int, int],
    ) -> Tuple[Tuple[int, int, int, int], cv2.Mat] | Tuple[None, None]:
        """Run adaptive thresholding around a keypoint and return the ROI mask."""
        search_w, search_h = search_window
        cx, cy = origin

        x1 = max(0, cx - search_w // 2)
        y1 = max(0, cy - search_h // 2)
        x2 = min(image.shape[1], cx + search_w // 2)
        y2 = min(image.shape[0], cy + search_h // 2)
        if x2 <= x1 or y2 <= y1:
            return None, None

        local_roi = image[y1:y2, x1:x2]
        local_binary = cv2.adaptiveThreshold(
            local_roi,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
        return (x1, y1, x2, y2), local_binary


    def find_containing_contour(
        contours: Sequence[cv2.Mat],
        local_point: Tuple[int, int],
    ) -> cv2.Mat | None:
        """Return the contour that contains the given point, if any."""
        for contour in contours:
            if cv2.pointPolygonTest(contour, local_point, False) >= 0:
                return contour
        return None


    def expand_to_global_roi(
        contour: cv2.Mat,
        roi_bounds: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> ROI | None:
        """Convert contour bounds to global coordinates with padding."""
        x1, y1, x2, y2 = roi_bounds
        pad_x, pad_y = padding
        rx, ry, rw, rh = cv2.boundingRect(contour)

        global_x = max(0, rx + x1 - pad_x)
        global_y = max(0, ry + y1 - pad_y)
        global_w = min(image_shape[1] - global_x, rw + 2 * pad_x)
        global_h = min(image_shape[0] - global_y, rh + 2 * pad_y)
        if global_w <= 0 or global_h <= 0:
            return None
        return global_x, global_y, global_w, global_h


    def compute_rois(
        image: cv2.Mat,
        keypoints: Iterable[cv2.KeyPoint],
        search_window: Tuple[int, int] = (150, 80),
        padding: Tuple[int, int] = (5, 5),
    ) -> List[ROI]:
        """Compute global ROIs for each keypoint."""
        auto_rois: List[ROI] = []
        for keypoint in keypoints:
            cx, cy = int(keypoint.pt[0]), int(keypoint.pt[1])
            roi_bounds, local_binary = adaptive_threshold_roi(
                image, search_window, (cx, cy)
            )
            if roi_bounds is None or local_binary is None:
                continue

            contours, _ = cv2.findContours(
                local_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            local_point = (cx - roi_bounds[0], cy - roi_bounds[1])
            target_contour = find_containing_contour(contours, local_point)
            if target_contour is None:
                continue

            global_roi = expand_to_global_roi(
                target_contour, roi_bounds, image.shape, padding
            )
            if global_roi is None:
                continue

            auto_rois.append(global_roi)
        return auto_rois


    def annotate_keypoints(
        image: cv2.Mat,
        keypoints: Iterable[cv2.KeyPoint],
        rois: Iterable[ROI],
    ) -> cv2.Mat:
        """Draw ROIs and index labels on the image."""
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for roi in rois:
            x, y, w, h = roi
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

        for index, keypoint in enumerate(keypoints):
            x, y = keypoint.pt
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


    image = load_grayscale_image(image_path)
    detector = build_blob_detector()
    keypoints = detect_blobs(image, detector)
    rois = compute_rois(image, keypoints)
    show_rois(rois)

    annotated_image = annotate_keypoints(image, keypoints, rois)
    # save the annotated image
    annotated_image_path = Path(image_path).parent / f"{Path(image_path).stem}_annotated.png"
    cv2.imwrite(str(annotated_image_path), annotated_image)
    return str(annotated_image_path.resolve())