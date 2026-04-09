"""
ROI extractor: crops regions inside a bus bounding box where the fleet number
is likely to appear.

Since we cannot know the camera angle automatically yet, we define several
named zones that together cover all the positions observed in the field:

  - rear_center       → rear-view cameras: number painted on upper-center panel
  - lateral_front_low → lateral cameras: number on lower-front area of the body
  - front_lateral_low → front-lateral cameras: number on lower-right of the front

Each zone is expressed as (x1_frac, y1_frac, x2_frac, y2_frac) relative to the
bus bounding box.  All fractions are clamped to [0, 1] so partial bboxes near
the frame edges are handled safely.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from detectors.yolo_detector import BusDetection


# ---------------------------------------------------------------------------
# Zone definitions  (relative to bus bounding box)
# ---------------------------------------------------------------------------

ZONES: dict[str, tuple[float, float, float, float]] = {
    # Rear-view: number is painted on the upper-center panel of the rear face
    "rear_center": (0.05, 0.25, 0.70, 0.58),

    # Lateral overhead: number appears on the lower-front section of the body
    "lateral_front_low": (0.00, 0.68, 0.22, 0.98),

    # Front-lateral: number on the lower-right area of the front face
    "front_lateral_low": (0.45, 0.68, 0.80, 0.98),

}

# Minimum crop size in pixels; smaller crops are skipped (not enough info for OCR)
MIN_CROP_PX = 30


@dataclass
class ROICrop:
    """A single named region of interest cropped from a bus detection."""
    zone_name: str
    image: np.ndarray          # BGR crop, ready for preprocessing
    bus_bbox: tuple[int, int, int, int]   # original bus bbox for reference
    abs_bbox: tuple[int, int, int, int]   # absolute pixel coords of this crop


def extract_full_bus_crop(
    frame: np.ndarray,
    detection: BusDetection,
    min_px: int = MIN_CROP_PX,
) -> np.ndarray | None:
    """
    Return the full bus bounding box as a BGR crop, or None if too small.

    Used by the Claude vision backend, which doesn't need tight ROI zones
    and can locate the number anywhere in the bus image.
    """
    x1, y1, x2, y2 = detection.bbox
    # Add horizontal padding so edge digits aren't clipped by the YOLO bbox.
    pad_x = int((x2 - x1) * 0.08)
    x1 = max(0, x1 - pad_x)
    x2 = min(detection.frame_width, x2 + pad_x)
    x1 = max(0, min(x1, detection.frame_width - 1))
    y1 = max(0, min(y1, detection.frame_height - 1))
    x2 = max(0, min(x2, detection.frame_width))
    y2 = max(0, min(y2, detection.frame_height))

    if (x2 - x1) < min_px or (y2 - y1) < min_px:
        return None

    return frame[y1:y2, x1:x2].copy()


def extract_rois(
    frame: np.ndarray,
    detection: BusDetection,
    zones: dict[str, tuple[float, float, float, float]] = ZONES,
) -> list[ROICrop]:
    """
    Extract all zone crops from a single bus detection.

    Args:
        frame:      Full BGR frame from the camera.
        detection:  A BusDetection (contains bbox and frame dimensions).
        zones:      Zone definitions to use (defaults to ZONES above).

    Returns:
        List of ROICrop, one per zone that produced a valid (non-empty) crop.
    """
    x1, y1, x2, y2 = detection.bbox
    bus_w = x2 - x1
    bus_h = y2 - y1

    crops: list[ROICrop] = []

    for zone_name, (rx1, ry1, rx2, ry2) in zones.items():
        # Convert relative fractions to absolute pixel coords
        ax1 = x1 + int(rx1 * bus_w)
        ay1 = y1 + int(ry1 * bus_h)
        ax2 = x1 + int(rx2 * bus_w)
        ay2 = y1 + int(ry2 * bus_h)

        # Clamp to frame bounds
        ax1 = max(0, min(ax1, detection.frame_width - 1))
        ay1 = max(0, min(ay1, detection.frame_height - 1))
        ax2 = max(0, min(ax2, detection.frame_width))
        ay2 = max(0, min(ay2, detection.frame_height))

        crop_w = ax2 - ax1
        crop_h = ay2 - ay1

        if crop_w < MIN_CROP_PX or crop_h < MIN_CROP_PX:
            continue

        crop = frame[ay1:ay2, ax1:ax2].copy()

        crops.append(
            ROICrop(
                zone_name=zone_name,
                image=crop,
                bus_bbox=detection.bbox,
                abs_bbox=(ax1, ay1, ax2, ay2),
            )
        )

    return crops
