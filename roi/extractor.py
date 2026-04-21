"""
ROI extractor: crops regions inside a bus bounding box where the fleet number
is likely to appear.

Moondream uses the full bus bounding box — no sub-zone cropping.
"""

import numpy as np

from config.settings import (
    CROP_PAD_X_FRAC,
    CROP_PAD_Y_FRAC,
    CROP_MIN_PAD_PX,
    CAPTURE_TIMESTAMP_TOP_PX,
    CAPTURE_TIMESTAMP_TOP_FRAC,
)
from detectors.yolo_detector import BusDetection
from preprocessing.night_enhancer import enhance, is_dark_frame


MIN_CROP_PX = 30


def _timestamp_band_px(frame_height: int) -> int:
    """Alto efectivo de la franja del timestamp del DVR, en píxeles."""
    return min(
        frame_height,
        max(CAPTURE_TIMESTAMP_TOP_PX, int(frame_height * CAPTURE_TIMESTAMP_TOP_FRAC)),
    )


def mask_dvr_timestamp(frame: np.ndarray) -> np.ndarray:
    """
    Devuelve una copia del frame con la franja superior del timestamp del DVR
    pintada de negro. Conserva la resolución — útil para pasar a Moondream sin
    que lea la fecha como número de flota, y para guardar capturas a máxima
    resolución sin recortar.
    """
    if frame is None or frame.size == 0:
        return frame
    out = frame.copy()
    band = _timestamp_band_px(out.shape[0])
    if band > 0:
        out[:band, :] = 0
    return out


def extract_full_bus_crop(
    frame: np.ndarray,
    detection: BusDetection,
    min_px: int = MIN_CROP_PX,
) -> np.ndarray | None:
    """
    Return the full bus bounding box as a BGR crop, or None if too small.
    La franja del timestamp se pinta negro antes del slice para no achicar el crop.
    """
    x1, y1, x2, y2 = detection.bbox
    pad_x = max(CROP_MIN_PAD_PX, int((x2 - x1) * CROP_PAD_X_FRAC))
    pad_y = max(CROP_MIN_PAD_PX, int((y2 - y1) * CROP_PAD_Y_FRAC))
    x1 = max(0, x1 - pad_x)
    x2 = min(detection.frame_width, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(detection.frame_height, y2 + pad_y)
    x1 = max(0, min(x1, detection.frame_width - 1))
    y1 = max(0, min(y1, detection.frame_height - 1))
    x2 = max(0, min(x2, detection.frame_width))
    y2 = max(0, min(y2, detection.frame_height))

    if (x2 - x1) < min_px or (y2 - y1) < min_px:
        return None

    source = mask_dvr_timestamp(frame)
    crop = source[y1:y2, x1:x2].copy()
    if is_dark_frame(frame):
        crop = enhance(crop)
    return crop
