"""
Image preprocessor: generates multiple enhanced variants of an ROI crop.

Each variant is a different attempt to make the fleet number more readable
for OCR. We run OCR on all variants and keep the best result.

Variants produced per crop:
  - upscaled      : 3x bicubic upscale (more pixels for OCR)
  - gray_clahe    : grayscale + CLAHE (contrast normalization)
  - otsu          : Otsu binarization on top of CLAHE
  - otsu_inv      : inverted Otsu (white text on black background)
  - morph         : denoised + sharpened version
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ProcessedVariant:
    name: str
    image: np.ndarray   # grayscale or binary, ready for OCR


def process(crop: np.ndarray, upscale: int = 3) -> list[ProcessedVariant]:
    """
    Generate preprocessing variants for a single BGR crop.

    Args:
        crop:    BGR image (ROI from the bus bounding box).
        upscale: Upscale factor applied before all other transforms.

    Returns:
        List of ProcessedVariant, one per technique.
    """
    if crop is None or crop.size == 0:
        return []

    variants: list[ProcessedVariant] = []

    # --- 1. Upscale ---
    h, w = crop.shape[:2]
    big = cv2.resize(crop, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)
    variants.append(ProcessedVariant("upscaled", big))

    # --- 2. Grayscale + CLAHE ---
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    variants.append(ProcessedVariant("gray_clahe", gray_clahe))

    # --- 3. Otsu threshold (dark text on light background) ---
    _, otsu = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(ProcessedVariant("otsu", otsu))

    # --- 4. Inverted Otsu (light text on dark background) ---
    variants.append(ProcessedVariant("otsu_inv", cv2.bitwise_not(otsu)))

    # --- 5. Morphological denoising + sharpening ---
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(gray_clahe, cv2.MORPH_CLOSE, kernel)
    sharpen_kernel = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
    morph = cv2.filter2D(morph, -1, sharpen_kernel)
    variants.append(ProcessedVariant("morph", morph))

    return variants
