"""
OCR reader: runs EasyOCR on preprocessed image variants and returns
raw numeric candidates with their confidence scores.
"""

import re
from dataclasses import dataclass

import easyocr
import numpy as np

# Shared reader instance (expensive to init — load once)
_reader: easyocr.Reader | None = None


def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


@dataclass
class RawCandidate:
    text: str               # raw OCR string (digits only)
    confidence: float       # EasyOCR confidence 0-1
    zone_name: str
    variant_name: str


# Accept 1-4 digit strings; 5+ are discarded (no bus has a 5-digit fleet number)
_DIGIT_RE = re.compile(r"\d{1,4}")


def read_candidates(
    variant_image: np.ndarray,
    zone_name: str,
    variant_name: str,
) -> list[RawCandidate]:
    """
    Run OCR on a single preprocessed image and return numeric candidates.

    Only strings of 2-4 digits are kept; everything else is discarded.
    """
    reader = get_reader()

    # EasyOCR accepts BGR, grayscale, or binary images
    results = reader.readtext(variant_image, allowlist="0123456789", detail=1)

    candidates: list[RawCandidate] = []
    for (_bbox, text, conf) in results:
        text = text.strip()
        if _DIGIT_RE.fullmatch(text):
            candidates.append(
                RawCandidate(
                    text=text,
                    confidence=conf,
                    zone_name=zone_name,
                    variant_name=variant_name,
                )
            )

    return candidates
