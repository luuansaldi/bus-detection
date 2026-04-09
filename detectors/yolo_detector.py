"""
Bus detector using YOLOv8 (Ultralytics).

Detects buses in an image or video frame and returns their bounding boxes.
Only the COCO "bus" class (id=5) is returned.
"""

from dataclasses import dataclass
from pathlib import Path

import threading

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config.settings import YOLO_BUS_CLASS_IDS, YOLO_MIN_CONFIDENCE, YOLO_MODEL


@dataclass
class BusDetection:
    """A single bus detection result."""
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixels
    confidence: float
    frame_width: int
    frame_height: int

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


class BusDetector:
    """
    Wraps YOLOv8 to detect buses in images or frames.

    Usage:
        detector = BusDetector()
        detections = detector.detect(image)
    """

    def __init__(
        self,
        model_path: str = YOLO_MODEL,
        min_confidence: float = YOLO_MIN_CONFIDENCE,
        class_ids: list[int] = YOLO_BUS_CLASS_IDS,
    ) -> None:
        self.min_confidence = min_confidence
        self.class_ids = class_ids
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = YOLO(model_path)
        self._model.to(self._device)
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray) -> list[BusDetection]:
        """
        Run detection on a single BGR image (as returned by cv2.imread).

        Returns a list of BusDetection, one per bus found.
        Results are sorted by confidence descending.
        """
        frame_h, frame_w = image.shape[:2]

        with self._lock:
            results = self._model(
                image,
                classes=self.class_ids,
                conf=self.min_confidence,
                verbose=False,
                device=self._device,
            )

        detections: list[BusDetection] = []

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

                # Clamp to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_w, x2)
                y2 = min(frame_h, y2)

                detections.append(
                    BusDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        frame_width=frame_w,
                        frame_height=frame_h,
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_from_file(self, image_path: str | Path) -> list[BusDetection]:
        """Convenience wrapper that loads an image file and runs detection."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not decode image: {path}")

        return self.detect(image)
