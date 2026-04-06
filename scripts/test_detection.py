"""
CLI script to test bus detection on a single image.

Usage:
    python scripts/test_detection.py --image path/to/frame.jpg
    python scripts/test_detection.py --image path/to/frame.jpg --conf 0.3 --save
"""

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from detectors.yolo_detector import BusDetector


def draw_detections(image, detections):
    """Draw bounding boxes and confidence scores on image."""
    output = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        label = f"bus {det.confidence:.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color=(0, 200, 0), thickness=2)
        cv2.putText(
            output,
            label,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 200, 0),
            thickness=2,
        )

    return output


def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8 bus detection on an image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.40, help="Min confidence threshold")
    parser.add_argument("--save", action="store_true", help="Save annotated output image")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    print(f"[INFO] Loading image: {image_path}")
    detector = BusDetector(min_confidence=args.conf)

    detections = detector.detect_from_file(image_path)

    if not detections:
        print("[RESULT] No buses detected.")
        return

    print(f"[RESULT] {len(detections)} bus(es) detected:")
    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det.bbox
        print(
            f"  [{i}] confidence={det.confidence:.3f}  "
            f"bbox=({x1},{y1})-({x2},{y2})  "
            f"size={det.width}x{det.height}px"
        )

    if args.save:
        import cv2 as _cv2
        image = _cv2.imread(str(image_path))
        output = draw_detections(image, detections)
        out_path = image_path.parent / f"{image_path.stem}_detected{image_path.suffix}"
        _cv2.imwrite(str(out_path), output)
        print(f"[INFO] Annotated image saved to: {out_path}")


if __name__ == "__main__":
    main()
