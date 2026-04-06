"""
CLI script to test ROI extraction on a single image.

Draws all zone crops on the original image and saves one output file.
Also saves each crop individually so you can inspect them.

Usage:
    python scripts/test_roi.py --image path/to/frame.jpg
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from detectors.yolo_detector import BusDetector
from roi.extractor import extract_rois

ZONE_COLORS = {
    "rear_center":        (0,   200, 0),    # green
    "lateral_front_low":  (0,   150, 255),  # orange
    "front_lateral_low":  (255, 50,  50),   # blue
    "lower_half":         (180, 0,   180),  # purple
}


def main():
    parser = argparse.ArgumentParser(description="Test ROI extraction on an image")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    image_path = Path(args.image)
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)

    # --- Step 1: detect bus ---
    detector = BusDetector()
    detections = detector.detect(frame)

    if not detections:
        print("[RESULT] No bus detected — cannot extract ROIs.")
        sys.exit(0)

    print(f"[INFO] {len(detections)} bus(es) detected, using best one.")
    best = detections[0]
    x1, y1, x2, y2 = best.bbox
    print(f"  Bus bbox: ({x1},{y1})-({x2},{y2})  conf={best.confidence:.3f}")

    # Draw bus bbox on annotated image
    annotated = frame.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # --- Step 2: extract ROIs ---
    crops = extract_rois(frame, best)
    print(f"\n[INFO] {len(crops)} ROI zone(s) extracted:")

    out_dir = image_path.parent / f"{image_path.stem}_rois"
    out_dir.mkdir(exist_ok=True)

    for crop in crops:
        ax1, ay1, ax2, ay2 = crop.abs_bbox
        color = ZONE_COLORS.get(crop.zone_name, (200, 200, 200))
        w = ax2 - ax1
        h = ay2 - ay1

        print(f"  [{crop.zone_name}]  ({ax1},{ay1})-({ax2},{ay2})  {w}x{h}px")

        # Draw zone on annotated frame
        cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), color, 2)
        cv2.putText(
            annotated, crop.zone_name,
            (ax1 + 4, ay1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )

        # Save individual crop
        crop_path = out_dir / f"{crop.zone_name}.jpg"
        cv2.imwrite(str(crop_path), crop.image)

    # Save annotated full frame
    out_path = image_path.parent / f"{image_path.stem}_roi_zones.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"\n[INFO] Annotated image → {out_path}")
    print(f"[INFO] Individual crops → {out_dir}/")


if __name__ == "__main__":
    main()
