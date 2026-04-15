#!/usr/bin/env python3
"""
Scan all captures in the DB, remove entries where no bus is visible,
and reassign the main image to the best available crop per detection.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3
import cv2
from detectors.yolo_detector import BusDetector

DB_PATH = PROJECT_ROOT / "detecciones.db"
CAPTURES_DIR = PROJECT_ROOT / "captures"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def has_bus(detector: BusDetector, image_path: str) -> bool:
    """Return True if YOLO detects at least one bus in the image."""
    if not os.path.exists(image_path):
        return False
    img = cv2.imread(image_path)
    if img is None:
        return False
    detections = detector.detect(img)
    return len(detections) > 0


def best_crop_for_detection(detector: BusDetector, crops: list[dict]) -> dict | None:
    """Return the crop with the best-framed bus (largest bbox area, not too close)."""
    best = None
    best_score = -1

    for crop in crops:
        path = crop["crop_path"]
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        detections = detector.detect(img)
        if not detections:
            continue

        fh, fw = img.shape[:2]
        frame_area = fh * fw
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            ratio = bbox_area / frame_area
            # Best: bus visible but not filling entire image
            # Prefer ratio 0.10-0.50 (bus well-framed)
            if ratio > 0.85:
                score = 0.1  # too close
            elif ratio < 0.03:
                score = 0.05  # too far
            else:
                ideal = 0.25
                score = max(0.0, 1.0 - abs(ratio - ideal) / ideal)
                score += bbox_area / 500_000  # bonus for resolution

            if score > best_score:
                best_score = score
                best = crop

    return best


def main():
    detector = BusDetector()
    conn = get_conn()
    cur = conn.cursor()

    # Get all detections
    detections = cur.execute(
        "SELECT id, numero_flota, direccion, imagen_path FROM detecciones ORDER BY id"
    ).fetchall()

    ids_to_delete = []
    images_to_delete = []
    updates = []

    print(f"Scanning {len(detections)} detections...\n")

    for det in detections:
        det_id = det["id"]
        flota = det["numero_flota"]
        main_path = det["imagen_path"]

        # Get crops for this detection
        crops = cur.execute(
            "SELECT cam_label, crop_path FROM detection_crops WHERE detection_id = ?",
            (det_id,),
        ).fetchall()
        crops = [dict(c) for c in crops]

        # Check if main image has a bus
        main_has_bus = has_bus(detector, main_path) if main_path else False

        # Check which crops have a bus
        valid_crops = []
        invalid_crop_paths = []
        for crop in crops:
            if has_bus(detector, crop["crop_path"]):
                valid_crops.append(crop)
            else:
                invalid_crop_paths.append(crop["crop_path"])

        # Decide action
        if not main_has_bus and not valid_crops:
            # No bus anywhere — delete this detection entirely
            print(f"  DELETE id={det_id} flota={flota}: no bus in any image")
            ids_to_delete.append(det_id)
            if main_path and os.path.exists(main_path):
                images_to_delete.append(main_path)
            for crop in crops:
                if os.path.exists(crop["crop_path"]):
                    images_to_delete.append(crop["crop_path"])
            continue

        # Remove invalid crops from DB
        for path in invalid_crop_paths:
            print(f"  REMOVE CROP id={det_id} flota={flota}: no bus in {Path(path).name}")
            cur.execute("DELETE FROM detection_crops WHERE detection_id = ? AND crop_path = ?",
                        (det_id, path))
            if os.path.exists(path):
                images_to_delete.append(path)

        # If main image has no bus but crops do, reassign main to best crop
        if not main_has_bus and valid_crops:
            best = best_crop_for_detection(detector, valid_crops)
            if best:
                print(f"  REASSIGN id={det_id} flota={flota}: main → {Path(best['crop_path']).name}")
                cur.execute("UPDATE detecciones SET imagen_path = ? WHERE id = ?",
                            (best["crop_path"], det_id))
                if main_path and os.path.exists(main_path):
                    images_to_delete.append(main_path)
        elif main_has_bus and valid_crops:
            # Main has bus, but maybe a crop is better framed?
            all_candidates = [{"crop_path": main_path, "cam_label": "main"}] + valid_crops
            best = best_crop_for_detection(detector, all_candidates)
            if best and best["crop_path"] != main_path:
                print(f"  UPGRADE id={det_id} flota={flota}: main → {Path(best['crop_path']).name}")
                cur.execute("UPDATE detecciones SET imagen_path = ? WHERE id = ?",
                            (best["crop_path"], det_id))

    # Delete detections with no bus
    if ids_to_delete:
        placeholders = ','.join('?' * len(ids_to_delete))
        cur.execute(f"DELETE FROM detection_crops WHERE detection_id IN ({placeholders})",
                    ids_to_delete)
        cur.execute(f"DELETE FROM comparaciones WHERE salida_id IN ({placeholders}) OR entrada_id IN ({placeholders})",
                    ids_to_delete + ids_to_delete)
        cur.execute(f"DELETE FROM detecciones WHERE id IN ({placeholders})",
                    ids_to_delete)

    conn.commit()

    # Delete image files
    deleted_files = 0
    for path in images_to_delete:
        try:
            os.unlink(path)
            deleted_files += 1
        except OSError:
            pass

    print(f"\nDone:")
    print(f"  Detections deleted: {len(ids_to_delete)}")
    print(f"  Image files deleted: {deleted_files}")
    print(f"  Main images reassigned: {sum(1 for u in updates)}")

    conn.close()


if __name__ == "__main__":
    main()
