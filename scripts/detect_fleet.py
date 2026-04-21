"""
MVP: detect the fleet number of a bus from a single image (Moondream).

Usage:
    python scripts/detect_fleet.py --image path/to/frame.jpg
    python scripts/detect_fleet.py --image path/to/frame.jpg --verbose
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from detectors.yolo_detector import BusDetector
from roi.extractor import extract_full_bus_crop
from ocr.moondream_reader import get_moondream_reader


def run(image_path: Path, verbose: bool = False) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERROR] No se pudo leer la imagen: {image_path}")
        sys.exit(1)

    detector = BusDetector()
    detections = detector.detect(frame)

    if not detections:
        print("[RESULTADO] No se detectó ningún bus.")
        return

    best_detection = detections[0]
    if verbose:
        x1, y1, x2, y2 = best_detection.bbox
        print(f"[YOLO] Bus detectado  conf={best_detection.confidence:.2f}  bbox=({x1},{y1})-({x2},{y2})")

    bus_crop = extract_full_bus_crop(frame, best_detection)
    if bus_crop is None:
        print("\n[RESULTADO] Crop del bus demasiado pequeño.")
        return
    if verbose:
        print(f"[CROP] Crop {bus_crop.shape[1]}×{bus_crop.shape[0]}px → moondream")

    number = get_moondream_reader().read(bus_crop, cam_label="detect_fleet")
    if number:
        print(f"\n[RESULTADO] Número de flota: {number}")
    else:
        print("\n[RESULTADO] No se encontró número de flota válido.")


def main():
    parser = argparse.ArgumentParser(description="Detectar número de flota de un bus")
    parser.add_argument("--image", required=True, help="Ruta a la imagen")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mostrar logs detallados")
    args = parser.parse_args()

    run(Path(args.image), verbose=args.verbose)


if __name__ == "__main__":
    main()
