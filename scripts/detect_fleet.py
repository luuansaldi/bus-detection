"""
MVP: detect the fleet number of a bus from a single image.

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
from roi.extractor import extract_rois, extract_full_bus_crop
from preprocessing.image_processor import process
from ocr.reader import read_candidates
from filters.candidate_filter import select_best


def run(image_path: Path, verbose: bool = False, ocr_backend: str = "claude") -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERROR] No se pudo leer la imagen: {image_path}")
        sys.exit(1)

    # ── Step 1: detect bus ────────────────────────────────────────────────
    detector = BusDetector()
    detections = detector.detect(frame)

    if not detections:
        print("[RESULTADO] No se detectó ningún bus.")
        return

    best_detection = detections[0]
    if verbose:
        x1, y1, x2, y2 = best_detection.bbox
        print(f"[YOLO] Bus detectado  conf={best_detection.confidence:.2f}  bbox=({x1},{y1})-({x2},{y2})")

    if ocr_backend == "moondream":
        # ── Vision model path: send full bus crop ─────────────────────────
        from ocr.moondream_reader import get_moondream_reader
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
        return

    # ── EasyOCR path (legacy) ─────────────────────────────────────────────

    # ── Step 2: extract ROIs ──────────────────────────────────────────────
    crops = extract_rois(frame, best_detection)
    if verbose:
        print(f"[ROI]  {len(crops)} zona(s) extraídas: {[c.zone_name for c in crops]}")

    # ── Step 3 + 4: preprocess each crop and run OCR ──────────────────────
    all_candidates = []

    for crop in crops:
        variants = process(crop.image)
        for variant in variants:
            candidates = read_candidates(variant.image, crop.zone_name, variant.name)
            if verbose and candidates:
                for c in candidates:
                    print(f"[OCR]  zona={c.zone_name}  variante={c.variant_name}  "
                          f"texto={c.text}  conf={c.confidence:.2f}")
            all_candidates.extend(candidates)

    # ── Step 5: filter and select ─────────────────────────────────────────
    result = select_best(all_candidates)

    if result:
        print(f"\n[RESULTADO] Número de flota: {result.number}")
        if verbose:
            print(f"            score={result.score:.2f}  conf_raw={result.raw_confidence:.2f}"
                  f"  zona={result.zone_name}  variante={result.variant_name}")
    else:
        print("\n[RESULTADO] No se encontró número de flota válido.")


def main():
    parser = argparse.ArgumentParser(description="Detectar número de flota de un bus")
    parser.add_argument("--image", required=True, help="Ruta a la imagen")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mostrar logs detallados")
    parser.add_argument(
        "--ocr-backend",
        choices=["moondream", "easyocr"],
        default="moondream",
        help="Backend OCR: 'moondream' (default, local) o 'easyocr' (legacy)",
    )
    args = parser.parse_args()

    run(Path(args.image), verbose=args.verbose, ocr_backend=args.ocr_backend)


if __name__ == "__main__":
    main()
