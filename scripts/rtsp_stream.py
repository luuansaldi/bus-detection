"""
Real-time bus number detection from RTSP stream.
Switch cameras with keys 1 / 2 / 3 / 4. Quit with Q.

Usage:
    python scripts/rtsp_stream.py
    python scripts/rtsp_stream.py --skip 5 --confirm 3
"""

import argparse
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import os
# Must be set before importing cv2
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|loglevel;error"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from detectors.yolo_detector import BusDetector
from roi.extractor import extract_rois
from preprocessing.image_processor import process
from ocr.reader import read_candidates
from filters.candidate_filter import select_best

BASE_URL = "rtsp://test:fono1234@190.220.138.178:34224/cam/realmonitor"
RECONNECT_DELAY = 5  # seconds between reconnect attempts

CAMERAS = {
    1: {"channel": 1,  "label": "Camara 1"},
    2: {"channel": 5,  "label": "Camara 2"},
    3: {"channel": 9,  "label": "Camara 3"},
    4: {"channel": 13, "label": "Camara 4"},
}


def build_url(channel: int) -> str:
    return f"{BASE_URL}?channel={channel}&subtype=0"


def process_frame(frame, detector: BusDetector) -> int | None:
    """Run the full detection pipeline on a single frame. Returns fleet number or None."""
    detections = detector.detect(frame)

    if not detections:
        return None

    crops = extract_rois(frame, detections[0])
    if not crops:
        return None

    all_candidates = []
    for crop in crops:
        variants = process(crop.image)
        for variant in variants:
            candidates = read_candidates(variant.image, crop.zone_name, variant.name)
            all_candidates.extend(candidates)

    result = select_best(all_candidates)
    return result.number if result else None


def overlay_hud(frame, cam_key: int, confirmed_bus: int | None, pending_count: int, confirm_needed: int) -> None:
    """Draw camera label, confirmation progress, and confirmed bus number on frame."""
    cam = CAMERAS[cam_key]
    h, w = frame.shape[:2]

    # camera label — top left
    cv2.putText(frame, cam["label"], (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # confirmation progress — top left below label
    if pending_count > 0:
        bar = f"Confirmando: {'|' * pending_count}{'.' * (confirm_needed - pending_count)}  ({pending_count}/{confirm_needed})"
        cv2.putText(frame, bar, (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

    # hint — bottom left
    hint = "[ 1 | 2 | 3 | 4 ] camara   [ Q ] salir"
    cv2.putText(frame, hint, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # confirmed bus number — top right
    if confirmed_bus is not None:
        label = f"BUS: {confirmed_bus}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.putText(frame, label, (w - tw - 15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)


class ConfirmationBuffer:
    """
    Sliding window confirmation: confirms a number if it appears `needed` times
    within the last `window` processed frames. Much better for fast-moving buses
    than requiring N consecutive identical detections.
    """

    def __init__(self, needed: int = 2, window: int = 5):
        self.needed = needed
        self.window = window
        self._history: list[int | None] = []
        self._reported: set[int] = set()

    def feed(self, number: int | None) -> int | None:
        self._history.append(number)
        if len(self._history) > self.window:
            self._history.pop(0)

        counts: dict[int, int] = {}
        for n in self._history:
            if n is not None:
                counts[n] = counts.get(n, 0) + 1

        for n, count in counts.items():
            if count >= self.needed and n not in self._reported:
                self._reported.add(n)
                return n

        visible = {n for n in self._history if n is not None}
        self._reported &= visible

        return None

    @property
    def pending_count(self) -> int:
        counts: dict[int, int] = {}
        for n in self._history:
            if n is not None:
                counts[n] = counts.get(n, 0) + 1
        return max(counts.values(), default=0)

    @property
    def pending_candidate(self) -> int | None:
        counts: dict[int, int] = {}
        for n in self._history:
            if n is not None:
                counts[n] = counts.get(n, 0) + 1
        if not counts:
            return None
        return max(counts, key=lambda n: counts[n])


def connect_rtsp(url: str) -> cv2.VideoCapture | None:
    """Open RTSP stream. Returns capture object or None on failure."""
    print(f"[INFO] Conectando a {url} ...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir el stream.")
        return None
    print("[INFO] Stream conectado.")
    return cap


def main_loop(skip: int = 3, confirm: int = 3) -> None:
    print("[INFO] Cargando modelo YOLO ...")
    detector = BusDetector()
    print("[INFO] Modelo listo.\n")

    cap = None
    frame_count = 0
    current_cam = 1
    pending_switch = None
    confirmed_bus = None
    buffer = ConfirmationBuffer(needed=confirm)

    while True:
        # ── Handle camera switch ──────────────────────────────────────────
        if pending_switch is not None and pending_switch != current_cam:
            print(f"[INFO] Cambiando a {CAMERAS[pending_switch]['label']} ...")
            if cap:
                cap.release()
                cap = None
            current_cam = pending_switch
            frame_count = 0
            confirmed_bus = None
            buffer = ConfirmationBuffer(needed=confirm)
            pending_switch = None

        # ── Connect / reconnect ───────────────────────────────────────────
        if cap is None or not cap.isOpened():
            url = build_url(CAMERAS[current_cam]["channel"])
            cap = connect_rtsp(url)
            if cap is None:
                print(f"[INFO] Reintentando en {RECONNECT_DELAY}s ...")
                time.sleep(RECONNECT_DELAY)
                continue
            frame_count = 0

        # ── Read frame ────────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Frame perdido, reconectando ...")
            cap.release()
            cap = None
            continue

        frame_count += 1

        # ── Process every Nth frame ───────────────────────────────────────
        if frame_count % skip == 0:
            try:
                raw = process_frame(frame, detector)
            except Exception as e:
                print(f"[ERROR] Pipeline falló en frame {frame_count}: {e}")
                raw = None

            newly_confirmed = buffer.feed(raw)

            if newly_confirmed is not None:
                confirmed_bus = newly_confirmed
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{ts}] {CAMERAS[current_cam]['label'].upper()} — BUS DETECTED: {confirmed_bus}")

        # ── Display ───────────────────────────────────────────────────────
        overlay_hud(frame, current_cam, confirmed_bus,
                    buffer.pending_count, confirm)
        cv2.imshow("FonoBus", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            pending_switch = 1
        elif key == ord("2"):
            pending_switch = 2
        elif key == ord("3"):
            pending_switch = 3
        elif key == ord("4"):
            pending_switch = 4

    # ── Cleanup ───────────────────────────────────────────────────────────
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stream cerrado.")


def main():
    parser = argparse.ArgumentParser(description="Detectar buses en stream RTSP en tiempo real")
    parser.add_argument("--skip",    type=int, default=3, help="Procesar 1 de cada N frames (default: 3)")
    parser.add_argument("--confirm", type=int, default=3, help="Detecciones consecutivas para confirmar (default: 3)")
    args = parser.parse_args()

    print("Controles: [ 1 | 2 | 3 | 4 ] camara   [ Q ] salir\n")
    main_loop(skip=args.skip, confirm=args.confirm)


if __name__ == "__main__":
    main()
