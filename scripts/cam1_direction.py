"""
Cam 1 — detección de dirección y movimiento de buses.

Solo procesa la cámara de la barrera (Cam 1). Muestra video con overlay
de zonas y trayectoria, y loguea en consola cada bus detectado con su dirección.

Uso:
    python scripts/cam1_direction.py
    python scripts/cam1_direction.py --debug
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|loglevel;error"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from detectors.yolo_detector import BusDetector, BusDetection

BASE_URL = "rtsp://test:fono1234@190.220.138.178:34224/cam/realmonitor"
CAM1_CHANNEL = 1
RECONNECT_DELAY = 5

# Líneas virtuales de barrera (fracción del ancho del frame)
LINE_RED_X    = 0.80   # zona A: exterior (derecha)
LINE_YELLOW_X = 0.18   # zona B: depósito (izquierda)

# Parámetros de movimiento
ACTIVATION_DISTANCE_PX = 30   # desplazamiento EMA mínimo para activar
MIN_DIRECTION_PX       = 40   # dx mínimo para determinar dirección por desplazamiento
MOVE_THRESHOLD_PX      = 50   # movimiento mínimo entre frames para resetear static_count
STATIC_LIMIT           = 8    # máx frames consecutivos sin moverse antes de marcar estático
SLOT_RADIUS_PX         = 400  # radio para reasignar detecciones al mismo slot
PARKED_TIMEOUT_SEC     = 30.0 # tiempo máximo en slot sin cruzar línea → estacionado
DIRECTION_MIN_FRAMES   = 2    # frames consistentes para confirmar dirección por desplazamiento

# Cooldown para no reportar el mismo slot dos veces seguidas
REPORT_COOLDOWN_SEC = 15.0


class MotionTracker:
    """
    Rastrea buses entre frames y determina dirección (entering/exiting).
    Usa EMA para filtrar jitter de YOLO.
    """

    def __init__(self):
        self._slots: dict[int, dict] = {}
        self._next_slot = 0

    def _find_slot(self, cx: float, cy: float) -> int | None:
        best_id, best_dist = None, float("inf")
        for sid, slot in self._slots.items():
            dist = ((cx - slot["cx_last"]) ** 2 + (cy - slot["cy"]) ** 2) ** 0.5
            if dist < SLOT_RADIUS_PX and dist < best_dist:
                best_dist, best_id = dist, sid
        return best_id

    def update(self, detection: BusDetection, frame_width: int) -> dict | None:
        """
        Actualiza el tracker con una nueva detección.
        Devuelve el estado del slot si el bus está activo (moviéndose), o None si estático.
        """
        x1, y1, x2, y2 = detection.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        now = time.monotonic()

        sid = self._find_slot(cx, cy)
        if sid is None:
            # Bus nuevo
            self._slots[self._next_slot] = {
                "cx_last": cx, "cy": cy,
                "cx_first": cx, "cy_first": cy,
                "ema_cx": cx, "ema_cy": cy,
                "static_count": 1,
                "first_crossing": None,
                "confirmed_direction": None,
                "direction_candidate": None,
                "direction_streak": 0,
                "first_seen_time": now,
                "last_seen_time": now,
                "zones_visited": [],
                "last_reported": 0.0,
            }
            self._next_slot += 1
            return None

        slot = self._slots[sid]
        prev_cx = slot["cx_last"]
        dist = ((cx - prev_cx) ** 2 + (cy - slot["cy"]) ** 2) ** 0.5

        slot["static_count"] = 0 if dist > MOVE_THRESHOLD_PX else slot["static_count"] + 1

        # EMA (α=0.35)
        alpha = 0.35
        slot["ema_cx"] = alpha * cx + (1 - alpha) * slot["ema_cx"]
        slot["ema_cy"] = alpha * cy + (1 - alpha) * slot["ema_cy"]

        # Detección de zona (barrera Cam 1)
        red_x    = frame_width * LINE_RED_X
        yellow_x = frame_width * LINE_YELLOW_X
        if cx > red_x:
            current_zone = "A"
        elif cx < yellow_x:
            current_zone = "B"
        else:
            current_zone = None

        zones = slot["zones_visited"]
        if current_zone and (not zones or zones[-1] != current_zone):
            zones.append(current_zone)

        if len(zones) >= 2 and slot["first_crossing"] is None:
            if zones[0] == "A":
                slot["first_crossing"] = "entering"
            elif zones[0] == "B":
                slot["first_crossing"] = "exiting"

        slot["cx_last"] = cx
        slot["cy"] = cy
        slot["last_seen_time"] = now

        # Activación: EMA debe haber viajado ACTIVATION_DISTANCE_PX desde el origen
        ema_disp = ((slot["ema_cx"] - slot["cx_first"]) ** 2 +
                    (slot["ema_cy"] - slot["cy_first"]) ** 2) ** 0.5
        if ema_disp < ACTIVATION_DISTANCE_PX:
            return None

        # Timeout de estacionado
        if slot["first_crossing"] is None and now - slot["first_seen_time"] > PARKED_TIMEOUT_SEC:
            return None

        # Verificar que no esté completamente quieto
        if slot["static_count"] >= STATIC_LIMIT:
            return None

        return slot

    def get_direction(self, slot: dict) -> str:
        # 1. Cruce de línea (señal dura)
        if slot["first_crossing"]:
            return slot["first_crossing"]

        # 2. Dirección ya confirmada por histéresis
        if slot["confirmed_direction"]:
            return slot["confirmed_direction"]

        # 3. Desplazamiento neto con histéresis
        net_dx = slot["ema_cx"] - slot["cx_first"]
        if net_dx < -MIN_DIRECTION_PX:
            candidate = "entering"
        elif net_dx > MIN_DIRECTION_PX:
            candidate = "exiting"
        else:
            candidate = None

        if candidate is None:
            slot["direction_candidate"] = None
            slot["direction_streak"] = 0
        elif candidate == slot["direction_candidate"]:
            slot["direction_streak"] += 1
            if slot["direction_streak"] >= DIRECTION_MIN_FRAMES:
                slot["confirmed_direction"] = candidate
        else:
            slot["direction_candidate"] = candidate
            slot["direction_streak"] = 1

        return slot["confirmed_direction"] or "unknown"

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        now = time.monotonic()

        red_x    = int(w * LINE_RED_X)
        yellow_x = int(w * LINE_YELLOW_X)

        # Zonas semitransparentes
        overlay = out.copy()
        cv2.rectangle(overlay, (red_x, 0), (w, h), (0, 0, 200), -1)
        cv2.rectangle(overlay, (0, 0), (yellow_x, h), (0, 200, 200), -1)
        cv2.addWeighted(overlay, 0.08, out, 0.92, 0, out)

        # Líneas de barrera
        cv2.line(out, (red_x, 0),    (red_x, h),    (0, 0, 255), 2)
        cv2.line(out, (yellow_x, 0), (yellow_x, h), (0, 255, 255), 2)
        cv2.putText(out, "ENTRA", (red_x - 65, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        cv2.putText(out, "SALE",  (yellow_x + 5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        for slot in self._slots.values():
            if now - slot["last_seen_time"] > 5.0:
                continue

            cx_first = int(slot["cx_first"])
            cx_last  = int(slot["cx_last"])
            cy       = int(slot["cy"])
            confirmed = slot["confirmed_direction"] or slot["first_crossing"]
            candidate = slot["direction_candidate"]
            streak    = slot["direction_streak"]
            zones_str = "→".join(slot["zones_visited"])
            net_dx    = slot["cx_last"] - slot["cx_first"]

            color = (0, 255, 0) if confirmed else (0, 165, 255)
            cv2.arrowedLine(out, (cx_first, cy), (cx_last, cy), color, 2, tipLength=0.3)
            cv2.circle(out, (cx_last, cy), 6, color, -1)

            dir_text = confirmed or (f"{candidate}?({streak}/{DIRECTION_MIN_FRAMES})" if candidate else "?")
            label = f"dx={int(net_dx)} zones={zones_str} [{dir_text}]"
            cv2.putText(out, label, (cx_last + 8, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return out

    def cleanup_stale(self, max_age_sec: float = 60.0) -> None:
        now = time.monotonic()
        stale = [sid for sid, s in self._slots.items()
                 if now - s["last_seen_time"] > max_age_sec]
        for sid in stale:
            del self._slots[sid]


def main():
    parser = argparse.ArgumentParser(description="Cam 1 — detección de dirección de buses")
    parser.add_argument("--debug", action="store_true", help="Logs detallados")
    parser.add_argument("--skip",  type=int, default=2, help="Procesar 1 de cada N frames (default: 2)")
    parser.add_argument("--fps",   type=int, default=10, help="FPS máximo de lectura (default: 10)")
    parser.add_argument("--min-bbox", type=int, default=80, help="Tamaño mínimo de bbox (default: 80)")
    args = parser.parse_args()

    url = f"{BASE_URL}?channel={CAM1_CHANNEL}&subtype=0"
    frame_interval = 1.0 / args.fps

    print("[INFO] Cargando YOLO...")
    detector = BusDetector()
    tracker  = MotionTracker()
    print("[INFO] Listo. Conectando a Cam 1...")
    print("Controles: [ Q ] salir\n")

    cap = None
    frame_count = 0
    last_cleanup = time.monotonic()

    while True:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print(f"[WARN] Sin conexión, reintentando en {RECONNECT_DELAY}s...")
                time.sleep(RECONNECT_DELAY)
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print("[INFO] Conectado a Cam 1.")
            frame_count = 0

        t0 = time.monotonic()
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            cap = None
            continue

        frame_count += 1
        h, w = frame.shape[:2]

        if frame_count % args.skip == 0:
            detections = detector.detect(frame)

            for det in detections:
                x1, y1, x2, y2 = det.bbox
                bw, bh = x2 - x1, y2 - y1

                if bw < args.min_bbox or bh < args.min_bbox:
                    if args.debug:
                        print(f"  SKIP: bbox pequeño ({bw}×{bh}px)")
                    continue

                if args.debug:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    print(f"  YOLO: bus conf={det.confidence:.2f} bbox=({x1},{y1})-({x2},{y2}) cx={cx:.0f}")

                slot = tracker.update(det, frame_width=w)
                if slot is None:
                    if args.debug:
                        print(f"  SKIP: bus estático o nuevo")
                    continue

                direction = tracker.get_direction(slot)
                now = time.monotonic()

                # Log solo si hay dirección y no está en cooldown
                if direction != "unknown":
                    if now - slot["last_reported"] > REPORT_COOLDOWN_SEC:
                        slot["last_reported"] = now
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        action = "ENTRÓ" if direction == "entering" else "SALIÓ"
                        print(f"[{ts}] Bus detectado → {action}  (zones={('→'.join(slot['zones_visited']))})")
                elif args.debug:
                    net_dx = slot["ema_cx"] - slot["cx_first"]
                    print(f"  ACTIVO: dx={net_dx:.0f} zones={'→'.join(slot['zones_visited'])} dir=unknown")

        # Cleanup slots viejos cada 30s
        if time.monotonic() - last_cleanup > 30:
            tracker.cleanup_stale()
            last_cleanup = time.monotonic()

        # Overlay y display
        display = tracker.draw_overlay(frame)
        cv2.imshow("Cam 1 — Detección de dirección", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        elapsed = time.monotonic() - t0
        remaining = frame_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    cv2.destroyAllWindows()
    if cap:
        cap.release()
    print("[INFO] Cerrado.")


if __name__ == "__main__":
    main()
