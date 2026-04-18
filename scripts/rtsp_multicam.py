"""
Multi-camera view: all 4 channels simultaneously with bus detection.
Displays a 2x2 grid. Quit with Q.

Optimizations:
  A - Static bus filter: ignores buses that haven't moved between frames
  B - Minimum bbox size: skips OCR on buses too small to read
  C - yolov8n.pt (set in config/settings.py)
  D - Reader FPS cap: limits how fast frames are read to reduce CPU load

Usage:
    python scripts/rtsp_multicam.py
    python scripts/rtsp_multicam.py --fps 10 --min-bbox 120 --debug
"""

import argparse
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import os
# Must be set before importing cv2
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|loglevel;error"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from detectors.yolo_detector import BusDetector, BusDetection
from roi.extractor import extract_rois, extract_full_bus_crop, mask_dvr_timestamp
from preprocessing.image_processor import process
from preprocessing.night_enhancer import enhance as night_enhance, is_dark_frame
from config.settings import YOLO_MIN_CONFIDENCE_NIGHT
from ocr.reader import read_candidates
from filters.candidate_filter import select_best
from web.database import init_db, insertar as db_insertar, actualizar_direccion as db_actualizar_direccion
from web.app import emit_event

BASE_URL = "rtsp://test:fono1234@192.168.1.63:34224/cam/realmonitor"
RECONNECT_DELAY = 5

CAMERAS = {
    1: {"channel": 1,  "label": "Cam 1"},
    2: {"channel": 5,  "label": "Cam 2"},
    3: {"channel": 9,  "label": "Cam 3"},
    4: {"channel": 13, "label": "Cam 4"},
}

# Cameras active for detection
ACTIVE_CAMS = {1, 2, 3, 4}

TILE_W = 640
TILE_H = 360

# Cam 1 is at the barrier — the only camera where direction is unambiguous.
DIRECTION_CAM = "Cam 1"

# Cameras with a fixed direction assignment — any moving bus on these cameras
# is always reported with this direction, regardless of motion analysis.
# Cam 1 (barrera): solo detecta entradas (buses que vienen del exterior).
# Cam 2 (lateral):  solo detecta salidas (buses que salen al exterior).
CAM_FIXED_DIRECTION: dict[str, str] = {
    "Cam 1": "entering",
    "Cam 2": "exiting",
}

# Per-camera exclusion zones: detections whose center falls inside are ignored
# completely (no MotionFilter slot created). Coordinates as fractions of frame size.
# Format: cam_label → list of (x1_frac, y1_frac, x2_frac, y2_frac)
# Cam 3: upper-right area where 3 parked patio buses always sit.
EXCLUDE_ZONES: dict[str, list[tuple[float, float, float, float]]] = {
    "Cam 3": [(0.35, 0.00, 1.00, 0.50)],
}

# Encuadre por cámara para las capturas que van a la DB (Gemini).
# Cada valor es una región relativa (x1, y1, x2, y2) del frame. La clave interna
# es la dirección del bus — "both" aplica a entering y exiting por igual. Hoy
# todas las cámaras devuelven el frame completo; las fracciones permiten
# recortar después sin tocar código.
# Cam 1/2: laterales del bus (frame entero cubre el lateral).
# Cam 3: cola del bus entrando, frente saliendo.
# Cam 4: frente del bus entrando, cola saliendo.
CAPTURE_FRAMING: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "Cam 1": {"both":     (0.0, 0.0, 1.0, 1.0)},
    "Cam 2": {"both":     (0.0, 0.0, 1.0, 1.0)},
    "Cam 3": {"entering": (0.0, 0.0, 1.0, 1.0),
              "exiting":  (0.0, 0.0, 1.0, 1.0)},
    "Cam 4": {"entering": (0.0, 0.0, 1.0, 1.0),
              "exiting":  (0.0, 0.0, 1.0, 1.0)},
}

# Cameras that detect direction via Moondream orientation (single call, same crop).
# Maps cam_label → {"front": direction, "rear": direction}
# Cam 1 (barrier): front of bus = entering, rear = exiting.
# Cam 4 (rear view): sees rear of entering buses, front of exiting buses.
ORIENTATION_TO_DIRECTION: dict[str, dict[str, str]] = {
    "Cam 1": {"front": "exiting", "rear": "entering"},
}

# Virtual barrier lines for Cam 1, as a fraction of frame width.
# A bus whose center crosses the RED line first is entering (comes from outside/right).
CAM1_LINE_RED_X    = 0.65   # right barrier (entrada) — also shown as red zone in overlay
CAM1_LINE_YELLOW_X = 0.18   # left barrier — used only for zone detection logic (no longer shown in overlay)

# Yellow zone for Cam 2 (salida): fraction of frame width where the exit zone is drawn.
CAM2_LINE_YELLOW_X = 0.75   # adjust to match where exiting buses appear in Cam 2

# Zone crossing trigger: bus must completely traverse this zone (right→left) to fire a detection.
# Maps cam_label → fraction of frame width that defines the LEFT edge of the trigger zone.
# The zone spans from this x to the right edge of the frame.
ZONE_X_FRACS: dict[str, float] = {
    "Cam 1": CAM1_LINE_RED_X,
    "Cam 2": CAM2_LINE_YELLOW_X,
}

# Fallback: if no line crossing is detected, use net horizontal displacement.
# net_dx < -MIN_DIRECTION_PX → entering (moving left), net_dx > MIN_DIRECTION_PX → exiting.
MIN_DIRECTION_PX = 40

# Hysteresis: direction candidate must be consistent for this many consecutive frames
# before it's confirmed. Prevents a single noisy frame from triggering a wrong direction.
DIRECTION_MIN_FRAMES = 2

# Minimum total displacement from first-seen position before a bus is considered
# "active" and fed into OCR. Filters YOLO bbox jitter on parked buses — jitter
# oscillates around the same point, real movement accumulates away from it.
ACTIVATION_DISTANCE_PX = 30

# Zone-based direction detection for Cam 1.
# Zona A: exterior (derecha del frame, cx > CAM1_LINE_RED_X).
# Zona B: interior/depósito (izquierda del frame, cx < CAM1_LINE_YELLOW_X).
# If bus visited A first then B → entering; B first then A → exiting.
CAM1_ZONE_A_MIN_X = CAM1_LINE_RED_X      # zona A: cx > this fraction of frame width
CAM1_ZONE_B_MAX_X = CAM1_LINE_YELLOW_X   # zona B: cx < this fraction of frame width

# Bus visible in the same slot for this many seconds without crossing either line
# → treated as parked; will not vote in the consensus buffer.
PARKED_TIMEOUT_SEC = 30.0

# Fallback for large/fast buses on zone-trigger cameras:
# YOLO sometimes first detects a bus AFTER it has already crossed zone_x
# (bus enters from off-screen and traverses zone in < 1 YOLO frame).
# If the bus bbox area is > this threshold (close to camera = in exit lane)
# AND has moved leftward by > ZONE_FALLBACK_MIN_DX_PX, fire the crossing.
ZONE_FALLBACK_AREA   = 300_000   # px² — large bus = close to camera
ZONE_FALLBACK_MIN_DX = 20        # px  — minimum leftward movement to fire fallback


# ── Global barrier crossing signals ───────────────────────────────────────────
# YOLO workers write here when a bus crosses a barrier line (independent of OCR).
# Maps cam_label → monotonic timestamp of last crossing detected.
_barrier_crossings: dict[str, float] = {}
_barrier_lock = threading.Lock()

BARRIER_SIGNAL_MAX_AGE = 15.0  # seconds — crossing signal is valid for this long


def record_barrier_crossing(cam_label: str) -> None:
    with _barrier_lock:
        _barrier_crossings[cam_label] = time.monotonic()


def get_recent_barrier_direction(max_age: float = BARRIER_SIGNAL_MAX_AGE) -> str:
    """Return direction from the most recent barrier crossing, or 'unknown'."""
    now = time.monotonic()
    with _barrier_lock:
        for cam, direction in CAM_FIXED_DIRECTION.items():
            ts = _barrier_crossings.get(cam, 0.0)
            if now - ts < max_age:
                return direction
    return "unknown"


# ── A: Static bus filter + direction tracker ──────────────────────────────────

class MotionFilter:
    """
    Rejects bus detections that haven't moved significantly since the last frame.

    For Cam 1: detects direction by tracking which virtual barrier line the bus
    center crosses first (red=entering, yellow=exiting).
    Buses visible >PARKED_TIMEOUT_SEC without any line crossing are treated as parked.
    """

    def __init__(self, move_threshold: int = 50, static_limit: int = 2, slot_radius: int = 400):
        self.move_threshold = move_threshold
        self.static_limit = static_limit
        self.slot_radius = slot_radius
        # slot_id -> dict(cx_last, cy, static_count, first_crossing, first_seen_time)
        self._slots: dict[int, dict] = {}
        self._next_slot: int = 0

    def _purge_stale_slots(self, max_age_sec: float = 8.0) -> None:
        """Remove slots not seen in the last max_age_sec seconds."""
        now = time.monotonic()
        stale = [sid for sid, slot in self._slots.items()
                 if now - slot["last_seen_time"] > max_age_sec]
        for sid in stale:
            del self._slots[sid]

    def _find_slot(self, cx: float, cy: float) -> int | None:
        """Return the nearest slot within slot_radius, or None."""
        self._purge_stale_slots()
        best_id = None
        best_dist = float("inf")
        for sid, slot in self._slots.items():
            dist = ((cx - slot["cx_last"]) ** 2 + (cy - slot["cy"]) ** 2) ** 0.5
            if dist < self.slot_radius and dist < best_dist:
                best_dist = dist
                best_id = sid
        return best_id

    def is_moving(self, detection: BusDetection, frame_width: int = 0,
                  zone_x_frac: float | None = None) -> bool:
        """
        Returns True if the bus is actively moving.
        frame_width: pass frame.shape[1] for Cam 1 to enable line-crossing tracking.
        zone_x_frac: if set, tracks whether bus center is inside the trigger zone (cx > zone_x).
        """
        x1, y1, x2, y2 = detection.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        now = time.monotonic()

        sid = self._find_slot(cx, cy)
        if sid is None:
            # New bus — create slot, skip this frame.
            # in_zone is set immediately if the bus appears inside the trigger zone
            # on its first detection (fast buses may only appear in 1-2 YOLO frames).
            in_zone_now = (zone_x_frac is not None and frame_width > 0
                           and cx > frame_width * zone_x_frac)
            self._slots[self._next_slot] = {
                "cx_last": cx, "cy": cy,
                "cx_first": cx, "cy_first": cy,
                "ema_cx": cx, "ema_cy": cy,   # smoothed position (EMA)
                "static_count": 1,
                "first_crossing": None,
                "confirmed_direction": None,  # direction confirmed after hysteresis
                "direction_candidate": None,  # last candidate seen
                "direction_streak": 0,        # consecutive frames with same candidate
                "first_seen_time": now,
                "last_seen_time": now,
                "zones_visited": [],          # ordered list of zones visited ("A", "B")
                "in_zone": in_zone_now,       # bus center was seen inside trigger zone
                "crossing_fired": False,      # zone crossing already triggered once
                "barrier_signaled": False,   # barrier crossing signal sent to global
                "max_area": 0,               # largest bbox area seen (for fallback)
            }
            self._next_slot += 1
            return False

        slot = self._slots[sid]
        prev_cx = slot["cx_last"]
        dist = ((cx - prev_cx) ** 2 + (cy - slot["cy"]) ** 2) ** 0.5

        if dist > self.move_threshold:
            slot["static_count"] = 0
        else:
            slot["static_count"] += 1

        # Update EMA (α=0.35): jitter averages out near the true position;
        # real sustained movement shifts the EMA steadily away from the origin.
        alpha = 0.35
        slot["ema_cx"] = alpha * cx + (1 - alpha) * slot["ema_cx"]
        slot["ema_cy"] = alpha * cy + (1 - alpha) * slot["ema_cy"]

        # Zone-based direction detection (Cam 1 only)
        if frame_width > 0 and slot["first_crossing"] is None:
            red_x    = frame_width * CAM1_ZONE_A_MIN_X
            yellow_x = frame_width * CAM1_ZONE_B_MAX_X

            # Determine which zone the current centroid is in
            current_zone = None
            if cx > red_x:
                current_zone = "A"   # exterior (right side)
            elif cx < yellow_x:
                current_zone = "B"   # interior/depósito (left side)

            # Record zone if it's new
            zones = slot["zones_visited"]
            if current_zone is not None and (not zones or zones[-1] != current_zone):
                zones.append(current_zone)

            # Once both zones have been visited, determine direction by order
            if len(zones) >= 2:
                if zones[0] == "A" and "B" in zones:
                    slot["first_crossing"] = "entering"   # A (right/exterior) → B (left/depot) = entering
                elif zones[0] == "B" and "A" in zones:
                    slot["first_crossing"] = "exiting"    # B (left/depot) → A (right/exterior) = exiting

        slot["cx_last"] = cx
        slot["cy"] = cy
        slot["last_seen_time"] = now
        slot["max_area"] = max(slot.get("max_area", 0), detection.area)

        # Zone tracking: mark if bus center is inside the trigger zone
        if zone_x_frac is not None and frame_width > 0:
            if cx > frame_width * zone_x_frac:
                slot["in_zone"] = True

        # Activation check: use the EMA position instead of raw cx to filter YOLO
        # bbox jitter. Jitter oscillates → EMA stays near cx_first. Real movement
        # accumulates → EMA drifts away. Only activate when EMA displacement exceeds
        # the threshold.
        ema_disp = ((slot["ema_cx"] - slot["cx_first"]) ** 2 +
                    (slot["ema_cy"] - slot["cy_first"]) ** 2) ** 0.5
        if ema_disp < ACTIVATION_DISTANCE_PX:
            return False

        # Parked timeout: bus in slot too long without crossing any line → parked
        if slot["first_crossing"] is None and now - slot["first_seen_time"] > PARKED_TIMEOUT_SEC:
            return False

        # If the EMA already confirmed real movement, don't block on per-frame threshold.
        # static_count only blocks when the bus hasn't moved at all recently.
        return slot["static_count"] < max(self.static_limit, 8)

    def check_zone_crossing(self, detection: BusDetection, frame_width: int,
                            zone_x_frac: float) -> bool:
        """
        Returns True (once per slot) when the bus has completely crossed the trigger zone
        from right to left: was seen with cx > zone_x, now seen with cx < zone_x.
        After returning True, sets crossing_fired to prevent re-triggering.
        """
        x1, y1, x2, y2 = detection.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        sid = self._find_slot(cx, cy)
        if sid is None:
            return False
        slot = self._slots[sid]
        if slot.get("crossing_fired"):
            return False
        zone_x = frame_width * zone_x_frac
        # Use cx_last (most recent YOLO position) rather than the queued detection's cx.
        # The crop is queued while the bus is still *inside* the zone (cx > zone_x);
        # by the time OCR finishes, YOLO has advanced and cx_last reflects the exit.
        current_cx = slot["cx_last"]

        # Primary path: bus was explicitly seen inside zone then exited.
        if slot.get("in_zone") and current_cx < zone_x:
            slot["crossing_fired"] = True
            return True

        # Fallback for large/fast buses: YOLO only caught the bus AFTER it had
        # already crossed zone_x (bus entered from off-screen on the right and
        # the zone traversal happened between YOLO frames).
        # Use max_area (largest bbox seen in this slot) instead of the queued
        # detection's area — the bus grows as it approaches and may have been
        # small when first enqueued.
        net_leftward = slot["cx_first"] - current_cx  # positive = moved left
        effective_area = max(slot.get("max_area", 0), detection.area)
        if (effective_area >= ZONE_FALLBACK_AREA and
                net_leftward >= ZONE_FALLBACK_MIN_DX):
            slot["crossing_fired"] = True
            return True

        return False

    def get_direction(self, detection: BusDetection) -> str | None:
        """
        Return "entering", "exiting", or None (unknown).
        Only meaningful for DIRECTION_CAM. Must be called after is_moving().

        Priority:
        1. Line crossing (hard signal — bus crossed a virtual barrier). Instant confirm.
        2. Net horizontal displacement with hysteresis: candidate must be the same for
           DIRECTION_MIN_FRAMES consecutive frames before it's confirmed as direction.
           Prevents a single noisy detection from triggering a wrong result.
        """
        x1, y1, x2, y2 = detection.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        sid = self._find_slot(cx, cy)
        if sid is None:
            return None
        slot = self._slots[sid]

        # Hard signal: line crossing always wins immediately.
        if slot["first_crossing"] is not None:
            return slot["first_crossing"]

        # Already confirmed via hysteresis — return cached result.
        if slot["confirmed_direction"] is not None:
            return slot["confirmed_direction"]

        # Compute candidate from smoothed (EMA) displacement, consistent with
        # activation check — raw cx_last has jitter, ema_cx does not.
        net_dx = slot["ema_cx"] - slot["cx_first"]
        if net_dx < -MIN_DIRECTION_PX:
            candidate = "entering"
        elif net_dx > MIN_DIRECTION_PX:
            candidate = "exiting"
        else:
            candidate = None

        # Hysteresis: accumulate streak for consistent candidates.
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

        return slot["confirmed_direction"]

    def draw_cam1_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw direction debug overlay on a copy of a Cam 1 frame.
        Shows the red entry zone, bus trajectory, net_dx, and direction state.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # Red barrier line (entrada only)
        red_x = int(w * CAM1_LINE_RED_X)
        cv2.line(out, (red_x, 0), (red_x, h), (0, 0, 255), 2)
        cv2.putText(out, "ENTRA", (red_x - 60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Red zone overlay (semitransparent)
        overlay_zones = out.copy()
        cv2.rectangle(overlay_zones, (red_x, 0), (w, h), (0, 0, 255), -1)
        cv2.addWeighted(overlay_zones, 0.08, out, 0.92, 0, out)
        cv2.putText(out, "ZONA A (ext)", (red_x + 5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        now = time.monotonic()
        for slot in self._slots.values():
            # Only draw slots seen in the last 5 seconds — skip stale/parked buses
            if now - slot["last_seen_time"] > 5.0:
                continue

            cx_first = int(slot["cx_first"])
            cx_last  = int(slot["cx_last"])
            cy       = int(slot["cy"])
            net_dx   = slot["cx_last"] - slot["cx_first"]
            candidate = slot["direction_candidate"]
            confirmed = slot["confirmed_direction"] or slot["first_crossing"]
            streak    = slot["direction_streak"]

            # Trajectory arrow: from first seen to current position
            color = (0, 255, 0) if confirmed else (0, 165, 255)
            cv2.arrowedLine(out, (cx_first, cy), (cx_last, cy), color, 2, tipLength=0.3)
            cv2.circle(out, (cx_last, cy), 6, color, -1)

            # Info text
            dir_text  = confirmed or (f"{candidate}? ({streak}/{DIRECTION_MIN_FRAMES})" if candidate else "desconocido")
            zones_str = "→".join(slot.get("zones_visited", []))
            dx_text   = f"dx={int(net_dx)} zones={zones_str} {dir_text}"
            cv2.putText(out, dx_text, (cx_last + 8, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

        return out

    def draw_cam2_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw exit zone overlay on a copy of a Cam 2 frame.
        Shows the yellow exit zone where exiting buses are tracked.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # Yellow barrier line (salida)
        yellow_x = int(w * CAM2_LINE_YELLOW_X)
        cv2.line(out, (yellow_x, 0), (yellow_x, h), (0, 255, 255), 2)
        cv2.putText(out, "SALE", (yellow_x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Yellow zone overlay (semitransparent)
        overlay_zones = out.copy()
        cv2.rectangle(overlay_zones, (yellow_x, 0), (w, h), (0, 255, 255), -1)
        cv2.addWeighted(overlay_zones, 0.08, out, 0.92, 0, out)
        cv2.putText(out, "ZONA B (sal)", (yellow_x + 5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        return out


# ── Pipeline ──────────────────────────────────────────────────────────────────

def process_frame(
    frame,
    detector: BusDetector,
    motion_filter: MotionFilter,
    min_bbox: int = 120,
    label: str = "",
    debug: bool = False,
    ocr_backend: str = "claude",
) -> tuple[tuple | None, tuple | None]:
    """Returns (result, bbox) where bbox is (x1,y1,x2,y2) of the bus sent to OCR."""
    detections = detector.detect(frame)
    if not detections:
        if debug:
            print(f"  [{label}] YOLO: sin bus")
        return None, None

    # Evaluate all detections: update motion slots for every bus, collect moving ones.
    # Among moving buses, pick the largest bbox (closest to camera).
    fh_frame, fw_frame = frame.shape[:2]
    fw = fw_frame if label == DIRECTION_CAM else 0
    exclusions = EXCLUDE_ZONES.get(label, [])
    moving = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        bus_w, bus_h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        if debug:
            print(f"  [{label}] YOLO: bus conf={det.confidence:.2f} bbox=({x1},{y1})-({x2},{y2})")

        # Exclusion zones: skip detections whose center is in a masked area
        excluded = any(
            ex1 * fw_frame <= cx <= ex2 * fw_frame and
            ey1 * fh_frame <= cy <= ey2 * fh_frame
            for ex1, ey1, ex2, ey2 in exclusions
        )
        if excluded:
            if debug:
                print(f"  [{label}] SKIP: zona excluida ({cx:.0f},{cy:.0f})")
            continue

        # B: minimum bbox size
        if bus_w < min_bbox or bus_h < min_bbox:
            if debug:
                print(f"  [{label}] SKIP: bbox demasiado pequeño ({bus_w}×{bus_h}px < {min_bbox}px)")
            # Still update motion slot even if skipped for size
            motion_filter.is_moving(det, frame_width=fw)
            continue

        # Freeze zone tracking when bus fills most of the frame — cx is near center
        # and not reliable for direction detection.
        det_ratio = (bus_w * bus_h) / (fw_frame * fh_frame)
        fw_for_motion = 0 if (fw > 0 and det_ratio > 0.60) else fw

        # A: static bus filter — always call to update slot state
        if not motion_filter.is_moving(det, frame_width=fw_for_motion):
            if debug:
                print(f"  [{label}] SKIP: bus estático (no se movió o estacionado)")
            continue

        moving.append(det)

    if not moving:
        return None, None

    # Pick the moving bus closest to the camera (largest bbox area)
    best = max(moving, key=lambda d: d.area)
    if debug and len(moving) > 1:
        print(f"  [{label}] {len(moving)} buses en movimiento → usando el más cercano (área={best.area}px²)")

    if ocr_backend == "moondream":
        from ocr.moondream_reader import get_moondream_reader
        bus_crop = extract_full_bus_crop(frame, best)
        if bus_crop is None:
            if debug:
                print(f"  [{label}] CROP: demasiado pequeño")
            return None, None

        # If the bus bbox covers > 40% of the frame the bus is too close and
        # the full crop won't show the number. Split the frame into 4 quadrants
        # and try each one — the number will be in one of the corners.
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = best.bbox
        bbox_ratio = ((x2 - x1) * (y2 - y1)) / (fw * fh)
        if bbox_ratio > 0.60 and label in ORIENTATION_TO_DIRECTION:
            # Cam 1 / Cam 2: bus too close, number not readable. Skip OCR and let
            # the other camera handle it. Direction still tracked via MotionFilter.
            if debug:
                print(f"  [{label}] BUS GRANDE ({bbox_ratio:.0%} frame) → skip OCR, bus demasiado cerca")
            direction = motion_filter.get_direction(best) or "unknown"
            return None, best.bbox
        elif bbox_ratio > 0.60:
            # Cam 3 (OCR-only): bus large but still try sub-crops — no other cam to rely on.
            hw, hh = fw // 2, fh // 2
            subcrops = [
                frame[0:hh,  0:hw],
                frame[0:hh,  hw:fw],
                frame[hh:fh, 0:hw],
                frame[hh:fh, hw:fw],
            ]
            if debug:
                print(f"  [{label}] BUS GRANDE ({bbox_ratio:.0%} frame) → probando sub-crops")
            number = get_moondream_reader().read_subcrops(subcrops)
            direction = "unknown"
        elif label in ORIENTATION_TO_DIRECTION:
            # Single call: read number + orientation, then map orientation → direction.
            number, orientation = get_moondream_reader().read_with_orientation(bus_crop, cam_label=label)
            orientation_map = ORIENTATION_TO_DIRECTION[label]
            direction = orientation_map.get(orientation or "", "unknown")
            # Moondream couldn't determine orientation — fall back to MotionFilter
            if direction == "unknown":
                direction = motion_filter.get_direction(best) or "unknown"
            if debug and number is None:
                dump_dir = Path(__file__).resolve().parent.parent / "captures" / "crop_debug"
                dump_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%H%M%S_%f")[:9]
                cv2.imwrite(str(dump_dir / f"{ts}_{label.replace(' ','')}_normal.jpg"), bus_crop)
        else:
            number = get_moondream_reader().read(bus_crop, cam_label=label)
            direction = "unknown"
            if debug and number is None:
                dump_dir = Path(__file__).resolve().parent.parent / "captures" / "crop_debug"
                dump_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%H%M%S_%f")[:9]
                cv2.imwrite(str(dump_dir / f"{ts}_{label.replace(' ','')}_normal.jpg"), bus_crop)

        if debug:
            print(f"  [{label}] MOONDREAM: {number if number else 'none'} dir={direction}")
        return ((number, direction) if number else None), best.bbox

    # ── EasyOCR path (legacy) ──────────────────────────────────────────────
    crops = extract_rois(frame, best)
    if not crops:
        if debug:
            print(f"  [{label}] ROI: sin crops válidos")
        return None, None

    all_candidates = []
    for crop in crops:
        variants = process(crop.image)
        for variant in variants:
            candidates = read_candidates(variant.image, crop.zone_name, variant.name)
            if debug and candidates:
                for c in candidates:
                    print(f"  [{label}] OCR: zona={c.zone_name} texto='{c.text}' conf={c.confidence:.2f}")
            all_candidates.extend(candidates)

    if debug and not all_candidates:
        print(f"  [{label}] OCR: sin candidatos")

    result = select_best(all_candidates)
    if debug:
        if result:
            print(f"  [{label}] RESULTADO: {result.number} (score={result.score:.2f})")
        else:
            print(f"  [{label}] RESULTADO: descartado por filtro")
    return ((result.number, direction) if result else None), best.bbox


# ── Consensus buffer (shared across all cameras) ──────────────────────────────

class ConsensusBuffer:
    """
    Two-step detection flow:
      1. First vote → emits a "provisional" event immediately (shown on HUD).
      2. Window closes (on second confirming vote or timeout) → emits:
           "confirmed"  if all votes agree on the same number
           "conflict"   if votes disagree (logs an error, no save)
    Cooldown is PER NUMBER (applies only to confirmed events).

    Events are 4-tuples: (event_type, number, direction, extra)
      provisional : (type, number, direction, cam_label)
      confirmed   : (type, number, direction, None)
      conflict    : (type, first_number, direction, second_number)

    Las capturas que van a la DB las arma `_report` desde `state.frame` de
    cada cámara — el buffer ya no guarda imágenes.
    """

    def __init__(self, max_window_sec: float = 6.0, number_cooldown_sec: float = 10.0):
        self.max_window_sec      = max_window_sec
        self.number_cooldown_sec = number_cooldown_sec

        self._votes: list[tuple[int, str, str]] = []  # (number, direction, cam_label)
        self._window_start: float | None = None
        self._reported: dict[int, float] = {}
        self._reported_dir: dict[int, str] = {}   # direction at confirmation time
        self._lock = threading.Lock()

    def feed(self, result: tuple[int, str] | None, cam_label: str) -> list[tuple]:
        now = time.monotonic()
        events: list[tuple] = []
        with self._lock:
            if self._window_start is None:
                if result is not None:
                    number, direction = result
                    # Direction upgrade: bus was just confirmed with unknown direction,
                    # and now a direction camera reads the same number within cooldown.
                    last_report = self._reported.get(number, 0.0)
                    if (cam_label in CAM_FIXED_DIRECTION and
                            direction != "unknown" and
                            now - last_report < self.number_cooldown_sec and
                            self._reported_dir.get(number) == "unknown"):
                        self._reported_dir[number] = direction
                        events.append(("direction_update", number, direction, None))
                        return events
                    self._window_start = now
                    self._votes = [(number, direction, cam_label)]
                    events.append(("provisional", number, direction, cam_label))
                    if cam_label in CAM_FIXED_DIRECTION:
                        events.extend(self._close(now))
            else:
                if result is not None:
                    number, direction = result
                    self._votes.append((number, direction, cam_label))
                    cams_voted = {lbl for _, _, lbl in self._votes}
                    if cam_label in CAM_FIXED_DIRECTION or len(cams_voted) >= 2:
                        events.extend(self._close(now))
                elif now - self._window_start >= self.max_window_sec:
                    events.extend(self._close(now))
        return events

    def tick(self) -> list[tuple]:
        """Close expired window even when no new detections arrive."""
        now = time.monotonic()
        with self._lock:
            if (self._window_start is not None and
                    now - self._window_start >= self.max_window_sec):
                return self._close(now)
        return []

    def _close(self, now: float) -> list[tuple]:
        """Evaluate votes and emit confirmed or conflict. Called with lock held."""
        votes = self._votes
        self._votes = []
        self._window_start = None

        if not votes:
            return []

        # Majority number
        counts: dict[int, int] = {}
        for n, _, _ in votes:
            counts[n] = counts.get(n, 0) + 1
        winner = max(counts, key=counts.__getitem__)
        distinct = [n for n in counts if n != winner]

        # Direction: majority non-unknown across all votes for the winner
        dir_votes = [d for n, d, _ in votes if n == winner and d != "unknown"]
        direction = max(set(dir_votes), key=dir_votes.count) if dir_votes else "unknown"

        # If no camera voted with a direction, check if a barrier crossing
        # happened recently (bus crossed Cam 1 or Cam 2 but OCR failed there).
        if direction == "unknown":
            direction = get_recent_barrier_direction()

        last = self._reported.get(winner, 0.0)
        if now - last < self.number_cooldown_sec:
            return []

        if not distinct:
            # All votes agree
            self._reported[winner] = now
            self._reported_dir[winner] = direction
            return [("confirmed", winner, direction, None)]
        else:
            rival = max(distinct, key=counts.__getitem__)
            return [("conflict", winner, direction, rival)]

    @property
    def window_active(self) -> bool:
        with self._lock:
            return self._window_start is not None

    def window_progress(self) -> tuple[float, float]:
        now = time.monotonic()
        with self._lock:
            if self._window_start is None:
                return 0.0, self.max_window_sec
            return min(now - self._window_start, self.max_window_sec), self.max_window_sec

    def vote_count(self, cam_label: str) -> int:
        with self._lock:
            return sum(1 for _, _, lbl in self._votes if lbl == cam_label)

    def all_votes(self) -> dict[int, list[str]]:
        with self._lock:
            result: dict[int, list[str]] = {}
            for n, _, lbl in self._votes:
                result.setdefault(n, []).append(lbl)
            return result


# ── Per-camera state ──────────────────────────────────────────────────────────

class CameraState:
    def __init__(self, cam_key: int):
        self.cam_key = cam_key
        self.label = CAMERAS[cam_key]["label"]
        self.frame: np.ndarray | None = None
        self.process_frame: np.ndarray | None = None
        self.overlay_frame: np.ndarray | None = None  # Cam 1 debug overlay
        self.overlay_ts: float = 0.0
        self.confirmed_label: str | None = None  # e.g. "325 Entró"
        self.pending_count: int = 0              # this camera's votes in window
        self.connected: bool = False
        self.lock = threading.Lock()
        self.process_event = threading.Event()
        self.last_capture: np.ndarray | None = None
        # OCR queue: YOLO puts (crop, best_detection, frame) here; Moondream worker consumes.
        # maxsize=1 so we always process the freshest crop and drop stale ones.
        self.ocr_queue: queue.Queue = queue.Queue(maxsize=1)


# ── Reader thread ─────────────────────────────────────────────────────────────

def reader_worker(state: CameraState, skip: int, fps_cap: int) -> None:
    """
    Reads frames from RTSP.
    D: fps_cap limits how many frames/sec are read to reduce CPU load.
    """
    url = f"{BASE_URL}?channel={CAMERAS[state.cam_key]['channel']}&subtype=0"
    frame_interval = 1.0 / fps_cap  # D: minimum seconds between reads
    cap = None
    frame_count = 0

    while True:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                with state.lock:
                    state.connected = False
                emit_event({"type": "camera_status", "cam": state.label, "connected": False})
                time.sleep(RECONNECT_DELAY)
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize lag — always show latest frame
            with state.lock:
                state.connected = True
            emit_event({"type": "camera_status", "cam": state.label, "connected": True})
            frame_count = 0

        t0 = time.monotonic()

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            cap = None
            with state.lock:
                state.connected = False
            emit_event({"type": "camera_status", "cam": state.label, "connected": False})
            continue

        frame_count += 1

        with state.lock:
            state.frame = frame

        # queue for processor every N frames, only if processor is free
        if frame_count % skip == 0 and not state.process_event.is_set():
            with state.lock:
                state.process_frame = frame.copy()
            state.process_event.set()

        # D: sleep remainder of frame interval to cap FPS
        elapsed = time.monotonic() - t0
        remaining = frame_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)


# ── YOLO worker (fast, runs every frame) ─────────────────────────────────────

def yolo_worker(
    state: CameraState,
    detector: BusDetector,
    motion_filter: MotionFilter,
    min_bbox: int,
    debug: bool = False,
) -> None:
    """
    Runs YOLO on every queued frame as fast as possible.
    When a moving bus is found, puts the crop + context into state.ocr_queue
    for the OCR worker to pick up. Drops the crop if OCR is still busy (queue full).
    """
    print(f"[INFO] {state.label}: listo.")

    while True:
        try:
            state.process_event.wait()
            state.process_event.clear()

            with state.lock:
                frame = state.process_frame
            if frame is None:
                continue

            try:
                best, bus_crop, bbox_ratio = _yolo_detect(
                    frame, detector, motion_filter, min_bbox,
                    label=state.label, debug=debug,
                )
            except Exception as e:
                print(f"  [{state.label}] YOLO ERROR: {e}")
                best, bus_crop, bbox_ratio = None, None, 0.0

            # Update direction overlays regardless of OCR result
            if state.label == DIRECTION_CAM:
                try:
                    annotated = motion_filter.draw_cam1_overlay(frame)
                except Exception as e:
                    print(f"  [{state.label}] OVERLAY ERROR: {e}")
                    annotated = None
                with state.lock:
                    state.overlay_frame = annotated
                    state.overlay_ts = time.monotonic()
            elif state.label == "Cam 2":
                try:
                    annotated = motion_filter.draw_cam2_overlay(frame)
                except Exception as e:
                    print(f"  [{state.label}] OVERLAY ERROR: {e}")
                    annotated = None
                with state.lock:
                    state.overlay_frame = annotated
                    state.overlay_ts = time.monotonic()

            if best is None or bus_crop is None:
                # No moving bus — clear overlay so display falls back to live frame
                if state.label not in (DIRECTION_CAM, "Cam 2"):
                    with state.lock:
                        state.overlay_frame = None
                        state.overlay_ts = time.monotonic()
                continue

            # Draw green bbox immediately (don't wait for Moondream)
            x1, y1, x2, y2 = best.bbox
            annotated = frame.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            with state.lock:
                state.overlay_frame = annotated
                state.overlay_ts = time.monotonic()

            # Hand off to OCR worker — drop if busy (queue full = stale crop)
            try:
                state.ocr_queue.put_nowait((bus_crop, best, frame, bbox_ratio))
            except queue.Full:
                if debug:
                    print(f"  [{state.label}] OCR ocupado, descartando crop")

        except Exception as e:
            print(f"  [{state.label}] YOLO WORKER ERROR (thread vivo): {e}")


def _yolo_detect(
    frame,
    detector: BusDetector,
    motion_filter: MotionFilter,
    min_bbox: int,
    label: str = "",
    debug: bool = False,
):
    """
    Run YOLO + MotionFilter. Returns (best_detection, bus_crop, bbox_ratio) or
    (None, None, 0) if no moving bus found worth sending to OCR.
    """
    from roi.extractor import extract_full_bus_crop

    if is_dark_frame(frame):
        detections = detector.detect(night_enhance(frame), conf=YOLO_MIN_CONFIDENCE_NIGHT)
        if debug and detections:
            print(f"  [{label}] NIGHT MODE: enhance + conf={YOLO_MIN_CONFIDENCE_NIGHT}")
    else:
        detections = detector.detect(frame)
    if not detections:
        if debug:
            print(f"  [{label}] YOLO: sin bus")
        return None, None, 0.0

    fh_frame, fw_frame = frame.shape[:2]
    zone_x_frac = ZONE_X_FRACS.get(label)
    fw = fw_frame if (label == DIRECTION_CAM or zone_x_frac is not None) else 0
    exclusions = EXCLUDE_ZONES.get(label, [])
    moving = []

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        bus_w, bus_h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        if debug:
            print(f"  [{label}] YOLO: bus conf={det.confidence:.2f} bbox=({x1},{y1})-({x2},{y2})")

        excluded = any(
            ex1 * fw_frame <= cx <= ex2 * fw_frame and
            ey1 * fh_frame <= cy <= ey2 * fh_frame
            for ex1, ey1, ex2, ey2 in exclusions
        )
        if excluded:
            if debug:
                print(f"  [{label}] SKIP: zona excluida ({cx:.0f},{cy:.0f})")
            continue

        if bus_w < min_bbox or bus_h < min_bbox:
            if debug:
                print(f"  [{label}] SKIP: bbox demasiado pequeño ({bus_w}×{bus_h}px < {min_bbox}px)")
            motion_filter.is_moving(det, frame_width=fw, zone_x_frac=zone_x_frac)
            continue

        # Freeze zone tracking when bus fills most of the frame — cx is near center
        # and not reliable for direction detection.
        det_ratio = (bus_w * bus_h) / (fw_frame * fh_frame)
        fw_for_motion = 0 if (fw > 0 and det_ratio > 0.60) else fw

        if not motion_filter.is_moving(det, frame_width=fw_for_motion, zone_x_frac=zone_x_frac):
            if debug:
                print(f"  [{label}] SKIP: bus estático (no se movió o estacionado)")
            continue

        # Record barrier crossing from YOLO worker (independent of OCR).
        # This fires even if OCR can't read the number — allows other cameras
        # that DO read the number to inherit the direction.
        if zone_x_frac is not None and fw_for_motion > 0:
            x1b, y1b, x2b, y2b = det.bbox
            cxb = (x1b + x2b) / 2
            sid = motion_filter._find_slot(cxb, (y1b + y2b) / 2)
            if sid is not None:
                slot = motion_filter._slots[sid]
                if slot.get("in_zone") and slot["cx_last"] < fw_for_motion * zone_x_frac:
                    if not slot.get("barrier_signaled"):
                        slot["barrier_signaled"] = True
                        record_barrier_crossing(label)
                        if debug:
                            print(f"  [{label}] BARRERA: cruce detectado → {CAM_FIXED_DIRECTION.get(label)}")

        moving.append(det)

    if not moving:
        return None, None, 0.0

    best = max(moving, key=lambda d: d.area)
    if debug and len(moving) > 1:
        print(f"  [{label}] {len(moving)} buses en movimiento → usando el más cercano (área={best.area}px²)")

    bus_crop = extract_full_bus_crop(frame, best)
    if bus_crop is None:
        if debug:
            print(f"  [{label}] CROP: demasiado pequeño")
        return None, None, 0.0

    fh, fw2 = frame.shape[:2]
    x1, y1, x2, y2 = best.bbox
    bbox_ratio = ((x2 - x1) * (y2 - y1)) / (fw2 * fh)
    return best, bus_crop, bbox_ratio


# ── OCR worker (Moondream, async) ─────────────────────────────────────────────

def ocr_worker(
    state: CameraState,
    global_buffer: ConsensusBuffer,
    all_states: dict,
    motion_filter: MotionFilter,
    debug: bool = False,
    ocr_backend: str = "moondream",
) -> None:
    """
    Consumes crops from state.ocr_queue and runs Moondream OCR.
    Runs independently of YOLO so YOLO is never blocked waiting for OCR.
    """
    while True:
        try:
            bus_crop, best, frame, bbox_ratio = state.ocr_queue.get()
            label = state.label
            fh, fw = frame.shape[:2]

            try:
                raw = _run_ocr(
                    bus_crop, best, frame, bbox_ratio, fw, fh,
                    motion_filter, label, debug, ocr_backend,
                )
            except Exception as e:
                print(f"  [{label}] OCR ERROR: {e}")
                raw = None

            # El buffer solo contabiliza votos — las capturas para DB se
            # construyen en `_report` tomando el frame actual de cada cámara.
            events = global_buffer.feed(raw, label)

            if debug and raw is not None:
                votes = global_buffer.all_votes()
                print(f"  [{label}] VOTO: {raw}  |  votos en ventana: {votes}")

            for event in events:
                _handle_event(event, frame, all_states)

            with state.lock:
                state.pending_count = global_buffer.vote_count(label)

        except Exception as e:
            print(f"  [{state.label}] OCR WORKER ERROR (thread vivo): {e}")


def _run_ocr(bus_crop, best, frame, bbox_ratio, fw, fh,
             motion_filter, label, debug, ocr_backend):
    """Runs the OCR backend and returns (number, direction) or None."""
    if ocr_backend != "moondream":
        # EasyOCR legacy path
        from roi.extractor import extract_rois
        crops = extract_rois(frame, best)
        if not crops:
            if debug:
                print(f"  [{label}] ROI: sin crops válidos")
            return None
        all_candidates = []
        for crop in crops:
            variants = process(crop.image)
            for variant in variants:
                candidates = read_candidates(variant.image, crop.zone_name, variant.name)
                if debug and candidates:
                    for c in candidates:
                        print(f"  [{label}] OCR: zona={c.zone_name} texto='{c.text}' conf={c.confidence:.2f}")
                all_candidates.extend(candidates)
        if debug and not all_candidates:
            print(f"  [{label}] OCR: sin candidatos")
        result = select_best(all_candidates)
        if debug:
            if result:
                print(f"  [{label}] RESULTADO: {result.number} (score={result.score:.2f})")
            else:
                print(f"  [{label}] RESULTADO: descartado por filtro")
        direction = motion_filter.get_direction(best) or "unknown"
        return (result.number, direction) if result else None

    from ocr.moondream_reader import get_moondream_reader

    if bbox_ratio > 0.60 and label in ORIENTATION_TO_DIRECTION:
        if debug:
            print(f"  [{label}] BUS GRANDE ({bbox_ratio:.0%} frame) → skip OCR, bus demasiado cerca")
        direction = motion_filter.get_direction(best) or "unknown"
        return None
    elif bbox_ratio > 0.60:
        from roi.extractor import mask_dvr_timestamp
        masked = mask_dvr_timestamp(frame)
        hw, hh = fw // 2, fh // 2
        subcrops = [
            masked[0:hh,  0:hw],
            masked[0:hh,  hw:fw],
            masked[hh:fh, 0:hw],
            masked[hh:fh, hw:fw],
        ]
        if debug:
            print(f"  [{label}] BUS GRANDE ({bbox_ratio:.0%} frame) → probando sub-crops")
        number = get_moondream_reader().read_subcrops(subcrops)
        direction = "unknown"
    elif label in ORIENTATION_TO_DIRECTION:
        number, orientation = get_moondream_reader().read_with_orientation(bus_crop, cam_label=label)
        # MotionFilter zone crossing is the primary direction signal (more reliable).
        # Moondream orientation is fallback only when MotionFilter has no result yet.
        motion_direction = motion_filter.get_direction(best)
        if motion_direction:
            direction = motion_direction
        else:
            orientation_map = ORIENTATION_TO_DIRECTION[label]
            direction = orientation_map.get(orientation or "", "unknown")
        if debug and number is None:
            dump_dir = Path(__file__).resolve().parent.parent / "captures" / "crop_debug"
            dump_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%H%M%S_%f")[:9]
            cv2.imwrite(str(dump_dir / f"{ts}_{label.replace(' ','')}_normal.jpg"), bus_crop)
    else:
        number = get_moondream_reader().read(bus_crop, cam_label=label)
        direction = "unknown"
        if debug and number is None:
            dump_dir = Path(__file__).resolve().parent.parent / "captures" / "crop_debug"
            dump_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%H%M%S_%f")[:9]
            cv2.imwrite(str(dump_dir / f"{ts}_{label.replace(' ','')}_normal.jpg"), bus_crop)

    # For zone-trigger cameras (Cam 1 / Cam 2): only report if bus has completely
    # crossed through the trigger zone (right→left). Without a completed crossing,
    # discard the OCR result — the bus hasn't passed the gate yet.
    zone_x_frac = ZONE_X_FRACS.get(label)
    if zone_x_frac is not None:
        if debug:
            # Peek at slot state to diagnose zone crossing failures
            x1d, y1d, x2d, y2d = best.bbox
            cxd = (x1d + x2d) / 2
            cyd = (y1d + y2d) / 2
            sid = motion_filter._find_slot(cxd, cyd)
            if sid is not None:
                sl = motion_filter._slots[sid]
                net_left = sl['cx_first'] - sl['cx_last']
                print(f"  [{label}] ZONA CHECK: queued_cx={cxd:.0f} cx_first={sl['cx_first']:.0f} "
                      f"cx_last={sl['cx_last']:.0f} zone_x={fw * zone_x_frac:.0f} "
                      f"in_zone={sl.get('in_zone')} fired={sl.get('crossing_fired')} "
                      f"area={best.area:.0f} net_left={net_left:.0f}")
            else:
                print(f"  [{label}] ZONA CHECK: slot no encontrado para queued_cx={cxd:.0f}")
        if not motion_filter.check_zone_crossing(best, fw, zone_x_frac):
            if debug:
                print(f"  [{label}] MOONDREAM: {number if number else 'none'} — sin cruce de zona, descartando")
            return None
        # Crossing confirmed — assign fixed direction and record global signal
        direction = CAM_FIXED_DIRECTION[label]
        record_barrier_crossing(label)
    else:
        # Override direction with fixed assignment if configured for this camera
        fixed_dir = CAM_FIXED_DIRECTION.get(label)
        if fixed_dir:
            direction = fixed_dir

    if debug:
        print(f"  [{label}] MOONDREAM: {number if number else 'none'} dir={direction}")

    if number is None and zone_x_frac is not None:
        # OCR failed on this crop — reset crossing_fired so the next crop can retry.
        # The ConsensusBuffer per-number cooldown handles dedup; we don't need
        # crossing_fired to stay True when we haven't read anything yet.
        x1r, y1r, x2r, y2r = best.bbox
        sid = motion_filter._find_slot((x1r + x2r) / 2, (y1r + y2r) / 2)
        if sid is not None:
            motion_filter._slots[sid]["crossing_fired"] = False

    return (number, direction) if number else None


# ── Report helper ────────────────────────────────────────────────────────────

def _handle_event(event: tuple, frame, all_states: dict) -> None:
    event_type = event[0]
    if event_type == "provisional":
        _, number, direction, cam_label = event
        action = {"entering": "entrando", "exiting": "saliendo"}.get(direction, "")
        suffix = f" ({action})" if action else ""
        print(f"[provisional] Bus {number}{suffix} — visto por {cam_label}, esperando confirmación...")
        # Update HUD on all tiles
        label = f"{number} {action}".strip()
        for s in all_states.values():
            with s.lock:
                s.confirmed_label = label
    elif event_type == "confirmed":
        _, number, direction, _ = event
        _report(number, direction, all_states)
    elif event_type == "direction_update":
        _, number, direction, _ = event
        if direction == "entering":
            action = "Entro"
        elif direction == "exiting":
            action = "Salio"
        else:
            action = ""
        label = f"{number} {action}".strip()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] Bus {number} direccion actualizada → {action}")
        db_actualizar_direccion(number, direction)
        for s in all_states.values():
            with s.lock:
                s.confirmed_label = label
    elif event_type == "conflict":
        _, first_number, direction, rival = event
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] ⚠️  CONFLICTO: primera lectura={first_number}, segunda lectura={rival} — descartado")


def build_capture(frame: np.ndarray, cam_label: str, direction: str) -> np.ndarray | None:
    """
    Construye la captura final que va a disco/DB para una cámara:
    - pinta la franja del timestamp del DVR
    - aplica night-enhance si está oscuro
    - recorta según `CAPTURE_FRAMING[cam_label]` (entering/exiting/both) en
      coordenadas relativas del frame.
    Conserva la resolución nativa del stream — no se hace resize.
    """
    if frame is None or frame.size == 0:
        return None

    framing_map = CAPTURE_FRAMING.get(cam_label, {})
    rect = framing_map.get(direction) or framing_map.get("both") or framing_map.get("entering")
    if rect is None:
        rect = (0.0, 0.0, 1.0, 1.0)

    out = mask_dvr_timestamp(frame)
    if is_dark_frame(out):
        out = night_enhance(out)

    h, w = out.shape[:2]
    x1 = max(0, min(int(rect[0] * w), w - 1))
    y1 = max(0, min(int(rect[1] * h), h - 1))
    x2 = max(x1 + 1, min(int(rect[2] * w), w))
    y2 = max(y1 + 1, min(int(rect[3] * h), h))
    return out[y1:y2, x1:x2].copy()


def _report(number: int, direction: str, all_states: dict) -> None:
    ts_log  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    if direction == "entering":
        action = "Entro"
    elif direction == "exiting":
        action = "Salio"
    else:
        action = ""

    label = f"{number} {action}".strip()
    suffix = f" {action}" if action else ""
    print(f"[{ts_log}] Bus: {number} ✅{suffix}")

    captures_dir = Path(__file__).resolve().parent.parent / "captures"
    captures_dir.mkdir(exist_ok=True)
    tag = f"_{direction}" if direction != "unknown" else ""

    # Snapshot del frame actual de cada cámara y armado de captura independiente.
    # `all_states` está indexado por cam_key (int 1..4) — ordenamos por esa clave.
    saved_crop_paths: dict[str, str] = {}
    for cam_key in sorted(all_states.keys()):
        state = all_states[cam_key]
        cam_label = state.label
        with state.lock:
            snap = state.frame.copy() if state.frame is not None else None
        if snap is None:
            print(f"  [CROP] {cam_label} → sin señal")
            continue
        capture = build_capture(snap, cam_label, direction)
        if capture is None or capture.size == 0:
            print(f"  [CROP] {cam_label} → frame inválido")
            continue
        cam_slug = cam_label.lower().replace(" ", "")
        crop_filename = captures_dir / f"{ts_file}_{number}_{cam_slug}{tag}.jpg"
        cv2.imwrite(str(crop_filename), capture)
        saved_crop_paths[cam_label] = str(crop_filename)
        print(f"  [CROP] {cam_label} → {crop_filename.name} ({capture.shape[1]}×{capture.shape[0]})")

    # Imagen principal: preferimos Cam 1 si hay (barrera), si no la primera disponible.
    main_path = saved_crop_paths.get("Cam 1") or next(iter(saved_crop_paths.values()), None)
    if main_path is None:
        # Ninguna cámara tenía frame — grabamos la detección igual sin imagen
        main_path = str(captures_dir / f"{ts_file}_{number}{tag}.jpg")

    nueva_id, is_new = db_insertar(number, direction, main_path, crop_paths=saved_crop_paths)
    if not is_new:
        print(f"  [MERGE] Agregado a detección existente id={nueva_id}")

    emit_event({
        "type": "detection",
        "id": nueva_id,
        "timestamp": ts_log,
        "numero_flota": number,
        "direccion": direction,
        "imagen_path": main_path,
        "crops": {cam: Path(p).name for cam, p in saved_crop_paths.items()},
    })

    # Every new detection triggers an individual Gemini analysis. When both
    # legs of a round trip have analyses, the module fires the text comparison.
    if is_new:
        from web.damage_detector import analizar_individual_async
        analizar_individual_async(
            detection_id=nueva_id,
            numero_flota=number,
            direccion=direction,
            fallback_path=main_path,
        )

    for s in all_states.values():
        with s.lock:
            s.confirmed_label = label


# ── Tile rendering ────────────────────────────────────────────────────────────

OVERLAY_MAX_AGE = 0.5  # segundos — si el overlay es más viejo que esto, mostrar frame en vivo

def render_tile(state: CameraState, global_buffer: ConsensusBuffer) -> np.ndarray:
    now = time.monotonic()
    with state.lock:
        fresh           = state.overlay_frame is not None and (now - state.overlay_ts) < OVERLAY_MAX_AGE
        overlay         = state.overlay_frame.copy() if fresh else None
        frame           = state.frame.copy() if state.frame is not None else None
        connected       = state.connected
        confirmed_label = state.confirmed_label
        cam_votes       = state.pending_count
        label           = state.label

    # Use overlay only while fresh — otherwise show live frame
    if overlay is not None:
        frame = overlay

    if frame is None:
        tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
        status = "Conectando..." if not connected else "Sin señal"
        cv2.putText(tile, label,  (10, 30),        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(tile, status, (10, TILE_H//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        return tile

    tile = cv2.resize(frame, (TILE_W, TILE_H))

    # camera label
    cv2.putText(tile, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # voting window progress bar (shown on all tiles while window is active)
    if global_buffer.window_active:
        elapsed, total = global_buffer.window_progress()
        bar_w = int((elapsed / total) * (TILE_W - 20))
        cv2.rectangle(tile, (10, 38), (TILE_W - 10, 50), (60, 60, 60), -1)
        cv2.rectangle(tile, (10, 38), (10 + bar_w, 50), (0, 200, 255), -1)
        label_votes = f"analizando... {elapsed:.1f}s/{total:.0f}s  [{cam_votes} votos esta cam]"
        cv2.putText(tile, label_votes, (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)

    # confirmed number + direction (only show if window is not active)
    if confirmed_label is not None and not global_buffer.window_active:
        bus_label = f"BUS: {confirmed_label}"
        (tw, _), _ = cv2.getTextSize(bus_label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        cv2.putText(tile, bus_label, (TILE_W - tw - 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

    return tile


# ── Web server (background thread) ───────────────────────────────────────────

def _start_web_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Arranca el servidor FastAPI en un thread de fondo. No bloquea."""
    import uvicorn

    config = uvicorn.Config("web.app:app", host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    t = threading.Thread(target=server.run, daemon=True, name="web-server")
    t.start()
    print(f"[INFO] Servidor web en http://localhost:{port}/api/detecciones")


# ── Main ──────────────────────────────────────────────────────────────────────

def main_loop(skip: int, confirm: int, min_window: float, max_window: float,
              fps_cap: int, min_bbox: int, debug: bool, exclude: set[int],
              ocr_backend: str = "moondream") -> None:
    init_db()
    _start_web_server()
    states = {k: CameraState(k) for k in CAMERAS}
    global_buffer = ConsensusBuffer(max_window_sec=max_window, number_cooldown_sec=10.0)

    print("[INFO] Cargando modelo YOLO (instancia compartida) ...")
    detector = BusDetector()
    print(f"[INFO] YOLO listo en {detector._device}.")

    from ocr.moondream_reader import get_moondream_reader
    print("[INFO] Precargando Moondream ...")
    _reader = get_moondream_reader()
    if _reader._model is None:
        _reader._load()
    print("[INFO] Moondream listo.")

    for k, state in states.items():
        threading.Thread(target=reader_worker,
                         args=(state, skip, fps_cap), daemon=True).start()
        # excluded cameras or inactive cams: reader only, no detection
        if k not in exclude and k in ACTIVE_CAMS:
            motion_filter = MotionFilter()
            threading.Thread(target=yolo_worker,
                             args=(state, detector, motion_filter, min_bbox, debug),
                             daemon=True).start()
            threading.Thread(target=ocr_worker,
                             args=(state, global_buffer, states, motion_filter, debug, ocr_backend),
                             daemon=True).start()
        else:
            print(f"[INFO] {state.label}: solo video, sin detección (excluida)")

    print("Controles: [ Q ] salir\n")

    while True:
        # tick: close window if it expired with no new detections coming in
        for event in global_buffer.tick():
            f = None
            for s in states.values():
                with s.lock:
                    f = s.frame
                if f is not None:
                    break
            _handle_event(event, f, states)

        row0 = np.hstack([render_tile(states[1], global_buffer),
                          render_tile(states[2], global_buffer)])
        row1 = np.hstack([render_tile(states[3], global_buffer),
                          render_tile(states[4], global_buffer)])
        cv2.imshow("FonoBus - Multicam", np.vstack([row0, row1]))

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("[INFO] Cerrado.")


def main():
    parser = argparse.ArgumentParser(description="Vista multicámara con detección de buses")
    parser.add_argument("--skip",     type=int,   default=2,   help="Procesar 1 de cada N frames (default: 2)")
    parser.add_argument("--confirm",  type=int,   default=2,   help="Votos mínimos para confirmar (default: 2)")
    parser.add_argument("--min-window", type=float, default=2.0, help="Segundos mínimos de ventana (default: 2)")
    parser.add_argument("--max-window", type=float, default=6.0, help="Segundos máximos de ventana (default: 6)")
    parser.add_argument("--fps",      type=int,   default=10,  help="FPS máximo del reader (default: 10)")
    parser.add_argument("--min-bbox", type=int,   default=80,   help="Tamaño mínimo de bbox en px (default: 80)")
    parser.add_argument("--exclude",  type=int,   nargs="*", default=[], help="Cámaras a excluir de detección (ej: --exclude 4)")
    parser.add_argument("--debug",    action="store_true",      help="Logs detallados del pipeline")
    parser.add_argument(
        "--ocr-backend",
        choices=["moondream", "easyocr"],
        default="moondream",
        help="Backend OCR: 'moondream' (default, local) o 'easyocr' (legacy)",
    )
    args = parser.parse_args()

    print(f"Config: skip={args.skip} confirm={args.confirm} window={args.min_window}-{args.max_window}s fps={args.fps} min-bbox={args.min_bbox} exclude={args.exclude} ocr={args.ocr_backend}\n")
    main_loop(
        skip=args.skip,
        confirm=args.confirm,
        min_window=args.min_window,
        max_window=args.max_window,
        fps_cap=args.fps,
        min_bbox=args.min_bbox,
        debug=args.debug,
        exclude=set(args.exclude),
        ocr_backend=args.ocr_backend,
    )


if __name__ == "__main__":
    main()
