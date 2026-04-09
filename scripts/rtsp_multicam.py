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
from roi.extractor import extract_rois, extract_full_bus_crop
from preprocessing.image_processor import process
from ocr.reader import read_candidates
from filters.candidate_filter import select_best

BASE_URL = "rtsp://test:fono1234@190.220.138.178:34224/cam/realmonitor"
RECONNECT_DELAY = 5

CAMERAS = {
    1: {"channel": 1,  "label": "Cam 1"},
    2: {"channel": 5,  "label": "Cam 2"},
    3: {"channel": 9,  "label": "Cam 3"},
    4: {"channel": 13, "label": "Cam 4"},
}

TILE_W = 640
TILE_H = 360

# Cam 1 is at the barrier — the only camera where direction is unambiguous.
DIRECTION_CAM = "Cam 1"

# Virtual barrier lines for Cam 1, as a fraction of frame width.
# A bus whose center crosses the RED line first is entering (comes from outside/right).
# A bus whose center crosses the YELLOW line first is exiting (leaves to outside/right).
CAM1_LINE_RED_X    = 0.80   # right barrier
CAM1_LINE_YELLOW_X = 0.18   # left barrier

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

    def _find_slot(self, cx: float, cy: float) -> int | None:
        """Return the nearest slot within slot_radius, or None."""
        best_id = None
        best_dist = float("inf")
        for sid, slot in self._slots.items():
            dist = ((cx - slot["cx_last"]) ** 2 + (cy - slot["cy"]) ** 2) ** 0.5
            if dist < self.slot_radius and dist < best_dist:
                best_dist = dist
                best_id = sid
        return best_id

    def is_moving(self, detection: BusDetection, frame_width: int = 0) -> bool:
        """
        Returns True if the bus is actively moving.
        frame_width: pass frame.shape[1] for Cam 1 to enable line-crossing tracking.
        """
        x1, y1, x2, y2 = detection.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        now = time.monotonic()

        sid = self._find_slot(cx, cy)
        if sid is None:
            # New bus — create slot, skip this frame
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
                    slot["first_crossing"] = "entering"   # came from outside (A) to inside (B)
                elif zones[0] == "B" and "A" in zones:
                    slot["first_crossing"] = "exiting"    # came from inside (B) to outside (A)

        slot["cx_last"] = cx
        slot["cy"] = cy
        slot["last_seen_time"] = now

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
        Shows virtual barrier lines, bus trajectory, net_dx, and direction state.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # Virtual barrier lines
        red_x    = int(w * CAM1_LINE_RED_X)
        yellow_x = int(w * CAM1_LINE_YELLOW_X)
        cv2.line(out, (red_x, 0),    (red_x, h),    (0, 0, 255),   2)
        cv2.line(out, (yellow_x, 0), (yellow_x, h), (0, 255, 255), 2)
        cv2.putText(out, "ENTRA", (red_x - 60, 20),    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),   1)
        cv2.putText(out, "SALE",  (yellow_x + 5, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Zone overlays (semitransparent)
        overlay_zones = out.copy()
        cv2.rectangle(overlay_zones, (red_x, 0), (w, h), (0, 0, 255), -1)       # red = Zona A (exterior)
        cv2.rectangle(overlay_zones, (0, 0), (yellow_x, h), (0, 255, 255), -1)  # yellow = Zona B (depósito)
        cv2.addWeighted(overlay_zones, 0.08, out, 0.92, 0, out)
        cv2.putText(out, "ZONA A (ext)", (red_x + 5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(out, "ZONA B (dep)", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

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


# ── Pipeline ──────────────────────────────────────────────────────────────────

def process_frame(
    frame,
    detector: BusDetector,
    motion_filter: MotionFilter,
    min_bbox: int = 120,
    label: str = "",
    debug: bool = False,
    ocr_backend: str = "claude",
) -> int | None:
    detections = detector.detect(frame)
    if not detections:
        if debug:
            print(f"  [{label}] YOLO: sin bus")
        return None

    # Evaluate all detections: update motion slots for every bus, collect moving ones.
    # Among moving buses, pick the largest bbox (closest to camera).
    fw = frame.shape[1] if label == DIRECTION_CAM else 0
    moving = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        bus_w, bus_h = x2 - x1, y2 - y1

        if debug:
            print(f"  [{label}] YOLO: bus conf={det.confidence:.2f} bbox=({x1},{y1})-({x2},{y2})")

        # B: minimum bbox size
        if bus_w < min_bbox or bus_h < min_bbox:
            if debug:
                print(f"  [{label}] SKIP: bbox demasiado pequeño ({bus_w}×{bus_h}px < {min_bbox}px)")
            # Still update motion slot even if skipped for size
            motion_filter.is_moving(det, frame_width=fw)
            continue

        # A: static bus filter — always call to update slot state
        if not motion_filter.is_moving(det, frame_width=fw):
            if debug:
                print(f"  [{label}] SKIP: bus estático (no se movió o estacionado)")
            continue

        moving.append(det)

    if not moving:
        return None

    # Pick the moving bus closest to the camera (largest bbox area)
    best = max(moving, key=lambda d: d.area)
    if debug and len(moving) > 1:
        print(f"  [{label}] {len(moving)} buses en movimiento → usando el más cercano (área={best.area}px²)")

    # Direction: only Cam 1 (barrier camera) can reliably distinguish entering vs exiting
    if label == DIRECTION_CAM:
        direction = motion_filter.get_direction(best) or "unknown"
        if debug and direction != "unknown":
            print(f"  [{label}] DIRECCIÓN DETECTADA: {direction}")
    else:
        direction = "unknown"

    if ocr_backend == "moondream":
        from ocr.moondream_reader import get_moondream_reader
        bus_crop = extract_full_bus_crop(frame, best)
        if bus_crop is None:
            if debug:
                print(f"  [{label}] CROP: demasiado pequeño")
            return None

        # If the bus bbox covers > 40% of the frame the bus is too close and
        # the full crop won't show the number. Split the frame into 4 quadrants
        # and try each one — the number will be in one of the corners.
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = best.bbox
        bbox_ratio = ((x2 - x1) * (y2 - y1)) / (fw * fh)
        if bbox_ratio > 0.40:
            hw, hh = fw // 2, fh // 2
            subcrops = [
                frame[0:hh,  0:hw],       # top-left
                frame[0:hh,  hw:fw],      # top-right
                frame[hh:fh, 0:hw],       # bottom-left
                frame[hh:fh, hw:fw],      # bottom-right
            ]
            if debug:
                print(f"  [{label}] BUS GRANDE ({bbox_ratio:.0%} frame) → probando sub-crops")
                # Dump crops to disk for inspection
                dump_dir = Path(__file__).resolve().parent.parent / "captures" / "crop_debug"
                dump_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%H%M%S_%f")[:9]
                cv2.imwrite(str(dump_dir / f"{ts}_{label.replace(' ','')}_full.jpg"), bus_crop)
                names = ["tl", "tr", "bl", "br"]
                for name, sc in zip(names, subcrops):
                    cv2.imwrite(str(dump_dir / f"{ts}_{label.replace(' ','')}_sub_{name}.jpg"), sc)
                print(f"  [{label}] DUMP → captures/crop_debug/{ts}_{label.replace(' ','')}_*.jpg")
            number = get_moondream_reader().read_subcrops(subcrops)
        else:
            number = get_moondream_reader().read(bus_crop, cam_label=label)
            if debug and number is None:
                # Dump normal crop too for comparison
                dump_dir = Path(__file__).resolve().parent.parent / "captures" / "crop_debug"
                dump_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%H%M%S_%f")[:9]
                cv2.imwrite(str(dump_dir / f"{ts}_{label.replace(' ','')}_normal.jpg"), bus_crop)

        if debug:
            print(f"  [{label}] MOONDREAM: {number if number else 'none'} dir={direction}")
        return (number, direction) if number else None

    # ── EasyOCR path (legacy) ──────────────────────────────────────────────
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
    return (result.number, direction) if result else None


# ── Consensus buffer (shared across all cameras) ──────────────────────────────

class ConsensusBuffer:
    """
    Adaptive voting window: opens on first detection, closes early once
    min_votes are reached (after min_window_sec), or after max_window_sec.

    Supports multiple simultaneous buses (e.g. one entering, one exiting):
    each distinct bus number that reaches min_votes is confirmed independently.
    Votes carry direction ("entering"/"exiting"/"unknown"); majority direction wins.
    Cooldown is PER NUMBER.
    """

    def __init__(self, min_window_sec: float = 2.0, max_window_sec: float = 6.0,
                 min_votes: int = 2, number_cooldown_sec: float = 10.0):
        self.min_window_sec      = min_window_sec
        self.max_window_sec      = max_window_sec
        self.min_votes           = min_votes
        self.number_cooldown_sec = number_cooldown_sec

        self._votes: list[tuple[int, str, str]] = []  # (number, direction, cam_label)
        self._window_start: float | None = None
        self._reported: dict[int, float] = {}          # number → last reported timestamp
        self._lock = threading.Lock()

    def feed(self, result: tuple[int, str] | None, cam_label: str) -> list[tuple[int, str]]:
        now = time.monotonic()
        with self._lock:
            if self._window_start is not None:
                if result is not None:
                    number, direction = result
                    self._votes.append((number, direction, cam_label))
                elapsed = now - self._window_start
                if elapsed >= self.min_window_sec:
                    counts: dict[int, int] = {}
                    for n, _, _ in self._votes:
                        counts[n] = counts.get(n, 0) + 1
                    if max(counts.values(), default=0) >= self.min_votes:
                        return self._close(now)
                if elapsed >= self.max_window_sec:
                    return self._close(now)
                return []

            if result is not None:
                number, direction = result
                self._window_start = now
                self._votes = [(number, direction, cam_label)]

        return []

    def tick(self) -> list[tuple[int, str]]:
        """Close expired window even when no new detections arrive."""
        now = time.monotonic()
        with self._lock:
            if self._window_start is not None:
                elapsed = now - self._window_start
                if elapsed >= self.min_window_sec:
                    counts: dict[int, int] = {}
                    for n, _, _ in self._votes:
                        counts[n] = counts.get(n, 0) + 1
                    if max(counts.values(), default=0) >= self.min_votes:
                        return self._close(now)
                if elapsed >= self.max_window_sec:
                    return self._close(now)
        return []

    def _close(self, now: float) -> list[tuple[int, str]]:
        """Confirm all numbers that reached min_votes. Must be called with lock held."""
        votes = self._votes
        self._votes = []
        self._window_start = None

        if not votes:
            return []

        # Count votes and collect direction votes per number
        counts: dict[int, int] = {}
        dir_votes: dict[int, dict[str, int]] = {}
        for n, d, _ in votes:
            counts[n] = counts.get(n, 0) + 1
            dir_votes.setdefault(n, {})
            dir_votes[n][d] = dir_votes[n].get(d, 0) + 1

        confirmed: list[tuple[int, str]] = []
        for n in sorted(counts, key=lambda x: counts[x], reverse=True):
            if counts[n] < self.min_votes:
                continue
            last = self._reported.get(n, 0.0)
            if now - last < self.number_cooldown_sec:
                continue
            real_dir_votes = {d: c for d, c in dir_votes[n].items() if d != "unknown"}
            majority_dir = max(real_dir_votes, key=real_dir_votes.get) if real_dir_votes else "unknown"
            self._reported[n] = now
            confirmed.append((n, majority_dir))

        return confirmed

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
        self.confirmed_label: str | None = None  # e.g. "325 Entró"
        self.pending_count: int = 0              # this camera's votes in window
        self.connected: bool = False
        self.lock = threading.Lock()
        self.process_event = threading.Event()
        self.last_capture: np.ndarray | None = None


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
                time.sleep(RECONNECT_DELAY)
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize lag — always show latest frame
            with state.lock:
                state.connected = True
            frame_count = 0

        t0 = time.monotonic()

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            cap = None
            with state.lock:
                state.connected = False
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


# ── Processor thread ──────────────────────────────────────────────────────────

def processor_worker(
    state: CameraState,
    global_buffer: ConsensusBuffer,
    all_states: dict,
    detector: BusDetector,
    min_bbox: int,
    debug: bool = False,
    ocr_backend: str = "claude",
) -> None:
    motion_filter = MotionFilter()
    print(f"[INFO] {state.label}: listo.")

    while True:
        state.process_event.wait()
        state.process_event.clear()

        with state.lock:
            frame = state.process_frame

        if frame is None:
            continue

        try:
            raw = process_frame(
                frame, detector, motion_filter,
                min_bbox=min_bbox, label=state.label, debug=debug,
                ocr_backend=ocr_backend,
            )
        except Exception as e:
            if debug:
                print(f"  [{state.label}] ERROR: {e}")
            raw = None

        # For Cam 1: update the direction debug overlay (separate try so it
        # can never nullify a successful detection if it fails).
        if state.label == DIRECTION_CAM:
            try:
                annotated = motion_filter.draw_cam1_overlay(frame)
                with state.lock:
                    state.overlay_frame = annotated
            except Exception as e:
                if debug:
                    print(f"  [{state.label}] OVERLAY ERROR: {e}")

        # feed into the shared consensus buffer
        confirmed_list = global_buffer.feed(raw, state.label)

        if debug and raw is not None:
            votes = global_buffer.all_votes()
            print(f"  [{state.label}] VOTO: {raw}  |  votos en ventana: {votes}")

        for confirmed_number, confirmed_direction in confirmed_list:
            _report(confirmed_number, confirmed_direction, frame, all_states)

        # update this camera's vote count for the HUD
        with state.lock:
            state.pending_count = global_buffer.vote_count(state.label)


# ── Report helper ────────────────────────────────────────────────────────────

def _report(number: int, direction: str, frame: np.ndarray, all_states: dict) -> None:
    ts_log  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    if direction == "entering":
        action = "Entró"
    elif direction == "exiting":
        action = "Salió"
    else:
        action = ""

    label = f"{number} {action}".strip()
    suffix = f" {action}" if action else ""
    print(f"[{ts_log}] Bus: {number} ✅{suffix}")

    captures_dir = Path(__file__).resolve().parent.parent / "captures"
    captures_dir.mkdir(exist_ok=True)
    tag = f"_{direction}" if direction != "unknown" else ""
    filename = captures_dir / f"{ts_file}_{number}{tag}.jpg"
    cv2.imwrite(str(filename), frame)

    for s in all_states.values():
        with s.lock:
            s.confirmed_label = label


# ── Tile rendering ────────────────────────────────────────────────────────────

def render_tile(state: CameraState, global_buffer: ConsensusBuffer) -> np.ndarray:
    with state.lock:
        overlay         = state.overlay_frame.copy() if state.overlay_frame is not None else None
        frame           = state.frame.copy() if state.frame is not None else None
        connected       = state.connected
        confirmed_label = state.confirmed_label
        cam_votes       = state.pending_count
        label           = state.label

    # Cam 1: prefer the annotated overlay frame so the debug info is always visible
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main_loop(skip: int, confirm: int, min_window: float, max_window: float,
              fps_cap: int, min_bbox: int, debug: bool, exclude: set[int],
              ocr_backend: str = "moondream") -> None:
    states = {k: CameraState(k) for k in CAMERAS}
    global_buffer = ConsensusBuffer(min_window_sec=min_window, max_window_sec=max_window, min_votes=confirm)

    print("[INFO] Cargando modelo YOLO (instancia compartida) ...")
    detector = BusDetector()
    print(f"[INFO] YOLO listo en {detector._device}.")

    for k, state in states.items():
        threading.Thread(target=reader_worker,
                         args=(state, skip, fps_cap), daemon=True).start()
        # excluded cameras: start reader (to show video) but no processor
        if k not in exclude:
            threading.Thread(target=processor_worker,
                             args=(state, global_buffer, states, detector, min_bbox, debug, ocr_backend),
                             daemon=True).start()
        else:
            print(f"[INFO] {state.label}: solo video, sin detección (excluida)")

    print("Controles: [ Q ] salir\n")

    while True:
        # tick: close window if it expired with no new detections coming in
        for confirmed_number, confirmed_direction in global_buffer.tick():
            f = None
            for s in states.values():
                with s.lock:
                    f = s.frame
                if f is not None:
                    break
            if f is None:
                continue  # no frame available yet, skip capture
            _report(confirmed_number, confirmed_direction, f, states)

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
