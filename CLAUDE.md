# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt   # ultralytics, opencv-python, transformers, torch, pillow, easyocr, numpy
```

Python 3.11. All scripts use `sys.path.insert(0, project_root)` so they must be run from the repo root.

## Running scripts

```bash
# Full pipeline on a single image (main debug tool)
python scripts/detect_fleet.py --image test_images/frame.jpg --verbose

# Test only YOLO detection
python scripts/test_detection.py --image test_images/frame.jpg --save

# Test ROI zone extraction (saves annotated image + individual crops)
python scripts/test_roi.py --image test_images/frame.jpg

# Live RTSP multi-camera stream (4 cameras) — default OCR: moondream
python scripts/rtsp_multicam.py --debug

# Live RTSP single camera with keyboard switching (1/2/3/4) — legacy EasyOCR only
python scripts/rtsp_stream.py
```

## Architecture

### OCR backends

There are two backends, selectable via `--ocr-backend`:

**Moondream (default):** sends the full bus bounding box crop to a local vision model (`vikhyatk/moondream2`). Skips the ROI/preprocessing/EasyOCR stages entirely.
```
Frame → BusDetector → extract_full_bus_crop → MoondreamReader → int | None
```

**EasyOCR (legacy):** crops sub-zones from the bus bbox, generates image variants, and runs character-level OCR.
```
Frame → BusDetector → extract_rois → process (variants) → read_candidates → select_best → int | None
```

Key files for each stage:
```
detectors/yolo_detector.py       BusDetection objects
roi/extractor.py                 ROICrop (EasyOCR path) / full crop (Moondream path)
preprocessing/image_processor.py ProcessedVariant (EasyOCR path only)
ocr/reader.py                    EasyOCR — module-level singleton via get_reader()
ocr/moondream_reader.py          Moondream2 — module-level singleton via get_moondream_reader()
filters/candidate_filter.py      EasyOCR path only — select_best()
```

`scripts/detect_fleet.py` is the reference implementation that wires all stages together on a single image.

**Key design decisions:**
- `BusDetector` loads YOLO on `__init__` — instantiate once per thread, never per frame.
- `MoondreamReader` lazy-loads on first call (~1.8 GB download). Uses MPS on Apple Silicon, CUDA if available, CPU otherwise. Thread-safe via an internal lock (moondream is not thread-safe).
- ROI zones (EasyOCR path) are relative fractions of the bus bounding box (defined in `roi/extractor.py:ZONES`). Four zones cover rear, lateral, front-lateral, and a wide fallback.
- Valid fleet numbers: 10–1500, configured in `config/settings.py`.

## Live stream architecture (`rtsp_multicam.py`)

Three threads per camera: **reader** + **yolo_worker** + **ocr_worker**. A single `ConsensusBuffer` is shared across all camera threads.

- **reader**: reads frames from RTSP continuously, stores latest in `state.frame`, queues every Nth frame for YOLO via `state.process_event`.
- **yolo_worker**: runs YOLO + `MotionFilter` on every queued frame. When a moving bus is found, puts the crop into `state.ocr_queue` (maxsize=1 — drops stale crops if OCR is busy) and updates the display overlay immediately.
- **ocr_worker**: blocks on `state.ocr_queue`, runs Moondream (or EasyOCR) on the crop, feeds result to `ConsensusBuffer`. Runs independently so YOLO is never blocked by OCR inference.

This decoupling means YOLO tracks buses at full frame rate regardless of how long Moondream takes.

- `MotionFilter` (per camera): skips buses that haven't moved > 50px between detections (ignores parked buses). Also tracks net horizontal displacement for direction detection.
- `ConsensusBuffer`: opens a time window (default 2–6s) on first detection, collects votes from all cameras, closes early once min_votes are reached. Per-number cooldown (10s) prevents re-reporting the same bus.
- RTSP transport is forced to TCP via `OPENCV_FFMPEG_CAPTURE_OPTIONS` set before `import cv2`.
- `overlay_frame` expires after 0.5s (`OVERLAY_MAX_AGE`) so the display always falls back to the live reader frame rather than freezing on the last detected bus.

## Direction detection

Each vote includes a direction (`"entering"` / `"exiting"` / `"unknown"`). Direction is determined by the net horizontal displacement tracked in `MotionFilter.get_direction()`.

**Only Cam 1 (barrier camera) votes on direction.** It's the only camera where direction is unambiguous: buses crossing right→left are entering, left→right are exiting. Other cameras see maneuvering buses inside the depot and cannot reliably determine direction.

| Camera | Entering | Exiting | Used for direction? |
|--------|----------|---------|-----------------|
| Cam 1  | der→izq  | izq→der | **Yes** — barrier camera |
| Cam 2  | varies   | varies  | No |
| Cam 3  | varies   | varies  | No — maneuvering buses confuse it |
| Cam 4  | head-on  | head-on | No |

`DIRECTION_CAM = "Cam 1"`, `DIRECTION_CAM_ENTERING_LEFT = True` (net_dx < 0 = entering).

Minimum net displacement to assign a direction: `MIN_DIRECTION_PX = 80`.

`MotionFilter` uses `slot_radius=400px` to track buses across frames — buses can move up to 400px between processed frames and still be matched to their existing slot.

## Configuration (`config/settings.py`)

| Setting | Value | Notes |
|---|---|---|
| `YOLO_MODEL` | `yolov8m.pt` | Reliable on all 3 CCTV angles; `yolov8n.pt` misses awkward shots |
| `YOLO_MIN_CONFIDENCE` | 0.40 | |
| `YOLO_BUS_CLASS_IDS` | `[5, 7]` | 5=bus, 7=truck (rear close-ups sometimes classified as truck) |
| `FLEET_MIN` / `FLEET_MAX` | 10 / 1500 | |
| `FLEET_BLACKLIST` | `{90}` | Numbers explicitly rejected (e.g. speed limit signs painted on buses) |
| `FLEET_YEAR_THRESHOLD` | 2000 | 4-digit numbers ≥ this are rejected as years (not applicable given max=1500) |
| `OCR_MIN_CONFIDENCE` | 0.65 | EasyOCR path only |
| `CLAUDE_OCR_MODEL` | `claude-haiku-4-5-20251001` | Unused in current default flow |

## Captured images

Confirmed detections are saved to `captures/` as `YYYYMMDD_HHMMSS_<number>_<direction>.jpg` (direction tag omitted if unknown).

---

## Estado actual del proyecto (2026-04-10)

### Ramas
| Rama | Responsable | Propósito |
|------|-------------|-----------|
| `main` | — | Base estable |
| `interfaz-web` | Lucia | Interfaz web para integrar el sistema |
| `cambios-direcciones-deteccion` | Compañero | Mejoras en detección y dirección |

### Infraestructura
- **IP cámaras (externa):** `190.220.138.178:34224`
- **Credenciales RTSP:** `test:fono1234`
- **Canales:** Cam 1 → ch1, Cam 2 → ch5, Cam 3 → ch9, Cam 4 → ch13

### Qué funciona
- Pipeline completo: YOLO → Moondream OCR → ConsensusBuffer → captura
- 4 cámaras en paralelo con threads separados (reader / yolo_worker / ocr_worker)
- Detección de dirección por zona en Cam 1 (barrera): zona A (exterior) → zona B (depósito) = entrando; inverso = saliendo
- Blacklist de número 90 (carteles de límite de velocidad pintados en buses)

### Pendiente / en progreso
- Validar dirección en vivo con buses reales cruzando Cam 1
- Interfaz web (`interfaz-web`): por iniciar
- Mejoras de detección (`cambios-direcciones-deteccion`): en progreso por compañero

### Cómo correr
```bash
cd fonobus && .venv/bin/python scripts/rtsp_multicam.py --debug
```
