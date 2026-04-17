# COCO class indices accepted as "bus"
# 5 = bus, 7 = truck (rear close-up views of buses are sometimes classified as truck)
YOLO_BUS_CLASS_IDS = [5, 7]

# Minimum confidence to accept a YOLO detection
YOLO_MIN_CONFIDENCE = 0.40

# YOLOv8 model weights.
# yolov8n.pt is fast but misses awkward CCTV angles.
# yolov8m.pt gives reliable detection on all 3 camera types we tested.
YOLO_MODEL = "yolov8m.pt"

# Minimum OCR confidence to accept a candidate (0-1)
# Values below this are usually background text, noise, or partial reads
OCR_MIN_CONFIDENCE = 0.65

# ── Fleet whitelist (loaded from config/internos.csv) ────────────────────────
# Only numbers in this set are accepted. Replaces the old min/max/blacklist approach.

import csv as _csv
from pathlib import Path as _Path

def _load_whitelist() -> set[int]:
    csv_path = _Path(__file__).resolve().parent / "internos.csv"
    nums: set[int] = set()
    with open(csv_path, encoding="latin-1") as f:
        reader = _csv.DictReader(f, delimiter=";")
        for row in reader:
            code = row.get("\ufeffCódigo") or row.get("Código") or row.get("C\udcdcódigo") or list(row.values())[0]
            try:
                nums.add(int(code.strip()))
            except (ValueError, AttributeError):
                continue
    return nums

FLEET_WHITELIST: set[int] = _load_whitelist()

# Legacy aliases kept for backward compatibility in scoring logic
FLEET_MIN = min(FLEET_WHITELIST) if FLEET_WHITELIST else 1
FLEET_MAX = max(FLEET_WHITELIST) if FLEET_WHITELIST else 9999

# 4-digit numbers >= this value are assumed to be years (e.g. 2026 from CCTV timestamp)
FLEET_YEAR_THRESHOLD = 2000

# ── Captures cleanup ─────────────────────────────────────────────────────────

# Capturas más viejas que este número de días se eliminan automáticamente.
# El cleanup corre al iniciar el servidor y cada 24 horas.
CAPTURES_RETENTION_DAYS = 7

# ── Claude Vision OCR settings ────────────────────────────────────────────────

# Model used for fleet number reading via vision API
CLAUDE_OCR_MODEL = "claude-haiku-4-5-20251001"

# Minimum seconds between API calls per camera (rate limiting).
# At 1.5s per camera, 4 cameras = ~2.7 calls/sec total.
CLAUDE_MIN_INTERVAL_SEC = 1.5

# Max tokens for the OCR response (answer is at most 4 digits or "none")
CLAUDE_MAX_TOKENS = 16

# ── Crop padding (fed to Gemini / Moondream) ────────────────────────────────
# Padding alrededor del bbox de YOLO antes de mandar a la IA. Generoso para que
# el bus completo (espejos, paragolpes, parte trasera) entre en el crop aun si
# el bbox de YOLO quedó apretado.
CROP_PAD_X_FRAC = 0.40
CROP_PAD_Y_FRAC = 0.25
CROP_MIN_PAD_PX = 40

# Píxeles a cortar en el borde superior para tapar el timestamp del DVR (que
# Moondream/Gemini leen como número y confunden con el interno del bus).
# Aplicado tanto a `extract_full_bus_crop` como a frames completos guardados
# para análisis de daños.
CAPTURE_TIMESTAMP_TOP_PX = 60

# ── Night-mode preprocessing ────────────────────────────────────────────────
# Brillo medio (0-255) por debajo del cual el frame se considera "noche" y se
# aplica CLAHE + gamma antes de YOLO y a los crops para Gemini/Moondream.
NIGHT_BRIGHTNESS_THRESHOLD = 70.0
# Gamma >1 levanta sombras; rango seguro 1.4–1.8.
NIGHT_GAMMA = 1.6
# Threshold de YOLO reducido en noche: el contraste enhancado puede generar
# bbox ligeramente menos confiables y conviene ser más permisivo.
YOLO_MIN_CONFIDENCE_NIGHT = 0.30
