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

# Fleet number valid range
FLEET_MIN = 10
FLEET_MAX = 1500

# Numbers explicitly blacklisted — common false positives (e.g. speed limit signs)
FLEET_BLACKLIST = {10, 90}

# 4-digit numbers >= this value are assumed to be years — not applicable with max=1500
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
