"""
Moondream2 local vision OCR reader.

Uses the vikhyatk/moondream2 model (HuggingFace) to read fleet numbers
from bus crop images. Runs 100% offline after the first download (~1.8 GB).

On Apple Silicon (MPS) inference takes ~1-3s. On CPU ~5-15s.
"""

import re
import threading
import time

import cv2
import numpy as np
from PIL import Image

from config.settings import FLEET_MAX, FLEET_MIN, FLEET_YEAR_THRESHOLD, FLEET_BLACKLIST

_MODEL_ID = "vikhyatk/moondream2"
_REVISION = "2024-08-26"

_QUERY = (
    "This image is a crop from a CCTV camera at a bus depot. "
    "STEP 1 — Identify the vehicle: the buses in this depot are FONO BUS passenger coaches, "
    "typically white with green or pink details, or all black. "
    "If the main vehicle visible is a fuel tanker, cistern truck, cargo truck, pickup, or car — reply with 'none'. "
    "Also reply 'none' if a truck or tanker is blocking most of the bus making the number unclear. "
    "STEP 2 — If it IS a FONO BUS passenger coach, find the FLEET NUMBER. "
    "IMPORTANT: 'FONO BUS' is the company name/brand — it is NOT the fleet number. Ignore it completely. "
    "The fleet number is a small number between 10 and 1500 painted separately from the branding (examples: 30, 325, 731, 764, 1022). "
    "Depending on the camera angle, look here: "
    "REAR view: lower section of the rear panel, near the bumper or below the rear window. "
    "FRONT view: lower-left or lower-right corner near the bumper. "
    "LATERAL view: check all four lower corners of the visible side — "
    "lower-rear-right, lower-rear-left, lower-front-right, lower-front-left — "
    "the number may appear near the rear wheel arch, below the driver window, below the door, or near the front bumper corner. "
    "Ignore: the brand name 'FONO BUS', street signs, license plates, capacity markings, timestamps, "
    "and speed limit markings (e.g. '90' painted on the upper lateral). "
    "Reply with ONLY the fleet number digits, or 'none' if you cannot find it."
)

_ORIENTATION_QUERY = (
    "This is a crop of a bus from a CCTV camera. "
    "Is the FRONT (headlights, windshield, front bumper visible) or the REAR (taillights, exhaust pipe, rear bumper visible) of the bus facing the camera? "
    "Reply with only 'front' or 'rear'. If you cannot tell, reply 'unknown'."
)

_DIGIT_RE = re.compile(r"\b(\d{1,4})\b")

# Common OCR confusions: letter → digit
_OCR_FIXES = str.maketrans("GgOoSsBbIilZz", "6600558811122")


def _normalize_ocr(text: str) -> str:
    """Replace letters commonly misread instead of digits."""
    return text.translate(_OCR_FIXES)


def _validate(value: int) -> bool:
    if not (FLEET_MIN <= value <= FLEET_MAX):
        return False
    if len(str(value)) == 4 and value >= FLEET_YEAR_THRESHOLD:
        return False
    if value in FLEET_BLACKLIST:
        return False
    return True


def _parse(text: str) -> int | None:
    text = text.strip().lower()
    if text == "none":
        return None
    text = _normalize_ocr(text)
    try:
        value = int(text)
        return value if _validate(value) else None
    except ValueError:
        pass
    match = _DIGIT_RE.search(text)
    if not match:
        return None
    value = int(match.group(1))
    return value if _validate(value) else None


class MoondreamReader:
    """
    Wraps vikhyatk/moondream2 for fleet-number OCR.

    The model is loaded once on first use (expensive). Subsequent calls
    are inference-only. Thread-safe via a lock (moondream is not thread-safe).
    """

    def __init__(self, min_interval_sec: float = 1.5) -> None:
        self._min_interval = min_interval_sec
        self._last_calls: dict[str, float] = {}
        self._rate_lock = threading.Lock()
        self._infer_lock = threading.Lock()  # moondream not thread-safe
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self) -> None:
        """Lazy-load the model on first call."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print(f"[Moondream] Cargando modelo en {device} (primera vez: descarga ~1.8 GB)...")

        dtype = torch.float16 if device == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_ID, revision=_REVISION, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            revision=_REVISION,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device)
        self._model.eval()
        self._device = device
        print(f"[Moondream] Listo en {device}.")

    def read(self, crop_bgr: np.ndarray, cam_label: str = "") -> int | None:
        """
        Read the fleet number from a BGR bus crop.

        Returns the integer fleet number, or None if unreadable or throttled.
        Never raises.
        """
        # Per-camera rate limit
        with self._rate_lock:
            now = time.monotonic()
            if now - self._last_calls.get(cam_label, 0.0) < self._min_interval:
                return None
            self._last_calls[cam_label] = now

        try:
            with self._infer_lock:
                if self._model is None:
                    self._load()

                # Convert BGR numpy → RGB PIL
                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                enc = self._model.encode_image(pil_img)
                answer = self._model.answer_question(enc, _QUERY, self._tokenizer)

            import torch
            if self._device == "cuda":
                torch.cuda.empty_cache()

            return _parse(answer)

        except Exception as e:
            print(f"[Moondream] ERROR: {e}")
            return None

    def read_subcrops(self, crops: list[np.ndarray]) -> int | None:
        """
        Try multiple crops in sequence (no rate limit).
        Returns the first valid fleet number found, or None.
        Used as fallback when the bus bbox covers most of the frame.
        """
        try:
            with self._infer_lock:
                if self._model is None:
                    self._load()
                for crop in crops:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    enc = self._model.encode_image(pil_img)
                    answer = self._model.answer_question(enc, _QUERY, self._tokenizer)
                    result = _parse(answer)
                    if result is not None:
                        return result
            return None
        except Exception as e:
            print(f"[Moondream] ERROR subcrop: {e}")
            return None

    def read_with_orientation(
        self, crop_bgr: np.ndarray, cam_label: str = ""
    ) -> tuple[int | None, str | None]:
        """
        Read fleet number AND bus orientation (front/rear) from the same crop.
        Encodes the image once and asks two questions.

        Returns (number, orientation) where orientation is "front", "rear", or None.
        Returns (None, None) if throttled or on error.
        """
        with self._rate_lock:
            now = time.monotonic()
            if now - self._last_calls.get(cam_label, 0.0) < self._min_interval:
                return None, None
            self._last_calls[cam_label] = now

        try:
            with self._infer_lock:
                if self._model is None:
                    self._load()

                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                enc = self._model.encode_image(pil_img)
                number_answer = self._model.answer_question(enc, _QUERY, self._tokenizer)
                orientation_answer = self._model.answer_question(enc, _ORIENTATION_QUERY, self._tokenizer)

            number = _parse(number_answer)
            orientation_raw = orientation_answer.strip().lower()
            if "front" in orientation_raw:
                orientation = "front"
            elif "rear" in orientation_raw:
                orientation = "rear"
            else:
                orientation = None

            return number, orientation

        except Exception as e:
            print(f"[Moondream] ERROR: {e}")
            return None, None


# Module-level singleton
_reader: MoondreamReader | None = None


def get_moondream_reader() -> MoondreamReader:
    global _reader
    if _reader is None:
        _reader = MoondreamReader()
    return _reader
