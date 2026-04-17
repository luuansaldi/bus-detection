"""
Night-mode image enhancement.

Aplica CLAHE en el canal L de LAB + gamma correction para levantar sombras y
recuperar contraste en frames oscuros (cámaras CCTV de noche). Preserva el
color (no convierte a grises).

Uso típico:
    from preprocessing.night_enhancer import is_dark_frame, enhance
    if is_dark_frame(frame):
        frame_for_yolo = enhance(frame)

Funciones puras y thread-safe (sin estado compartido).
"""

import cv2
import numpy as np

from config.settings import NIGHT_BRIGHTNESS_THRESHOLD, NIGHT_GAMMA


def frame_brightness(frame: np.ndarray) -> float:
    """Brillo medio del frame en escala 0-255."""
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def is_dark_frame(frame: np.ndarray, threshold: float = NIGHT_BRIGHTNESS_THRESHOLD) -> bool:
    """True si el frame está por debajo del umbral de brillo nocturno."""
    return frame_brightness(frame) < threshold


_GAMMA_LUT_CACHE: dict[float, np.ndarray] = {}


def _gamma_lut(gamma: float) -> np.ndarray:
    """LUT para gamma correction. Cacheada por valor de gamma."""
    lut = _GAMMA_LUT_CACHE.get(gamma)
    if lut is None:
        inv = 1.0 / max(gamma, 1e-6)
        lut = np.array(
            [((i / 255.0) ** inv) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        _GAMMA_LUT_CACHE[gamma] = lut
    return lut


def sharpness(frame: np.ndarray) -> float:
    """
    Laplacian variance — valor alto = nítido, bajo = movido.
    En CCTV típicamente: >300 muy nítido, 100-300 aceptable, <100 movido.
    """
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def enhance(frame: np.ndarray, gamma: float = NIGHT_GAMMA) -> np.ndarray:
    """CLAHE en canal L de LAB + gamma. Devuelve BGR."""
    if frame is None or frame.size == 0:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.LUT(enhanced, _gamma_lut(gamma))
