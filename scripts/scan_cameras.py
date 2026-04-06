"""
Scan available RTSP channels on the DVR/NVR.

Usage:
    python scripts/scan_cameras.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

BASE_URL = "rtsp://test:fono1234@190.220.138.178:34224/cam/realmonitor"
MAX_CHANNELS = 16  # most DVRs have 4, 8, or 16 channels


def probe(channel: int, subtype: int = 0, timeout_ms: int = 4000) -> bool:
    url = f"{BASE_URL}?channel={channel}&subtype={subtype}"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
    ok = cap.isOpened()
    if ok:
        ret, frame = cap.read()
        ok = ret and frame is not None
    cap.release()
    return ok


def main():
    print(f"Escaneando canales 1–{MAX_CHANNELS} ...\n")
    found = []

    for ch in range(1, MAX_CHANNELS + 1):
        print(f"  canal {ch:2d} ... ", end="", flush=True)
        if probe(ch):
            print("OK")
            found.append(ch)
        else:
            print("sin señal")

    print()
    if found:
        print(f"Canales disponibles: {found}")
        print()
        for ch in found:
            print(f"  channel={ch}  →  {BASE_URL}?channel={ch}&subtype=0")
    else:
        print("No se encontró ningún canal activo.")


if __name__ == "__main__":
    main()
