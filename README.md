# FonoBus

Sistema de detección automática de números de flota de buses y dirección de circulación (entrada/salida del depósito) a partir de cámaras RTSP.

## Qué hace

- Detecta buses en tiempo real usando YOLOv8
- Lee el número de flota pintado en el bus (ej: `325`, `1022`) con un modelo de visión local (Moondream2)
- Determina si el bus **está entrando o saliendo** del depósito
- Consolida votos de 4 cámaras simultáneas antes de confirmar una detección
- Guarda una captura de imagen por cada bus confirmado

## Setup

Python 3.11 requerido.

**Mac / Linux**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows** (PowerShell)
```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\activate
pip install -r requirements.txt
```

> La primera vez que corra el proyecto se descargan ~1.8 GB del modelo Moondream2 automáticamente.

## Uso rápido

**Mac / Linux**
```bash
# Probar en una imagen fija
python scripts/detect_fleet.py --image test_images/frame.jpg --verbose

# Stream en vivo, 4 cámaras simultáneas
python scripts/rtsp_multicam.py --debug

# Stream en vivo, 1 cámara (cambiar con teclas 1/2/3/4)
python scripts/rtsp_stream.py
```

**Windows** (PowerShell)
```powershell
# Probar en una imagen fija
.venv\Scripts\python scripts\detect_fleet.py --image test_images\frame.jpg --verbose

# Stream en vivo, 4 cámaras simultáneas
.venv\Scripts\python scripts\rtsp_multicam.py --debug

# Stream en vivo, 1 cámara (cambiar con teclas 1/2/3/4)
.venv\Scripts\python scripts\rtsp_stream.py
```

## Pipeline

```
Frame RTSP
  └─→ YOLOv8 (detectors/yolo_detector.py)
        └─→ Crop completo del bus
              └─→ Moondream2 (ocr/moondream_reader.py)   ← backend default
                    └─→ número de flota (int) o None
```

El backend legacy EasyOCR (`--ocr-backend easyocr`) extrae sub-zonas del bus, genera 5 variantes de imagen y corre OCR de caracteres.

## Detección de dirección

Las 4 cámaras ven el depósito desde ángulos distintos. Solo **Cam 3** puede distinguir entrada de salida de forma confiable (los buses se mueven en direcciones opuestas según el caso). Las cámaras 1, 2 y 4 votan el número pero no la dirección.

La dirección se determina por el desplazamiento horizontal neto del bus a través de los frames (`MotionFilter` en `rtsp_multicam.py`).

## Arquitectura multi-cámara

`rtsp_multicam.py` corre 2 hilos por cámara:

- **reader**: lee frames del stream RTSP continuamente
- **processor**: corre el pipeline de detección sobre los frames encolados

Un `ConsensusBuffer` compartido acumula votos de las 4 cámaras en una ventana de tiempo (2–6s). El número que alcanza el mínimo de votos se confirma. Cooldown de 10s por número para evitar re-reportar el mismo bus.

## Estructura

```
fonobus/
├── config/settings.py          Parámetros globales (modelo, rangos, umbrales)
├── detectors/yolo_detector.py  Detección de buses con YOLOv8
├── roi/extractor.py            Extracción de crops (full bus o sub-zonas)
├── preprocessing/              Variantes de imagen para EasyOCR
├── ocr/
│   ├── moondream_reader.py     Backend default — modelo de visión local
│   └── reader.py               Backend legacy — EasyOCR
├── filters/candidate_filter.py Selección del mejor candidato (EasyOCR)
└── scripts/
    ├── rtsp_multicam.py        Stream en vivo multi-cámara (script principal)
    ├── rtsp_stream.py          Stream en vivo cámara única
    ├── detect_fleet.py         Pipeline completo sobre imagen fija
    ├── test_detection.py       Solo YOLO, sin OCR
    ├── test_roi.py             Visualizar extracción de zonas ROI
    └── scan_cameras.py         Escanear canales RTSP disponibles
```

## Capturas

Cada bus confirmado se guarda en `captures/` como `YYYYMMDD_HHMMSS_<numero>_<direccion>.jpg`.

## Configuración clave

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `YOLO_MODEL` | `yolov8m.pt` | Modelo YOLO — `n` es más rápido, `m` más preciso |
| `FLEET_MIN/MAX` | 10 / 1500 | Rango válido de números de flota |
| `YOLO_MIN_CONFIDENCE` | 0.40 | Umbral mínimo de confianza YOLO |

Ver `config/settings.py` para todos los parámetros.
