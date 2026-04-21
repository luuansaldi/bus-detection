"""
Análisis de daños con Gemini 2.5 Flash — todos manuales desde el dashboard.

1. `analizar_individual_async(detection_id, ...)`
   Lo dispara el usuario desde el botón "Analizar" de una captura. Manda los
   crops disponibles (o la imagen principal) a Gemini y guarda el reporte en
   `detecciones.analisis_individual`.

2. `comparar_visual_async(numero_flota, detection_ids)`
   Lo dispara el usuario desde "Comparar seleccionadas" (N ≥ 2 fotos). Manda
   las imágenes directamente a Gemini y guarda un resumen de diferencias en
   `comparaciones`.

Requiere: GOOGLE_API_KEY en el entorno.
"""

import sys
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web.database import (
    actualizar_analisis_individual,
    get_deteccion_by_id,
    insertar_comparacion_n,
    obtener_crops,
)


PROMPT_INDIVIDUAL = (
    "Estas son {n_imgs} fotos del bus interno {numero_flota} tomadas por distintas cámaras "
    "({cam_labels}) cuando {evento} del depósito.\n"
    "Describí el ESTADO VISIBLE del bus. Buscá ÚNICAMENTE daños físicos: "
    "rayones profundos, abolladuras, espejos rotos, vidrios rotos, paragolpes dañados. "
    "Ignorá suciedad, barro, diferencias de iluminación o reflejos.\n"
    "Respondé SOLO en este formato exacto (4 líneas):\n"
    "ESTADO: OK | DAÑADO | INCONCLUYE\n"
    "SEVERIDAD: ninguna | menor | mayor\n"
    "ZONAS: <lista separada por comas de: frente, lateral izquierdo, lateral derecho, trasero> o 'no aplica'\n"
    "DESCRIPCION: <una oración en español, específica, mencionando zona y tipo de daño si lo hay>\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# Análisis individual
# ─────────────────────────────────────────────────────────────────────────────

def analizar_individual_async(detection_id: int, numero_flota: int, direccion: str,
                               fallback_path: str | None = None) -> None:
    """Lanza el análisis individual en un thread daemon y retorna inmediato."""
    t = threading.Thread(
        target=_analizar_individual,
        args=(detection_id, numero_flota, direccion, fallback_path),
        daemon=True,
        name=f"individual-{detection_id}",
    )
    t.start()


def _parse_individual(text: str) -> tuple[str, str]:
    """Extrae (estado, texto_completo_trimmed)."""
    estado = "inconcluye"
    for line in text.splitlines():
        if ":" in line and line.upper().startswith("ESTADO"):
            raw = line.split(":", 1)[1].strip().upper()
            if raw.startswith("OK"):
                estado = "ok"
            elif "DA" in raw:
                estado = "danado"
            else:
                estado = "inconcluye"
            break
    return estado, text.strip()


def _analizar_individual(detection_id: int, numero_flota: int, direccion: str,
                         fallback_path: str | None) -> None:
    evento = "salió" if direccion == "exiting" else ("entró" if direccion == "entering" else "pasó")
    try:
        from google import genai
        from google.genai import types
        import PIL.Image
    except ImportError:
        actualizar_analisis_individual(
            detection_id,
            "ERROR: google-genai no instalado",
            "error",
        )
        return

    # Reunir imágenes: crops por cámara si existen, si no el fallback.
    crops = obtener_crops(detection_id)
    imagenes: list[tuple[str, str]] = [(c["cam_label"], c["crop_path"]) for c in crops]
    if not imagenes and fallback_path:
        imagenes = [("principal", fallback_path)]

    if not imagenes:
        actualizar_analisis_individual(
            detection_id, "ERROR: sin imágenes disponibles", "error",
        )
        return

    try:
        client = genai.Client()
        pil_imgs = [PIL.Image.open(path) for _, path in imagenes]
        cam_labels = ", ".join(lbl for lbl, _ in imagenes)
        prompt = PROMPT_INDIVIDUAL.format(
            n_imgs=len(imagenes),
            numero_flota=numero_flota,
            cam_labels=cam_labels,
            evento=evento,
        )

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[*pil_imgs, prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=300,
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        estado, texto = _parse_individual(resp.text or "")
    except Exception as e:
        actualizar_analisis_individual(
            detection_id, f"ERROR: {e}", "error",
        )
        print(f"[DamageDetector] individual id={detection_id} ERROR: {e}")
        return

    actualizar_analisis_individual(detection_id, texto, estado)
    print(f"[DamageDetector] individual id={detection_id} bus={numero_flota} "
          f"dir={direccion} estado={estado}")

    try:
        from web.app import emit_event
        emit_event({
            "type": "individual_analysis",
            "detection_id": detection_id,
            "numero_flota": numero_flota,
            "direccion": direccion,
            "estado": estado,
            "analisis": texto,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Comparación visual N-way (manual)
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_COMPARACION_VISUAL = (
    "Te muestro {n} fotos del bus interno {numero_flota} tomadas en distintos "
    "momentos (entradas y salidas del depósito). El orden en que te las paso "
    "coincide con la cronología.\n\n"
    "{labels}\n\n"
    "Comparálas entre sí y buscá ÚNICAMENTE daños físicos NUEVOS o que hayan "
    "empeorado: rayones profundos, abolladuras, espejos rotos, vidrios rotos, "
    "paragolpes dañados, faltantes. Ignorá suciedad, barro, reflejos y "
    "diferencias de iluminación o ángulo.\n"
    "Respondé SOLO en este formato (5 líneas, obligatorio):\n"
    "RESULTADO: OK | DAÑOS | INCONCLUYE\n"
    "SEVERIDAD: ninguna | menor | mayor\n"
    "ZONAS: <lista separada por comas de: frente, lateral izquierdo, lateral derecho, trasero> o 'no aplica'\n"
    "ENTRE_FOTOS: <indicá entre qué fotos aparece el cambio, ej: '2 vs 3' o 'no aplica'>\n"
    "DESCRIPCION: <una oración en español, específica>\n"
)


def comparar_visual_async(numero_flota: int, detection_ids: list[int]) -> None:
    """Dispara la comparación N-way en un thread y retorna inmediato."""
    t = threading.Thread(
        target=_comparar_visual,
        args=(numero_flota, list(detection_ids)),
        daemon=True,
        name=f"compare-visual-{numero_flota}",
    )
    t.start()


def _parse_comparacion(text: str) -> tuple[str, str, str]:
    lines = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            lines[k.strip().upper()] = v.strip()

    raw = lines.get("RESULTADO", "").upper()
    if "OK" in raw:
        resultado = "ok"
    elif "DA" in raw:
        resultado = "danos"
    else:
        resultado = "inconcluye"

    severidad = lines.get("SEVERIDAD", "desconocida")
    descripcion = lines.get("DESCRIPCION", text[:200])
    return resultado, severidad, descripcion


def _comparar_visual(numero_flota: int, detection_ids: list[int]) -> None:
    if len(detection_ids) < 2:
        return

    detecciones = []
    for did in detection_ids:
        det = get_deteccion_by_id(did)
        if not det:
            continue
        path = det.get("imagen_path")
        if not path:
            crops = obtener_crops(did)
            if crops:
                path = crops[0]["crop_path"]
        if not path:
            continue
        detecciones.append((det, path))

    if len(detecciones) < 2:
        _insertar_error(numero_flota, detection_ids, "No se pudieron cargar imágenes suficientes")
        return

    print(f"[DamageDetector] comparando visual bus {numero_flota}: {len(detecciones)} fotos")
    try:
        from google import genai
        from google.genai import types
        import PIL.Image

        client = genai.Client()
        labels_lines = []
        pil_imgs = []
        for idx, (det, path) in enumerate(detecciones, start=1):
            evento = ("saliendo" if det["direccion"] == "exiting"
                      else "entrando" if det["direccion"] == "entering"
                      else "paso")
            labels_lines.append(f"Foto {idx} — {det['timestamp']} ({evento})")
            pil_imgs.append(PIL.Image.open(path))
        labels = "\n".join(labels_lines)
        prompt = PROMPT_COMPARACION_VISUAL.format(
            n=len(detecciones),
            numero_flota=numero_flota,
            labels=labels,
        )

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[*pil_imgs, prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        resultado, severidad, descripcion = _parse_comparacion(resp.text or "")
    except ImportError:
        resultado, severidad, descripcion = "error", "desconocida", "google-genai no instalado"
    except Exception as e:
        resultado, severidad, descripcion = "error", "desconocida", str(e)

    det_ids_used = [d["id"] for d, _ in detecciones]
    comp_id = insertar_comparacion_n(
        numero_flota, det_ids_used,
        resultado, descripcion, severidad,
    )
    print(f"[DamageDetector] bus {numero_flota} comp_id={comp_id}: {resultado} ({severidad}) — {descripcion}")

    try:
        from web.app import emit_event
        emit_event({
            "type": "damage_analysis",
            "comp_id": comp_id,
            "numero_flota": numero_flota,
            "detection_ids": det_ids_used,
            "resultado": resultado,
            "severidad": severidad,
            "descripcion": descripcion,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception:
        pass


def _insertar_error(numero_flota: int, detection_ids: list[int], msg: str) -> None:
    comp_id = insertar_comparacion_n(
        numero_flota, detection_ids, "error", msg, "desconocida",
    )
    try:
        from web.app import emit_event
        emit_event({
            "type": "damage_analysis",
            "comp_id": comp_id,
            "numero_flota": numero_flota,
            "detection_ids": detection_ids,
            "resultado": "error",
            "severidad": "desconocida",
            "descripcion": msg,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception:
        pass
