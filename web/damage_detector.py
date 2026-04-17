"""
Análisis de daños con Gemini 2.0 Flash.

Dos operaciones independientes:

1. `analizar_individual_async(detection_id, ...)`
   Se dispara apenas se guarda una detección (entrada o salida). Manda todas las
   capturas por cámara a Gemini y pide un reporte del estado VISIBLE del bus
   (OK / DAÑADO / dónde). El texto se guarda en `detecciones.analisis_individual`.
   Este paso permite probar Gemini sin esperar un ida y vuelta completo.

2. `comparar_pair_async(salida_id, entrada_id, ...)`
   Cuando la segunda detección (entrada) tiene su análisis listo y hay una salida
   anterior con análisis, se comparan los DOS TEXTOS (no las imágenes) y se
   guarda el resultado en `comparaciones`.

Requiere: GOOGLE_API_KEY en el entorno.
"""

import sys
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web.database import (
    actualizar_analisis_individual,
    get_ultima_salida_con_analisis,
    get_deteccion_entrada_pendiente,
    get_deteccion_by_id,
    insertar_comparacion,
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

PROMPT_COMPARACION = (
    "Comparar dos reportes del bus interno {numero_flota}.\n\n"
    "REPORTE DE SALIDA (cuando salió del depósito):\n{salida}\n\n"
    "REPORTE DE ENTRADA (cuando volvió):\n{entrada}\n\n"
    "¿Aparecieron daños NUEVOS entre salida y entrada? Un daño es 'nuevo' si está "
    "mencionado en el reporte de ENTRADA pero no en el de SALIDA, o si empeoró. "
    "Ignorá diferencias de redacción o nivel de detalle.\n"
    "Respondé SOLO en este formato (4 líneas):\n"
    "RESULTADO: OK | DAÑOS | INCONCLUYE\n"
    "SEVERIDAD: ninguna | menor | mayor\n"
    "ZONA: frente | lateral izquierdo | lateral derecho | trasero | múltiple | no aplica\n"
    "DESCRIPCION: <una oración en español>\n"
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

    # Si este fue el análisis que completó el par, disparar comparación textual.
    _maybe_trigger_comparison(numero_flota, detection_id, direccion)


# ─────────────────────────────────────────────────────────────────────────────
# Comparación text-vs-text
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_trigger_comparison(numero_flota: int, detection_id: int, direccion: str) -> None:
    """
    Llamado al terminar un análisis individual. Busca el par salida+entrada con
    análisis listos y dispara la comparación de textos si no existe ya.
    """
    if direccion == "entering":
        salida = get_ultima_salida_con_analisis(numero_flota)
        if salida:
            comparar_pair_async(
                numero_flota=numero_flota,
                salida_id=salida["id"],
                entrada_id=detection_id,
            )
    elif direccion == "exiting":
        entrada = get_deteccion_entrada_pendiente(
            numero_flota,
            desde_ts=datetime.now().strftime("%Y-%m-%d 00:00:00"),
        )
        # Una entrada sólo tiene sentido si fue POSTERIOR a esta salida.
        salida_det = get_deteccion_by_id(detection_id)
        if entrada and salida_det and entrada["timestamp"] > salida_det["timestamp"]:
            comparar_pair_async(
                numero_flota=numero_flota,
                salida_id=detection_id,
                entrada_id=entrada["id"],
            )


def comparar_pair_async(numero_flota: int, salida_id: int, entrada_id: int) -> None:
    t = threading.Thread(
        target=_comparar_textos,
        args=(numero_flota, salida_id, entrada_id),
        daemon=True,
        name=f"compare-{numero_flota}",
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


def _comparar_textos(numero_flota: int, salida_id: int, entrada_id: int) -> None:
    salida_det = get_deteccion_by_id(salida_id)
    entrada_det = get_deteccion_by_id(entrada_id)
    if not salida_det or not entrada_det:
        return
    if not salida_det.get("analisis_individual") or not entrada_det.get("analisis_individual"):
        return

    print(f"[DamageDetector] comparando bus {numero_flota}: salida #{salida_id} vs entrada #{entrada_id}")
    try:
        from google import genai
        from google.genai import types

        client = genai.Client()
        prompt = PROMPT_COMPARACION.format(
            numero_flota=numero_flota,
            salida=salida_det["analisis_individual"],
            entrada=entrada_det["analisis_individual"],
        )
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=300,
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        resultado, severidad, descripcion = _parse_comparacion(resp.text or "")
    except ImportError:
        resultado, severidad, descripcion = "error", "desconocida", "google-genai no instalado"
    except Exception as e:
        resultado, severidad, descripcion = "error", "desconocida", str(e)

    insertar_comparacion(
        numero_flota, salida_id, entrada_id,
        resultado, descripcion, severidad,
    )
    print(f"[DamageDetector] bus {numero_flota}: {resultado} ({severidad}) — {descripcion}")

    try:
        from web.app import emit_event
        emit_event({
            "type": "damage_analysis",
            "numero_flota": numero_flota,
            "resultado": resultado,
            "severidad": severidad,
            "descripcion": descripcion,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception:
        pass
