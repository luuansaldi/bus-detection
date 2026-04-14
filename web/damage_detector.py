"""
Comparación de daños entre la captura de salida y la de retorno de un bus.

Usa Gemini 1.5 Flash (gratis) para determinar si el bus volvió con daños nuevos.
Se invoca en background (thread daemon) desde _report() en rtsp_multicam.py.

Requiere: GOOGLE_API_KEY en el entorno.
Obtené una gratis en: https://aistudio.google.com/apikey
"""

import sys
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web.database import insertar_comparacion

PROMPT = (
    "Estas son dos fotos del bus de flota #{numero_flota}. "
    "La PRIMERA imagen es cuando SALIÓ del depósito. "
    "La SEGUNDA imagen es cuando REGRESÓ. "
    "Comparalas y determiná si el bus tiene daños nuevos visibles "
    "(rayones, abolladuras, espejos rotos, vidrios rotos, etc.). "
    "Respondé ÚNICAMENTE con una de estas dos formas:\n"
    "OK: <descripción breve en español>\n"
    "DAÑOS: <descripción breve del daño en español>"
)


def comparar_viaje_async(numero_flota: int, salida_path: str, entrada_path: str,
                          salida_id: int, entrada_id: int) -> None:
    """Lanza la comparación en un thread daemon y retorna inmediatamente."""
    t = threading.Thread(
        target=_comparar,
        args=(numero_flota, salida_path, entrada_path, salida_id, entrada_id),
        daemon=True,
        name=f"damage-{numero_flota}",
    )
    t.start()


def _comparar(numero_flota: int, salida_path: str, entrada_path: str,
               salida_id: int, entrada_id: int) -> None:
    print(f"[DamageDetector] Bus {numero_flota}: comparando "
          f"salida={Path(salida_path).name} vs entrada={Path(entrada_path).name}")
    try:
        from google import genai
        from google.genai import types
        import PIL.Image

        client = genai.Client()  # lee GOOGLE_API_KEY del entorno automáticamente

        img_salida  = PIL.Image.open(salida_path)
        img_entrada = PIL.Image.open(entrada_path)

        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[img_salida, img_entrada, PROMPT.format(numero_flota=numero_flota)],
            config=types.GenerateContentConfig(
                max_output_tokens=300,
                temperature=0.1,
            ),
        )
        text = resp.text.strip()

        if text.upper().startswith("OK"):
            resultado = "ok"
        elif text.upper().startswith("DA"):
            resultado = "danos"
        else:
            resultado = "ok"  # conservador: respuesta inesperada → asumir OK

        descripcion = text.split(":", 1)[1].strip() if ":" in text else text

    except ImportError:
        resultado, descripcion = "error", "google-genai no instalado"
    except Exception as e:
        resultado, descripcion = "error", str(e)

    insertar_comparacion(numero_flota, salida_id, entrada_id, resultado, descripcion)
    print(f"[DamageDetector] Bus {numero_flota}: {resultado} — {descripcion}")

    try:
        from web.app import emit_event
        emit_event({
            "type": "damage_analysis",
            "numero_flota": numero_flota,
            "resultado": resultado,
            "descripcion": descripcion,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception:
        pass
