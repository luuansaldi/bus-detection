"""
Comparación de daños entre la captura de salida y la de retorno de un bus.

Usa Gemini 2.0 Flash (gratis) para determinar si el bus volvió con daños nuevos.
Se invoca en background (thread daemon) desde _report() en rtsp_multicam.py.

Requiere: GOOGLE_API_KEY en el entorno.
Obtené una gratis en: https://aistudio.google.com/apikey
"""

import sys
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web.database import insertar_comparacion, obtener_crops

PROMPT_CAMARA = (
    "Estas son dos fotos del bus {numero_flota} tomadas por {cam_label}. "
    "PRIMERA imagen = cuando SALIÓ del depósito. SEGUNDA = cuando REGRESÓ. "
    "Las condiciones de luz pueden variar — ignorá diferencias de brillo o cielo. "
    "Buscá ÚNICAMENTE daños físicos nuevos: rayones, abolladuras, espejos/vidrios rotos, paragolpes dañados. "
    "Respondé SOLO en este formato (4 líneas):\n"
    "RESULTADO: OK | DAÑOS | INCONCLUYE\n"
    "SEVERIDAD: ninguna | menor | mayor\n"
    "ZONA: frente | lateral izquierdo | lateral derecho | trasero | múltiple | no aplica\n"
    "DESCRIPCION: <una oración en español>\n"
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


def _parse_gemini_response(text: str) -> tuple[str, str, str]:
    """Parsea la respuesta estructurada de Gemini. Retorna (resultado, severidad, descripcion)."""
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

    severidad   = lines.get("SEVERIDAD", "desconocida")
    descripcion = lines.get("DESCRIPCION", text[:200])
    return resultado, severidad, descripcion


def _comparar_par(client, cam_label: str, img_path_1: str, img_path_2: str,
                  numero_flota: int) -> tuple[str, str, str]:
    """
    Hace una llamada a Gemini comparando dos imágenes de la misma cámara.
    Retorna (resultado, severidad, descripcion).
    """
    from google.genai import types
    import PIL.Image

    img1 = PIL.Image.open(img_path_1)
    img2 = PIL.Image.open(img_path_2)
    prompt = PROMPT_CAMARA.format(numero_flota=numero_flota, cam_label=cam_label)

    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[img1, img2, prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=300,
            temperature=0.1,
        ),
    )
    return _parse_gemini_response(resp.text.strip())


def _comparar(numero_flota: int, salida_path: str, entrada_path: str,
               salida_id: int, entrada_id: int) -> None:
    print(f"[DamageDetector] Bus {numero_flota}: comparando "
          f"salida={Path(salida_path).name} vs entrada={Path(entrada_path).name}")
    try:
        from google import genai
        client = genai.Client()

        # Buscar crops por cámara para ambas detecciones
        crops_salida  = {c["cam_label"]: c["crop_path"] for c in obtener_crops(salida_id)}
        crops_entrada = {c["cam_label"]: c["crop_path"] for c in obtener_crops(entrada_id)}
        camaras_comunes = [cam for cam in crops_salida if cam in crops_entrada]

        resultados_por_cam: dict[str, tuple[str, str, str]] = {}

        if camaras_comunes:
            print(f"[DamageDetector] Bus {numero_flota}: comparando {len(camaras_comunes)} cámara(s): {camaras_comunes}")
            for cam in camaras_comunes:
                res, sev, desc = _comparar_par(
                    client, cam, crops_salida[cam], crops_entrada[cam], numero_flota
                )
                resultados_por_cam[cam] = (res, sev, desc)
                print(f"[DamageDetector]   {cam}: {res} / {sev} — {desc}")
        else:
            # Fallback: usar imágenes full-frame principales
            print(f"[DamageDetector] Bus {numero_flota}: sin crops matcheados, usando full-frame")
            res, sev, desc = _comparar_par(
                client, "imagen principal", salida_path, entrada_path, numero_flota
            )
            resultados_por_cam["full"] = (res, sev, desc)

        # Agregar resultados: cualquier "danos" → resultado final danos
        resultado_final = "ok"
        severidad_final = "ninguna"
        descripciones = []

        for cam, (res, sev, desc) in resultados_por_cam.items():
            if res == "danos":
                resultado_final = "danos"
            elif res == "inconcluye" and resultado_final == "ok":
                resultado_final = "inconcluye"
            if sev == "mayor":
                severidad_final = "mayor"
            elif sev == "menor" and severidad_final == "ninguna":
                severidad_final = "menor"
            descripciones.append(f"{cam}: {desc}")

        descripcion_final = " | ".join(descripciones)

    except ImportError:
        resultado_final  = "error"
        severidad_final  = "desconocida"
        descripcion_final = "google-genai no instalado"
    except Exception as e:
        resultado_final  = "error"
        severidad_final  = "desconocida"
        descripcion_final = str(e)

    insertar_comparacion(
        numero_flota, salida_id, entrada_id,
        resultado_final, descripcion_final, severidad_final,
    )
    print(f"[DamageDetector] Bus {numero_flota}: {resultado_final} ({severidad_final}) — {descripcion_final}")

    try:
        from web.app import emit_event
        emit_event({
            "type":          "damage_analysis",
            "numero_flota":  numero_flota,
            "resultado":     resultado_final,
            "severidad":     severidad_final,
            "descripcion":   descripcion_final,
            "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception:
        pass
