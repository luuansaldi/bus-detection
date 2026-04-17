"""
Smoke-test del análisis de daños con Gemini.

Uso:
    .venv/bin/python scripts/test_damage_analysis.py                          # ping API + auto-pick imagen reciente
    .venv/bin/python scripts/test_damage_analysis.py captures/foo.jpg         # 1 imagen
    .venv/bin/python scripts/test_damage_analysis.py captures/a.jpg b.jpg     # multi-cámara

Verifica que GOOGLE_API_KEY funcione y muestra exactamente el reporte que
Gemini devuelve para esa(s) imagen(es), usando el mismo prompt que corre en
producción dentro de `web.damage_detector`.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def ping_api() -> bool:
    """Verifica que GOOGLE_API_KEY esté seteada y responda."""
    print("─" * 60)
    print("1) Verificando GOOGLE_API_KEY…")
    if not os.environ.get("GOOGLE_API_KEY"):
        print("   ✗ GOOGLE_API_KEY no está seteada en el entorno")
        print("   Conseguir una en https://aistudio.google.com/apikey")
        print("   Después: export GOOGLE_API_KEY=tu_key  (en la shell o en .env)")
        return False
    print(f"   ✓ GOOGLE_API_KEY presente (prefijo: {os.environ['GOOGLE_API_KEY'][:8]}…)")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("   ✗ google-genai no instalado: .venv/bin/pip install google-genai")
        return False
    print("   ✓ google-genai importable")

    try:
        client = genai.Client()
        t0 = time.time()
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=["Respondé SOLO con la palabra: OK"],
            config=types.GenerateContentConfig(max_output_tokens=10, temperature=0),
        )
        dt = time.time() - t0
        print(f"   ✓ Gemini respondió en {dt:.2f}s: {resp.text.strip()!r}")
        return True
    except Exception as e:
        print(f"   ✗ Error al llamar a Gemini: {e}")
        return False


def auto_pick_image() -> list[Path]:
    """Devuelve la imagen más reciente de captures/, o vacío si no hay."""
    captures = Path(__file__).resolve().parent.parent / "captures"
    if not captures.exists():
        return []
    imgs = sorted(
        [p for p in captures.glob("*.jpg") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return imgs[:1]


def run_analysis(image_paths: list[Path]) -> None:
    """Corre el prompt de análisis individual contra las imágenes dadas."""
    from google import genai
    from google.genai import types
    import PIL.Image
    from web.damage_detector import PROMPT_INDIVIDUAL, _parse_individual

    print("─" * 60)
    print(f"2) Análisis individual sobre {len(image_paths)} imagen(es):")
    for p in image_paths:
        print(f"   • {p.name}  ({p.stat().st_size / 1024:.0f} KB)")

    cam_labels = ", ".join(p.stem.split("_")[-2] if "_" in p.stem else "principal"
                           for p in image_paths)
    prompt = PROMPT_INDIVIDUAL.format(
        n_imgs=len(image_paths),
        numero_flota=999,  # ficticio, sólo para el prompt
        cam_labels=cam_labels,
        evento="pasó",
    )
    pil_imgs = [PIL.Image.open(p) for p in image_paths]

    client = genai.Client()
    t0 = time.time()
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[*pil_imgs, prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=300,
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    dt = time.time() - t0

    print(f"\n   Respuesta cruda (en {dt:.2f}s):\n")
    print("   " + "\n   ".join((resp.text or "").splitlines()))

    estado, texto = _parse_individual(resp.text or "")
    print(f"\n   Estado parseado: {estado}")
    print("─" * 60)


def main() -> None:
    if not ping_api():
        sys.exit(1)

    args = sys.argv[1:]
    if args:
        image_paths = [Path(a) for a in args]
        for p in image_paths:
            if not p.exists():
                print(f"✗ No existe: {p}")
                sys.exit(1)
    else:
        image_paths = auto_pick_image()
        if not image_paths:
            print("✗ No hay imágenes en captures/. Pasá una ruta como argumento.")
            sys.exit(1)
        print(f"\n(no se pasó imagen, usando la más reciente: {image_paths[0].name})")

    run_analysis(image_paths)


if __name__ == "__main__":
    main()
