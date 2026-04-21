"""
Servidor web del sistema fonobus.

Endpoints:
  GET  /                        → dashboard HTML
  GET  /api/detecciones         → lista de detecciones (JSON)
  GET  /captures/<filename>     → sirve imágenes capturadas
  WS   /ws                      → WebSocket: eventos en tiempo real

Cómo correr (desde la raíz del proyecto):
    .venv/bin/python -m uvicorn web.app:app --port 8000
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from web.database import (init_db, obtener, obtener_crops, stats_por_hora, stats_por_dia,
                           stats_frecuentes, stats_por_numero, limpiar_capturas_antiguas,
                           get_deteccion_by_id,
                           eliminar_crop, set_main_image, eliminar_deteccion,
                           eliminar_bus)
from config.settings import CAPTURES_RETENTION_DAYS

CAPTURES_DIR = Path(__file__).resolve().parent.parent / "captures"
STATIC_DIR   = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Fonobus API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "PUT"],
    allow_headers=["*"],
)

# ── WebSocket manager ─────────────────────────────────────────────────────────

class _ConnectionManager:
    def __init__(self):
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self._clients:
            self._clients.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self._clients:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self._clients:
                self._clients.remove(ws)


_manager = _ConnectionManager()
_event_queue: asyncio.Queue | None = None
_loop: asyncio.AbstractEventLoop | None = None


def emit_event(data: dict) -> None:
    """
    Thread-safe. Llamar desde cualquier thread de detección.
    Pone el evento en la cola para que el worker async lo transmita.
    """
    if _loop is not None and _event_queue is not None:
        asyncio.run_coroutine_threadsafe(_event_queue.put(data), _loop)


async def _broadcast_worker():
    """Consume la cola y hace broadcast a todos los clientes WebSocket."""
    while True:
        data = await _event_queue.get()
        await _manager.broadcast(data)


async def _cleanup_worker():
    """Borra imágenes antiguas cada 7 días. Los registros DB se mantienen."""
    while True:
        await asyncio.sleep(7 * 24 * 60 * 60)
        archivos = await asyncio.get_event_loop().run_in_executor(
            None, limpiar_capturas_antiguas, CAPTURES_RETENTION_DAYS
        )
        if archivos:
            print(f"[Cleanup] {archivos} imagen(es) eliminada(s) (>{CAPTURES_RETENTION_DAYS} días).")
        else:
            print(f"[Cleanup] Sin imágenes antiguas (retención: {CAPTURES_RETENTION_DAYS} días).")


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _event_queue, _loop
    init_db()
    _event_queue = asyncio.Queue()
    _loop = asyncio.get_event_loop()
    asyncio.create_task(_broadcast_worker())
    asyncio.create_task(_cleanup_worker())


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await _manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # mantiene la conexión viva
    except WebSocketDisconnect:
        _manager.disconnect(ws)


@app.get("/api/detecciones")
def get_detecciones(
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
):
    rows = obtener(limit=limit, offset=offset)
    return JSONResponse(content=rows)


@app.get("/api/stats/por-hora")
def get_stats_hora():
    return JSONResponse(content=stats_por_hora())

@app.get("/api/stats/por-dia")
def get_stats_dia():
    return JSONResponse(content=stats_por_dia())

@app.get("/api/stats/frecuentes")
def get_stats_frecuentes():
    return JSONResponse(content=stats_frecuentes())

@app.get("/api/stats/por-numero")
def get_stats_por_numero():
    return JSONResponse(content=stats_por_numero())

@app.get("/api/detecciones/{detection_id}/crops")
def get_detection_crops(detection_id: int):
    crops = obtener_crops(detection_id)
    return JSONResponse(content=crops)


@app.post("/api/detecciones/{detection_id}/analizar")
def analizar_deteccion(detection_id: int):
    det = get_deteccion_by_id(detection_id)
    if not det:
        raise HTTPException(status_code=404, detail="Detección no encontrada")
    from web.damage_detector import analizar_individual_async
    analizar_individual_async(
        detection_id=detection_id,
        numero_flota=det["numero_flota"],
        direccion=det["direccion"],
        fallback_path=det.get("imagen_path"),
    )
    return JSONResponse(status_code=202, content={"status": "analizando", "detection_id": detection_id})


@app.post("/api/comparaciones/nueva")
def comparar_nueva(payload: dict = Body(...)):
    numero_flota = payload.get("numero_flota")
    detection_ids = payload.get("detection_ids") or []
    if not isinstance(numero_flota, int):
        raise HTTPException(status_code=400, detail="numero_flota requerido (int)")
    if not isinstance(detection_ids, list) or len(detection_ids) < 2:
        raise HTTPException(status_code=400, detail="detection_ids debe ser lista con al menos 2 ids")
    # Validar que todas las detecciones existan y pertenezcan al mismo bus
    for did in detection_ids:
        det = get_deteccion_by_id(did)
        if not det:
            raise HTTPException(status_code=404, detail=f"Detección {did} no encontrada")
        if det["numero_flota"] != numero_flota:
            raise HTTPException(status_code=400, detail=f"Detección {did} no pertenece al bus {numero_flota}")
    from web.damage_detector import comparar_visual_async
    comparar_visual_async(numero_flota, detection_ids)
    return JSONResponse(status_code=202, content={"status": "comparando", "count": len(detection_ids)})


@app.delete("/api/buses/{numero_flota}")
def delete_bus(numero_flota: int):
    count, paths = eliminar_bus(numero_flota)
    if count == 0:
        raise HTTPException(status_code=404, detail="Bus no encontrado")
    import os
    for p in paths:
        if p and os.path.exists(p):
            try: os.unlink(p)
            except OSError: pass
    return JSONResponse(content={"deleted_detections": count, "deleted_files": len(paths)})


@app.post("/api/buses/borrar-varios")
def delete_buses_bulk(payload: dict = Body(...)):
    nums = payload.get("numeros_flota") or []
    if not isinstance(nums, list) or not nums:
        raise HTTPException(status_code=400, detail="numeros_flota debe ser lista no vacía")
    import os
    total_det = 0
    total_files = 0
    borrados = []
    for n in nums:
        if not isinstance(n, int):
            continue
        count, paths = eliminar_bus(n)
        if count == 0:
            continue
        total_det += count
        for p in paths:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                    total_files += 1
                except OSError:
                    pass
        borrados.append(n)
    return JSONResponse(content={
        "deleted_buses": borrados,
        "deleted_detections": total_det,
        "deleted_files": total_files,
    })


@app.delete("/api/detecciones/{detection_id}")
def delete_detection(detection_id: int):
    paths = eliminar_deteccion(detection_id)
    import os
    for p in paths:
        if p and os.path.exists(p):
            os.unlink(p)
    if not paths and not get_deteccion_by_id(detection_id):
        raise HTTPException(status_code=404, detail="Detección no encontrada")
    return JSONResponse(content={"deleted": len(paths)})


@app.delete("/api/detecciones/{detection_id}/crops/{cam_label}")
def delete_crop(detection_id: int, cam_label: str):
    import os
    path = eliminar_crop(detection_id, cam_label)
    if path is None:
        raise HTTPException(status_code=404, detail="Crop no encontrado")
    if os.path.exists(path):
        os.unlink(path)
    return JSONResponse(content={"deleted": path})


@app.put("/api/detecciones/{detection_id}/main-image/{cam_label}")
def update_main_image(detection_id: int, cam_label: str):
    ok = set_main_image(detection_id, cam_label)
    if not ok:
        raise HTTPException(status_code=404, detail="Crop no encontrado")
    return JSONResponse(content={"status": "ok"})


@app.get("/captures/{filename}")
def get_capture(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse(status_code=400, content={"error": "nombre de archivo inválido"})
    path = CAPTURES_DIR / filename
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "imagen no encontrada"})
    return FileResponse(path, media_type="image/jpeg")
