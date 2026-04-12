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

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from web.database import init_db, obtener, stats_por_hora, stats_por_dia, stats_frecuentes, stats_por_numero

CAPTURES_DIR = Path(__file__).resolve().parent.parent / "captures"
STATIC_DIR   = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Fonobus API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
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


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _event_queue, _loop
    init_db()
    _event_queue = asyncio.Queue()
    _loop = asyncio.get_event_loop()
    asyncio.create_task(_broadcast_worker())


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


@app.get("/captures/{filename}")
def get_capture(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse(status_code=400, content={"error": "nombre de archivo inválido"})
    path = CAPTURES_DIR / filename
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "imagen no encontrada"})
    return FileResponse(path, media_type="image/jpeg")
