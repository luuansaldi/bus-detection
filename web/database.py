"""
SQLite para persistir detecciones confirmadas de buses.

Cada vez que el sistema detecta un bus con consenso, se guarda una fila con:
  - timestamp exacto
  - número de flota
  - dirección (entering / exiting / unknown)
  - ruta a la imagen capturada
"""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "detecciones.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # devuelve dicts en vez de tuplas
    return conn


def init_db() -> None:
    """Crea la tabla si no existe. Seguro de llamar múltiples veces."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detecciones (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT    NOT NULL,
                numero_flota INTEGER NOT NULL,
                direccion    TEXT    NOT NULL,
                imagen_path  TEXT
            )
        """)


def insertar(numero_flota: int, direccion: str, imagen_path: str | None = None) -> None:
    """Inserta una detección confirmada."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO detecciones (timestamp, numero_flota, direccion, imagen_path) VALUES (?, ?, ?, ?)",
            (ts, numero_flota, direccion, imagen_path),
        )


def stats_por_hora() -> list[dict]:
    """Cantidad de detecciones agrupadas por hora del día (0-23)."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT CAST(strftime('%H', timestamp) AS INTEGER) AS hora,
                   COUNT(*) AS total,
                   SUM(CASE WHEN direccion='entering' THEN 1 ELSE 0 END) AS entraron,
                   SUM(CASE WHEN direccion='exiting'  THEN 1 ELSE 0 END) AS salieron
            FROM detecciones
            GROUP BY hora
            ORDER BY hora
        """).fetchall()
    return [dict(r) for r in rows]


def stats_por_dia() -> list[dict]:
    """Cantidad de detecciones agrupadas por día (últimos 30 días)."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT DATE(timestamp) AS dia,
                   COUNT(*) AS total,
                   SUM(CASE WHEN direccion='entering' THEN 1 ELSE 0 END) AS entraron,
                   SUM(CASE WHEN direccion='exiting'  THEN 1 ELSE 0 END) AS salieron
            FROM detecciones
            WHERE timestamp >= DATE('now', '-30 days')
            GROUP BY dia
            ORDER BY dia DESC
        """).fetchall()
    return [dict(r) for r in rows]


def stats_frecuentes(limit: int = 10) -> list[dict]:
    """Los N números de flota más detectados."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT numero_flota,
                   COUNT(*) AS total,
                   SUM(CASE WHEN direccion='entering' THEN 1 ELSE 0 END) AS entraron,
                   SUM(CASE WHEN direccion='exiting'  THEN 1 ELSE 0 END) AS salieron,
                   MAX(timestamp) AS ultima_vez
            FROM detecciones
            GROUP BY numero_flota
            ORDER BY total DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def obtener(limit: int = 100, offset: int = 0) -> list[dict]:
    """Devuelve las últimas detecciones, más recientes primero."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM detecciones ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]
