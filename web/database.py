"""
SQLite para persistir detecciones confirmadas de buses.

Cada vez que el sistema detecta un bus con consenso, se guarda una fila con:
  - timestamp exacto
  - número de flota
  - dirección (entering / exiting / unknown)
  - ruta a la imagen capturada
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "detecciones.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # devuelve dicts en vez de tuplas
    return conn


def init_db() -> None:
    """Crea las tablas si no existen. Seguro de llamar múltiples veces."""
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_crops (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id  INTEGER NOT NULL,
                cam_label     TEXT    NOT NULL,
                crop_path     TEXT    NOT NULL,
                FOREIGN KEY (detection_id) REFERENCES detecciones(id)
            )
        """)


def insertar(numero_flota: int, direccion: str, imagen_path: str | None = None,
             crop_paths: dict[str, str] | None = None) -> None:
    """Inserta una detección confirmada y opcionalmente los crops por cámara."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO detecciones (timestamp, numero_flota, direccion, imagen_path) VALUES (?, ?, ?, ?)",
            (ts, numero_flota, direccion, imagen_path),
        )
        if crop_paths:
            det_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO detection_crops (detection_id, cam_label, crop_path) VALUES (?, ?, ?)",
                [(det_id, cam, path) for cam, path in crop_paths.items()],
            )


def obtener_crops(detection_id: int) -> list[dict]:
    """Devuelve los crops por cámara de una detección."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT cam_label, crop_path FROM detection_crops WHERE detection_id = ? ORDER BY cam_label",
            (detection_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def actualizar_direccion(numero_flota: int, direccion: str) -> bool:
    """Actualiza la dirección de la detección más reciente de este bus (si era 'unknown')."""
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE detecciones SET direccion = ? "
            "WHERE id = (SELECT id FROM detecciones WHERE numero_flota = ? "
            "AND direccion = 'unknown' ORDER BY id DESC LIMIT 1)",
            (direccion, numero_flota),
        )
        return cur.rowcount > 0


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


def stats_por_numero() -> list[dict]:
    """Detecciones agrupadas por número de flota, con todas sus capturas."""
    with get_conn() as conn:
        buses = conn.execute("""
            SELECT numero_flota,
                   COUNT(*) AS total,
                   SUM(CASE WHEN direccion='entering' THEN 1 ELSE 0 END) AS entradas,
                   SUM(CASE WHEN direccion='exiting'  THEN 1 ELSE 0 END) AS salidas
            FROM detecciones
            GROUP BY numero_flota
            ORDER BY total DESC
        """).fetchall()

        result = []
        for bus in buses:
            capturas = conn.execute("""
                SELECT id, timestamp, direccion, imagen_path
                FROM detecciones
                WHERE numero_flota = ?
                ORDER BY timestamp ASC
            """, (bus["numero_flota"],)).fetchall()
            capturas_list = []
            for c in capturas:
                cap = dict(c)
                crops = conn.execute(
                    "SELECT cam_label, crop_path FROM detection_crops WHERE detection_id = ? ORDER BY cam_label",
                    (c["id"],),
                ).fetchall()
                cap["crops"] = [dict(cr) for cr in crops]
                capturas_list.append(cap)
            result.append({
                "numero_flota": bus["numero_flota"],
                "total":        bus["total"],
                "entradas":     bus["entradas"],
                "salidas":      bus["salidas"],
                "capturas":     capturas_list,
            })
    return result


def limpiar_capturas_antiguas(retention_days: int) -> int:
    """
    Elimina solo los archivos de imagen de captures/ más viejos que retention_days.
    Los registros en DB se mantienen (imagen_path queda en NULL).
    Retorna la cantidad de archivos borrados.
    """
    from pathlib import Path as _Path

    cutoff = datetime.now() - timedelta(days=retention_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    captures_dir = DB_PATH.parent / "captures"
    archivos_borrados = 0

    with get_conn() as conn:
        filas = conn.execute(
            "SELECT id, imagen_path FROM detecciones WHERE timestamp < ? AND imagen_path IS NOT NULL",
            (cutoff_str,),
        ).fetchall()

        ids_limpiados = []
        for fila in filas:
            img = _Path(fila["imagen_path"])
            if not img.is_absolute():
                img = captures_dir / img.name
            if img.exists():
                try:
                    img.unlink()
                    archivos_borrados += 1
                except OSError:
                    pass
            # Borrar crops asociados
            crop_filas = conn.execute(
                "SELECT crop_path FROM detection_crops WHERE detection_id = ?",
                (fila["id"],),
            ).fetchall()
            for cf in crop_filas:
                cp = _Path(cf["crop_path"])
                if not cp.is_absolute():
                    cp = captures_dir / cp.name
                if cp.exists():
                    try:
                        cp.unlink()
                        archivos_borrados += 1
                    except OSError:
                        pass
            ids_limpiados.append(fila["id"])

        if ids_limpiados:
            placeholders = ','.join('?' * len(ids_limpiados))
            conn.execute(
                f"UPDATE detecciones SET imagen_path = NULL WHERE id IN ({placeholders})",
                ids_limpiados,
            )
            conn.execute(
                f"DELETE FROM detection_crops WHERE detection_id IN ({placeholders})",
                ids_limpiados,
            )

    return archivos_borrados


def obtener(limit: int = 100, offset: int = 0) -> list[dict]:
    """Devuelve las últimas detecciones, más recientes primero."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM detecciones ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]
