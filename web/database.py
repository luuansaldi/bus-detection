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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS comparaciones (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                numero_flota       INTEGER NOT NULL,
                salida_id          INTEGER REFERENCES detecciones(id),
                entrada_id         INTEGER REFERENCES detecciones(id),
                resultado          TEXT,
                severidad          TEXT,
                descripcion        TEXT,
                timestamp_analisis TEXT NOT NULL
            )
        """)
        # Migración: agrega columna severidad si no existe (para DBs anteriores)
        try:
            conn.execute("ALTER TABLE comparaciones ADD COLUMN severidad TEXT DEFAULT 'desconocida'")
        except Exception:
            pass
        # Migración: análisis individual por detección (estado visible del bus).
        # Se guarda apenas se captura; se usa después para comparar sin volver a
        # cargar las imágenes.
        try:
            conn.execute("ALTER TABLE detecciones ADD COLUMN analisis_individual TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE detecciones ADD COLUMN analisis_estado TEXT")  # ok / danado / inconcluye / error / pendiente
        except Exception:
            pass


_DEDUP_WINDOW_SEC = 60  # same bus within this window → merge into one record


def insertar(numero_flota: int, direccion: str, imagen_path: str | None = None,
             crop_paths: dict[str, str] | None = None) -> tuple[int, bool]:
    """Insert or merge a confirmed detection.

    If the same bus (numero_flota) was already recorded within the last
    DEDUP_WINDOW_SEC seconds, the new crops are added to the existing
    record instead of creating a duplicate row.  The main image is
    upgraded if the new one comes from a camera with a better angle.
    Returns (detection_id, is_new).
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cutoff = (datetime.now() - timedelta(seconds=_DEDUP_WINDOW_SEC)).strftime("%Y-%m-%d %H:%M:%S")

    with get_conn() as conn:
        # Look for a recent detection of the same bus
        existing = conn.execute(
            """SELECT id, direccion, imagen_path FROM detecciones
               WHERE numero_flota = ? AND timestamp >= ?
               ORDER BY id DESC LIMIT 1""",
            (numero_flota, cutoff),
        ).fetchone()

        if existing is not None:
            det_id = existing["id"]

            # Upgrade direction if the existing one was unknown
            if existing["direccion"] == "unknown" and direccion != "unknown":
                conn.execute(
                    "UPDATE detecciones SET direccion = ? WHERE id = ?",
                    (direccion, det_id),
                )

            # Update timestamp to latest sighting
            conn.execute(
                "UPDATE detecciones SET timestamp = ? WHERE id = ?",
                (ts, det_id),
            )

            # Add new crops that we don't already have for this detection
            if crop_paths:
                existing_cams = {
                    r["cam_label"]
                    for r in conn.execute(
                        "SELECT cam_label FROM detection_crops WHERE detection_id = ?",
                        (det_id,),
                    ).fetchall()
                }
                new_crops = [
                    (det_id, cam, path)
                    for cam, path in crop_paths.items()
                    if cam not in existing_cams
                ]
                if new_crops:
                    conn.executemany(
                        "INSERT INTO detection_crops (detection_id, cam_label, crop_path) VALUES (?, ?, ?)",
                        new_crops,
                    )

            # Upgrade main image if new one is better (prefer non-cam1 crops)
            if imagen_path and existing["imagen_path"]:
                old_name = Path(existing["imagen_path"]).name
                new_name = Path(imagen_path).name
                # Prefer cam3/cam4 over cam1 (cenital), prefer any cam crop over generic
                cam_priority = {"cam3": 3, "cam4": 3, "cam2": 2, "cam1": 1}
                old_score = max((v for k, v in cam_priority.items() if k in old_name), default=0)
                new_score = max((v for k, v in cam_priority.items() if k in new_name), default=0)
                if new_score > old_score:
                    conn.execute(
                        "UPDATE detecciones SET imagen_path = ? WHERE id = ?",
                        (imagen_path, det_id),
                    )

            return det_id, False

        # No recent duplicate — insert new row
        cur = conn.execute(
            "INSERT INTO detecciones (timestamp, numero_flota, direccion, imagen_path) VALUES (?, ?, ?, ?)",
            (ts, numero_flota, direccion, imagen_path),
        )
        det_id = cur.lastrowid
        if crop_paths:
            conn.executemany(
                "INSERT INTO detection_crops (detection_id, cam_label, crop_path) VALUES (?, ?, ?)",
                [(det_id, cam, path) for cam, path in crop_paths.items()],
            )
        return det_id, True


def obtener_crops(detection_id: int) -> list[dict]:
    """Devuelve los crops por cámara de una detección."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT cam_label, crop_path FROM detection_crops WHERE detection_id = ? ORDER BY cam_label",
            (detection_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def actualizar_analisis_individual(detection_id: int, analisis: str, estado: str) -> None:
    """Guarda el texto del análisis individual de una detección."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE detecciones SET analisis_individual = ?, analisis_estado = ? WHERE id = ?",
            (analisis, estado, detection_id),
        )


def get_ultima_salida_con_analisis(numero_flota: int) -> dict | None:
    """
    Retorna la última detección 'exiting' del bus que YA tiene análisis individual
    y todavía no fue comparada contra una entrada. None si no existe.
    """
    with get_conn() as conn:
        row = conn.execute(
            """SELECT d.id, d.timestamp, d.imagen_path, d.analisis_individual
               FROM detecciones d
               WHERE d.numero_flota = ?
                 AND d.direccion = 'exiting'
                 AND d.analisis_individual IS NOT NULL
                 AND NOT EXISTS (
                     SELECT 1 FROM comparaciones c WHERE c.salida_id = d.id
                 )
               ORDER BY d.timestamp DESC LIMIT 1""",
            (numero_flota,),
        ).fetchone()
    return dict(row) if row else None


def get_deteccion_entrada_pendiente(numero_flota: int, desde_ts: str) -> dict | None:
    """
    Retorna la primera detección 'entering' con análisis individual posterior a
    desde_ts que aún no tiene comparación registrada. None si no existe.
    """
    with get_conn() as conn:
        row = conn.execute(
            """SELECT d.id, d.timestamp, d.imagen_path, d.analisis_individual
               FROM detecciones d
               WHERE d.numero_flota = ?
                 AND d.direccion = 'entering'
                 AND d.timestamp > ?
                 AND d.analisis_individual IS NOT NULL
                 AND NOT EXISTS (
                     SELECT 1 FROM comparaciones c WHERE c.entrada_id = d.id
                 )
               ORDER BY d.timestamp ASC LIMIT 1""",
            (numero_flota, desde_ts),
        ).fetchone()
    return dict(row) if row else None


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


def get_ultima_salida(numero_flota: int) -> dict | None:
    """Retorna la detección exiting más reciente del bus, o None si no existe."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT id, imagen_path FROM detecciones
               WHERE numero_flota = ? AND direccion = 'exiting'
               ORDER BY timestamp DESC LIMIT 1""",
            (numero_flota,),
        ).fetchone()
    return dict(row) if row else None


def insertar_comparacion(numero_flota: int, salida_id: int, entrada_id: int,
                          resultado: str, descripcion: str,
                          severidad: str = "desconocida") -> int:
    """Guarda el resultado de la comparación de daños. Retorna el id."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO comparaciones
               (numero_flota, salida_id, entrada_id, resultado, severidad, descripcion, timestamp_analisis)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (numero_flota, salida_id, entrada_id, resultado, severidad, descripcion, ts),
        )
        return cur.lastrowid


def get_comparacion(comp_id: int) -> dict | None:
    """Retorna una comparación por id."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM comparaciones WHERE id = ?", (comp_id,)
        ).fetchone()
    return dict(row) if row else None


def get_deteccion_by_id(det_id: int) -> dict | None:
    """Retorna una detección por id."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM detecciones WHERE id = ?", (det_id,)
        ).fetchone()
    return dict(row) if row else None


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
            ORDER BY numero_flota ASC
        """).fetchall()

        result = []
        for bus in buses:
            capturas = conn.execute("""
                SELECT id, timestamp, direccion, imagen_path,
                       analisis_individual, analisis_estado
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
            ultimo_analisis = conn.execute("""
                SELECT id, resultado, severidad, descripcion, timestamp_analisis
                FROM comparaciones
                WHERE numero_flota = ?
                ORDER BY id DESC LIMIT 1
            """, (bus["numero_flota"],)).fetchone()
            result.append({
                "numero_flota":    bus["numero_flota"],
                "total":           bus["total"],
                "entradas":        bus["entradas"],
                "salidas":         bus["salidas"],
                "capturas":        capturas_list,
                "ultimo_analisis": dict(ultimo_analisis) if ultimo_analisis else None,
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


def eliminar_crop(detection_id: int, cam_label: str) -> str | None:
    """Delete a single crop from a detection. Returns the deleted file path, or None."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT crop_path FROM detection_crops WHERE detection_id = ? AND cam_label = ?",
            (detection_id, cam_label),
        ).fetchone()
        if not row:
            return None
        path = row["crop_path"]
        conn.execute(
            "DELETE FROM detection_crops WHERE detection_id = ? AND cam_label = ?",
            (detection_id, cam_label),
        )
        # If the main image was this crop, reassign to another crop
        det = conn.execute("SELECT imagen_path FROM detecciones WHERE id = ?", (detection_id,)).fetchone()
        if det and det["imagen_path"] == path:
            other = conn.execute(
                "SELECT crop_path FROM detection_crops WHERE detection_id = ? LIMIT 1",
                (detection_id,),
            ).fetchone()
            conn.execute(
                "UPDATE detecciones SET imagen_path = ? WHERE id = ?",
                (other["crop_path"] if other else None, detection_id),
            )
    return path


def set_main_image(detection_id: int, cam_label: str) -> bool:
    """Set a crop as the main image for a detection."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT crop_path FROM detection_crops WHERE detection_id = ? AND cam_label = ?",
            (detection_id, cam_label),
        ).fetchone()
        if not row:
            return False
        conn.execute(
            "UPDATE detecciones SET imagen_path = ? WHERE id = ?",
            (row["crop_path"], detection_id),
        )
    return True


def eliminar_deteccion(detection_id: int) -> list[str]:
    """Delete a detection and all its crops. Returns list of file paths deleted."""
    paths = []
    with get_conn() as conn:
        det = conn.execute("SELECT imagen_path FROM detecciones WHERE id = ?", (detection_id,)).fetchone()
        if not det:
            return paths
        if det["imagen_path"]:
            paths.append(det["imagen_path"])
        crops = conn.execute(
            "SELECT crop_path FROM detection_crops WHERE detection_id = ?", (detection_id,),
        ).fetchall()
        for c in crops:
            if c["crop_path"] not in paths:
                paths.append(c["crop_path"])
        conn.execute("DELETE FROM detection_crops WHERE detection_id = ?", (detection_id,))
        conn.execute("DELETE FROM comparaciones WHERE salida_id = ? OR entrada_id = ?",
                     (detection_id, detection_id))
        conn.execute("DELETE FROM detecciones WHERE id = ?", (detection_id,))
    return paths
