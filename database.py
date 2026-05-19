import sqlite3
import json
from datetime import datetime

DB_PATH = "turf.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # permite acceder a columnas por nombre
    return conn


def init_db():
    """Crea todas las tablas si no existen."""
    conn = get_connection()
    c = conn.cursor()

    # Cada carrera que analizás
    c.execute("""
        CREATE TABLE IF NOT EXISTS carreras (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha           TEXT,
            hipodromo       TEXT,
            numero_carrera  TEXT,
            distancia       INTEGER,
            tipo_pista      TEXT,
            condicion_pista TEXT,
            creada_en       TEXT DEFAULT (datetime('now'))
        )
    """)

    # Cada caballo dentro de una carrera
    c.execute("""
        CREATE TABLE IF NOT EXISTS caballos (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            carrera_id      INTEGER REFERENCES carreras(id),
            numero_cuerpo   INTEGER,
            nombre          TEXT,
            -- Historial: guardamos las 5 actuaciones como JSON
            -- Cada actuacion: {vr, posicion, cuerpos, distancia, ritmo}
            historial       TEXT,
            -- Ritmo predominante del caballo
            perfil_ritmo    TEXT,  -- 'front' | 'delantero' | 'mid' | 'closer'
            -- Resultado real en esta carrera (se carga despues)
            posicion_final  INTEGER
        )
    """)

    # Dividendos cargados antes de la largada
    c.execute("""
        CREATE TABLE IF NOT EXISTS dividendos (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            carrera_id      INTEGER REFERENCES carreras(id),
            tipo_apuesta    TEXT,  -- 'ganador' | 'imperfecta' | 'exacta' | 'trifecta' | 'cuatrifecta'
            combinacion     TEXT,  -- JSON con los numeros de cuerpo, ej: [3] o [3,7] o [3,7,1]
            dividendo       REAL
        )
    """)

    # Predicciones generadas por el modelo para cada caballo
    c.execute("""
        CREATE TABLE IF NOT EXISTS predicciones (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            carrera_id      INTEGER REFERENCES carreras(id),
            caballo_id      INTEGER REFERENCES caballos(id),
            probabilidad    REAL,   -- probabilidad de ganar segun el modelo
            score_bt        REAL    -- fuerza latente de Bradley-Terry
        )
    """)

    # Pesos del modelo — se actualizan con cada carrera
    c.execute("""
        CREATE TABLE IF NOT EXISTS modelo_pesos (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre          TEXT UNIQUE,  -- nombre del peso/feature
            valor           REAL,
            actualizado_en  TEXT DEFAULT (datetime('now'))
        )
    """)

    # Log de aprendizaje — para visualizar como mejora el modelo
    c.execute("""
        CREATE TABLE IF NOT EXISTS aprendizaje_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            carrera_id      INTEGER REFERENCES carreras(id),
            log_loss        REAL,   -- error del modelo en esa carrera
            fecha           TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()


# ── CARRERAS ──────────────────────────────────────────────────────────────────

def insertar_carrera(fecha, hipodromo, numero_carrera, distancia, tipo_pista, condicion_pista):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO carreras (fecha, hipodromo, numero_carrera, distancia, tipo_pista, condicion_pista)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (fecha, hipodromo, numero_carrera, distancia, tipo_pista, condicion_pista))
    conn.commit()
    carrera_id = c.lastrowid
    conn.close()
    return carrera_id


def obtener_carrera(carrera_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM carreras WHERE id = ?", (carrera_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def obtener_todas_carreras():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM carreras ORDER BY creada_en DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── CABALLOS ──────────────────────────────────────────────────────────────────

def insertar_caballo(carrera_id, numero_cuerpo, nombre, historial, perfil_ritmo):
    """
    historial: lista de dicts con las ultimas 5 actuaciones
    [
      {"vr": 95.2, "posicion": 1, "cuerpos": 2.5, "distancia": 1200, "ritmo": "front"},
      ...
    ]
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO caballos (carrera_id, numero_cuerpo, nombre, historial, perfil_ritmo)
        VALUES (?, ?, ?, ?, ?)
    """, (carrera_id, numero_cuerpo, nombre, json.dumps(historial), perfil_ritmo))
    conn.commit()
    caballo_id = c.lastrowid
    conn.close()
    return caballo_id


def obtener_caballos_de_carrera(carrera_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM caballos WHERE carrera_id = ?", (carrera_id,))
    rows = c.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["historial"] = json.loads(d["historial"])
        result.append(d)
    return result


def registrar_resultado(caballo_id, posicion_final):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        UPDATE caballos SET posicion_final = ? WHERE id = ?
    """, (posicion_final, caballo_id))
    conn.commit()
    conn.close()


# ── DIVIDENDOS ────────────────────────────────────────────────────────────────

def insertar_dividendo(carrera_id, tipo_apuesta, combinacion, dividendo):
    """
    combinacion: lista de numeros de cuerpo, ej: [3] para ganador, [3,7] para exacta
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO dividendos (carrera_id, tipo_apuesta, combinacion, dividendo)
        VALUES (?, ?, ?, ?)
    """, (carrera_id, tipo_apuesta, json.dumps(combinacion), dividendo))
    conn.commit()
    conn.close()


def obtener_dividendos_de_carrera(carrera_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM dividendos WHERE carrera_id = ?", (carrera_id,))
    rows = c.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["combinacion"] = json.loads(d["combinacion"])
        result.append(d)
    return result


# ── PREDICCIONES ──────────────────────────────────────────────────────────────

def guardar_prediccion(carrera_id, caballo_id, probabilidad, score_bt):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO predicciones (carrera_id, caballo_id, probabilidad, score_bt)
        VALUES (?, ?, ?, ?)
    """, (carrera_id, caballo_id, probabilidad, score_bt))
    conn.commit()
    conn.close()


def obtener_predicciones_de_carrera(carrera_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT p.*, ca.nombre, ca.numero_cuerpo, ca.posicion_final
        FROM predicciones p
        JOIN caballos ca ON ca.id = p.caballo_id
        WHERE p.carrera_id = ?
        ORDER BY p.probabilidad DESC
    """, (carrera_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── PESOS DEL MODELO ──────────────────────────────────────────────────────────

PESOS_INICIALES = {
    "w_vr":            1.5,   # peso del VR ajustado por distancia
    "w_competitividad": 1.0,  # peso del indice posicion+cuerpos
    "w_ritmo_campo":   0.8,   # peso del ajuste por dinamica de campo
    "lr":              0.05,  # learning rate del descenso por gradiente
}

def inicializar_pesos():
    """Inserta los pesos iniciales si no existen."""
    conn = get_connection()
    c = conn.cursor()
    for nombre, valor in PESOS_INICIALES.items():
        c.execute("""
            INSERT OR IGNORE INTO modelo_pesos (nombre, valor) VALUES (?, ?)
        """, (nombre, valor))
    conn.commit()
    conn.close()


def obtener_pesos():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT nombre, valor FROM modelo_pesos")
    rows = c.fetchall()
    conn.close()
    return {r["nombre"]: r["valor"] for r in rows}


def actualizar_peso(nombre, nuevo_valor):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        UPDATE modelo_pesos SET valor = ?, actualizado_en = datetime('now')
        WHERE nombre = ?
    """, (nuevo_valor, nombre))
    conn.commit()
    conn.close()


# ── LOG DE APRENDIZAJE ────────────────────────────────────────────────────────

def registrar_log_loss(carrera_id, log_loss):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO aprendizaje_log (carrera_id, log_loss) VALUES (?, ?)
    """, (carrera_id, log_loss))
    conn.commit()
    conn.close()


def obtener_historial_aprendizaje():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT al.*, ca.hipodromo, ca.numero_carrera
        FROM aprendizaje_log al
        JOIN carreras ca ON ca.id = al.carrera_id
        ORDER BY al.fecha ASC
    """)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    inicializar_pesos()
    print("Base de datos inicializada correctamente.")
    print("Tablas: carreras, caballos, dividendos, predicciones, modelo_pesos, aprendizaje_log")


# ── BORRAR CARRERA ────────────────────────────────────────────────────────────

def borrar_carrera(carrera_id):
    """Borra una carrera y todos sus datos asociados en cascada."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM aprendizaje_log  WHERE carrera_id = ?", (carrera_id,))
    c.execute("DELETE FROM predicciones     WHERE carrera_id = ?", (carrera_id,))
    c.execute("DELETE FROM dividendos       WHERE carrera_id = ?", (carrera_id,))
    c.execute("DELETE FROM caballos         WHERE carrera_id = ?", (carrera_id,))
    c.execute("DELETE FROM carreras         WHERE id = ?",         (carrera_id,))
    conn.commit()
    conn.close()
