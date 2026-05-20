import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor

# En local usa SQLite via DATABASE_URL no definida
# En Render usa la variable de entorno DATABASE_URL que provee PostgreSQL
DATABASE_URL = os.environ.get("DATABASE_URL")

# psycopg2 necesita que la URL empiece con postgresql:// no postgres://
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


def get_connection():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS carreras (
            id              SERIAL PRIMARY KEY,
            fecha           TEXT,
            hipodromo       TEXT,
            numero_carrera  TEXT,
            distancia       INTEGER,
            tipo_pista      TEXT,
            condicion_pista TEXT,
            creada_en       TIMESTAMP DEFAULT NOW()
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS caballos (
            id              SERIAL PRIMARY KEY,
            carrera_id      INTEGER REFERENCES carreras(id),
            numero_cuerpo   INTEGER,
            nombre          TEXT,
            historial       TEXT,
            perfil_ritmo    TEXT,
            posicion_final  INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS dividendos (
            id              SERIAL PRIMARY KEY,
            carrera_id      INTEGER REFERENCES carreras(id),
            tipo_apuesta    TEXT,
            combinacion     TEXT,
            dividendo       REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS predicciones (
            id              SERIAL PRIMARY KEY,
            carrera_id      INTEGER REFERENCES carreras(id),
            caballo_id      INTEGER REFERENCES caballos(id),
            probabilidad    REAL,
            score_bt        REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS modelo_pesos (
            id              SERIAL PRIMARY KEY,
            nombre          TEXT UNIQUE,
            valor           REAL,
            actualizado_en  TIMESTAMP DEFAULT NOW()
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS aprendizaje_log (
            id              SERIAL PRIMARY KEY,
            carrera_id      INTEGER REFERENCES carreras(id),
            log_loss        REAL,
            fecha           TIMESTAMP DEFAULT NOW()
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
        VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
    """, (fecha, hipodromo, numero_carrera, distancia, tipo_pista, condicion_pista))
    carrera_id = c.fetchone()["id"]
    conn.commit()
    conn.close()
    return carrera_id


def obtener_carrera(carrera_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM carreras WHERE id = %s", (carrera_id,))
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
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO caballos (carrera_id, numero_cuerpo, nombre, historial, perfil_ritmo)
        VALUES (%s, %s, %s, %s, %s) RETURNING id
    """, (carrera_id, numero_cuerpo, nombre, json.dumps(historial), perfil_ritmo))
    caballo_id = c.fetchone()["id"]
    conn.commit()
    conn.close()
    return caballo_id


def obtener_caballos_de_carrera(carrera_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM caballos WHERE carrera_id = %s", (carrera_id,))
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
    c.execute("UPDATE caballos SET posicion_final = %s WHERE id = %s", (posicion_final, caballo_id))
    conn.commit()
    conn.close()


# ── DIVIDENDOS ────────────────────────────────────────────────────────────────

def insertar_dividendo(carrera_id, tipo_apuesta, combinacion, dividendo):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO dividendos (carrera_id, tipo_apuesta, combinacion, dividendo)
        VALUES (%s, %s, %s, %s)
    """, (carrera_id, tipo_apuesta, json.dumps(combinacion), dividendo))
    conn.commit()
    conn.close()


def obtener_dividendos_de_carrera(carrera_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM dividendos WHERE carrera_id = %s", (carrera_id,))
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
        VALUES (%s, %s, %s, %s)
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
        WHERE p.carrera_id = %s
        ORDER BY p.probabilidad DESC
    """, (carrera_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── PESOS DEL MODELO ──────────────────────────────────────────────────────────

PESOS_INICIALES = {
    "w_vr":             1.5,
    "w_competitividad": 1.0,
    "w_ritmo_campo":    0.8,
    "lr":               0.05,
}

def inicializar_pesos():
    conn = get_connection()
    c = conn.cursor()
    for nombre, valor in PESOS_INICIALES.items():
        c.execute("""
            INSERT INTO modelo_pesos (nombre, valor)
            VALUES (%s, %s)
            ON CONFLICT (nombre) DO NOTHING
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
        UPDATE modelo_pesos SET valor = %s, actualizado_en = NOW() WHERE nombre = %s
    """, (nuevo_valor, nombre))
    conn.commit()
    conn.close()


# ── LOG DE APRENDIZAJE ────────────────────────────────────────────────────────

def registrar_log_loss(carrera_id, log_loss):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO aprendizaje_log (carrera_id, log_loss) VALUES (%s, %s)", (carrera_id, log_loss))
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


# ── BORRAR CARRERA ────────────────────────────────────────────────────────────

def borrar_carrera(carrera_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM aprendizaje_log WHERE carrera_id = %s", (carrera_id,))
    c.execute("DELETE FROM predicciones    WHERE carrera_id = %s", (carrera_id,))
    c.execute("DELETE FROM dividendos      WHERE carrera_id = %s", (carrera_id,))
    c.execute("DELETE FROM caballos        WHERE carrera_id = %s", (carrera_id,))
    c.execute("DELETE FROM carreras        WHERE id = %s",         (carrera_id,))
    conn.commit()
    conn.close()
