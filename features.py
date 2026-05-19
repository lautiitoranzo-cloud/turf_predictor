import numpy as np

# ── VR AJUSTADO POR DISTANCIA ─────────────────────────────────────────────────

# Factor de ajuste por diferencia de distancia.
# Un caballo que corrió en 1000m y hoy corre en 1600m tiene un VR
# que no es directamente comparable. Este factor lo penaliza/bonifica.
# Basado en la curva energética de los equinos — a mayor diferencia, mayor ajuste.

FACTOR_DISTANCIA = 0.00015  # calibrable con el tiempo


def ajustar_vr_por_distancia(vr, distancia_carrera_pasada, distancia_hoy):
    """
    Ajusta el VR historico segun la diferencia de distancia con la carrera actual.
    Si corrió en la misma distancia, el VR no se toca.
    Si la diferencia es grande, se aplica una penalizacion suave.
    """
    diferencia = abs(distancia_carrera_pasada - distancia_hoy)
    factor = 1 - (FACTOR_DISTANCIA * diferencia)
    factor = max(factor, 0.7)  # nunca ajustamos mas del 30%
    return vr * factor


def calcular_vr_promedio_ajustado(historial, distancia_hoy):
    """
    Toma las ultimas 5 actuaciones y calcula el VR promedio ajustado.
    Las ultimas 3 tienen mas peso que las primeras 2, como en tu analisis.

    historial: lista de dicts [{vr, posicion, cuerpos, distancia, ritmo}, ...]
               ordenada de mas reciente a mas antigua
    """
    pesos = [0.28, 0.24, 0.20, 0.16, 0.12]  # suma 1.0, mas peso a las recientes

    vr_ajustados = []
    for actuacion in historial[:5]:
        if actuacion.get("vr") is None:
            continue
        vr_aj = ajustar_vr_por_distancia(
            actuacion["vr"],
            actuacion["distancia"],
            distancia_hoy
        )
        vr_ajustados.append(vr_aj)

    if not vr_ajustados:
        return None

    # Usamos solo los pesos correspondientes a los VRs que tenemos
    pesos_usados = pesos[:len(vr_ajustados)]
    suma_pesos = sum(pesos_usados)
    pesos_norm = [p / suma_pesos for p in pesos_usados]

    return sum(vr * p for vr, p in zip(vr_ajustados, pesos_norm))


def normalizar_vr_en_campo(vrs_promedio):
    """
    Normaliza los VRs del campo completo para que sean comparables entre si.
    Resta la media y divide por el desvio estandar (z-score).
    Resultado: cada caballo tiene un VR relativo al campo, no absoluto.
    """
    valores = [v for v in vrs_promedio if v is not None]
    if len(valores) < 2:
        return vrs_promedio  # no hay suficientes datos para normalizar

    media = np.mean(valores)
    desvio = np.std(valores)

    if desvio == 0:
        return [0.0] * len(vrs_promedio)

    return [
        (v - media) / desvio if v is not None else 0.0
        for v in vrs_promedio
    ]


# ── INDICE DE COMPETITIVIDAD ──────────────────────────────────────────────────

def calcular_indice_competitividad(historial):
    """
    Combina posicion y cuerpos en un indice unico por actuacion.
    
    Logica:
    - La posicion base da un puntaje (1ro = 10, 2do = 7, 3ro = 5, etc.)
    - Los cuerpos ajustan ese puntaje: ganar por mucho sube, perder por mucho baja
    - Las ultimas 3 pesan mas que las primeras 2
    """
    pesos = [0.28, 0.24, 0.20, 0.16, 0.12]

    def puntaje_posicion(pos):
        tabla = {1: 10, 2: 7, 3: 5, 4: 3, 5: 2, 6: 1}
        return tabla.get(pos, 0.5)

    def ajuste_cuerpos(pos, cuerpos):
        if cuerpos is None:
            return 0
        # Si gano: mas cuerpos = mejor
        if pos == 1:
            return min(cuerpos * 0.3, 3.0)   # tope de +3 puntos
        # Si perdio: mas cuerpos = peor
        else:
            return -min(cuerpos * 0.2, 2.0)  # tope de -2 puntos

    indices = []
    for act in historial[:5]:
        pos = act.get("posicion")
        cuerpos = act.get("cuerpos")
        if pos is None:
            continue
        idx = puntaje_posicion(pos) + ajuste_cuerpos(pos, cuerpos)
        indices.append(max(idx, 0))

    if not indices:
        return 0.0

    pesos_usados = pesos[:len(indices)]
    suma_pesos = sum(pesos_usados)
    pesos_norm = [p / suma_pesos for p in pesos_usados]

    return sum(idx * p for idx, p in zip(indices, pesos_norm))


# ── ONE-HOT ENCODING DEL PERFIL DE RITMO ─────────────────────────────────────

RITMOS = ["front", "delantero", "mid", "closer"]

def one_hot_ritmo(perfil_ritmo):
    """
    Convierte el perfil de ritmo en un vector binario.
    Ej: 'closer' -> [0, 0, 0, 1]
         'front'  -> [1, 0, 0, 0]
    """
    return [1 if perfil_ritmo == r else 0 for r in RITMOS]


# ── DINAMICA DE CAMPO ─────────────────────────────────────────────────────────

def calcular_ajuste_dinamica(perfil_ritmo, perfiles_campo):
    """
    Calcula el ajuste de probabilidad segun la dinamica del campo completo.
    
    Logica basada en tu analisis:
    - Pocos fronts: los fronts se benefician, los closers se perjudican
    - Muchos fronts: los fronts se perjudican, los closers se benefician
    - Situacion balanceada: ajuste neutral
    
    Retorna un multiplicador (>1 beneficia, <1 perjudica)
    """
    n_total = len(perfiles_campo)
    if n_total == 0:
        return 1.0

    conteo = {r: perfiles_campo.count(r) for r in RITMOS}
    n_fronts = conteo["front"] + conteo["delantero"]
    ratio_fronts = n_fronts / n_total

    # Umbrales calibrables
    POCOS_FRONTS = 0.25   # menos del 25% del campo son fronts/delanteros
    MUCHOS_FRONTS = 0.50  # mas del 50% del campo son fronts/delanteros

    if perfil_ritmo in ["front", "delantero"]:
        if ratio_fronts <= POCOS_FRONTS:
            return 1.20  # punta libre, bonus del 20%
        elif ratio_fronts >= MUCHOS_FRONTS:
            return 0.85  # punta saturada, penalizacion del 15%
        else:
            return 1.0

    elif perfil_ritmo in ["mid", "closer"]:
        if ratio_fronts >= MUCHOS_FRONTS:
            return 1.15  # los de adelante se van a fundir, bonus del 15%
        elif ratio_fronts <= POCOS_FRONTS:
            return 0.90  # carrera lenta, los de atras no pueden explotar
        else:
            return 1.0

    return 1.0


# ── CONSTRUCCION DEL FEATURE VECTOR COMPLETO ─────────────────────────────────

def construir_features(caballo, distancia_hoy, perfiles_campo):
    """
    Construye el vector de features completo para un caballo.
    Este vector es lo que entra al modelo de Bradley-Terry.
    
    Retorna un dict con todos los features calculados.
    """
    historial = caballo.get("historial", [])

    # VR promedio ajustado (sin normalizar aun — se normaliza a nivel campo)
    vr_promedio = calcular_vr_promedio_ajustado(historial, distancia_hoy)

    # Indice de competitividad
    competitividad = calcular_indice_competitividad(historial)

    # One-hot del perfil de ritmo
    perfil_ritmo = caballo.get("perfil_ritmo", "mid")
    ritmo_vector = one_hot_ritmo(perfil_ritmo)

    # Ajuste de dinamica de campo
    ajuste_campo = calcular_ajuste_dinamica(perfil_ritmo, perfiles_campo)

    return {
        "vr_promedio": vr_promedio,           # se normalizara a nivel campo
        "competitividad": competitividad,
        "ritmo_vector": ritmo_vector,
        "ajuste_campo": ajuste_campo,
        "perfil_ritmo": perfil_ritmo,
        "nombre": caballo.get("nombre"),
        "numero_cuerpo": caballo.get("numero_cuerpo"),
    }


def construir_features_campo(caballos, distancia_hoy):
    """
    Construye los features de todos los caballos del campo,
    aplicando normalizacion del VR a nivel campo.
    """
    perfiles_campo = [c.get("perfil_ritmo", "mid") for c in caballos]

    # Primera pasada: features individuales
    features_lista = [
        construir_features(c, distancia_hoy, perfiles_campo)
        for c in caballos
    ]

    # Normalizar VR a nivel campo (z-score)
    vrs = [f["vr_promedio"] for f in features_lista]
    vrs_norm = normalizar_vr_en_campo(vrs)

    for f, vr_norm in zip(features_lista, vrs_norm):
        f["vr_normalizado"] = vr_norm

    return features_lista
