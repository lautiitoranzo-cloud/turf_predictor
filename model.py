import numpy as np
from database import obtener_pesos, actualizar_peso, registrar_log_loss
from features import construir_features_campo

# ── BRADLEY-TERRY ─────────────────────────────────────────────────────────────

def calcular_fuerza_bt(features, pesos):
    """
    Calcula la fuerza latente de Bradley-Terry para un caballo.
    
    Fuerza = exp( w_vr * vr_norm + w_competitividad * competitividad ) * ajuste_campo
    
    Usamos exp() para garantizar que la fuerza siempre sea positiva,
    que es un requisito del modelo Bradley-Terry.
    """
    w_vr = pesos["w_vr"]
    w_comp = pesos["w_competitividad"]
    w_campo = pesos["w_ritmo_campo"]

    score_lineal = (
        w_vr   * features["vr_normalizado"] +
        w_comp * features["competitividad"]
    )

    # El ajuste de campo multiplica la fuerza base
    ajuste = 1 + w_campo * (features["ajuste_campo"] - 1)

    fuerza = np.exp(score_lineal) * max(ajuste, 0.1)
    return fuerza


def calcular_probabilidades_bt(features_lista, pesos):
    """
    Calcula la probabilidad de ganar de cada caballo usando Bradley-Terry.
    
    P(caballo_i gana) = fuerza_i / sum(todas las fuerzas)
    
    Retorna lista de dicts con nombre, numero, fuerza y probabilidad.
    """
    fuerzas = [calcular_fuerza_bt(f, pesos) for f in features_lista]
    suma_fuerzas = sum(fuerzas)

    resultados = []
    for f, fuerza in zip(features_lista, fuerzas):
        prob = fuerza / suma_fuerzas if suma_fuerzas > 0 else 1 / len(fuerzas)
        resultados.append({
            "nombre":       f["nombre"],
            "numero_cuerpo": f["numero_cuerpo"],
            "perfil_ritmo": f["perfil_ritmo"],
            "fuerza_bt":    round(fuerza, 4),
            "probabilidad": round(prob, 4),
        })

    # Ordenar por probabilidad descendente
    resultados.sort(key=lambda x: x["probabilidad"], reverse=True)
    return resultados


# ── LOG-LOSS ──────────────────────────────────────────────────────────────────

def calcular_log_loss(probabilidades, ganador_numero_cuerpo):
    """
    Calcula el log-loss del modelo para esta carrera.
    
    Loss = -log(probabilidad asignada al ganador real)
    
    Cuanto mas cerca de 0, mejor predijo el modelo.
    """
    for pred in probabilidades:
        if pred["numero_cuerpo"] == ganador_numero_cuerpo:
            prob_ganador = max(pred["probabilidad"], 1e-10)  # evitar log(0)
            return -np.log(prob_ganador)

    return -np.log(1e-10)  # si no encontro al ganador, error maximo


# ── DESCENSO POR GRADIENTE ────────────────────────────────────────────────────

def actualizar_pesos(features_lista, ganador_numero_cuerpo, pesos):
    """
    Actualiza los pesos del modelo usando descenso por gradiente.
    
    El gradiente del log-loss respecto a cada peso nos dice en que
    direccion moverlo para reducir el error.
    
    Para Bradley-Terry con log-loss, el gradiente tiene forma cerrada:
    
    dL/dw = feature_ganador - sum(prob_i * feature_i)
    
    Es decir: la diferencia entre el feature del ganador real
    y el promedio ponderado del campo.
    """
    lr = pesos["lr"]

    # Calcular fuerzas y probabilidades actuales
    fuerzas = [calcular_fuerza_bt(f, pesos) for f in features_lista]
    suma = sum(fuerzas)
    probs = [fz / suma for fz in fuerzas]

    # Identificar el ganador
    ganador_features = None
    for f in features_lista:
        if f["numero_cuerpo"] == ganador_numero_cuerpo:
            ganador_features = f
            break

    if ganador_features is None:
        print(f"Advertencia: no se encontro el ganador #{ganador_numero_cuerpo}")
        return pesos

    # Calcular gradiente para w_vr
    # grad = vr_ganador - sum(prob_i * vr_i)
    vr_esperado = sum(p * f["vr_normalizado"] for p, f in zip(probs, features_lista))
    grad_vr = ganador_features["vr_normalizado"] - vr_esperado

    # Calcular gradiente para w_competitividad
    comp_esperado = sum(p * f["competitividad"] for p, f in zip(probs, features_lista))
    grad_comp = ganador_features["competitividad"] - comp_esperado

    # Calcular gradiente para w_ritmo_campo
    ajuste_esperado = sum(p * (f["ajuste_campo"] - 1) for p, f in zip(probs, features_lista))
    grad_campo = (ganador_features["ajuste_campo"] - 1) - ajuste_esperado

    # Actualizar pesos (ascenso del gradiente porque maximizamos log-likelihood)
    nuevos_pesos = {
        "w_vr":            pesos["w_vr"]            + lr * grad_vr,
        "w_competitividad": pesos["w_competitividad"] + lr * grad_comp,
        "w_ritmo_campo":   pesos["w_ritmo_campo"]   + lr * grad_campo,
        "lr":              lr
    }

    # Guardar en la base de datos
    for nombre, valor in nuevos_pesos.items():
        if nombre != "lr":
            actualizar_peso(nombre, valor)

    return nuevos_pesos


# ── VALOR ESPERADO DE APUESTAS ────────────────────────────────────────────────

def calcular_valor_esperado_ganador(predicciones, dividendos):
    """
    Para cada dividendo de ganador, calcula el valor esperado:
    VE = probabilidad_propia * dividendo
    
    Si VE > 1: hay valor, la apuesta es favorable a largo plazo.
    Si VE < 1: no hay valor, el hipódromo tiene ventaja en esa apuesta.
    """
    prob_por_numero = {p["numero_cuerpo"]: p["probabilidad"] for p in predicciones}

    resultados = []
    for div in dividendos:
        if div["tipo_apuesta"] != "ganador":
            continue
        numero = div["combinacion"][0]
        prob = prob_por_numero.get(numero, 0)
        ve = prob * div["dividendo"]
        resultados.append({
            "numero_cuerpo": numero,
            "dividendo":     div["dividendo"],
            "probabilidad":  round(prob, 4),
            "valor_esperado": round(ve, 3),
            "tiene_valor":   ve > 1.0
        })

    resultados.sort(key=lambda x: x["valor_esperado"], reverse=True)
    return resultados


def calcular_probabilidad_combinacion(predicciones, combinacion):
    """
    Estima la probabilidad de una combinacion de posiciones.
    Usa una aproximacion de Harville para exactas y trifectas:
    
    P(A 1ro, B 2do) ≈ P(A gana) * P(B gana | A ya gano)
                    = P(A) * P(B) / (1 - P(A))
    
    Es una aproximacion, no es exacta, pero es la mejor disponible
    sin simulaciones de Monte Carlo.
    """
    prob_por_numero = {p["numero_cuerpo"]: p["probabilidad"] for p in predicciones}

    if len(combinacion) == 1:
        return prob_por_numero.get(combinacion[0], 0)

    elif len(combinacion) == 2:
        pA = prob_por_numero.get(combinacion[0], 0)
        pB = prob_por_numero.get(combinacion[1], 0)
        if pA >= 1:
            return 0
        return pA * (pB / (1 - pA + 1e-10))

    elif len(combinacion) == 3:
        pA = prob_por_numero.get(combinacion[0], 0)
        pB = prob_por_numero.get(combinacion[1], 0)
        pC = prob_por_numero.get(combinacion[2], 0)
        if pA >= 1 or (pA + pB) >= 1:
            return 0
        pAB = pA * (pB / (1 - pA + 1e-10))
        pABC = pAB * (pC / (1 - pA - pB + 1e-10))
        return pABC

    elif len(combinacion) == 4:
        pA = prob_por_numero.get(combinacion[0], 0)
        pB = prob_por_numero.get(combinacion[1], 0)
        pC = prob_por_numero.get(combinacion[2], 0)
        pD = prob_por_numero.get(combinacion[3], 0)
        if pA >= 1 or (pA + pB) >= 1 or (pA + pB + pC) >= 1:
            return 0
        pAB   = pA * (pB / (1 - pA + 1e-10))
        pABC  = pAB * (pC / (1 - pA - pB + 1e-10))
        pABCD = pABC * (pD / (1 - pA - pB - pC + 1e-10))
        return pABCD

    return 0


def calcular_valor_esperado_combinaciones(predicciones, dividendos):
    """
    Calcula el valor esperado para todas las apuestas combinadas:
    imperfecta, exacta, trifecta, cuatrifecta.
    """
    resultados = []

    for div in dividendos:
        tipo = div["tipo_apuesta"]
        combinacion = div["combinacion"]

        if tipo == "ganador":
            continue

        if tipo == "imperfecta":
            # Cualquier orden — suma las dos permutaciones
            pAB = calcular_probabilidad_combinacion(predicciones, combinacion)
            pBA = calcular_probabilidad_combinacion(predicciones, combinacion[::-1])
            prob = pAB + pBA
        else:
            # Exacta, trifecta, cuatrifecta — orden especifico
            prob = calcular_probabilidad_combinacion(predicciones, combinacion)

        ve = prob * div["dividendo"]
        resultados.append({
            "tipo_apuesta":   tipo,
            "combinacion":    combinacion,
            "dividendo":      div["dividendo"],
            "probabilidad":   round(prob, 4),
            "valor_esperado": round(ve, 3),
            "tiene_valor":    ve > 1.0
        })

    resultados.sort(key=lambda x: x["valor_esperado"], reverse=True)
    return resultados


# ── PIPELINE PRINCIPAL ────────────────────────────────────────────────────────

def predecir_carrera(caballos, distancia_hoy):
    """
    Pipeline completo de prediccion para una carrera.
    Retorna las predicciones ordenadas por probabilidad.
    """
    pesos = obtener_pesos()
    features_lista = construir_features_campo(caballos, distancia_hoy)
    predicciones = calcular_probabilidades_bt(features_lista, pesos)
    return predicciones, features_lista


def aprender_de_resultado(carrera_id, features_lista, predicciones, ganador_numero_cuerpo):
    """
    Actualiza el modelo con el resultado real de la carrera.
    Calcula el log-loss, actualiza los pesos y lo registra en el log.
    """
    pesos = obtener_pesos()

    # Calcular error antes del update
    loss = calcular_log_loss(predicciones, ganador_numero_cuerpo)
    registrar_log_loss(carrera_id, loss)

    # Actualizar pesos
    nuevos_pesos = actualizar_pesos(features_lista, ganador_numero_cuerpo, pesos)

    return {
        "log_loss": round(loss, 4),
        "pesos_anteriores": pesos,
        "pesos_nuevos": nuevos_pesos
    }
