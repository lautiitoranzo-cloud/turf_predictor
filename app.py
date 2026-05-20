import streamlit as st
import pandas as pd
import json
from database import (
    init_db, inicializar_pesos,
    insertar_carrera, obtener_todas_carreras, obtener_carrera,
    insertar_caballo, obtener_caballos_de_carrera, registrar_resultado,
    insertar_dividendo, obtener_dividendos_de_carrera,
    guardar_prediccion, obtener_predicciones_de_carrera,
    obtener_pesos, obtener_historial_aprendizaje,
    borrar_carrera
)
from model import (
    predecir_carrera,
    calcular_valor_esperado_ganador,
    calcular_valor_esperado_combinaciones,
    aprender_de_resultado
)

init_db()
inicializar_pesos()

st.set_page_config(page_title="Turf Predictor", page_icon="🏇", layout="wide")
st.title("🏇 Turf Predictor")
st.caption("Modelo Bradley-Terry con aprendizaje incremental")

DISTANCIAS = [600, 800, 1000, 1100, 1200, 1400, 1600, 1800, 2000, 2200, 2400]
RITMOS = ["front", "delantero", "mid", "closer"]

tabs = st.tabs([
    "📋 Nueva carrera",
    "🔮 Predicción y apuestas",
    "✅ Cargar resultado",
    "📈 Aprendizaje del modelo",
    "📂 Historial",
    "🎯 Rendimiento"
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — NUEVA CARRERA
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Datos de la carrera")

    col1, col2, col3 = st.columns(3)
    with col1:
        hipodromo = st.selectbox("Hipódromo", ["Palermo", "San Isidro", "La Plata", "Rosario", "Mar del Plata", "Otro"])
        numero_carrera = st.text_input("N° de carrera", placeholder="Ej: 5ª")
    with col2:
        fecha = st.date_input("Fecha")
        distancia = st.selectbox("Distancia (m)", DISTANCIAS, index=5)
    with col3:
        tipo_pista = st.selectbox("Tipo de pista", ["Arena", "Césped", "Polvo"])
        condicion = st.selectbox("Condición", ["Seco", "Algo húmedo", "Húmedo", "Muy húmedo", "Fangoso"])

    if st.button("Crear carrera", type="primary"):
        carrera_id = insertar_carrera(str(fecha), hipodromo, numero_carrera, distancia, tipo_pista, condicion)
        st.session_state["carrera_id_activa"] = carrera_id
        st.success(f"Carrera creada — ID #{carrera_id}. Ahora cargá los caballos.")

    # ── PLANILLA EDITABLE ──────────────────────────────────────────────────────
    if "carrera_id_activa" in st.session_state:
        carrera_id = st.session_state["carrera_id_activa"]
        carrera = obtener_carrera(carrera_id)

        if carrera is None:
            st.session_state.pop("carrera_id_activa", None)
            st.rerun()

        st.divider()
        st.subheader(f"Caballos — {carrera['hipodromo']} · {carrera['distancia']}m · {carrera['condicion_pista']}")

        caballos_guardados = obtener_caballos_de_carrera(carrera_id)
        if caballos_guardados:
            st.success(f"✅ {len(caballos_guardados)} caballo(s) ya guardado(s): {', '.join(c['nombre'] for c in caballos_guardados)}")

        # ── REFERENCIA: carreras anteriores ───────────────────────────────────
        todas_carreras = obtener_todas_carreras()
        carreras_ant = [c for c in todas_carreras if c["id"] != carrera_id]
        if carreras_ant:
            with st.expander("📖 Consultar caballos de carreras anteriores"):
                opciones_ref = {f"#{c['id']} — {c['hipodromo']} {c['distancia']}m {c['fecha']}": c["id"] for c in carreras_ant}
                sel_ref = st.selectbox("Carrera de referencia", list(opciones_ref.keys()), key="ref_sel")
                caballos_ref = obtener_caballos_de_carrera(opciones_ref[sel_ref])
                if caballos_ref:
                    filas_ref = []
                    for c in caballos_ref:
                        fila = {"#": c["numero_cuerpo"], "Nombre": c["nombre"], "Ritmo": c["perfil_ritmo"]}
                        for j, act in enumerate(c["historial"][:5], 1):
                            fila[f"VR{j}"] = act.get("vr", 0)
                            fila[f"P{j}"]  = act.get("posicion", 1)
                            fila[f"C{j}"]  = act.get("cuerpos", 0)
                            fila[f"D{j}"]  = act.get("distancia", 1200)
                        for j in range(len(c["historial"])+1, 6):
                            fila[f"VR{j}"] = 0.0
                            fila[f"P{j}"] = 1
                            fila[f"C{j}"] = 0.0
                            fila[f"D{j}"] = 1200
                        filas_ref.append(fila)
                    st.dataframe(pd.DataFrame(filas_ref), use_container_width=True, hide_index=True)

        # ── FORMULARIO UN CABALLO POR VEZ ─────────────────────────────────────
        st.markdown("#### Agregar caballo")
        st.caption("Completá los datos y apretá **Agregar** — se guarda directo en la base de datos. Repetí para cada caballo.")

        with st.form(key=f"form_caballo_{carrera_id}_{len(caballos_guardados)}", clear_on_submit=True):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                f_num = st.number_input("N° cuerpo", min_value=1, max_value=20, value=len(caballos_guardados)+1, step=1)
            with fc2:
                f_nombre = st.text_input("Nombre *", placeholder="Nombre del caballo")
            with fc3:
                f_ritmo = st.selectbox("Ritmo", RITMOS)

            st.markdown("**Últimas actuaciones** — de más reciente (1) a más antigua (5). VR = 0 si no hay dato.")

            acts = []
            for j in range(1, 6):
                ac1, ac2, ac3, ac4 = st.columns(4)
                with ac1:
                    vr = st.number_input(f"VR{j}", min_value=0.0, max_value=200.0, value=0.0, step=0.1, format="%.1f", key=f"f_vr{j}")
                with ac2:
                    pos = st.number_input(f"P{j}", min_value=1, max_value=20, value=1, step=1, key=f"f_pos{j}")
                with ac3:
                    cuerpos = st.number_input(f"C{j}", min_value=0.0, max_value=50.0, value=0.0, step=0.5, format="%.1f", key=f"f_c{j}")
                with ac4:
                    dist_idx = DISTANCIAS.index(carrera["distancia"]) if carrera["distancia"] in DISTANCIAS else 5
                    dist = st.selectbox(f"D{j}", DISTANCIAS, index=dist_idx, key=f"f_d{j}")
                acts.append((vr, pos, cuerpos, dist))

            agregar = st.form_submit_button("➕ Agregar caballo", type="primary")

        if agregar:
            if not f_nombre.strip():
                st.error("Ingresá el nombre del caballo.")
            else:
                historial = []
                for vr, pos, cuerpos, dist in acts:
                    if float(vr) > 0:
                        historial.append({
                            "vr": float(vr),
                            "posicion": int(pos),
                            "cuerpos": float(cuerpos),
                            "distancia": int(dist),
                        })
                if not historial:
                    st.error("Ingresá al menos 1 VR mayor a 0.")
                else:
                    insertar_caballo(carrera_id, int(f_num), f_nombre.strip(), historial, f_ritmo)
                    st.success(f"✅ {f_nombre.strip()} agregado correctamente.")
                    st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICCION Y APUESTAS
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Predicción y valor esperado de apuestas")

    carreras = obtener_todas_carreras()
    if not carreras:
        st.info("Todavía no hay carreras cargadas.")
    else:
        opciones = {f"#{c['id']} — {c['hipodromo']} {c['distancia']}m {c['fecha']}": c["id"] for c in carreras}
        sel = st.selectbox("Seleccioná la carrera", list(opciones.keys()))
        carrera_id = opciones[sel]
        carrera = obtener_carrera(carrera_id)
        caballos = obtener_caballos_de_carrera(carrera_id)

        if len(caballos) < 2:
            st.warning("Necesitás al menos 2 caballos para predecir.")
        else:
            if st.button("Generar predicción", type="primary"):
                predicciones, features_lista = predecir_carrera(caballos, carrera["distancia"])
                for pred in predicciones:
                    cab = next((c for c in caballos if c["numero_cuerpo"] == pred["numero_cuerpo"]), None)
                    if cab:
                        guardar_prediccion(carrera_id, cab["id"], pred["probabilidad"], pred["fuerza_bt"])
                st.session_state[f"pred_{carrera_id}"] = predicciones
                st.session_state[f"feat_{carrera_id}"] = features_lista

            if f"pred_{carrera_id}" in st.session_state:
                predicciones = st.session_state[f"pred_{carrera_id}"]

                st.subheader("Orden de mérito")
                df_pred = pd.DataFrame([{
                    "Pos. predicha": i + 1,
                    "#": p["numero_cuerpo"],
                    "Caballo": p["nombre"],
                    "Ritmo": p["perfil_ritmo"],
                    "Probabilidad": f"{p['probabilidad']*100:.1f}%",
                    "Fuerza BT": p["fuerza_bt"]
                } for i, p in enumerate(predicciones)])
                st.dataframe(df_pred, use_container_width=True, hide_index=True)

                st.divider()
                st.subheader("Carga de dividendos")
                st.caption("Ingresá los dividendos del hipódromo antes de la largada.")

                TIPOS_APUESTA = ["ganador", "imperfecta", "exacta", "trifecta", "cuatrifecta"]
                COMBINACION_SIZE = {"ganador": 1, "imperfecta": 2, "exacta": 2, "trifecta": 3, "cuatrifecta": 4}

                with st.form("form_dividendos"):
                    tipo_ap = st.selectbox("Tipo de apuesta", TIPOS_APUESTA)
                    n = COMBINACION_SIZE[tipo_ap]
                    nums = []
                    cols_div = st.columns(n + 1)
                    for idx in range(n):
                        with cols_div[idx]:
                            nums.append(st.number_input(f"N° cuerpo {idx+1}", min_value=1, max_value=20, value=idx+1, key=f"div_num_{idx}"))
                    with cols_div[n]:
                        dividendo = st.number_input("Dividendo", min_value=1.0, value=3.0, step=0.1)
                    if st.form_submit_button("Agregar dividendo"):
                        insertar_dividendo(carrera_id, tipo_ap, nums, dividendo)
                        st.success("Dividendo agregado.")
                        st.rerun()

                dividendos = obtener_dividendos_de_carrera(carrera_id)
                if dividendos:
                    st.divider()
                    st.subheader("Valor esperado por apuesta")
                    st.caption("VE > 1.0 → la apuesta tiene valor positivo según el modelo.")

                    ve_ganador = calcular_valor_esperado_ganador(predicciones, dividendos)
                    ve_combos  = calcular_valor_esperado_combinaciones(predicciones, dividendos)
                    todos_ve   = sorted(ve_ganador + ve_combos, key=lambda x: x["valor_esperado"], reverse=True)

                    filas = []
                    for ve in todos_ve:
                        combo_str = " → ".join(str(n) for n in ve["combinacion"]) if "combinacion" in ve else str(ve.get("numero_cuerpo"))
                        filas.append({
                            "Tipo": ve.get("tipo_apuesta", "ganador"),
                            "Combinación": combo_str,
                            "Prob. modelo": f"{ve['probabilidad']*100:.2f}%",
                            "Dividendo": ve["dividendo"],
                            "Valor Esperado": ve["valor_esperado"],
                            "¿Vale?": "✅ SÍ" if ve["tiene_valor"] else "❌ NO"
                        })

                    st.dataframe(
                        pd.DataFrame(filas).style.apply(
                            lambda row: ["background-color: #1a3a1a" if row["¿Vale?"] == "✅ SÍ" else "" for _ in row], axis=1
                        ),
                        use_container_width=True, hide_index=True
                    )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — CARGAR RESULTADO
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Cargar resultado real")
    st.caption("Una vez terminada la carrera, registrá los resultados para que el modelo aprenda.")

    carreras = obtener_todas_carreras()
    if not carreras:
        st.info("No hay carreras cargadas.")
    else:
        opciones = {f"#{c['id']} — {c['hipodromo']} {c['distancia']}m {c['fecha']}": c["id"] for c in carreras}
        sel = st.selectbox("Seleccioná la carrera", list(opciones.keys()), key="sel_resultado")
        carrera_id = opciones[sel]
        caballos = obtener_caballos_de_carrera(carrera_id)

        if caballos:
            st.write("Ingresá la posición final de cada caballo:")
            resultados_input = {}
            cols_res = st.columns(4)
            for idx, cab in enumerate(caballos):
                with cols_res[idx % 4]:
                    pos_actual = cab.get("posicion_final") or 1
                    resultados_input[cab["id"]] = st.number_input(
                        f"#{cab['numero_cuerpo']} {cab['nombre']}",
                        min_value=1, max_value=20,
                        value=int(pos_actual),
                        key=f"res_{cab['id']}"
                    )

            if st.button("Guardar resultados y entrenar modelo", type="primary"):
                for cab_id, pos in resultados_input.items():
                    registrar_resultado(cab_id, pos)

                ganador_id = min(resultados_input, key=resultados_input.get)
                ganador = next((c for c in caballos if c["id"] == ganador_id), None)

                if ganador and f"feat_{carrera_id}" in st.session_state:
                    features_lista = st.session_state[f"feat_{carrera_id}"]
                    predicciones = st.session_state.get(f"pred_{carrera_id}", [])
                    resultado = aprender_de_resultado(carrera_id, features_lista, predicciones, ganador["numero_cuerpo"])
                    st.success(f"Modelo actualizado. Log-loss: **{resultado['log_loss']}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Pesos anteriores:**")
                        st.json({k: round(v, 4) for k, v in resultado["pesos_anteriores"].items() if k != "lr"})
                    with col2:
                        st.write("**Pesos nuevos:**")
                        st.json({k: round(v, 4) for k, v in resultado["pesos_nuevos"].items() if k != "lr"})
                else:
                    st.warning("Resultados guardados. Para entrenar el modelo, primero generá la predicción en la pestaña anterior.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — APRENDIZAJE DEL MODELO
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Evolución del modelo")

    pesos_actuales = obtener_pesos()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("w_vr", round(pesos_actuales.get("w_vr", 0), 4))
    with col2:
        st.metric("w_competitividad", round(pesos_actuales.get("w_competitividad", 0), 4))
    with col3:
        st.metric("w_ritmo_campo", round(pesos_actuales.get("w_ritmo_campo", 0), 4))

    historial = obtener_historial_aprendizaje()
    if historial:
        st.divider()
        st.subheader("Log-loss por carrera")
        st.caption("A medida que el modelo aprende, el log-loss debería tender a bajar.")
        df_log = pd.DataFrame([{
            "Carrera": f"#{h['carrera_id']} {h['hipodromo']}",
            "Log-Loss": h["log_loss"],
        } for h in historial])
        st.line_chart(df_log.set_index("Carrera")["Log-Loss"])
        st.dataframe(df_log, use_container_width=True, hide_index=True)
    else:
        st.info("El modelo todavía no procesó ningún resultado.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — HISTORIAL
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Historial de carreras")

    carreras = obtener_todas_carreras()
    if not carreras:
        st.info("No hay carreras registradas todavía.")
    else:
        for carrera in carreras:
            with st.expander(f"#{carrera['id']} — {carrera['hipodromo']} · {carrera['distancia']}m · {carrera['fecha']}"):
                predicciones = obtener_predicciones_de_carrera(carrera["id"])
                if predicciones:
                    df = pd.DataFrame([{
                        "Caballo": p["nombre"],
                        "#": p["numero_cuerpo"],
                        "Prob. predicha": f"{p['probabilidad']*100:.1f}%",
                        "Pos. real": p["posicion_final"] or "—"
                    } for p in predicciones])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.caption("Sin predicciones registradas.")

                st.divider()
                confirm_key = f"confirm_del_{carrera['id']}"
                if st.session_state.get(confirm_key):
                    st.warning(f"¿Seguro que querés borrar la carrera #{carrera['id']}? Esta acción no se puede deshacer.")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Sí, borrar", key=f"yes_{carrera['id']}", type="primary"):
                            borrar_carrera(carrera["id"])
                            st.session_state.pop(confirm_key, None)
                            st.rerun()
                    with col2:
                        if st.button("Cancelar", key=f"no_{carrera['id']}"):
                            st.session_state.pop(confirm_key, None)
                            st.rerun()
                else:
                    if st.button("🗑️ Borrar carrera", key=f"del_{carrera['id']}"):
                        st.session_state[confirm_key] = True
                        st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — RENDIMIENTO
# ════════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("Rendimiento del modelo")
    st.caption("Solo se cuentan carreras donde el modelo generó una predicción y se cargó el resultado real.")

    carreras = obtener_todas_carreras()

    total = 0
    acierto_1 = 0
    acierto_top2 = 0
    acierto_top3 = 0
    acierto_alguno_exacto = 0
    detalle = []

    for carrera in carreras:
        predicciones = obtener_predicciones_de_carrera(carrera["id"])
        # Solo carreras con prediccion y resultado cargado
        if not predicciones:
            continue
        if not any(p["posicion_final"] for p in predicciones):
            continue

        total += 1

        # Ordenar predicciones por probabilidad
        preds_sorted = sorted(predicciones, key=lambda x: x["probabilidad"], reverse=True)

        # Ganador real = el que tiene posicion_final == 1
        ganador_real = next((p for p in predicciones if p["posicion_final"] == 1), None)
        if not ganador_real:
            continue

        ganador_nombre = ganador_real["nombre"]
        ganador_num = ganador_real["numero_cuerpo"]

        # Posicion que le dio el modelo al ganador real
        pos_predicha = next((i+1 for i, p in enumerate(preds_sorted) if p["numero_cuerpo"] == ganador_num), None)

        acierto_gan = pos_predicha == 1
        acierto_t2  = pos_predicha <= 2 if pos_predicha else False
        acierto_t3  = pos_predicha <= 3 if pos_predicha else False

        # Algun caballo en posicion exacta
        alguno_exacto = any(
            p["posicion_final"] == (i+1)
            for i, p in enumerate(preds_sorted)
            if p["posicion_final"]
        )

        if acierto_gan: acierto_1 += 1
        if acierto_t2:  acierto_top2 += 1
        if acierto_t3:  acierto_top3 += 1
        if alguno_exacto: acierto_alguno_exacto += 1

        detalle.append({
            "Carrera": f"{carrera['hipodromo']} {carrera['distancia']}m",
            "Fecha": carrera["fecha"],
            "Ganador real": ganador_nombre,
            "Pos. predicha": f"{pos_predicha}°" if pos_predicha else "—",
            "Ganador exacto": "✅" if acierto_gan else "❌",
            "Top 2": "✅" if acierto_t2 else "❌",
            "Top 3": "✅" if acierto_t3 else "❌",
            "Alguno exacto": "✅" if alguno_exacto else "❌",
        })

    if total == 0:
        st.info("Todavía no hay carreras con predicción y resultado cargado.")
    else:
        pct = lambda n: f"{n/total*100:.1f}%"

        # Métricas principales
        st.subheader("Resumen global")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Carreras analizadas", total)
        with c2:
            st.metric("Ganador exacto (1°)", f"{acierto_1}/{total}", pct(acierto_1))
        with c3:
            st.metric("Ganador en top 2", f"{acierto_top2}/{total}", pct(acierto_top2))
        with c4:
            st.metric("Ganador en top 3", f"{acierto_top3}/{total}", pct(acierto_top3))

        st.divider()

        c5, c6 = st.columns(2)
        with c5:
            st.metric("Al menos 1 posición exacta", f"{acierto_alguno_exacto}/{total}", pct(acierto_alguno_exacto))
        with c6:
            st.caption("")
            st.caption("Una posición exacta = algún caballo quedó exactamente donde el modelo predijo.")

        st.divider()

        # Grafico de barras simple
        st.subheader("Comparativa visual")
        df_barras = pd.DataFrame({
            "Métrica": ["Ganador exacto", "Ganador en top 2", "Ganador en top 3", "Alguna pos. exacta"],
            "Aciertos (%)": [
                round(acierto_1/total*100, 1),
                round(acierto_top2/total*100, 1),
                round(acierto_top3/total*100, 1),
                round(acierto_alguno_exacto/total*100, 1),
            ]
        })
        st.bar_chart(df_barras.set_index("Métrica"))

        st.divider()

        # Detalle carrera por carrera
        st.subheader("Detalle por carrera")
        st.dataframe(pd.DataFrame(detalle), use_container_width=True, hide_index=True)
