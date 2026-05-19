# Turf Predictor 🏇

Modelo de predicción de carreras de turf argentino basado en **Bradley-Terry con aprendizaje incremental**.

## Instalación

```bash
pip install numpy scipy scikit-learn streamlit pandas
```

## Arrancar la app

Desde la carpeta del proyecto:

```bash
streamlit run app.py
```

Se abre automáticamente en el navegador en `http://localhost:8501`

---

## Flujo de uso

### 1. Nueva carrera
- Cargás los datos de la carrera (hipódromo, distancia, pista, condición)
- Agregás cada caballo con sus últimas 5 actuaciones:
  - **VR** de cada carrera
  - **Posición** en la que llegó
  - **Cuerpos** ganados o perdidos
  - **Distancia** de esa carrera
- Definís el **perfil de ritmo** del caballo: `front / delantero / mid / closer`

### 2. Predicción y apuestas
- El modelo genera el **orden de mérito** con probabilidades
- Cargás los **dividendos** del hipódromo antes de la largada
- El sistema calcula el **valor esperado** de cada apuesta:
  - `VE > 1.0` → hay valor, la apuesta es favorable a largo plazo
  - `VE < 1.0` → no hay valor

### 3. Cargar resultado
- Una vez terminada la carrera, ingresás las posiciones finales
- El modelo **actualiza sus pesos** automáticamente (descenso por gradiente)
- Podés ver el log-loss: qué tan bien o mal predijo

### 4. Aprendizaje del modelo
- Gráfico de log-loss carrera a carrera
- A medida que cargás más carreras, el modelo se calibra solo

---

## Arquitectura del modelo

### Bradley-Terry
Cada caballo tiene una **fuerza latente**:

```
Fuerza = exp( w_vr × VR_normalizado + w_comp × Competitividad ) × Ajuste_campo
```

La probabilidad de ganar de cada caballo es su fuerza relativa al campo:

```
P(caballo_i gana) = Fuerza_i / Σ(todas las fuerzas)
```

### Features
- **VR ajustado por distancia**: normalizado con z-score dentro del campo
- **Índice de competitividad**: combina posición + cuerpos con pesos recenciales
- **Dinámica de campo**: ajuste multiplicador según saturación de ritmos

### Aprendizaje incremental
El modelo minimiza el **log-loss** usando descenso por gradiente:

```
dL/dw = feature_ganador - Σ(prob_i × feature_i)
```

---

## Archivos

```
turf_predictor/
├── app.py          # Interfaz Streamlit
├── model.py        # Bradley-Terry + gradiente
├── database.py     # SQLite — toda la persistencia
├── features.py     # Construcción de features
└── turf.db         # Base de datos (se crea al arrancar)
```
