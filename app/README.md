# ElecPrevis · Aplicación (Entregable 2, Parte 5)

Aplicación web en Streamlit que implementa el modelo matemático del proyecto: muestra el
histórico de consumo, predice el siguiente intervalo de 30 minutos con un **ensamble de los cuatro
modelos** (y con cada modelo por separado), compara su desempeño y simula escenarios hipotéticos de
eficiencia sobre las submediciones.

## Cómo correrla

```bash
# desde la raíz del repo
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

La app lee todo de `app/artifacts/`, así que no necesita descargar el dataset ni tener
TensorFlow instalado.

## Cómo regenerar los modelos

```bash
pip install -r requirements.txt
python src/download_data.py
python src/preprocessing.py
python app/train_export.py        # unos 30 min en un portátil; la LSTM y el walk-forward son lo más lento
```

`train_export.py` entrena con las funciones de `src/models.py` (misma semilla y misma partición
80/20) y al final llama a `export_ensamble.py`. Exporta:

| Archivo | Contenido |
|---|---|
| `power_30min.parquet` | Serie remuestreada a 30 min (salida de `preprocessing.py`) |
| `lineal.json` | Coeficientes β, intercepto y escalador de la regresión lineal |
| `xgboost.json` | Modelo XGBoost |
| `lstm.npz` | Pesos de la LSTM y su escalador |
| `random_forest.joblib` | Random forest liviano (150 árboles, ≥ 10 observaciones por hoja, ≈15 MB) |
| `predicciones_prueba.parquet` | Predicción a un paso de los 4 modelos y del ensamble en todo el periodo de prueba |
| `resultados.json` | Métricas de entrenamiento y prueba, ensamble, walk-forward, importancia de variables e hiperparámetros |

Para rehacer solo el ensamble y el random forest, sin TensorFlow ni el dataset crudo (parte de lo
que ya está en `app/artifacts/`):

```bash
python app/export_ensamble.py            # unos 10 min, sobre todo por el walk-forward
python app/export_ensamble.py --sin-rf   # reutiliza el random_forest.joblib existente
```

**Random forest liviano:** el de 200 árboles sin límite de profundidad pesaba ≈160 MB y GitHub no
acepta archivos de más de 100 MB, así que la app publicada no lo tenía. Con 150 árboles y un mínimo
de 10 observaciones por hoja (`core.RF_PARAMS`) pesa ≈15 MB y en prueba rinde igual o mejor
(R² 0,68 frente a 0,67): el mínimo por hoja evita que cada árbol memorice intervalos sueltos.

**LSTM sin TensorFlow:** la app evalúa la red con NumPy (`core.LSTMNumpy`), escribiendo una a una
las ecuaciones de las compuertas de la Sección 6.4 del modelo matemático. `train_export.py`
verifica que la salida coincida con la de Keras (diferencia máxima ≈ 1e-6 kW).

## Ensamble de los cuatro modelos

La predicción principal de la app es el promedio de los cuatro modelos:

    ŷ_ens = (ŷ_lineal + ŷ_RF + ŷ_XGB + ŷ_LSTM) / 4

Es el mismo principio del random forest, que en regresión predice el promedio de sus árboles. Aquí
los "votantes" son cuatro modelos de naturaleza distinta (lineal, dos ensambles de árboles y una red
recurrente), que se equivocan en momentos distintos; al promediar, parte del error se compensa. La
app muestra la correlación entre los errores de los modelos como evidencia de esa diversidad.

`export_ensamble.py` compara además dos formas de combinar con parámetros, el **promedio ponderado**
(pesos ≥ 0 que suman 1) y el **stacking** (regresión lineal sobre las cuatro predicciones). Sus
parámetros se estiman en la primera mitad del periodo de prueba y todas se comparan en la segunda
mitad. Se usa el promedio porque rinde prácticamente igual, no tiene parámetros que estimar y es el
principio del random forest.

## Funcionalidades (11)

| # | Sección | Funcionalidad |
|---|---|---|
| F1 | Histórico | Serie histórica interactiva (30 min, hora, día, semana) con indicadores de energía, potencia media, pico y costo en pesos |
| F2 | Histórico | Patrones: mapa de calor hora × día de la semana y perfil mensual |
| F3 | Predicción | Predicción del siguiente intervalo (30 min) con el ensamble y los 4 modelos frente al valor real |
| F4 | Comparación | Tabla de métricas del ensamble y los 4 modelos (MAE, MSE, RMSE, R², sMAPE y precisión por tolerancia) |
| F5 | Comparación | Real vs. predicho por semana, error por hora del día y dispersión |
| F6 | Comparación | Estabilidad por cortes de la validación de ventana expansiva |
| F7 | Comparación | Registro de desempeño: métricas de entrenamiento frente a prueba (sobreajuste) |
| F8 | Comparación | Ensamble: tres formas de combinar (promedio, ponderado, stacking) y correlación de errores |
| F9 | Escenarios | Simulador de escenarios sobre el consumo medido o previsto, con desglose por grupo de carga, ahorro y costo (tarifa configurable) |
| F10 | Modelo | Importancia de variables (RF, XGBoost) y coeficientes (lineal) |
| F11 | Modelo | Defensa del modelo, ecuaciones clave (incluido el ensamble) y trazabilidad ecuación → código |

Todas las secciones permiten descargar en CSV los datos que muestran.

## Precisión

La precisión TP / (TP + FP) es una métrica de clasificación: cuenta aciertos cuando el modelo
responde sí o no. Este proyecto es de regresión (el modelo responde un número en kW), así que la
app reporta la **precisión por tolerancia**:

| Métrica | Fórmula | Qué mide |
|---|---|---|
| Precisión ±1 kW | % de intervalos con \|y − ŷ\| ≤ 1 kW | Qué tan seguido el modelo queda cerca del valor real |

Como referencia, la app reporta también la persistencia (predecir que se consume lo mismo que
ahora). Las métricas principales siguen siendo MAE, MSE, RMSE y R² (modelo matemático §8). La
tolerancia se cambia en `TOLERANCIA_KW`, en `app/core.py`.

## Supuestos de la app

- **Escenarios:** y′ₜ = yₜ − Σₖ rₖ·sₖ,ₜ, con sₖ la submedición k en kW (Wh/min × 60/1000).
  Son hipotéticos, no ahorros comprobados.

## Publicarla con link (Streamlit Community Cloud)

1. Entra a https://share.streamlit.io con la cuenta de GitHub dueña del repo.
2. **Create app** → *Deploy a public app from GitHub*.
3. Repositorio `NicolasMosquera23/elecprevis`, rama `main` y archivo principal
   `app/streamlit_app.py`.
4. En **Advanced settings**, Python 3.12.
5. **Deploy**. El primer arranque tarda unos minutos mientras instala las dependencias.
