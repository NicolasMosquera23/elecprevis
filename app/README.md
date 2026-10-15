# ElecPrevis · Aplicación (Entregable 2, Parte 5)

Aplicación web en Streamlit que implementa el modelo matemático del proyecto: muestra el
histórico de consumo, predice el siguiente intervalo de 30 minutos con los cuatro modelos,
compara su desempeño y simula escenarios hipotéticos de eficiencia sobre las submediciones.

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

`train_export.py` entrena con las funciones de `src/models.py` (mismos hiperparámetros, misma
semilla, misma partición 80/20) y exporta:

| Archivo | Contenido |
|---|---|
| `power_30min.parquet` | Serie remuestreada a 30 min (salida de `preprocessing.py`) |
| `lineal.json` | Coeficientes β, intercepto y escalador de la regresión lineal |
| `xgboost.json` | Modelo XGBoost |
| `lstm.npz` | Pesos de la LSTM y su escalador |
| `random_forest.joblib` | Random forest (≈160 MB, **no se sube a GitHub**, ver abajo) |
| `predicciones_prueba.parquet` | Predicción a un paso de los 4 modelos en todo el periodo de prueba |
| `resultados.json` | Métricas, walk-forward, importancia de variables e hiperparámetros |

**Random forest:** con 200 árboles pesa ≈160 MB comprimido y GitHub no acepta archivos de más
de 100 MB. Por eso no se versiona. Sus predicciones del periodo de prueba sí están en
`predicciones_prueba.parquet`, así que aparece en la predicción del siguiente intervalo y en
la comparación. Solo queda fuera del pronóstico de 24 h y de la calculadora cuando la app corre
sin el archivo. En local, después de correr `train_export.py`, también aparece ahí.

**LSTM sin TensorFlow:** la app evalúa la red con NumPy (`core.LSTMNumpy`), escribiendo una a una
las ecuaciones de las compuertas de la Sección 6.4 del modelo matemático. `train_export.py`
verifica que la salida coincida con la de Keras (diferencia máxima ≈ 1e-6 kW).

## Funcionalidades (15)

| # | Sección | Funcionalidad |
|---|---|---|
| F1 | Histórico | Serie histórica interactiva con rango de fechas y resolución (30 min, hora, día, semana) |
| F2 | Histórico | Indicadores del periodo: energía, potencia media, pico y costo en pesos |
| F3 | Histórico | Patrones: mapa de calor hora × día de la semana y perfil mensual |
| F4 | Predicción | Predicción del siguiente intervalo (30 min) con los 4 modelos frente al valor real |
| F5 | Predicción | Pronóstico encadenado de 24 h (48 pasos) con selector de modelos |
| F6 | Predicción | Calculadora: predicción a partir de condiciones ingresadas a mano |
| F7 | Comparación | Tabla de métricas de los 4 modelos (MAE, MSE, RMSE, R², sMAPE y precisión) |
| F8 | Comparación | Real vs. predicho por semana, error por hora del día y dispersión |
| F9 | Comparación | Estabilidad por cortes de la validación de ventana expansiva |
| F10 | Escenarios | Desglose de la energía por grupo de carga (submediciones) |
| F11 | Escenarios | Simulador de reducción por grupo de carga |
| F12 | Escenarios | Costo y ahorro en pesos con tarifa configurable (Electrohuila estrato 4 por defecto) |
| F13 | Modelo | Importancia de variables (RF, XGBoost) y coeficientes (lineal) |
| F14 | Modelo | Ecuaciones clave y trazabilidad ecuación → código |
| F15 | Todas | Descarga en CSV de datos, predicciones, pronósticos y escenarios |

## Precisión

El modelo matemático define MAE, MSE, RMSE, R² y sMAPE. La rúbrica pide "precisión > 95 %" sin
definirla para regresión, así que la app reporta tres lecturas (ver `core.py`):

| Lectura | Fórmula | Qué mide |
|---|---|---|
| Precisión energética (principal) | 1 − \|Σŷ − Σy\| / Σy | Acierto en la energía total del periodo de prueba |
| Precisión diaria | 100 − sMAPE de la energía diaria | Acierto día a día |
| Precisión por intervalo | 100 − sMAPE por intervalo | Acierto cada 30 min (la más exigente) |

La métrica principal se cambia en una línea: `PRECISION_PRINCIPAL` en `app/core.py`.

## Supuestos de la app

- **Pronóstico de 24 h:** cada paso usa la predicción anterior como dato. Las mediciones futuras
  distintas de la potencia activa se toman del mismo intervalo del día anterior. El error se
  acumula y no se cuantifica en el modelo matemático (§9).
- **Escenarios:** y′ₜ = yₜ − Σₖ rₖ·sₖ,ₜ, con sₖ la submedición k en kW (Wh/min × 60/1000).
  Son hipotéticos, no ahorros comprobados.

## Publicarla con link (Streamlit Community Cloud)

1. Entra a https://share.streamlit.io con la cuenta de GitHub dueña del repo.
2. **Create app** → *Deploy a public app from GitHub*.
3. Repositorio `NicolasMosquera23/elecprevis`, rama `main` y archivo principal
   `app/streamlit_app.py`.
4. En **Advanced settings**, Python 3.12.
5. **Deploy**. El primer arranque tarda unos minutos mientras instala las dependencias.
