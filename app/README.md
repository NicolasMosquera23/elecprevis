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
python app/train_export.py        # unos 35 min en un portátil; la LSTM y el walk-forward son lo más lento
python app/comprobaciones.py      # segundos; comprobaciones de robustez (ver más abajo)
```

`train_export.py` entrena con las funciones de `src/models.py` (misma semilla y misma partición
80/20) y al final llama a `validacion.py`, que reentrena los cuatro modelos sin el último año del
entrenamiento y lo predice (≈5 min), y este a `export_ensamble.py`, que toma con esas predicciones
las decisiones del ensamble (ver más abajo). Exportan:

| Archivo | Contenido |
|---|---|
| `power_30min.parquet` | Serie remuestreada a 30 min (salida de `preprocessing.py`) |
| `lineal.json` | Coeficientes β, intercepto y escalador de la regresión lineal |
| `xgboost.json` | Modelo XGBoost |
| `lstm.npz` | Pesos de la LSTM y su escalador |
| `random_forest.joblib` | Random forest liviano (150 árboles, ≥ 10 observaciones por hoja, ≈15 MB) |
| `predicciones_prueba.parquet` | Predicción a un paso de los 4 modelos, del ensamble y de la persistencia en todo el periodo de prueba |
| `predicciones_validacion.parquet` | Predicción del año de validación con modelos entrenados sin ese año (incluye la variante «peso picos») |
| `resultados.json` | Métricas de entrenamiento y prueba, validación, casos críticos, ensamble, walk-forward, importancia de variables e hiperparámetros |
| `comprobaciones.json` | Formas de combinar comparadas en orden temporal y métricas de prueba sin consumos rellenados (`comprobaciones.py`) |

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
(pesos ≥ 0 que suman 1) y el **stacking** (regresión lineal sobre las cuatro predicciones).

## Decisiones en validación, no en la prueba

La forma de combinar, sus pesos y el umbral de la alerta de pico se aprenden de datos. Para que el
periodo de prueba no intervenga en esas decisiones, se aprenden en una **validación**: el último
año del 80 % de entrenamiento (13/02/2009 – 13/02/2010, 17.520 intervalos, las cuatro estaciones).
`validacion.py` reentrena los cuatro modelos sin ese año y lo predice; `export_ensamble.py` toma
esas predicciones y aplica reglas que usan la validación y no la prueba (constantes en `core.py`):

| Decisión | Regla | Resultado |
|---|---|---|
| Forma de combinar | Menor RMSE en validación; si una más simple queda a menos de 0,5 %, la simple. Los pesos se estiman con una mitad del año y se evalúan en la otra, y al revés | **Promedio** (RMSE 0,4755, frente a 0,4912 del ponderado y 0,4842 del stacking) |
| Umbral de alerta de pico | Mayor F1 de la clase pico en validación | **1,85 kW** |
| Más peso a los picos al entrenar (×3) | Se adopta si el F1 sube al menos 0,01 y el RMSE no empeora más de 1 % | **No se adopta** (F1 0,657 → 0,654; RMSE +2,5 %) |

Después, la prueba se evalúa con los modelos entrenados con todo el 80 %. La prueba no está intacta:
una versión anterior estimaba con ella los pesos del ensamble, y con ella se evaluó y se retiró el
pronóstico a 24 horas. El procedimiento actual no la usa para decidir.

### Comprobaciones de robustez (`comprobaciones.py`)

No cambian ninguna decisión; guardan sus resultados en `comprobaciones.json`, que la app muestra.

- **Formas de combinar en orden temporal.** En el sentido segunda → primera mitad de la tabla de
  arriba, los pesos salen de datos posteriores a los evaluados, lo que no reproduce un pronóstico.
  Estimando solo con la primera mitad y evaluando las tres formas en la segunda (8.760 intervalos),
  el promedio también gana: RMSE 0,5105 kW, frente a 0,5304 del stacking y 0,5409 del ponderado.
- **Prueba sin consumos rellenados.** Con los datos minuto a minuto, 480 intervalos de prueba están
  totalmente rellenados por la imputación, 13 parcialmente y 1 tiene solo 3 minutos medidos. Con los
  13.274 intervalos completos y medidos, el ensamble obtiene RMSE 0,4350 kW y precisión ±1 kW de
  96,07 % (frente a 0,4285 y 96,20 % con todos). Necesita el dataset crudo (`src/download_data.py`).

## Casos críticos: picos de consumo

La precisión por tolerancia se infla porque la mayoría de los intervalos son de consumo bajo: predecir
siempre la media del entrenamiento (1,11 kW) ya acierta ±1 kW en el 91,2 % de la prueba. Por eso se
evalúa aparte el caso crítico: un **pico** es un intervalo con consumo real de más de 2 kW
(`core.UMBRAL_PICO_KW`), el 10,1 % de la prueba. La app da **alerta de pico** cuando el ensamble
prevé más que el umbral elegido en validación, y reporta recall, precisión, F1, PR-AUC, la matriz de
confusión y los falsos negativos y positivos por hora (`core.metricas_pico()`).

| Ensamble en prueba | Recall | Precisión | F1 | PR-AUC |
|---|---|---|---|---|
| Alerta si ŷ > 2 kW (sin ajustar) | 0,495 | 0,690 | 0,577 | 0,646 |
| Alerta si ŷ > 1,85 kW (umbral de validación) | 0,617 | 0,630 | 0,623 | 0,646 |

## Funcionalidades (12)

| # | Sección | Funcionalidad |
|---|---|---|
| F1 | Histórico | Serie histórica interactiva (30 min, hora, día, semana) con indicadores de energía, potencia media, pico y costo en pesos |
| F2 | Histórico | Patrones: mapa de calor hora × día de la semana y perfil mensual |
| F3 | Predicción | Predicción del siguiente intervalo (30 min) con el ensamble y los 4 modelos frente al valor real, con alerta de pico |
| F4 | Comparación | Tabla de métricas del ensamble y los 4 modelos (MAE, MSE, RMSE, R², sMAPE y precisión por tolerancia) |
| F5 | Comparación | Real vs. predicho por semana, error por hora del día y dispersión |
| F6 | Comparación | Estabilidad por cortes de la validación de ventana expansiva |
| F7 | Comparación | Registro de desempeño: métricas de entrenamiento frente a prueba (sobreajuste) |
| F8 | Comparación | Ensamble: tres formas de combinar comparadas en validación, forma elegida y correlación de errores |
| F9 | Comparación | Casos críticos: recall, precisión, F1 y PR-AUC de los picos, umbral de alerta, matriz de confusión, errores por hora y ajuste «peso picos» |
| F10 | Escenarios | Simulador de escenarios sobre el consumo medido o previsto, con desglose por grupo de carga, ahorro y costo (tarifa configurable) |
| F11 | Modelo | Importancia de variables (RF, XGBoost) y coeficientes (lineal) |
| F12 | Modelo | Defensa del modelo, ecuaciones clave (incluidos el ensamble y la alerta) y trazabilidad ecuación → código |

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
