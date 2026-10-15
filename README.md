# ElecPrevis

Previsión del consumo eléctrico residencial y escenarios de eficiencia energética mediante Machine Learning.

**Universidad Surcolombiana (USCO)** · Facultad de Ingeniería · Programa de Ingeniería de Software
Electiva en Ingeniería Aplicada II · Grupo 8 — Energía e Industria 4.0

Integrantes: Juan Diego Montenegro Segura, Sofía Montaña, Nicolás Mosquera Perdomo.

## Qué hace el proyecto

Compara cuatro modelos (regresión lineal, random forest, XGBoost, LSTM) para predecir el
consumo eléctrico residencial a 30 minutos, usando el dataset
[Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
del repositorio UCI, y simula escenarios de eficiencia energética a partir de las submediciones
de carga (cocina, lavandería, calentamiento de agua).

Ver `docs/Entregable_1.pdf` para la propuesta completa, el estado del arte y el marco teórico.

## Estructura del repo

```
elecprevis/
├── data/
│   ├── raw/           # dataset crudo descargado de UCI (no versionado)
│   └── processed/     # dataset limpio y remuestreado (no versionado)
├── notebooks/
│   └── 01_exploracion.ipynb
├── src/
│   ├── download_data.py    # descarga y extrae el dataset de UCI
│   ├── preprocessing.py    # limpieza, imputación, remuestreo a 30 min
│   ├── features.py         # variables temporales y rezagadas sin leakage
│   └── models.py           # entrenamiento y comparación de los 4 modelos
├── app/                    # aplicación Streamlit (Entregable 2, Parte 5), ver app/README.md
├── docs/
│   └── Entregable_1.pdf
├── requirements.txt
└── .gitignore
```

## Cómo correrlo

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/download_data.py      # descarga data/raw/household_power_consumption.txt
python src/preprocessing.py      # genera data/processed/power_30min.csv
```

Para abrir la aplicación (no necesita descargar el dataset):

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

El dataset no se sube al repo (pesa varios cientos de MB): cada integrante lo descarga
localmente con `download_data.py`, que lo trae directo desde UCI.

## Metodología (resumen)

1. **Adquisición y preparación** — descarga, limpieza, remuestreo a 30 min, variables temporales y rezagadas sin información futura.
2. **Análisis exploratorio** — descomposición de la serie, patrones horarios/semanales/estacionales.
3. **Construcción del problema predictivo** — horizonte de un paso, split temporal train/test.
4. **Modelado y evaluación** — regresión lineal, random forest, XGBoost, LSTM con walk-forward validation; métricas MAE, RMSE, R², sMAPE.
5. **Escenarios de eficiencia** — simulación de reducción sobre las submediciones, sin atribuir causalidad.
6. **Aplicación** — prototipo con histórico, predicción y escenarios simulados.

Detalle completo en `docs/Entregable_1.pdf`.
