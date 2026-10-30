"""
Núcleo de la aplicación ElecPrevis (Entregable 2, Parte 5).

Este módulo no reimplementa el modelo: importa la construcción de variables de
src/features.py y carga los cuatro estimadores que entrena app/train_export.py
con las funciones de src/models.py. Lo único propio de la app es:

    - Predictores ligeros para servir los modelos sin TensorFlow:
        * la regresión lineal se evalúa con sus coeficientes (ecuación de OLS),
        * la LSTM se evalúa con sus pesos en NumPy, escribiendo literalmente
          las ecuaciones de las compuertas de la Sección 6.4 del modelo
          matemático.
    - El pronóstico encadenado de 24 h (48 pasos), con la advertencia de la
      Sección 9: el error se acumula y no se cuantifica en la formulación.
    - Los escenarios hipotéticos de reducción sobre las submediciones.
    - La precisión por tolerancia en kW que pide la rúbrica (ver TOLERANCIA_KW).
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import features as ft  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"

TARGET = ft.TARGET
FEATURES = [
    "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "hour", "dayofweek", "is_weekend", "month",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "Global_active_power_lag1", "Global_active_power_lag48",
    "Global_active_power_lag336", "Global_active_power_roll_mean_48",
]
# Modelo matemático §6.1: la regresión lineal excluye Global_intensity (r = 0,999).
LINEAR_EXCLUDE = ["Global_intensity"]
LINEAR_FEATURES = [c for c in FEATURES if c not in LINEAR_EXCLUDE]
N_STEPS = 48  # ventana de la LSTM: 48 intervalos = 24 h

MODELOS = ["Regresión lineal", "Random Forest", "XGBoost", "LSTM"]

# Random forest de la app: 150 árboles con hojas de al menos 10 observaciones.
# El mínimo por hoja limita el sobreajuste (cada árbol deja de memorizar
# intervalos sueltos) y baja el archivo de ≈160 MB a ≈15 MB, así que se puede
# versionar y la app publicada lo usa. En prueba rinde igual o mejor que el
# de 200 árboles sin límite (R² 0,68 frente a 0,67).
RF_PARAMS = {"n_estimators": 150, "min_samples_leaf": 10}

# --------------------------------------------------------------------------
# Ensamble de los cuatro modelos (Modelo matemático §6.5)
# --------------------------------------------------------------------------
# Mismo principio del random forest: en regresión, el bosque predice el
# promedio de sus árboles. Aquí los "votantes" son los cuatro modelos, que son
# de naturaleza distinta (lineal, árboles, red recurrente) y por eso se
# equivocan en momentos distintos; al promediar, parte del error se compensa.
#
#     ŷ_ens = (ŷ_lineal + ŷ_RF + ŷ_XGB + ŷ_LSTM) / 4
#
# Se comparan además dos formas de combinar que necesitan estimar parámetros:
#     ponderado: ŷ = Σ wᵢ ŷᵢ, wᵢ ≥ 0, Σ wᵢ = 1 (mínimos cuadrados no negativos)
#     stacking:  ŷ = a + Σ bᵢ ŷᵢ (regresión lineal sobre las predicciones)
# Sus parámetros se estiman con la primera mitad del periodo de prueba
# (calibración) y las tres se comparan en la segunda mitad (evaluación), para
# no medir con los mismos datos con los que se ajustaron.
ENSAMBLE = "Ensamble"
COMBINACIONES = {
    "Promedio": "Cada modelo vale un voto, como los árboles de un random forest",
    "Ponderado": "Vota más el modelo que menos se equivoca (pesos ≥ 0 que suman 1)",
    "Stacking": "Una regresión lineal aprende a combinar las cuatro predicciones",
}


def ensamble_promedio(preds) -> np.ndarray:
    """preds: DataFrame o dict con una columna por modelo de MODELOS."""
    return np.mean(np.column_stack([np.asarray(preds[m], float) for m in MODELOS]), axis=1)


def combinar(preds, metodo: str = "Promedio", params: dict | None = None) -> np.ndarray:
    """Combina las predicciones de los cuatro modelos con el método indicado."""
    if metodo == "Promedio":
        return ensamble_promedio(preds)
    M = np.column_stack([np.asarray(preds[m], float) for m in MODELOS])
    if metodo == "Ponderado":
        w = np.array([params["pesos"][m] for m in MODELOS])
        return M @ w
    if metodo == "Stacking":
        b = np.array([params["coef"][m] for m in MODELOS])
        return np.clip(params["intercepto"] + M @ b, 0, None)
    raise ValueError(metodo)


# Paleta categórica validada (orden fijo, el color sigue al modelo, no al ranking).
COLOR_MODELO = {
    "Ensamble": "#7b3fbf",
    "Ensamble (sin LSTM)": "#7b3fbf",
    "Real": "#0b0b0b",
    "XGBoost": "#2a78d6",
    "Random Forest": "#eb6834",
    "LSTM": "#1baf7a",
    "Regresión lineal": "#eda100",
}
COLOR_CARGA = {
    "Sub_metering_1": "#2a78d6",
    "Sub_metering_2": "#eb6834",
    "Sub_metering_3": "#1baf7a",
    "Resto": "#a8a7a2",
}
TARIFA_DEFECTO = 1046  # COP/kWh, Electrohuila estrato 4, junio 2026 (Valora Analitik)

# Submediciones: el dataset las da en Wh de energía activa por minuto.
# Promedio de Wh/min en el intervalo × 60 / 1000 = potencia media en kW.
WH_MIN_A_KW = 60 / 1000
SUBMEDICIONES = {
    "Sub_metering_1": "Cocina (lavavajillas, horno, microondas)",
    "Sub_metering_2": "Lavandería (lavadora, secadora, nevera, luz)",
    "Sub_metering_3": "Calentador de agua y aire acondicionado",
}
HORAS_INTERVALO = 0.5  # 30 min

DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


# --------------------------------------------------------------------------
# Precisión por tolerancia (rúbrica: "precisión > 95 %")
# --------------------------------------------------------------------------
# La precisión TP / (TP + FP) es una métrica de clasificación y no aplica
# directamente a una regresión, cuya salida es un número continuo (kW). Por
# indicación de la docente se reporta la precisión por tolerancia: el
# porcentaje de predicciones cuyo error absoluto no supera TOLERANCIA_KW.
# Se acompaña del error P95, que dice lo mismo al revés: el error que el
# modelo no supera en el 95 % de los intervalos.
TOLERANCIA_KW = 1.0


def precision_tolerancia(y, yhat, tol: float = TOLERANCIA_KW) -> float:
    """% de intervalos con |y − ŷ| ≤ tol."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(100 * np.mean(np.abs(y - yhat) <= tol))


def error_p95(y, yhat) -> float:
    """Error absoluto que no se supera en el 95 % de los intervalos (kW)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.quantile(np.abs(y - yhat), 0.95))


def smape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    den = np.abs(y) + np.abs(yhat)
    den[den == 0] = 1e-9
    return float(100 * np.mean(2 * np.abs(y - yhat) / den))


def metricas(y, yhat, index=None) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = y - yhat
    mse = float(np.mean(err ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
        "sMAPE": smape(y, yhat),
        "Precisión ±1 kW": precision_tolerancia(y, yhat),
        "Error P95": error_p95(y, yhat),
    }


# --------------------------------------------------------------------------
# Predictores
# --------------------------------------------------------------------------
class LinealOLS:
    """ŷ = expm1(β0 + Σ βj · zj), con z la entrada estandarizada (§6.1 y §7)."""

    def __init__(self, d: dict):
        self.cols = d["features"]
        self.mean = np.array(d["scaler_mean"])
        self.scale = np.array(d["scaler_scale"])
        self.coef = np.array(d["coef"])
        self.intercept = float(d["intercept"])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        z = (X[self.cols].to_numpy(float) - self.mean) / self.scale
        return np.clip(np.expm1(z @ self.coef + self.intercept), 0, None)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class LSTMNumpy:
    """
    Evalúa la LSTM entrenada con Keras usando solo NumPy, ecuación por ecuación
    (Modelo matemático §6.4). Keras guarda las compuertas en el orden i, f, c, o.

        f_t = σ(W_f x_t + U_f h_{t−1} + b_f)        compuerta de olvido
        i_t = σ(W_i x_t + U_i h_{t−1} + b_i)        compuerta de entrada
        c̃_t = tanh(W_c x_t + U_c h_{t−1} + b_c)     estado candidato
        c_t = f_t ⊙ c_{t−1} + i_t ⊙ c̃_t             estado de la celda
        o_t = σ(W_o x_t + U_o h_{t−1} + b_o)        compuerta de salida
        h_t = o_t ⊙ tanh(c_t)
    """

    def __init__(self, npz):
        self.W = npz["kernel"]          # (19, 4u)
        self.U = npz["recurrent"]       # (u, 4u)
        self.b = npz["bias"]            # (4u,)
        self.W1, self.b1 = npz["dense1_w"], npz["dense1_b"]
        self.W2, self.b2 = npz["dense2_w"], npz["dense2_b"]
        self.mean, self.scale = npz["scaler_mean"], npz["scaler_scale"]
        self.u = self.U.shape[0]

    def predict(self, seq: np.ndarray) -> np.ndarray:
        """seq: (muestras, 48, 19) en unidades originales → kW."""
        x = (seq - self.mean) / self.scale
        n, pasos, _ = x.shape
        u = self.u
        h = np.zeros((n, u))
        c = np.zeros((n, u))
        for t in range(pasos):
            z = x[:, t, :] @ self.W + h @ self.U + self.b
            i = _sigmoid(z[:, :u])
            f = _sigmoid(z[:, u:2 * u])
            c_tilde = np.tanh(z[:, 2 * u:3 * u])
            o = _sigmoid(z[:, 3 * u:])
            c = f * c + i * c_tilde
            h = o * np.tanh(c)
        d1 = np.maximum(0, h @ self.W1 + self.b1)
        salida = (d1 @ self.W2 + self.b2).ravel()
        return np.clip(np.expm1(salida), 0, None)


@lru_cache(maxsize=1)
def cargar_modelos() -> dict:
    """Carga los cuatro modelos (el random forest es opcional por si falta el archivo)."""
    modelos = {}
    modelos["Regresión lineal"] = LinealOLS(json.loads((ART / "lineal.json").read_text()))
    from xgboost import XGBRegressor

    xgb = XGBRegressor()
    xgb.load_model(ART / "xgboost.json")
    modelos["XGBoost"] = xgb
    modelos["LSTM"] = LSTMNumpy(np.load(ART / "lstm.npz"))
    rf_path = ART / "random_forest.joblib"
    if rf_path.exists():
        import joblib

        modelos["Random Forest"] = joblib.load(rf_path)
    return modelos


def predecir(nombre: str, modelo, X: pd.DataFrame) -> np.ndarray:
    """
    Un paso: X con las 19 columnas (para la LSTM, las últimas 48 filas).
    Con nombre == ENSAMBLE, `modelo` es el dict de los cuatro modelos y la
    salida es el promedio de sus predicciones para la última fila.
    """
    if nombre == ENSAMBLE:
        return np.array([np.mean([predecir(m, modelo[m], X)[-1] for m in MODELOS])])
    if nombre == "LSTM":
        seq = X[FEATURES].to_numpy(float)[-N_STEPS:][None, :, :]
        return modelo.predict(seq)
    if nombre == "Regresión lineal":
        return modelo.predict(X)
    return np.clip(modelo.predict(X[FEATURES]), 0, None)


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def cargar_serie() -> pd.DataFrame:
    """Serie remuestreada a 30 min (salida de src/preprocessing.py)."""
    return pd.read_parquet(ART / "power_30min.parquet")


@lru_cache(maxsize=1)
def cargar_supervisado() -> pd.DataFrame:
    return ft.build_supervised_dataset(cargar_serie())


@lru_cache(maxsize=1)
def cargar_predicciones() -> pd.DataFrame:
    """Predicciones a un paso de los 4 modelos sobre todo el periodo de prueba."""
    return pd.read_parquet(ART / "predicciones_prueba.parquet")


@lru_cache(maxsize=1)
def cargar_resultados() -> dict:
    return json.loads((ART / "resultados.json").read_text(encoding="utf-8"))


def features_de_historia(hist: pd.DataFrame) -> pd.DataFrame:
    """Aplica exactamente las funciones de src/features.py sobre una historia."""
    df = ft.add_calendar_features(hist)
    df = ft.add_lag_features(df)
    df = ft.add_rolling_features(df)
    return df


# --------------------------------------------------------------------------
# Pronóstico encadenado de 24 h
# --------------------------------------------------------------------------
def pronostico_encadenado(nombre: str, modelo, serie: pd.DataFrame, t0, pasos: int = 48):
    """
    Encadena `pasos` predicciones de un paso desde t0 (inclusive).

    Supuestos (Modelo matemático §9, la predicción es a un paso):
      - La potencia activa futura es la que predice el propio modelo.
      - Las demás mediciones del intervalo futuro (tensión, reactiva,
        submediciones) se toman del mismo intervalo del día anterior.
      - La intensidad se ajusta a la potencia predicha con la relación I/P
        observada ese mismo intervalo del día anterior (r = 0,999).
    Devuelve una Serie con la potencia predicha en t0+30min ... t0+pasos·30min.
    """
    hist = serie.loc[:t0].tail(336 + N_STEPS + 2).copy()
    salidas = {}
    for _ in range(pasos):
        feats = features_de_historia(hist)
        X = feats[FEATURES].dropna()
        pred = float(predecir(nombre, modelo, X)[-1])
        nuevo_t = hist.index[-1] + pd.Timedelta(minutes=30)
        previo = hist.loc[nuevo_t - pd.Timedelta(days=1)] if (nuevo_t - pd.Timedelta(days=1)) in hist.index else hist.iloc[-48]
        fila = previo.copy()
        p_prev = previo["Global_active_power"]
        ratio = previo["Global_intensity"] / p_prev if p_prev > 0 else 4.2
        fila["Global_active_power"] = pred
        fila["Global_intensity"] = pred * ratio
        hist.loc[nuevo_t] = fila
        salidas[nuevo_t] = pred
    return pd.Series(salidas, name=nombre)


# --------------------------------------------------------------------------
# Escenarios de eficiencia (hipotéticos, sin causalidad: §9)
# --------------------------------------------------------------------------
def desglose_kw(serie: pd.DataFrame) -> pd.DataFrame:
    """Potencia media por grupo de carga, en kW. 'Resto' = total − submediciones."""
    out = pd.DataFrame(index=serie.index)
    for col in SUBMEDICIONES:
        out[col] = serie[col] * WH_MIN_A_KW
    out["Resto"] = (serie["Global_active_power"] - out[list(SUBMEDICIONES)].sum(axis=1)).clip(lower=0)
    return out


def simular_escenario(serie: pd.DataFrame, reducciones: dict) -> pd.DataFrame:
    """
    Escenario contrafactual: y'_t = y_t − Σ_k r_k · s_{k,t}, con s_k la
    submedición k en kW y r_k la reducción supuesta (0 a 1).
    """
    d = desglose_kw(serie)
    ahorro = sum(r * d[col] for col, r in reducciones.items())
    return pd.DataFrame({
        "Real": serie["Global_active_power"],
        "Escenario": serie["Global_active_power"] - ahorro,
        "Ahorro": ahorro,
    })
