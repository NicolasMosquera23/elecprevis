"""
Entrenamiento y comparación de los cuatro modelos (Fase 4):
regresión lineal, random forest, XGBoost y LSTM.

Protocolo común a los cuatro: partición temporal (nunca aleatoria), validación
de ventana expansiva (walk-forward) y las métricas MAE, RMSE, R² y sMAPE
definidas en el Entregable 1.

El preprocesamiento sí cambia según el modelo, y esa diferencia es deliberada:
    - Regresión lineal: se estandariza la entrada y se ajusta sobre log(1 + y),
      porque el objetivo tiene asimetría positiva severa (skew = 1,44) y los
      mínimos cuadrados quedan dominados por los pocos picos altos.
    - Random forest y XGBoost: sin escalado y sin transformar el objetivo; sus
      divisiones son invariantes ante cambios monótonos de escala.
    - LSTM: se estandariza la entrada con la media y la desviación del conjunto
      de entrenamiento. Sin esto la red recibe variables en rangos muy distintos
      (Voltage alrededor de 240 frente a Sub_metering_1 alrededor de 1) y el
      descenso de gradiente no converge bien.

La semilla se fija en un solo lugar para que los resultados se puedan repetir.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLS_EXCLUDE = {"y"}  # todo lo demás en el df supervisado es feature
SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Fija la semilla de random, numpy y Keras, si está disponible."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        from tensorflow import keras

        keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def smape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1e-9
    return float(100 * np.mean(2 * np.abs(y_true - y_pred) / denom))


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


def split_xy(df: pd.DataFrame, target_col: str = "y"):
    feature_cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]
    return df[feature_cols], df[target_col]


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def fit_linear_regression(X_train, y_train, log_target: bool = True):
    """
    Ajusta la regresión lineal sobre log(1 + y) cuando log_target es True.

    Devuelve (modelo, escalador, log_target). La transformación se justifica por
    la asimetría del objetivo; al invertir la predicción con expm1 las métricas
    vuelven a kW y siguen siendo comparables con las de los otros modelos.
    """
    scaler = StandardScaler().fit(X_train)
    y_fit = np.log1p(y_train) if log_target else y_train
    model = LinearRegression().fit(scaler.transform(X_train), y_fit)
    return model, scaler, log_target


def predict_linear(bundle, X) -> np.ndarray:
    """Predice con el paquete que devuelve fit_linear_regression."""
    model, scaler, log_target = bundle
    pred = model.predict(scaler.transform(X))
    return np.expm1(pred) if log_target else pred


def fit_random_forest(X_train, y_train, **kwargs):
    model = RandomForestRegressor(
        n_estimators=kwargs.pop("n_estimators", 200),
        random_state=SEED,
        n_jobs=-1,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def fit_xgboost(X_train, y_train, **kwargs):
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=kwargs.pop("n_estimators", 300),
        max_depth=kwargs.pop("max_depth", 6),
        learning_rate=kwargs.pop("learning_rate", 0.05),
        random_state=SEED,
        n_jobs=-1,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------------------------
# LSTM
# --------------------------------------------------------------------------
def build_lstm_sequences(X: np.ndarray, y: np.ndarray, n_steps: int = 48):
    """
    Convierte el dataset tabular en ventanas (muestras, n_steps, variables).

    Se aplica por separado a entrenamiento y a prueba: armar las ventanas sobre
    el conjunto completo y después partir mezclaría información de las dos
    mitades en la frontera.
    """
    Xs, ys = [], []
    for i in range(n_steps, len(X)):
        Xs.append(X[i - n_steps : i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def fit_lstm(X_train_seq, y_train_seq, n_features: int, epochs: int = 20,
             batch_size: int = 64, seed: int = SEED, log_target: bool = True):
    """
    Entrena la LSTM con los dos tratamientos que exigen sus datos.

    1. Entrada: se estandariza cada variable con la media y la desviación del
       entrenamiento. Sin esto la red recibe tensión alrededor de 240 junto a
       submediciones alrededor de 1, y el descenso de gradiente no converge.
    2. Objetivo: se ajusta sobre log(1 + y). El consumo es asimétrico y no puede
       ser negativo; sin la transformación la red devuelve consumos negativos y
       aplana los picos, porque el error cuadrático se concentra en la cola alta
       y la capa de salida lineal no tiene límite inferior.

    Devuelve (modelo, escalador, log_target), igual que fit_linear_regression,
    para poder invertir la transformación al predecir.
    """
    from tensorflow import keras

    set_global_seed(seed)

    n, pasos, f = X_train_seq.shape
    scaler = StandardScaler().fit(X_train_seq.reshape(-1, f))
    X_esc = scaler.transform(X_train_seq.reshape(-1, f)).reshape(n, pasos, f)

    y_ajuste = np.log1p(y_train_seq) if log_target else y_train_seq

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(pasos, n_features)),
            keras.layers.LSTM(64),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_esc, y_ajuste, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, scaler, log_target


def predict_lstm(bundle, X_seq) -> np.ndarray:
    """
    Predice con el paquete que devuelve fit_lstm y devuelve las unidades a kW.

    La predicción se recorta en cero porque el consumo no puede ser negativo:
    al invertir el logaritmo, un valor levemente negativo en la escala
    logarítmica se traduce en un consumo imposible.
    """
    model, scaler, log_target = bundle
    n, pasos, f = X_seq.shape
    X_esc = scaler.transform(X_seq.reshape(-1, f)).reshape(n, pasos, f)
    pred = model.predict(X_esc, verbose=0).ravel()
    if log_target:
        pred = np.expm1(pred)
    return np.clip(pred, 0, None)


# --------------------------------------------------------------------------
# Validación temporal
# --------------------------------------------------------------------------
def walk_forward_folds(n_samples: int, n_splits: int = 5, min_train_frac: float = 0.4):
    """
    Cortes de una validación de ventana expansiva: en cada corte se entrena con
    todo lo anterior y se predice el bloque siguiente. Nunca se entrena con
    datos posteriores al bloque evaluado.

    Devuelve tuplas (train_ini, train_fin, test_ini, test_fin) sobre posiciones.
    """
    inicio = int(n_samples * min_train_frac)
    bloque = (n_samples - inicio) // n_splits
    cortes = []
    for k in range(n_splits):
        train_fin = inicio + k * bloque
        test_fin = train_fin + bloque if k < n_splits - 1 else n_samples
        cortes.append((0, train_fin, train_fin, test_fin))
    return cortes


def walk_forward_evaluate(fit_predict, X, y, n_splits: int = 5,
                          min_train_frac: float = 0.4) -> pd.DataFrame:
    """
    Evalúa un modelo sobre todos los cortes. `fit_predict` recibe
    (X_train, y_train, X_test) y devuelve predicciones en kW.
    """
    filas = []
    for k, (_, train_fin, test_ini, test_fin) in enumerate(
        walk_forward_folds(len(X), n_splits, min_train_frac), start=1
    ):
        X_tr, y_tr = X.iloc[:train_fin], y.iloc[:train_fin]
        X_te, y_te = X.iloc[test_ini:test_fin], y.iloc[test_ini:test_fin]
        fila = evaluate(y_te, fit_predict(X_tr, y_tr, X_te))
        fila.update(
            {
                "corte": k,
                "n_train": len(X_tr),
                "n_test": len(X_te),
                "desde": str(X.index[test_ini])[:16],
                "hasta": str(X.index[test_fin - 1])[:16],
            }
        )
        filas.append(fila)
    return pd.DataFrame(filas)


def compare_models(results: dict) -> pd.DataFrame:
    """results: {"Regresión lineal": {...métricas...}, "Random Forest": {...}, ...}"""
    return pd.DataFrame(results).T.sort_values("RMSE")
