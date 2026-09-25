"""
Construcción de variables temporales y rezagadas (Fase 3).

Regla que se respeta en todo el módulo: ninguna variable puede usar
información posterior al instante que se predice (data leakage temporal).
Por eso todo se arma con .shift() hacia atrás y con rolling() sin centrar.
"""

import numpy as np
import pandas as pd

TARGET = "Global_active_power"

# Rezagos elegidos según los ciclos del consumo a resolución de 30 min:
#   1   -> intervalo inmediatamente anterior
#   48  -> mismo intervalo del día anterior (48 * 30min = 24h)
#   336 -> mismo intervalo de la semana anterior (336 * 30min = 7 días)
LAGS = [1, 48, 336]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek  # 0=lunes
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"] = df.index.month

    # Codificación cíclica: la hora y el mes son ciclos, no escalas lineales.
    # Sin esto, el modelo ve las 23:00 lejos de las 00:00 y diciembre lejos
    # de enero, lo que perjudica a los modelos lineales y a la LSTM.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def add_lag_features(df: pd.DataFrame, target: str = TARGET, lags=LAGS) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target: str = TARGET, windows=(48,)) -> pd.DataFrame:
    """Media móvil calculada solo con datos pasados (shift(1) antes de rolling)."""
    df = df.copy()
    for w in windows:
        df[f"{target}_roll_mean_{w}"] = df[target].shift(1).rolling(window=w).mean()
    return df


def build_supervised_dataset(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """
    Arma el dataset supervisado: y = target en t+1, X = todo lo disponible
    hasta t (rezagos, calendario, submediciones actuales).
    """
    df = add_calendar_features(df)
    df = add_lag_features(df, target)
    df = add_rolling_features(df, target)

    df["y"] = df[target].shift(-1)  # horizonte de 1 paso (30 min adelante)

    df = df.dropna()
    return df


def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split cronológico: los últimos `test_size` de las filas van a Test.
    Nunca mezclar aleatoriamente en series de tiempo.
    """
    n_test = int(len(df) * test_size)
    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]
    return train, test
