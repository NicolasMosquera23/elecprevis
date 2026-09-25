"""
Limpieza y preparación del dataset UCI (Fase 1 de la metodología ElecPrevis).

Pasos que sigue este módulo, en orden:
    1. Carga del .txt separado por ';', con '?' como marcador de nulo.
    2. Combinación de Date + Time en un índice datetime.
    3. Optimización de memoria (downcast numérico).
    4. Imputación causal de nulos: cada hueco se rellena con la última
       observación válida (forward fill). No se usa mediana global porque es
       una serie de tiempo, y no se usa interpolación centrada porque
       necesitaría el valor siguiente, es decir, información futura.
    5. Remuestreo de 1 minuto a 30 minutos.
    6. Reporte de outliers vía IQR (solo diagnóstico, no se recortan valores
       reales de consumo sin justificar por qué).

Ningún paso mira hacia adelante: el forward fill solo propaga valores ya
observados y el remuestreo promedia dentro de la ventana que abre en t. Cada
ventana de 30 minutos queda etiquetada por su inicio, así que la fila t
depende únicamente de minutos en [t, t+30).
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "household_power_consumption.txt"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

NUMERIC_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Carga el .txt crudo y arma el índice datetime."""
    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()
    return df


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast de columnas numéricas para reducir el consumo de RAM."""
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def report_missing(df: pd.DataFrame) -> pd.Series:
    """Porcentaje de nulos por columna, para documentar antes de imputar."""
    return (df.isna().mean() * 100).round(3)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward fill: cada hueco se rellena con la última observación válida.

    Se evita la interpolación centrada (limit_direction="both") porque para
    rellenar un hueco necesita también el valor siguiente, y en un pronóstico
    a 30 minutos esa información todavía no existe cuando se construye la
    variable. El forward fill solo propaga lo ya observado.
    """
    df[NUMERIC_COLS] = df[NUMERIC_COLS].ffill()
    return df


def report_outliers_iqr(df: pd.DataFrame, cols=NUMERIC_COLS) -> pd.DataFrame:
    """
    Reporta cuántos valores caen fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    por columna. Es diagnóstico: en consumo eléctrico un pico puede ser
    un evento real (encender el horno), no ruido a eliminar sin criterio.
    """
    rows = []
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        n_valid = int(df[col].notna().sum())
        rows.append(
            {"columna": col, "q1": q1, "q3": q3, "limite_inf": lower,
             "limite_sup": upper, "n_outliers": n_out,
             "pct_outliers": round(n_out / n_valid * 100, 3) if n_valid else 0.0}
        )
    return pd.DataFrame(rows)


def resample_30min(df: pd.DataFrame) -> pd.DataFrame:
    """Remuestreo de 1 minuto a intervalos de 30 minutos (promedio)."""
    return df[NUMERIC_COLS].resample("30min").mean()


def run_pipeline(save: bool = True, raw_path: Path = RAW_PATH) -> pd.DataFrame:
    print("Cargando datos crudos...")
    df = load_raw(raw_path)
    print(f"  {len(df):,} filas cargadas (resolución 1 min).")

    df = optimize_memory(df)

    missing = report_missing(df)
    print("Nulos por columna (%):")
    print(missing[missing > 0])

    # El diagnóstico de outliers se hace sobre los datos crudos: así no se
    # mezclan valores imputados con valores realmente medidos.
    outliers = report_outliers_iqr(df)
    print("\nOutliers sobre datos crudos (IQR, solo diagnóstico):")
    print(outliers.to_string(index=False))

    nulos_antes = int(df[NUMERIC_COLS].isna().sum().sum())
    df = impute_missing(df)
    pct_imp = nulos_antes / (len(df) * len(NUMERIC_COLS)) * 100
    print(f"\nValores imputados por forward fill: {nulos_antes:,} ({pct_imp:.3f}% de las celdas)")

    df_30min = resample_30min(df)
    print(f"\nRemuestreo a 30 min: {len(df_30min):,} intervalos.")

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "power_30min.csv"
        df_30min.to_csv(out_path)
        print(f"Guardado en {out_path}")

    return df_30min


if __name__ == "__main__":
    run_pipeline()
