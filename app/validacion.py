"""
Predicciones de validación: el último año del entrenamiento, que ningún modelo vio.

Uso (desde la raíz del repo; necesita TensorFlow por la LSTM):
    python app/validacion.py              # unos 10 min; al final llama a export_ensamble.py
    python app/validacion.py --cortes 5   # además rehace la validación por cortes (≈10 min más)

Por qué existe. Las decisiones de la integración (cómo combinar los cuatro modelos,
con qué pesos y desde qué consumo previsto se da la alerta de pico) se aprenden de
datos. Si se aprenden con el periodo de prueba, la evaluación final deja de ser
independiente de esas decisiones. Por eso el 80 % de entrenamiento se divide en dos:

    sub-entrenamiento  todo menos el último año (≈ dic 2006 – feb 2009)
    validación         el último año del entrenamiento (≈ feb 2009 – feb 2010),
                       17.520 intervalos: cubre las cuatro estaciones

    1. Se entrenan los cuatro modelos con el sub-entrenamiento, con los mismos
       hiperparámetros de la app.
    2. Se predice la validación y se guarda en predicciones_validacion.parquet.
    3. export_ensamble.py toma esas predicciones, elige la combinación y los
       umbrales con las reglas fijadas en core.py y los aplica sin cambios a la
       prueba, que se predice con los modelos entrenados con todo el 80 %.

También entrena la variante "peso picos" de la regresión lineal, el random forest
y XGBoost: cada intervalo con consumo siguiente > core.UMBRAL_PICO_KW pesa
core.PESO_PICOS veces más en el ajuste. Es un ajuste controlado: export_ensamble.py
lo compara contra la versión base en la validación y decide si se adopta.
La LSTM no tiene esa variante por su costo de entrenamiento.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "app"))

from src import features as ft  # noqa: E402
from src import models as md  # noqa: E402
import core  # noqa: E402
import export_ensamble  # noqa: E402

ART = core.ART


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(epocas: int = 20, cortes: int = 0):
    md.set_global_seed()
    serie = core.cargar_serie().astype("float64")
    sup = ft.build_supervised_dataset(serie)
    tr, _ = ft.temporal_train_test_split(sup)
    X_tr, y_tr = md.split_xy(tr)
    assert list(X_tr.columns) == core.FEATURES

    n_sub = len(X_tr) - core.N_VALIDACION
    X_sub, y_sub = X_tr.iloc[:n_sub], y_tr.iloc[:n_sub]
    X_val, y_val = X_tr.iloc[n_sub:], y_tr.iloc[n_sub:]
    log(f"sub-entrenamiento {len(X_sub)} ({X_sub.index[0]} – {X_sub.index[-1]})")
    log(f"validación        {len(X_val)} ({X_val.index[0]} – {X_val.index[-1]})")

    # El índice de cada predicción es el intervalo que se predice (t + 30 min),
    # igual que en predicciones_prueba.parquet.
    val = pd.DataFrame({"origen": X_val.index, "Real": y_val.to_numpy()},
                       index=X_val.index + pd.Timedelta(minutes=30))
    val.index.name = "intervalo"
    val["Persistencia"] = X_val["Global_active_power"].to_numpy()

    peso = np.where(y_sub.to_numpy() > core.UMBRAL_PICO_KW, core.PESO_PICOS, 1.0)
    log(f"intervalos pico en el sub-entrenamiento: {(peso > 1).mean() * 100:.1f} % (peso {core.PESO_PICOS})")

    # ---------------- Regresión lineal, random forest y XGBoost ----------------
    # Cada uno se entrena dos veces: base y con peso extra en los picos.
    for w, sufijo in [(None, ""), (peso, core.SUFIJO_PESO)]:
        log(f"regresión lineal, random forest {core.RF_PARAMS} y xgboost{sufijo}")
        lin = md.fit_linear_regression(X_sub, y_sub, sample_weight=w)
        val[f"Regresión lineal{sufijo}"] = np.clip(md.predict_linear(lin, X_val), 0, core.POTENCIA_MAX_KW)
        rf = md.fit_random_forest(X_sub, y_sub, sample_weight=w, **dict(core.RF_PARAMS))
        val[f"Random Forest{sufijo}"] = rf.predict(X_val)
        del rf
        val[f"XGBoost{sufijo}"] = np.clip(md.fit_xgboost(X_sub, y_sub, sample_weight=w).predict(X_val), 0, None)

    # ---------------- LSTM ----------------
    log(f"lstm ({epocas} épocas)")
    seq_sub, ys_sub = md.build_lstm_sequences(X_sub.values, y_sub.values, n_steps=core.N_STEPS)
    bundle = md.fit_lstm(seq_sub, ys_sub, n_features=seq_sub.shape[2], epochs=epocas)
    # La ventana de los primeros intervalos de validación toma contexto del final
    # del sub-entrenamiento (solo pasado).
    desde = n_sub - core.N_STEPS + 1
    seq_val, _ = md.build_lstm_sequences(X_tr.values[desde:], y_tr.values[desde:], n_steps=core.N_STEPS)
    assert len(seq_val) == len(X_val)
    val["LSTM"] = md.predict_lstm(bundle, seq_val)

    val.to_parquet(ART / "predicciones_validacion.parquet")
    for m in ["Persistencia"] + core.MODELOS:
        r = core.metricas(val["Real"], val[m])
        log(f"  {m:17s} R² {r['R2']:.4f} | RMSE {r['RMSE']:.4f}")

    log("decisiones y evaluación en prueba (export_ensamble.py)")
    export_ensamble.main(reentrenar_rf=False, cortes=cortes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--cortes", type=int, default=0,
                    help="cortes de la validación de ventana expansiva (0 = conservar los guardados)")
    args = ap.parse_args()
    main(epocas=args.epocas, cortes=args.cortes)
