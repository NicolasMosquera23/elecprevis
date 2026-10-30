"""
Entrena los cuatro modelos con src/models.py y exporta lo que la app necesita.

Uso (desde la raíz del repo, con el dataset ya procesado):
    python src/download_data.py && python src/preprocessing.py
    python app/train_export.py              # todo (la LSTM tarda varios minutos)

Salidas en app/artifacts/:
    power_30min.parquet         serie remuestreada (entrada de la app)
    lineal.json                 coeficientes y escalador de la regresión lineal
    xgboost.json                modelo XGBoost
    lstm.npz                    pesos de la LSTM + escalador (se evalúa en NumPy)
    random_forest.joblib        random forest liviano (core.RF_PARAMS, ≈15 MB)
    predicciones_prueba.parquet predicción a un paso de los 4 modelos y del ensamble en prueba
    resultados.json             métricas train/test, ensamble, walk-forward e importancia de variables

Al final llama a app/export_ensamble.py, que arma el ensamble de los 4 modelos.

El script no cambia el protocolo del modelo matemático: misma partición 80/20,
mismas variables y misma semilla. El random forest de la app usa core.RF_PARAMS.
"""

import argparse
import json
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

PROCESADO = RAIZ / "data" / "processed" / "power_30min.csv"
ART = core.ART


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--cortes", type=int, default=5)
    args = ap.parse_args()

    ART.mkdir(parents=True, exist_ok=True)
    md.set_global_seed()

    serie = pd.read_csv(PROCESADO, index_col=0, parse_dates=True).astype("float32")
    serie.index.name = "datetime"
    serie.to_parquet(ART / "power_30min.parquet")

    sup = ft.build_supervised_dataset(serie.astype("float64"))
    tr, te = ft.temporal_train_test_split(sup)
    X_tr, y_tr = md.split_xy(tr)
    X_te, y_te = md.split_xy(te)
    assert list(X_tr.columns) == core.FEATURES, X_tr.columns
    log(f"supervisado {len(sup)} | train {len(tr)} | prueba {len(te)}")

    # El índice de cada predicción es el intervalo que se predice (t + 30 min).
    idx_obj = te.index + pd.Timedelta(minutes=30)
    pred = pd.DataFrame({"origen": te.index, "Real": y_te.to_numpy()}, index=idx_obj)
    pred.index.name = "intervalo"

    # ---------------- Regresión lineal ----------------
    log("regresión lineal")
    bundle = md.fit_linear_regression(X_tr, y_tr)
    model, scaler, _, cols = bundle
    (ART / "lineal.json").write_text(json.dumps({
        "features": cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "log_target": True,
    }, indent=1))
    p_lin = md.predict_linear(bundle, X_te)
    p_lin_app = core.LinealOLS(json.loads((ART / "lineal.json").read_text())).predict(X_te)
    assert np.allclose(np.clip(p_lin, 0, None), p_lin_app, atol=1e-8)
    pred["Regresión lineal"] = p_lin_app

    # ---------------- Random forest ----------------
    log(f"random forest {core.RF_PARAMS}")
    rf = md.fit_random_forest(X_tr, y_tr, **dict(core.RF_PARAMS))
    pred["Random Forest"] = rf.predict(X_te)
    imp_rf = dict(zip(core.FEATURES, rf.feature_importances_.tolist()))
    import joblib
    joblib.dump(rf, ART / "random_forest.joblib", compress=("lzma", 6))
    log(f"  random_forest.joblib: {(ART / 'random_forest.joblib').stat().st_size / 1e6:.1f} MB")
    del rf

    # ---------------- XGBoost ----------------
    log("xgboost (300 árboles)")
    xgb = md.fit_xgboost(X_tr, y_tr)
    xgb.save_model(ART / "xgboost.json")
    pred["XGBoost"] = np.clip(xgb.predict(X_te), 0, None)
    imp_xgb = dict(zip(core.FEATURES, xgb.feature_importances_.astype(float).tolist()))

    # ---------------- LSTM ----------------
    log(f"lstm ({args.epocas} épocas)")
    seq_tr, ys_tr = md.build_lstm_sequences(X_tr.values, y_tr.values, n_steps=core.N_STEPS)
    lstm_bundle = md.fit_lstm(seq_tr, ys_tr, n_features=seq_tr.shape[2], epochs=args.epocas)
    k_model, k_scaler, _ = lstm_bundle
    capas = [l for l in k_model.layers]
    kernel, recurrent, bias = capas[0].get_weights()
    d1w, d1b = capas[1].get_weights()
    d2w, d2b = capas[2].get_weights()
    np.savez_compressed(
        ART / "lstm.npz", kernel=kernel, recurrent=recurrent, bias=bias,
        dense1_w=d1w, dense1_b=d1b, dense2_w=d2w, dense2_b=d2b,
        scaler_mean=k_scaler.mean_, scaler_scale=k_scaler.scale_,
    )
    # Para predecir todo el periodo de prueba, la ventana de los primeros
    # intervalos toma contexto del final del entrenamiento (solo pasado).
    X_all = pd.concat([X_tr, X_te]).values
    y_all = np.concatenate([y_tr.values, y_te.values])
    seq_all, _ = md.build_lstm_sequences(X_all, y_all, n_steps=core.N_STEPS)
    seq_te = seq_all[-len(X_te):]
    p_keras = md.predict_lstm(lstm_bundle, seq_te)
    lstm_np = core.LSTMNumpy(np.load(ART / "lstm.npz"))
    p_np = lstm_np.predict(seq_te)
    log(f"  diferencia máx. Keras vs NumPy: {np.abs(p_keras - p_np).max():.2e} kW")
    pred["LSTM"] = p_np

    pred.to_parquet(ART / "predicciones_prueba.parquet")

    # ---------------- Métricas ----------------
    metricas = {m: core.metricas(pred["Real"], pred[m], pred.index) for m in core.MODELOS}
    for m, v in metricas.items():
        log(f"  {m:17s} " + " | ".join(f"{k} {x:.4f}" for k, x in v.items()))

    # Referencia ingenua (persistencia): ŷ_{t+1} = y_t, sin modelo.
    persistencia = serie["Global_active_power"].astype("float64").reindex(pred["origen"]).to_numpy()
    referencia = core.metricas(pred["Real"], persistencia)

    lin = json.loads((ART / "lineal.json").read_text())
    resultados = {
        "dataset": {"filas": int(len(sup)), "train": int(len(tr)), "prueba": int(len(te)),
                    "inicio_prueba": str(te.index[0]), "fin_prueba": str(te.index[-1])},
        "metricas_prueba": metricas,
        "referencia_persistencia": referencia,
        "importancia": {"Random Forest": imp_rf, "XGBoost": imp_xgb,
                        "Regresión lineal": dict(zip(lin["features"], lin["coef"]))},
        "hiperparametros": {
            "Regresión lineal": "OLS sobre log(1+y), entradas estandarizadas, sin Global_intensity",
            "Random Forest": "",  # lo completa export_ensamble.py
            "XGBoost": "300 árboles, profundidad 6, η = 0,05, semilla 42",
            "LSTM": f"LSTM(64) → Dense(32, ReLU) → Dense(1); ventana 48; {args.epocas} épocas; batch 64; log(1+y)",
        },
    }
    (ART / "resultados.json").write_text(json.dumps(resultados, indent=1, ensure_ascii=False), encoding="utf-8")
    log("ensamble, métricas de entrenamiento y walk-forward")
    export_ensamble.main(reentrenar_rf=False, cortes=args.cortes)
    log("listo")


if __name__ == "__main__":
    main()
