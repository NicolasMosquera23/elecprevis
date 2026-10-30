"""
Arma el ensamble de los cuatro modelos y completa resultados.json.

Uso (desde la raíz del repo):
    python app/export_ensamble.py            # reentrena el random forest liviano y todo lo demás
    python app/export_ensamble.py --sin-rf   # usa el random_forest.joblib que ya existe

No necesita el dataset crudo ni TensorFlow: parte de app/artifacts/ (serie
procesada, lineal.json, xgboost.json, lstm.npz). app/train_export.py lo llama al
final, así que después de reentrenar todo no hace falta correrlo aparte.

Qué calcula:
    1. Random forest liviano (core.RF_PARAMS) y sus predicciones de prueba.
    2. Métricas de entrenamiento y de prueba de los 4 modelos y del ensamble,
       para revisar sobreajuste (registro de desempeño train / test).
    3. Ensamble por promedio sobre todo el periodo de prueba.
    4. Comparación de tres formas de combinar (promedio, ponderado, stacking):
       los parámetros se estiman en la primera mitad de la prueba y las tres se
       evalúan en la segunda mitad.
    5. Correlación entre los errores de los modelos (la diversidad que hace
       funcionar el ensamble).
    6. Validación de ventana expansiva de lineal, RF, XGBoost y su promedio
       (la LSTM se omite por costo de cómputo, igual que en run_models.py).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "app"))

from src import features as ft  # noqa: E402
from src import models as md  # noqa: E402
import core  # noqa: E402

ART = core.ART
E = core.ENSAMBLE


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def predecir_lstm_lote(lstm, X: np.ndarray, desde: int, lote: int = 4000) -> np.ndarray:
    """Predicción a un paso de la LSTM para las filas desde..fin de X (ventanas de 48)."""
    salidas = []
    for ini in range(desde, len(X), lote):
        fin = min(ini + lote, len(X))
        idx = np.arange(ini, fin)[:, None] + np.arange(-core.N_STEPS + 1, 1)[None, :]
        salidas.append(lstm.predict(X[idx]))
    return np.concatenate(salidas)


def main(reentrenar_rf: bool = True, cortes: int = 5):
    md.set_global_seed()
    serie = core.cargar_serie().astype("float64")
    sup = ft.build_supervised_dataset(serie)
    tr, te = ft.temporal_train_test_split(sup)
    X_tr, y_tr = md.split_xy(tr)
    X_te, y_te = md.split_xy(te)
    assert list(X_tr.columns) == core.FEATURES

    pred = pd.read_parquet(ART / "predicciones_prueba.parquet")
    assert np.allclose(pred["Real"].to_numpy(), y_te.to_numpy())
    resultados = json.loads((ART / "resultados.json").read_text(encoding="utf-8"))

    # ---------------- 1. Random forest liviano ----------------
    import joblib

    rf_path = ART / "random_forest.joblib"
    if reentrenar_rf or not rf_path.exists():
        log(f"random forest {core.RF_PARAMS}")
        rf = md.fit_random_forest(X_tr, y_tr, **dict(core.RF_PARAMS))
        joblib.dump(rf, rf_path, compress=("lzma", 6))
        log(f"  random_forest.joblib: {rf_path.stat().st_size / 1e6:.1f} MB")
    else:
        rf = joblib.load(rf_path)
    pred["Random Forest"] = rf.predict(X_te)
    resultados["importancia"]["Random Forest"] = dict(zip(core.FEATURES, rf.feature_importances_.tolist()))

    modelos = {
        "Regresión lineal": core.LinealOLS(json.loads((ART / "lineal.json").read_text())),
        "Random Forest": rf,
        "LSTM": core.LSTMNumpy(np.load(ART / "lstm.npz")),
    }
    from xgboost import XGBRegressor

    modelos["XGBoost"] = XGBRegressor()
    modelos["XGBoost"].load_model(ART / "xgboost.json")

    # ---------------- 2. Métricas de entrenamiento ----------------
    # La LSTM necesita 48 filas de historia: el entrenamiento se mide desde la
    # fila 48 para los cinco (mismas filas para todos).
    log("predicciones de entrenamiento")
    desde = core.N_STEPS - 1
    p_tr = pd.DataFrame({"Real": y_tr.to_numpy()[desde:]})
    p_tr["Regresión lineal"] = modelos["Regresión lineal"].predict(X_tr)[desde:]
    p_tr["Random Forest"] = rf.predict(X_tr)[desde:]
    p_tr["XGBoost"] = np.clip(modelos["XGBoost"].predict(X_tr), 0, None)[desde:]
    p_tr["LSTM"] = predecir_lstm_lote(modelos["LSTM"], X_tr[core.FEATURES].to_numpy(float), desde)
    p_tr[E] = core.ensamble_promedio(p_tr)
    pred[E] = core.ensamble_promedio(pred)
    pred.to_parquet(ART / "predicciones_prueba.parquet")

    nombres = core.MODELOS + [E]
    resultados["metricas_entrenamiento"] = {m: core.metricas(p_tr["Real"], p_tr[m]) for m in nombres}
    resultados["metricas_prueba"] = {m: core.metricas(pred["Real"], pred[m]) for m in nombres}
    for m in nombres:
        a, b = resultados["metricas_entrenamiento"][m], resultados["metricas_prueba"][m]
        log(f"  {m:17s} R² train {a['R2']:.4f} | test {b['R2']:.4f} | prec ±1 kW test {b['Precisión ±1 kW']:.2f} %")

    # ---------------- 4. Formas de combinar ----------------
    mitad = len(pred) // 2
    cal, ev = pred.iloc[:mitad], pred.iloc[mitad:]
    w, _ = nnls(cal[core.MODELOS].to_numpy(), cal["Real"].to_numpy())
    w = w / w.sum()
    stk = LinearRegression().fit(cal[core.MODELOS].to_numpy(), cal["Real"].to_numpy())
    params = {
        "Ponderado": {"pesos": dict(zip(core.MODELOS, w.tolist()))},
        "Stacking": {"coef": dict(zip(core.MODELOS, stk.coef_.tolist())), "intercepto": float(stk.intercept_)},
    }
    comparacion = {m: core.metricas(ev["Real"], ev[m]) for m in core.MODELOS}
    for metodo in core.COMBINACIONES:
        comparacion[f"{E} · {metodo}"] = core.metricas(ev["Real"], core.combinar(ev, metodo, params.get(metodo)))
    log("formas de combinar (segunda mitad de la prueba):")
    for k, v in comparacion.items():
        log(f"  {k:28s} R² {v['R2']:.4f} | MAE {v['MAE']:.4f} | prec {v['Precisión ±1 kW']:.2f} %")

    # ---------------- 5. Diversidad ----------------
    errores = pd.DataFrame({m: pred["Real"] - pred[m] for m in core.MODELOS})

    resultados["ensamble"] = {
        "formula": "ŷ_ens = (ŷ_lineal + ŷ_RF + ŷ_XGB + ŷ_LSTM) / 4",
        "parametros": params,
        "calibracion": {"desde": str(cal.index[0]), "hasta": str(cal.index[-1]), "n": int(len(cal))},
        "evaluacion": {"desde": str(ev.index[0]), "hasta": str(ev.index[-1]), "n": int(len(ev))},
        "comparacion_combinaciones": comparacion,
        "correlacion_errores": errores.corr().round(4).to_dict(),
    }

    # ---------------- 6. Walk-forward ----------------
    if cortes > 0:
        log(f"walk-forward ({cortes} cortes)")
        filas = {m: [] for m in ["Regresión lineal", "Random Forest", "XGBoost", "Ensamble (sin LSTM)"]}
        for k, (_, fin, ini, fin_te) in enumerate(md.walk_forward_folds(len(X_tr), cortes), start=1):
            a, b = X_tr.iloc[:fin], y_tr.iloc[:fin]
            c, yc = X_tr.iloc[ini:fin_te], y_tr.iloc[ini:fin_te]
            p = {
                "Regresión lineal": md.predict_linear(md.fit_linear_regression(a, b), c),
                "Random Forest": md.fit_random_forest(a, b, **dict(core.RF_PARAMS)).predict(c),
                "XGBoost": md.fit_xgboost(a, b).predict(c),
            }
            p["Ensamble (sin LSTM)"] = np.mean([p[m] for m in ["Regresión lineal", "Random Forest", "XGBoost"]], axis=0)
            for m, v in p.items():
                fila = {kk: round(vv, 4) for kk, vv in md.evaluate(yc, v).items()}
                fila.update({"corte": k, "n_train": len(a), "n_test": len(c),
                             "desde": str(X_tr.index[ini])[:16], "hasta": str(X_tr.index[fin_te - 1])[:16]})
                filas[m].append(fila)
            log("  corte %d: " % k + " | ".join(f"{m} R² {filas[m][-1]['R2']:.3f}" for m in filas))
        resultados["walk_forward"] = filas

    resultados["hiperparametros"]["Random Forest"] = (
        f"{core.RF_PARAMS['n_estimators']} árboles, mínimo {core.RF_PARAMS['min_samples_leaf']} "
        "observaciones por hoja, semilla 42"
    )
    resultados["hiperparametros"][E] = "Promedio simple de los 4 modelos (sin parámetros que entrenar)"
    (ART / "resultados.json").write_text(json.dumps(resultados, indent=1, ensure_ascii=False), encoding="utf-8")
    log("listo")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-rf", action="store_true", help="usa el random_forest.joblib existente")
    ap.add_argument("--cortes", type=int, default=5)
    args = ap.parse_args()
    main(reentrenar_rf=not args.sin_rf, cortes=args.cortes)
