"""
Arma el ensamble de los cuatro modelos y completa resultados.json.

Uso (desde la raíz del repo):
    python app/export_ensamble.py            # reentrena el random forest liviano y todo lo demás
    python app/export_ensamble.py --sin-rf   # usa el random_forest.joblib que ya existe

No necesita el dataset crudo ni TensorFlow: parte de app/artifacts/ (serie
procesada, lineal.json, xgboost.json, lstm.npz y predicciones_validacion.parquet,
que genera app/validacion.py). app/train_export.py y app/validacion.py lo llaman
al final.

Qué calcula:
    1. Random forest liviano (core.RF_PARAMS) y sus predicciones de prueba.
    2. Decisiones en la validación (último año del entrenamiento), con las reglas
       de core.py: forma de combinar (promedio, ponderado o stacking) y sus
       parámetros, umbral de alerta de pico de cada modelo y adopción o no del
       ajuste "más peso a los picos".
    3. Ensamble en la prueba con lo decidido en validación, y métricas de
       entrenamiento y de prueba (registro de desempeño train / test).
    4. Casos críticos en la prueba: recall, precisión, F1 y PR-AUC de los picos,
       antes y después de ajustar el umbral, y errores por hora del día.
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
SUF = core.SUFIJO_PESO


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pct(x: float) -> str:
    """0.005 → '0,5 %' (coma decimal, como en la app)."""
    return f"{100 * x:g} %".replace(".", ",")


def predecir_lstm_lote(lstm, X: np.ndarray, desde: int, lote: int = 4000) -> np.ndarray:
    """Predicción a un paso de la LSTM para las filas desde..fin de X (ventanas de 48)."""
    salidas = []
    for ini in range(desde, len(X), lote):
        fin = min(ini + lote, len(X))
        idx = np.arange(ini, fin)[:, None] + np.arange(-core.N_STEPS + 1, 1)[None, :]
        salidas.append(lstm.predict(X[idx]))
    return np.concatenate(salidas)


# --------------------------------------------------------------------------
# Decisiones en validación
# --------------------------------------------------------------------------
def ajustar_combinaciones(cal: pd.DataFrame) -> dict:
    """Parámetros del ponderado (mínimos cuadrados no negativos, normalizados) y del stacking."""
    M, y = cal[core.MODELOS].to_numpy(), cal["Real"].to_numpy()
    w, _ = nnls(M, y)
    w = w / w.sum()
    stk = LinearRegression().fit(M, y)
    return {
        "Ponderado": {"pesos": dict(zip(core.MODELOS, w.tolist()))},
        "Stacking": {"coef": dict(zip(core.MODELOS, stk.coef_.tolist())), "intercepto": float(stk.intercept_)},
    }


def combinaciones_fuera_de_muestra(val: pd.DataFrame) -> pd.DataFrame:
    """
    Predicción de las tres combinaciones en la validación sin ajustar y evaluar con
    el mismo dato: los parámetros se estiman con una mitad del año y se aplican a la
    otra, y luego al revés, de modo que todo el año queda evaluado.
    """
    mitad = len(val) // 2
    out = pd.DataFrame(index=val.index, columns=list(core.COMBINACIONES), dtype=float)
    for cal, ev in [(val.iloc[:mitad], val.iloc[mitad:]), (val.iloc[mitad:], val.iloc[:mitad])]:
        params = ajustar_combinaciones(cal)
        for metodo in core.COMBINACIONES:
            out.loc[ev.index, metodo] = core.combinar(ev, metodo, params.get(metodo))
    return out


def resumen_pico(y, yhat) -> dict:
    """RMSE y desempeño en picos con el umbral de mayor F1 (solo para validación)."""
    u = core.mejor_umbral(y, yhat)
    return {"RMSE": core.metricas(y, yhat)["RMSE"], **core.metricas_pico(y, yhat, u)}


def decidir(val: pd.DataFrame) -> dict:
    y = val["Real"]
    oof = combinaciones_fuera_de_muestra(val)
    comparacion = {m: core.metricas(y, val[m]) for m in ["Persistencia"] + core.MODELOS}
    comparacion.update({f"{E} · {m}": core.metricas(y, oof[m]) for m in core.COMBINACIONES})

    # Regla: menor RMSE; ante una diferencia menor que TOLERANCIA_SIMPLE, la forma más simple.
    rmse = {m: comparacion[f"{E} · {m}"]["RMSE"] for m in core.COMBINACIONES}
    mejor = min(rmse.values())
    elegida = next(m for m in core.COMBINACIONES if rmse[m] <= mejor * (1 + core.TOLERANCIA_SIMPLE))
    parametros = ajustar_combinaciones(val)

    # Regla: umbral de alerta con mayor F1 en validación, uno por modelo o combinación.
    puntajes = {"Persistencia": val["Persistencia"], **{m: val[m] for m in core.MODELOS},
                **{f"{E} · {m}": oof[m] for m in core.COMBINACIONES}}
    umbrales = {m: core.mejor_umbral(y, v) for m, v in puntajes.items()}
    umbrales[E] = umbrales[f"{E} · {elegida}"]
    pico_val = {m: core.metricas_pico(y, v, umbrales[m]) for m, v in puntajes.items()}

    # Ajuste controlado: más peso a los picos en lineal, RF y XGBoost (la LSTM queda igual).
    con_peso = ["Regresión lineal", "Random Forest", "XGBoost"]
    val_p = val[["Real", "LSTM"]].join(val[[m + SUF for m in con_peso]].rename(columns=lambda c: c.replace(SUF, "")))
    oof_p = combinaciones_fuera_de_muestra(val_p)[elegida]
    experimento = {m: {"base": resumen_pico(y, val[m]), "peso": resumen_pico(y, val[m + SUF])} for m in con_peso}
    experimento[E] = {"base": resumen_pico(y, oof[elegida]), "peso": resumen_pico(y, oof_p)}
    b, p = experimento[E]["base"], experimento[E]["peso"]
    adoptar = p["F1"] - b["F1"] >= core.MEJORA_F1 and p["RMSE"] <= b["RMSE"] * (1 + core.PEOR_RMSE)

    log(f"validación: RMSE {', '.join(f'{m} {v:.4f}' for m, v in rmse.items())} → {elegida}")
    log(f"  umbral de alerta del ensamble {umbrales[E]:.2f} kW | peso picos: F1 {b['F1']:.3f} → {p['F1']:.3f}, "
        f"RMSE {b['RMSE']:.4f} → {p['RMSE']:.4f} → {'se adopta' if adoptar else 'no se adopta'}")
    return {
        "combinacion": {"metodo": elegida, "params": parametros.get(elegida)},
        "parametros": parametros,
        "comparacion": comparacion,
        "umbrales": umbrales,
        "pico": pico_val,
        "experimento_peso_picos": {"peso": core.PESO_PICOS, "modelos": experimento, "adoptado": bool(adoptar)},
    }


def formula(comb: dict) -> str:
    abrev = {"Regresión lineal": "lineal", "Random Forest": "RF", "XGBoost": "XGB", "LSTM": "LSTM"}
    if comb["metodo"] == "Promedio":
        return "ŷ_ens = (ŷ_lineal + ŷ_RF + ŷ_XGB + ŷ_LSTM) / 4"
    if comb["metodo"] == "Ponderado":
        w = comb["params"]["pesos"]
        return "ŷ_ens = " + " + ".join(f"{w[m]:.3f}·ŷ_{abrev[m]}" for m in core.MODELOS)
    c = comb["params"]
    return f"ŷ_ens = {c['intercepto']:.3f} + " + " + ".join(f"{c['coef'][m]:.3f}·ŷ_{abrev[m]}" for m in core.MODELOS)


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
    val = pd.read_parquet(ART / "predicciones_validacion.parquet")
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

    # ---------------- 2. Decisiones en validación ----------------
    log("decisiones en la validación")
    d = decidir(val)
    comb = d["combinacion"]

    # ---------------- 3. Ensamble y métricas de entrenamiento / prueba ----------------
    # La LSTM necesita 48 filas de historia: el entrenamiento se mide desde la
    # fila 48 para los cinco (mismas filas para todos).
    log("predicciones de entrenamiento")
    desde = core.N_STEPS - 1
    p_tr = pd.DataFrame({"Real": y_tr.to_numpy()[desde:]})
    p_tr["Regresión lineal"] = modelos["Regresión lineal"].predict(X_tr)[desde:]
    p_tr["Random Forest"] = rf.predict(X_tr)[desde:]
    p_tr["XGBoost"] = np.clip(modelos["XGBoost"].predict(X_tr), 0, None)[desde:]
    p_tr["LSTM"] = predecir_lstm_lote(modelos["LSTM"], X_tr[core.FEATURES].to_numpy(float), desde)
    p_tr[E] = core.ensamble(p_tr, comb)
    pred[E] = core.ensamble(pred, comb)
    # Persistencia: ŷ_{t+1} = y_t, la referencia sin modelo.
    pred["Persistencia"] = serie["Global_active_power"].reindex(pred["origen"]).to_numpy()
    pred.to_parquet(ART / "predicciones_prueba.parquet")

    nombres = core.MODELOS + [E]
    resultados["metricas_entrenamiento"] = {m: core.metricas(p_tr["Real"], p_tr[m]) for m in nombres}
    resultados["metricas_prueba"] = {m: core.metricas(pred["Real"], pred[m]) for m in nombres}
    for m in nombres:
        a, b = resultados["metricas_entrenamiento"][m], resultados["metricas_prueba"][m]
        log(f"  {m:17s} R² train {a['R2']:.4f} | test {b['R2']:.4f} | prec ±1 kW test {b['Precisión ±1 kW']:.2f} %")

    # Las tres formas de combinar en la prueba, con los parámetros de la validación
    # (informativo: la elección ya se hizo en validación).
    comparacion = {m: resultados["metricas_prueba"][m] for m in core.MODELOS}
    for metodo in core.COMBINACIONES:
        comparacion[f"{E} · {metodo}"] = core.metricas(pred["Real"], core.combinar(pred, metodo, d["parametros"].get(metodo)))

    # ---------------- 4. Casos críticos en la prueba ----------------
    y = pred["Real"].to_numpy()
    puntajes = {"Persistencia": pred["Persistencia"], **{m: pred[m] for m in core.MODELOS},
                **{f"{E} · {m}": core.combinar(pred, m, d["parametros"].get(m)) for m in core.COMBINACIONES}}
    puntajes[E] = pred[E]
    pico_prueba = {
        m: {"umbral_2kW": core.metricas_pico(y, v, core.UMBRAL_PICO_KW),
            "umbral_validacion": core.metricas_pico(y, v, d["umbrales"][m])}
        for m, v in puntajes.items()
    }
    real_pico, alerta = y > core.UMBRAL_PICO_KW, pred[E].to_numpy() > d["umbrales"][E]
    hora = pd.Series(pred.index.hour, index=pred.index)
    por_hora = pd.DataFrame({"picos": real_pico, "FN": real_pico & ~alerta, "FP": ~real_pico & alerta}).groupby(hora.to_numpy()).sum()
    constante = float(y_tr.mean())
    resultados["casos_criticos"] = {
        "definicion": f"Pico: consumo real del intervalo > {core.UMBRAL_PICO_KW} kW",
        "proporcion_prueba": float(real_pico.mean()),
        "proporcion_validacion": float((val["Real"] > core.UMBRAL_PICO_KW).mean()),
        "proporcion_entrenamiento": float((y_tr > core.UMBRAL_PICO_KW).mean()),
        "constante": {"prediccion_kW": constante, **core.metricas(y, np.full_like(y, constante))},
        "prueba": pico_prueba,
        "errores_por_hora": {int(h): {k: int(v) for k, v in fila.items()} for h, fila in por_hora.iterrows()},
    }
    for m in ["Persistencia"] + nombres:
        a, b = pico_prueba[m]["umbral_2kW"], pico_prueba[m]["umbral_validacion"]
        log(f"  picos {m:17s} recall {a['Recall']:.3f} → {b['Recall']:.3f} | F1 {a['F1']:.3f} → {b['F1']:.3f} "
            f"(u = {b['Umbral de alerta (kW)']:.2f}) | PR-AUC {b['PR-AUC']:.3f}")

    # ---------------- 5. Diversidad ----------------
    errores = pd.DataFrame({m: pred["Real"] - pred[m] for m in core.MODELOS})

    resultados["validacion"] = {
        "sub_entrenamiento": {"desde": str(X_tr.index[0]), "hasta": str(val["origen"].iloc[0] - pd.Timedelta(minutes=30)),
                              "n": int(len(X_tr) - len(val))},
        "validacion": {"desde": str(val["origen"].iloc[0]), "hasta": str(val["origen"].iloc[-1]), "n": int(len(val))},
        "reglas": {
            "combinacion": f"menor RMSE en validación; si una forma más simple queda a menos de "
                           f"{pct(core.TOLERANCIA_SIMPLE)} del mejor RMSE, se elige la más simple",
            "umbral": "umbral de alerta con mayor F1 de la clase pico en validación",
            "peso_picos": f"se adopta si el F1 de validación sube al menos {core.MEJORA_F1:.2f}".replace(".", ",")
                          + f" y el RMSE no empeora más de {pct(core.PEOR_RMSE)}",
        },
        "comparacion": d["comparacion"],
        "umbrales": d["umbrales"],
        "pico": d["pico"],
        "experimento_peso_picos": d["experimento_peso_picos"],
    }
    resultados["ensamble"] = {
        "formula": formula(comb),
        "combinacion": comb,
        "parametros": d["parametros"],
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
    resultados["hiperparametros"][E] = (
        f"{comb['metodo']} de los 4 modelos, elegido en validación"
        + ("" if comb["metodo"] == "Promedio" else " (parámetros estimados en el año de validación)")
        + f"; alerta de pico si ŷ > {d['umbrales'][E]:.2f} kW"
    )
    (ART / "resultados.json").write_text(json.dumps(resultados, indent=1, ensure_ascii=False), encoding="utf-8")
    log("listo")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-rf", action="store_true", help="usa el random_forest.joblib existente")
    ap.add_argument("--cortes", type=int, default=5)
    args = ap.parse_args()
    main(reentrenar_rf=not args.sin_rf, cortes=args.cortes)
