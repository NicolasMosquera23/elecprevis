"""
Entrena y compara los cuatro modelos de la Fase 4 sobre el dataset procesado.

Uso:
    python src/run_models.py                    # los 3 modelos rápidos, partición única + walk-forward
    python src/run_models.py --lstm             # agrega la LSTM (20 épocas)
    python src/run_models.py --lstm --epocas 10
    python src/run_models.py --sin-walk-forward # solo la partición 80/20
    python src/run_models.py --salida results/fase4.json

Antes de correrlo hay que tener el dataset procesado:
    python src/download_data.py && python src/preprocessing.py

El script no reimplementa nada del modelado: importa src.models y src.features y
se limita a orquestar el protocolo, cronometrar y reportar.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    # Permite `python src/run_models.py` desde la raíz del repo: al ejecutar un
    # archivo dentro de src/, Python pone src/ en el path y no la raíz.
    sys.path.insert(0, str(RAIZ))

from src import features as ft
from src import models as md

PROCESADO = RAIZ / "data" / "processed" / "power_30min.csv"


def cronometrar(etiqueta: str, funcion, *args, **kwargs):
    """Ejecuta `funcion` y devuelve (resultado, segundos)."""
    t0 = time.time()
    resultado = funcion(*args, **kwargs)
    segundos = time.time() - t0
    print(f"  {etiqueta}: {segundos:.0f} s", flush=True)
    return resultado, segundos


def particion_unica(X_tr, y_tr, X_te, y_te, con_lstm: bool, epocas: int) -> dict:
    """Entrena los cuatro modelos sobre el 80 % inicial y evalúa en el 20 % final."""
    resultados = {}

    def corre_lineal():
        bundle = md.fit_linear_regression(X_tr, y_tr, log_target=True)
        return md.evaluate(y_te, md.predict_linear(bundle, X_te))

    def corre_rf():
        modelo = md.fit_random_forest(X_tr, y_tr)
        return md.evaluate(y_te, modelo.predict(X_te))

    def corre_xgb():
        modelo = md.fit_xgboost(X_tr, y_tr)
        return md.evaluate(y_te, modelo.predict(X_te))

    print("\nPartición única (entrenamiento 80 % / prueba 20 %):")
    resultados["Regresión lineal"], _ = cronometrar("regresión lineal", corre_lineal)
    resultados["Random Forest"], _ = cronometrar("random forest", corre_rf)
    resultados["XGBoost"], _ = cronometrar("xgboost", corre_xgb)

    if con_lstm:
        seq_tr, ys_tr = md.build_lstm_sequences(X_tr.values, y_tr.values, n_steps=48)
        seq_te, ys_te = md.build_lstm_sequences(X_te.values, y_te.values, n_steps=48)
        print(f"  secuencias: train {seq_tr.shape} | test {seq_te.shape}")

        def corre_lstm():
            bundle = md.fit_lstm(seq_tr, ys_tr, n_features=seq_tr.shape[2], epochs=epocas)
            return md.evaluate(ys_te, md.predict_lstm(bundle, seq_te))

        resultados["LSTM"], _ = cronometrar(f"lstm ({epocas} épocas)", corre_lstm)

    return resultados


def walk_forward(X_tr, y_tr, con_lstm: bool, epocas: int, n_splits: int) -> dict:
    """Validación de ventana expansiva sobre la partición de entrenamiento."""
    print(f"\nValidación de ventana expansiva ({n_splits} cortes):")
    resultados = {}

    estrategias = {
        "Regresión lineal": lambda a, b, c: md.predict_linear(
            md.fit_linear_regression(a, b, log_target=True), c),
        "Random Forest": lambda a, b, c: md.fit_random_forest(a, b).predict(c),
        "XGBoost": lambda a, b, c: md.fit_xgboost(a, b).predict(c),
    }
    for nombre, estrategia in estrategias.items():
        df, _ = cronometrar(nombre, md.walk_forward_evaluate, estrategia, X_tr, y_tr,
                            n_splits=n_splits)
        resultados[nombre] = df
        print(df[["corte", "n_train", "n_test", "MAE", "RMSE", "R2", "sMAPE"]].round(3).to_string(index=False))
        print(f"    promedio RMSE {df['RMSE'].mean():.4f} | MAE {df['MAE'].mean():.4f} "
              f"| R2 {df['R2'].mean():.4f}\n")

    if con_lstm:
        print("  (la LSTM se omite del walk-forward por costo de cómputo)")
    return resultados


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lstm", action="store_true", help="entrena también la LSTM")
    ap.add_argument("--epocas", type=int, default=20, help="épocas de la LSTM (por defecto 20)")
    ap.add_argument("--cortes", type=int, default=5, help="cortes del walk-forward (0 para omitir)")
    ap.add_argument("--sin-walk-forward", action="store_true")
    ap.add_argument("--salida", type=Path, default=RAIZ / "results" / "fase4_resultados.json")
    args = ap.parse_args()

    if not PROCESADO.exists():
        raise SystemExit(
            f"Falta {PROCESADO}. Corre primero:\n"
            "  python src/download_data.py\n  python src/preprocessing.py"
        )

    md.set_global_seed()
    print(f"Leyendo {PROCESADO.name} ...")
    df = pd.read_csv(PROCESADO, index_col=0, parse_dates=True)
    sup = ft.build_supervised_dataset(df)
    tr, te = ft.temporal_train_test_split(sup)
    X_tr, y_tr = md.split_xy(tr)
    X_te, y_te = md.split_xy(te)
    print(f"dataset supervisado: {len(sup):,} filas | train {len(tr):,} | prueba {len(te):,}".replace(",", "."))

    salida = {"dataset": {"filas": int(len(sup)), "train": int(len(tr)), "prueba": int(len(te))}}

    salida["particion_unica"] = particion_unica(X_tr, y_tr, X_te, y_te, args.lstm, args.epocas)
    print("\nResumen de la partición única:")
    print(pd.DataFrame(salida["particion_unica"]).T.round(4).to_string())

    if not args.sin_walk_forward and args.cortes > 0:
        wf = walk_forward(X_tr, y_tr, args.lstm, args.epocas, args.cortes)
        salida["walk_forward"] = {k: v.round(4).to_dict(orient="records") for k, v in wf.items()}

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados guardados en {args.salida}")


if __name__ == "__main__":
    main()
