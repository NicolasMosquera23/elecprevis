"""
Dos comprobaciones de robustez; no cambian ninguna decisión ni el ensamble.
Guarda sus resultados en app/artifacts/comprobaciones.json, que la app muestra.

Uso (desde la raíz del repo; necesita el dataset crudo de src/download_data.py):
    python app/comprobaciones.py

1. Formas de combinar en orden temporal. export_ensamble.py compara el promedio,
   el ponderado y el stacking estimando sus parámetros con una mitad del año de
   validación y evaluando en la otra, en los dos sentidos. En el sentido
   segunda → primera mitad, los parámetros salen de datos posteriores a los que
   se evalúan, lo que no reproduce un pronóstico. Aquí se estiman solo con la
   primera mitad y las tres formas se evalúan en la segunda, con las mismas
   filas, y se aplica la misma regla de core.py.
2. Prueba sin consumos rellenados. La imputación (forward fill) también rellena
   minutos del periodo de prueba, así que algunos valores reales con los que se
   evalúa son rellenos. Aquí se cuentan, con los datos minuto a minuto, los
   intervalos de prueba total o parcialmente rellenados y los incompletos
   (menos de 30 minutos medidos), y se recalculan las métricas solo con los
   intervalos completos y medidos.
"""

import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "app"))

from src import preprocessing as pp  # noqa: E402
import core  # noqa: E402
from export_ensamble import ajustar_combinaciones  # noqa: E402

ART = core.ART
E = core.ENSAMBLE
METRICAS = ["MAE", "RMSE", "R2", "Precisión ±1 kW"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def orden_temporal(val) -> dict:
    """Parámetros con la primera mitad de la validación; las tres formas, evaluadas en la segunda."""
    mitad = len(val) // 2
    cal, ev = val.iloc[:mitad], val.iloc[mitad:]
    params = ajustar_combinaciones(cal)
    res = {}
    for metodo in core.COMBINACIONES:
        m = core.metricas(ev["Real"], core.combinar(ev, metodo, params.get(metodo)))
        res[metodo] = {k: m[k] for k in METRICAS}
    rmse = {k: v["RMSE"] for k, v in res.items()}
    mejor = min(rmse.values())
    elegida = next(k for k in core.COMBINACIONES if rmse[k] <= mejor * (1 + core.TOLERANCIA_SIMPLE))
    return {
        "ajuste": [str(cal.index[0]), str(cal.index[-1])],
        "evaluacion": [str(ev.index[0]), str(ev.index[-1])],
        "n_evaluacion": len(ev),
        "combinaciones": res,
        "elegida": elegida,
    }


def prueba_sin_rellenos(pred) -> dict:
    """Clasifica los intervalos de prueba según sus minutos medidos y rellenados."""
    crudo = pp.load_raw()
    falta = crudo[pp.NUMERIC_COLS].isna().any(axis=1)  # en el crudo faltan todas las columnas a la vez
    minutos = falta.resample("30min").size().reindex(pred.index)
    rellenos = falta.resample("30min").sum().reindex(pred.index)
    total = rellenos == minutos
    parcial = (rellenos > 0) & (rellenos < minutos)
    incompleto = minutos < 30
    completo = ~(total | parcial | incompleto)
    modelos = core.MODELOS + [E]
    return {
        "intervalos": len(pred),
        "totalmente_rellenados": int(total.sum()),
        "parcialmente_rellenados": int(parcial.sum()),
        "incompletos": int(incompleto.sum()),
        "completos_y_medidos": int(completo.sum()),
        "meses_totalmente_rellenados": sorted(set(pred.index[total].strftime("%Y-%m"))),
        "todos": {m: {k: core.metricas(pred["Real"], pred[m])[k] for k in METRICAS} for m in modelos},
        "completos": {m: {k: core.metricas(pred["Real"][completo], pred[m][completo])[k] for k in METRICAS}
                      for m in modelos},
    }


def main():
    import pandas as pd

    val = pd.read_parquet(ART / "predicciones_validacion.parquet")
    pred = pd.read_parquet(ART / "predicciones_prueba.parquet")

    t = orden_temporal(val)
    log("orden temporal: " + ", ".join(f"{k} {v['RMSE']:.4f}" for k, v in t["combinaciones"].items())
        + f" → {t['elegida']}")
    r = prueba_sin_rellenos(pred)
    log(f"prueba: {r['totalmente_rellenados']} totalmente rellenados, {r['parcialmente_rellenados']} parcialmente, "
        f"{r['incompletos']} incompleto(s); quedan {r['completos_y_medidos']}")
    log(f"  ensamble RMSE {r['todos'][E]['RMSE']:.4f} → {r['completos'][E]['RMSE']:.4f}, "
        f"±1 kW {r['todos'][E]['Precisión ±1 kW']:.2f} → {r['completos'][E]['Precisión ±1 kW']:.2f} %")

    salida = {"orden_temporal": t, "prueba_sin_rellenos": r}
    (ART / "comprobaciones.json").write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf-8")
    log("listo")


if __name__ == "__main__":
    main()
