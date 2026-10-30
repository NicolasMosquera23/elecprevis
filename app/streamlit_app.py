"""
ElecPrevis · Aplicación (Entregable 2, Parte 5: Implementación de la aplicación)

Ejecutar desde la raíz del repo:
    pip install -r app/requirements.txt
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

st.set_page_config(page_title="ElecPrevis", page_icon="⚡", layout="wide")

# --------------------------------------------------------------------------
# Utilidades de formato (convención colombiana: punto miles, coma decimales)
# --------------------------------------------------------------------------
def fmt(x, d=2):
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def cop(x):
    return "$ " + fmt(x, 0)


def estilo(fig, alto=380, leyenda=True):
    fig.update_layout(
        height=alto, separators=",.", template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified", showlegend=leyenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(color="#0b0b0b"),
    )
    fig.update_xaxes(gridcolor="#ecebe8")
    fig.update_yaxes(gridcolor="#ecebe8", zerolinecolor="#d6d5d0")
    return fig


@st.cache_data(show_spinner=False)
def serie():
    return core.cargar_serie().astype("float64")


@st.cache_data(show_spinner=False)
def predicciones():
    return core.cargar_predicciones()


@st.cache_data(show_spinner=False)
def resultados():
    return core.cargar_resultados()


@st.cache_resource(show_spinner="Cargando modelos…")
def modelos():
    return core.cargar_modelos()


S = serie()
P = predicciones()
R = resultados()
INI_PRUEBA = pd.Timestamp(R["dataset"]["inicio_prueba"])

ETIQ_PREC = "Precisión ±1 kW"
TOL = fmt(core.TOLERANCIA_KW, 1)
REF = R.get("referencia_persistencia", {})
E = core.ENSAMBLE
TODOS = [E] + core.MODELOS  # el ensamble primero: es el modelo principal de la app
ENS = R.get("ensamble", {})

FUNCIONALIDADES = [
    ("Histórico", "F1. Serie histórica interactiva con rango de fechas y resolución ajustable"),
    ("Histórico", "F2. Indicadores del periodo: energía, potencia media, pico y costo"),
    ("Histórico", "F3. Patrones de consumo: mapa de calor hora × día y perfil mensual"),
    ("Predicción", "F4. Predicción del siguiente intervalo (30 min) con el ensamble y los 4 modelos"),
    ("Predicción", "F5. Pronóstico encadenado de 24 h con selector de modelo (incluye el ensamble)"),
    ("Predicción", "F6. Calculadora: predicción a partir de condiciones ingresadas a mano"),
    ("Comparación", "F7. Tabla de métricas del ensamble y los 4 modelos (MAE, MSE, RMSE, R², sMAPE, precisión por tolerancia)"),
    ("Comparación", "F8. Real vs. predicho y error por hora del día en el periodo de prueba"),
    ("Comparación", "F9. Estabilidad por cortes de la validación de ventana expansiva"),
    ("Comparación", "F10. Registro de desempeño: métricas de entrenamiento frente a prueba (sobreajuste)"),
    ("Comparación", "F11. Ensamble: comparación de tres formas de combinar y diversidad de los errores"),
    ("Escenarios", "F12. Desglose del consumo por grupo de carga (submediciones)"),
    ("Escenarios", "F13. Simulador de escenarios de reducción sobre el consumo medido o el previsto por el ensamble"),
    ("Escenarios", "F14. Estimación de costo y ahorro en pesos con tarifa configurable"),
    ("Modelo", "F15. Importancia de variables y coeficientes de cada modelo"),
    ("Modelo", "F16. Defensa del modelo: variables, objetivo, entrenamiento, evaluación, errores y limitaciones"),
    ("Modelo", "F17. Trazabilidad ecuación → código del modelo matemático"),
    ("Todas", "F18. Descarga en CSV de datos, predicciones y escenarios"),
]

# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚡ ElecPrevis")
    st.caption("Previsión del consumo eléctrico residencial y escenarios de eficiencia energética")
    pagina = st.radio(
        "Sección",
        ["Inicio", "Histórico", "Predicción", "Comparación de modelos", "Escenarios de eficiencia", "Modelo"],
        label_visibility="collapsed",
    )
    st.divider()
    tarifa = st.number_input("Tarifa (COP/kWh)", min_value=100, max_value=3000,
                             value=core.TARIFA_DEFECTO, step=10,
                             help="Por defecto: Electrohuila, estrato 4, junio de 2026. Cámbiala según tu recibo.")
    st.caption("USCO · Electiva en Ingeniería Aplicada II · Grupo 8")


def descargar(df, nombre, etiqueta="Descargar CSV"):
    st.download_button(etiqueta, df.to_csv(sep=";", decimal=",").encode("utf-8-sig"),
                       file_name=nombre, mime="text/csv")


# ==========================================================================
# INICIO
# ==========================================================================
if pagina == "Inicio":
    st.title("ElecPrevis")
    st.markdown(
        "Predice el **consumo eléctrico de un hogar en los próximos 30 minutos** con un **ensamble de cuatro "
        "modelos** de machine learning y simula **escenarios hipotéticos de ahorro** a partir de las "
        "submediciones de cocina, lavandería y calentador de agua."
    )
    m = R["metricas_prueba"]
    mejor_ind = min(core.MODELOS, key=lambda k: m[k]["RMSE"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros del dataset", fmt(R["dataset"]["filas"], 0))
    c2.metric("Modelo principal", "Ensamble de los 4",
              help=f"Mejor modelo individual: {mejor_ind} (R² {fmt(m[mejor_ind]['R2'], 3)})")
    c3.metric(f"{ETIQ_PREC} · ensamble", fmt(m[E][ETIQ_PREC], 1) + " %",
              delta=fmt(m[E][ETIQ_PREC] - m[mejor_ind][ETIQ_PREC], 2) + f" pp vs. {mejor_ind}")
    c4.metric("R² · ensamble", fmt(m[E]["R2"], 3),
              delta=fmt(m[E]["R2"] - m[mejor_ind]["R2"], 3) + f" vs. {mejor_ind}")

    st.subheader("Cómo funciona")
    st.markdown(
        "1. **Datos:** 2.075.259 mediciones minuto a minuto de una vivienda (UCI, 2006–2010), "
        "promediadas en 69.177 intervalos de 30 minutos.\n"
        "2. **Variables:** 19 por intervalo: 7 mediciones, 8 de calendario y 4 de memoria del consumo "
        "(hace 30 min, ayer, la semana pasada y la media de 24 h).\n"
        "3. **Modelos:** regresión lineal, random forest, XGBoost y LSTM, entrenados con el 80 % más antiguo "
        "de la serie y evaluados en el 20 % final, que nunca vieron.\n"
        "4. **Ensamble:** la predicción final es el promedio de los cuatro modelos, igual que un random forest "
        "promedia sus árboles. Como los modelos son de tipos distintos, se equivocan en momentos distintos y "
        "al promediar parte del error se compensa.\n"
        "5. **Escenarios:** reducciones porcentuales supuestas sobre cada grupo de carga. Son hipotéticos: "
        "los datos son observacionales y no prueban causalidad."
    )
    st.subheader("Funcionalidades")
    st.dataframe(pd.DataFrame(FUNCIONALIDADES, columns=["Sección", "Funcionalidad"]),
                 hide_index=True, use_container_width=True)
    st.info(
        f"**Sobre la precisión.** Como el modelo predice un número (kW) y no una clase, la precisión se mide "
        f"por tolerancia: el porcentaje de predicciones que quedan a ±{TOL} kW o menos del valor real. "
        "La explicación completa está en *Comparación de modelos*."
    )

# ==========================================================================
# HISTÓRICO (F1, F2, F3)
# ==========================================================================
elif pagina == "Histórico":
    st.title("Histórico de consumo")
    min_d, max_d = S.index.min().date(), S.index.max().date()
    c1, c2 = st.columns([3, 1])
    rango = c1.date_input("Rango de fechas", value=(pd.Timestamp("2010-01-01").date(), pd.Timestamp("2010-01-31").date()),
                          min_value=min_d, max_value=max_d)
    res = c2.selectbox("Resolución", ["30 min", "Hora", "Día", "Semana"], index=1)
    if not isinstance(rango, tuple) or len(rango) != 2:
        st.stop()
    d0, d1 = pd.Timestamp(rango[0]), pd.Timestamp(rango[1]) + pd.Timedelta(days=1) - pd.Timedelta(minutes=30)
    sub = S.loc[d0:d1]
    if sub.empty:
        st.warning("No hay datos en ese rango.")
        st.stop()

    # F2
    energia = sub["Global_active_power"].sum() * core.HORAS_INTERVALO
    dias = max((d1 - d0).total_seconds() / 86400, 1 / 48)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Energía consumida", fmt(energia, 1) + " kWh")
    k2.metric("Potencia media", fmt(sub["Global_active_power"].mean(), 2) + " kW")
    k3.metric("Pico (media de 30 min)", fmt(sub["Global_active_power"].max(), 2) + " kW",
              help=f"Ocurrió el {sub['Global_active_power'].idxmax():%d/%m/%Y %H:%M}")
    k4.metric("Costo estimado", cop(energia * tarifa), help=f"≈ {cop(energia * tarifa / dias * 30)} por mes")

    # F1
    regla = {"30 min": None, "Hora": "h", "Día": "D", "Semana": "W"}[res]
    y = sub["Global_active_power"] if regla is None else sub["Global_active_power"].resample(regla).mean()
    fig = go.Figure(go.Scatter(x=y.index, y=y.values, mode="lines", name="Potencia activa",
                               line=dict(color=core.COLOR_MODELO["XGBoost"], width=2),
                               hovertemplate="%{y:.2f} kW<extra></extra>"))
    fig.update_yaxes(title="Potencia activa media (kW)", rangemode="tozero")
    st.plotly_chart(estilo(fig, leyenda=False), use_container_width=True)

    # F3
    st.subheader("Patrones de consumo")
    c1, c2 = st.columns(2)
    tmp = sub.assign(hora=sub.index.hour, dia=sub.index.dayofweek)
    mapa = tmp.pivot_table(index="dia", columns="hora", values="Global_active_power", aggfunc="mean")
    mapa.index = [core.DIAS[i] for i in mapa.index]
    figh = px.imshow(mapa, color_continuous_scale="Blues", aspect="auto",
                     labels=dict(x="Hora del día", y="", color="kW"))
    figh.update_traces(hovertemplate="%{y} %{x}:00 · %{z:.2f} kW<extra></extra>")
    c1.markdown("**Potencia media por hora y día de la semana**")
    c1.plotly_chart(estilo(figh, 320, leyenda=False).update_layout(hovermode="closest"), use_container_width=True)

    perfil = S.groupby(S.index.month)["Global_active_power"].mean()
    figm = go.Figure(go.Bar(x=[core.MESES[i - 1] for i in perfil.index], y=perfil.values,
                            marker_color=core.COLOR_MODELO["XGBoost"], hovertemplate="%{x}: %{y:.2f} kW<extra></extra>"))
    figm.update_yaxes(title="kW", rangemode="tozero")
    c2.markdown("**Potencia media por mes (serie completa, 2006–2010)**")
    c2.plotly_chart(estilo(figm, 320, leyenda=False), use_container_width=True)

    descargar(sub, f"elecprevis_historico_{rango[0]}_{rango[1]}.csv", "Descargar datos del rango (CSV)")

# ==========================================================================
# PREDICCIÓN (F4, F5, F6)
# ==========================================================================
elif pagina == "Predicción":
    st.title("Predicción")
    st.caption("Los instantes disponibles pertenecen al periodo de prueba (13/02/2010 – 26/11/2010), "
               "que ningún modelo vio durante el entrenamiento.")
    tab1, tab2, tab3 = st.tabs(["Siguiente intervalo (30 min)", "Pronóstico 24 h", "Calculadora"])

    with tab1:  # F4
        c1, c2 = st.columns(2)
        fecha = c1.date_input("Fecha", value=pd.Timestamp("2010-06-15").date(),
                              min_value=INI_PRUEBA.date(), max_value=P["origen"].max().date())
        horas = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
        hora = c2.selectbox("Hora actual (t)", horas, index=horas.index("19:00"))
        t = pd.Timestamp(f"{fecha} {hora}")
        fila = P[P["origen"] == t]
        if fila.empty:
            st.warning("No hay predicción para ese instante (inicio o fin de la serie).")
        else:
            fila = fila.iloc[0]
            obj = t + pd.Timedelta(minutes=30)
            st.markdown(f"Con la información disponible hasta **{t:%d/%m/%Y %H:%M}**, el consumo medio "
                        f"previsto para el intervalo **{obj:%H:%M}–{obj + pd.Timedelta(minutes=30):%H:%M}** es:")
            cols = st.columns(6)
            cols[0].metric("Real (medido)", fmt(fila["Real"], 3) + " kW")
            for c, m in zip(cols[1:], TODOS):
                err = fila[m] - fila["Real"]
                c.metric(f"**{m}**" if m == E else m, fmt(fila[m], 3) + " kW",
                         delta=("+" if err >= 0 else "") + fmt(err, 3) + " kW", delta_color="off")
            st.caption(f"Ensamble = ({' + '.join(fmt(fila[m], 3) for m in core.MODELOS)}) / 4 = "
                       f"{fmt(fila[E], 3)} kW: cada modelo aporta un voto, como los árboles de un random forest.")
            ventana = S.loc[t - pd.Timedelta(hours=12):t, "Global_active_power"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ventana.index, y=ventana.values, mode="lines", name="Histórico (hasta t)",
                                     line=dict(color=core.COLOR_MODELO["Real"], width=2)))
            fig.add_trace(go.Scatter(x=[obj], y=[fila["Real"]], mode="markers", name="Real en t+1",
                                     marker=dict(color=core.COLOR_MODELO["Real"], size=11, symbol="circle-open", line=dict(width=2))))
            for m in TODOS:
                fig.add_trace(go.Scatter(x=[obj], y=[fila[m]], mode="markers", name=m,
                                         marker=dict(color=core.COLOR_MODELO[m], size=14 if m == E else 10,
                                                     symbol="diamond" if m == E else "circle",
                                                     line=dict(color="white", width=2))))
            fig.update_yaxes(title="kW", rangemode="tozero")
            st.plotly_chart(estilo(fig, 360), use_container_width=True)

    with tab2:  # F5
        mods = modelos()
        disponibles = [m for m in core.MODELOS if m in mods]
        if len(disponibles) == len(core.MODELOS):
            disponibles = [E] + disponibles
        c1, c2, c3 = st.columns(3)
        fecha2 = c1.date_input("Fecha de inicio", value=pd.Timestamp("2010-06-15").date(),
                               min_value=INI_PRUEBA.date() + pd.Timedelta(days=1),
                               max_value=P["origen"].max().date() - pd.Timedelta(days=1), key="f2")
        hora2 = c2.selectbox("Hora de inicio", [f"{h:02d}:00" for h in range(24)], index=0, key="h2")
        elegidos = c3.multiselect("Modelos", disponibles, default=[E, "XGBoost"] if E in disponibles else disponibles[:2])
        st.caption("Con el ensamble, en cada paso los cuatro modelos predicen y su promedio pasa al paso siguiente.")
        st.warning("**Cada paso usa la predicción anterior como si fuera real**, así que el error se acumula a lo "
                   "largo de las 24 h. El modelo matemático (§9) define la predicción a un paso; esta vista es "
                   "ilustrativa y su error no se cuantifica en la formulación.", icon="⚠️")
        if st.button("Calcular pronóstico de 24 h", type="primary") and elegidos:
            t0 = pd.Timestamp(f"{fecha2} {hora2}")
            curvas = {}
            with st.spinner("Encadenando 48 predicciones…"):
                for m in elegidos:
                    curvas[m] = core.pronostico_encadenado(m, mods if m == E else mods[m], S, t0)
            df = pd.DataFrame(curvas)
            df.insert(0, "Real", S["Global_active_power"].reindex(df.index))
            fig = go.Figure()
            hist = S.loc[t0 - pd.Timedelta(hours=12):t0, "Global_active_power"]
            fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Histórico", mode="lines",
                                     line=dict(color="#a8a7a2", width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df["Real"], name="Real", mode="lines",
                                     line=dict(color=core.COLOR_MODELO["Real"], width=2)))
            for m in elegidos:
                fig.add_trace(go.Scatter(x=df.index, y=df[m], name=m, mode="lines",
                                         line=dict(color=core.COLOR_MODELO[m], width=2)))
            fig.add_vline(x=t0, line_dash="dot", line_color="#8a8984")
            fig.update_yaxes(title="kW", rangemode="tozero")
            st.plotly_chart(estilo(fig, 400), use_container_width=True)
            filas = []
            for m in elegidos:
                e_pred = df[m].sum() * core.HORAS_INTERVALO
                e_real = df["Real"].sum() * core.HORAS_INTERVALO
                filas.append({"Modelo": m, "Energía prevista 24 h (kWh)": fmt(e_pred, 2),
                              "Energía real 24 h (kWh)": fmt(e_real, 2),
                              "Costo previsto": cop(e_pred * tarifa),
                              "MAE de las 48 predicciones (kW)": fmt(np.mean(np.abs(df[m] - df["Real"])), 3)})
            st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
            descargar(df, f"elecprevis_pronostico24h_{fecha2}.csv")

    with tab3:  # F6
        st.markdown("Ingresa las condiciones del momento actual y obtén la predicción para los próximos 30 minutos. "
                    "La LSTM no aparece aquí porque necesita las 48 filas de las últimas 24 h, no una sola; por eso "
                    "el ensamble de esta calculadora promedia solo los otros tres modelos.")
        mods = modelos()
        c1, c2, c3 = st.columns(3)
        f_cal = c1.date_input("Fecha", value=pd.Timestamp("2010-06-15").date(), key="fc")
        h_cal = c1.slider("Hora", 0, 23, 19)
        mi_cal = c1.radio("Minuto", [0, 30], horizontal=True)
        gap = c2.number_input("Consumo actual (kW)", 0.0, 10.0, 1.5, 0.1)
        lag1 = c2.number_input("Consumo hace 30 min (kW)", 0.0, 10.0, 1.3, 0.1)
        lag48 = c2.number_input("Consumo ayer a esta hora (kW)", 0.0, 10.0, 1.6, 0.1)
        lag336 = c2.number_input("Consumo hace una semana a esta hora (kW)", 0.0, 10.0, 1.4, 0.1)
        roll = c2.number_input("Consumo medio de las últimas 24 h (kW)", 0.0, 10.0, 1.1, 0.1)
        s1 = c3.slider("Cocina (Wh por minuto)", 0.0, 40.0, 0.0, 0.5)
        s2 = c3.slider("Lavandería (Wh por minuto)", 0.0, 40.0, 1.0, 0.5)
        s3 = c3.slider("Calentador / A.A. (Wh por minuto)", 0.0, 31.0, 6.0, 0.5)
        volt = c3.slider("Tensión (V)", 225.0, 252.0, 240.0, 0.5)
        reac = c3.slider("Potencia reactiva (kW)", 0.0, 1.0, 0.1, 0.01)
        ts = pd.Timestamp(f"{f_cal} {h_cal:02d}:{mi_cal:02d}")
        base = pd.DataFrame({
            "Global_active_power": [gap], "Global_reactive_power": [reac], "Voltage": [volt],
            "Global_intensity": [gap * 1000 / volt], "Sub_metering_1": [s1], "Sub_metering_2": [s2],
            "Sub_metering_3": [s3],
        }, index=pd.DatetimeIndex([ts]))
        X = core.ft.add_calendar_features(base)
        X["Global_active_power_lag1"] = lag1
        X["Global_active_power_lag48"] = lag48
        X["Global_active_power_lag336"] = lag336
        X["Global_active_power_roll_mean_48"] = roll
        X = X[core.FEATURES]
        usados = [m for m in core.MODELOS if m in mods and m != "LSTM"]
        valores = {m: float(core.predecir(m, mods[m], X)[0]) for m in usados}
        cols = st.columns(len(usados) + 1)
        cols[0].metric(f"**{E} (sin LSTM)**", fmt(np.mean(list(valores.values())), 3) + " kW")
        for c, m in zip(cols[1:], usados):
            c.metric(m, fmt(valores[m], 3) + " kW")
        if (s1 + s2 + s3) * core.WH_MIN_A_KW > gap:
            st.caption("⚠️ Las submediciones suman más que el consumo actual; revisa los valores.")

# ==========================================================================
# COMPARACIÓN DE MODELOS (F7, F8, F9, F10, F11)
# ==========================================================================
elif pagina == "Comparación de modelos":
    st.title("Comparación: los cuatro modelos y su ensamble")
    st.caption(f"Periodo de prueba: {fmt(R['dataset']['prueba'], 0)} intervalos de 30 min "
               f"({INI_PRUEBA:%d/%m/%Y} – {pd.Timestamp(R['dataset']['fin_prueba']):%d/%m/%Y}). "
               "Partición cronológica 80/20, sin mezclar al azar.")

    def tabla_metricas(m: dict, filas: list) -> pd.DataFrame:
        tabla = pd.DataFrame(m).T.loc[filas]
        return pd.DataFrame({
            "MAE (kW)": tabla["MAE"].map(lambda v: fmt(v, 4)),
            "MSE (kW²)": tabla["MSE"].map(lambda v: fmt(v, 4)),
            "RMSE (kW)": tabla["RMSE"].map(lambda v: fmt(v, 4)),
            "R²": tabla["R2"].map(lambda v: fmt(v, 4)),
            "sMAPE (%)": tabla["sMAPE"].map(lambda v: fmt(v, 2)),
            f"Precisión ±{TOL} kW (%)": tabla["Precisión ±1 kW"].map(lambda v: fmt(v, 2)),
            "Error P95 (kW)": tabla["Error P95"].map(lambda v: fmt(v, 3)),
        })

    m = R["metricas_prueba"]
    st.dataframe(tabla_metricas(m, TODOS), use_container_width=True)
    mejor_ind = min(core.MODELOS, key=lambda k: m[k]["RMSE"])
    st.markdown(
        f"El **ensamble** (promedio de los cuatro) supera al mejor modelo individual, **{mejor_ind}**: "
        f"R² {fmt(m[mejor_ind]['R2'], 3)} → **{fmt(m[E]['R2'], 3)}**, MAE {fmt(m[mejor_ind]['MAE'], 3)} → "
        f"**{fmt(m[E]['MAE'], 3)} kW** y precisión ±{TOL} kW {fmt(m[mejor_ind][ETIQ_PREC], 2)} % → "
        f"**{fmt(m[E][ETIQ_PREC], 2)} %**. La mejora es moderada: lo que más gana el ensamble es estabilidad."
    )
    with st.expander("¿Por qué precisión por tolerancia y no TP / (TP + FP)?"):
        ref_txt = (f" Como referencia, predecir que en la próxima media hora se consume lo mismo que ahora, "
                   f"sin ningún modelo, alcanza {fmt(REF['Precisión ±1 kW'], 1)} % con la misma tolerancia."
                   if REF else "")
        st.markdown(
            "- **Precision = TP / (TP + FP)** es una métrica de **clasificación**: cuenta aciertos cuando el "
            "modelo responde sí o no. Este proyecto es de **regresión**: el modelo responde un número (kW), "
            "así que no hay positivos ni negativos que contar.\n"
            f"- Por eso se usa la **precisión por tolerancia**: el porcentaje de intervalos en los que "
            f"|y − ŷ| ≤ {TOL} kW. Con esa tolerancia, el ensamble, XGBoost, Random Forest y la LSTM superan el "
            f"95 %.{ref_txt}\n"
            "- **Error P95** dice lo mismo al revés: el error que el modelo no supera en el 95 % de los "
            "intervalos.\n"
            "- Las métricas principales de una regresión son **MAE, MSE, RMSE y R²**, definidas en la "
            "Sección 8 del modelo matemático. La precisión por tolerancia las complementa, no las reemplaza."
        )

    # F8
    st.subheader("Real vs. predicho")
    c1, c2 = st.columns([3, 1])
    dsel = c1.date_input("Semana desde", value=pd.Timestamp("2010-08-02").date(),
                         min_value=INI_PRUEBA.date(), max_value=P.index.max().date() - pd.Timedelta(days=7), key="cmp")
    vis = c2.multiselect("Modelos", TODOS, default=[E, "Regresión lineal"])
    w = P.loc[pd.Timestamp(dsel):pd.Timestamp(dsel) + pd.Timedelta(days=7)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w.index, y=w["Real"], name="Real", line=dict(color=core.COLOR_MODELO["Real"], width=2)))
    for mm in vis:
        fig.add_trace(go.Scatter(x=w.index, y=w[mm], name=mm, line=dict(color=core.COLOR_MODELO[mm], width=2)))
    fig.update_yaxes(title="kW", rangemode="tozero")
    st.plotly_chart(estilo(fig, 380), use_container_width=True)

    c1, c2 = st.columns(2)
    err_h = pd.DataFrame({mm: (P[mm] - P["Real"]).abs() for mm in TODOS})
    err_h = err_h.groupby(P.index.hour).mean()
    fige = go.Figure()
    for mm in TODOS:
        fige.add_trace(go.Scatter(x=err_h.index, y=err_h[mm], name=mm, mode="lines+markers",
                                  line=dict(color=core.COLOR_MODELO[mm], width=3 if mm == E else 2),
                                  marker=dict(size=8)))
    fige.update_xaxes(title="Hora del día", dtick=2)
    fige.update_yaxes(title="Error absoluto medio (kW)", rangemode="tozero")
    c1.markdown("**¿A qué hora se equivocan más?**")
    c1.plotly_chart(estilo(fige, 340), use_container_width=True)

    sel = c2.selectbox("Dispersión real vs. predicho", TODOS, index=0)
    muestra = P.sample(3000, random_state=0)
    figs = go.Figure(go.Scattergl(x=muestra["Real"], y=muestra[sel], mode="markers",
                                  marker=dict(color=core.COLOR_MODELO[sel], size=6, opacity=0.35),
                                  hovertemplate="real %{x:.2f} · predicho %{y:.2f}<extra></extra>"))
    lim = float(max(muestra["Real"].max(), muestra[sel].max()))
    figs.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color="#8a8984", dash="dot"), name="ŷ = y"))
    figs.update_xaxes(title="Real (kW)")
    figs.update_yaxes(title="Predicho (kW)")
    c2.plotly_chart(estilo(figs, 340, leyenda=False).update_layout(hovermode="closest"), use_container_width=True)

    # F9
    if R.get("walk_forward"):
        st.subheader("Estabilidad: validación de ventana expansiva (5 cortes)")
        st.caption("Cada corte entrena con todo lo anterior y evalúa el bloque siguiente, dentro del 80 % de "
                   "entrenamiento. La LSTM se omite aquí por costo de cómputo, como en run_models.py; por eso el "
                   "ensamble de esta vista promedia los otros tres modelos.")
        filas = []
        for mm, cortes in R["walk_forward"].items():
            for c in cortes:
                filas.append({"Modelo": mm, "Corte": c["corte"], "RMSE": c["RMSE"], "R2": c["R2"],
                              "Periodo": f"{c['desde'][:10]} → {c['hasta'][:10]}"})
        wf = pd.DataFrame(filas)
        figw = go.Figure()
        for mm in wf["Modelo"].unique():
            d = wf[wf["Modelo"] == mm]
            figw.add_trace(go.Scatter(x=d["Corte"], y=d["RMSE"], name=mm, mode="lines+markers",
                                      line=dict(color=core.COLOR_MODELO[mm], width=2), marker=dict(size=9),
                                      customdata=d["Periodo"], hovertemplate="%{customdata}: RMSE %{y:.3f}<extra>" + mm + "</extra>"))
        figw.update_xaxes(title="Corte", dtick=1)
        figw.update_yaxes(title="RMSE (kW)", rangemode="tozero")
        st.plotly_chart(estilo(figw, 320).update_layout(hovermode="closest"), use_container_width=True)
        resumen = wf.groupby("Modelo")[["RMSE", "R2"]].agg(["mean", "std"]).round(4)
        resumen.columns = ["RMSE media", "RMSE desv.", "R² media", "R² desv."]
        st.dataframe(resumen.map(lambda v: fmt(v, 4)), use_container_width=True)

    # F10
    if R.get("metricas_entrenamiento"):
        st.subheader("Registro de desempeño: entrenamiento frente a prueba")
        st.caption("Si un modelo es mucho mejor en entrenamiento que en prueba, memorizó los datos en vez de "
                   "aprender el patrón (sobreajuste). El entrenamiento se mide desde el intervalo 48 para que "
                   "la LSTM tenga su ventana de 24 h.")
        mtr = R["metricas_entrenamiento"]
        filas = []
        for mm in TODOS:
            filas.append({
                "Modelo": mm,
                "RMSE train (kW)": fmt(mtr[mm]["RMSE"], 4), "RMSE test (kW)": fmt(m[mm]["RMSE"], 4),
                "R² train": fmt(mtr[mm]["R2"], 4), "R² test": fmt(m[mm]["R2"], 4),
                "Caída de R²": fmt(mtr[mm]["R2"] - m[mm]["R2"], 4),
                f"Precisión ±{TOL} kW train (%)": fmt(mtr[mm][ETIQ_PREC], 2),
                f"Precisión ±{TOL} kW test (%)": fmt(m[mm][ETIQ_PREC], 2),
            })
        st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)

    # F11
    if ENS:
        st.subheader("¿Cómo se combinan los cuatro modelos?")
        st.markdown(
            "Igual que un **random forest**, que en regresión predice el promedio de sus árboles, el ensamble "
            "promedia las predicciones de los cuatro modelos. Se probaron tres formas de combinar:"
        )
        st.dataframe(pd.DataFrame(core.COMBINACIONES.items(), columns=["Forma", "Idea"]),
                     hide_index=True, use_container_width=True)
        cal, ev = ENS["calibracion"], ENS["evaluacion"]
        st.caption(f"El ponderado y el stacking tienen parámetros: se estiman con la primera mitad de la prueba "
                   f"({cal['desde'][:10]} – {cal['hasta'][:10]}) y todas las filas de la tabla se evalúan en la "
                   f"segunda mitad ({ev['desde'][:10]} – {ev['hasta'][:10]}), que ninguna forma de combinar vio.")
        comp = ENS["comparacion_combinaciones"]
        orden = [f"{E} · {k}" for k in core.COMBINACIONES] + core.MODELOS
        st.dataframe(tabla_metricas(comp, orden), use_container_width=True)
        pes = ENS["parametros"]["Ponderado"]["pesos"]
        stk = ENS["parametros"]["Stacking"]
        c1, c2 = st.columns(2)
        c1.markdown("**Pesos del ponderado**")
        c1.dataframe(pd.DataFrame({"Peso": {k: fmt(v, 3) for k, v in pes.items()}}), use_container_width=True)
        c2.markdown("**Coeficientes del stacking**")
        c2.dataframe(pd.DataFrame({"Coeficiente": {**{k: fmt(v, 3) for k, v in stk["coef"].items()},
                                                   "Intercepto": fmt(stk["intercepto"], 3)}}),
                     use_container_width=True)
        r_prom = comp[f"{E} · Promedio"]["R2"]
        r_otros = max(comp[f"{E} · Ponderado"]["R2"], comp[f"{E} · Stacking"]["R2"])
        st.markdown(
            f"**Se usa el promedio.** Las formas con parámetros apenas cambian el resultado (R² {fmt(r_prom, 3)} "
            f"frente a {fmt(r_otros, 3)} como máximo), agregan parámetros que estimar y riesgo de sobreajuste. "
            "El promedio no entrena nada y es el mismo principio del random forest."
        )

        st.markdown("**¿Por qué funciona? Los modelos se equivocan en momentos distintos**")
        corr = pd.DataFrame(ENS["correlacion_errores"]).loc[core.MODELOS, core.MODELOS]
        figc = px.imshow(corr, zmin=0.5, zmax=1, color_continuous_scale="Purples", text_auto=".2f",
                         labels=dict(color="Correlación"))
        figc.update_traces(hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>")
        c1, c2 = st.columns([1, 1])
        c1.plotly_chart(estilo(figc, 360, leyenda=False).update_layout(hovermode="closest"),
                        use_container_width=True)
        c2.markdown(
            "Correlación entre los errores (y − ŷ) de cada par de modelos en el periodo de prueba. "
            "1 significaría que se equivocan exactamente igual y promediarlos no ayudaría.\n\n"
            f"- **Random Forest y XGBoost** se equivocan casi igual "
            f"({fmt(corr.loc['Random Forest', 'XGBoost'], 2)}): los dos son ensambles de árboles.\n"
            f"- **La regresión lineal y la LSTM** aportan la diversidad: ven el problema de otra forma "
            f"(relaciones lineales y secuencia de 24 h) y fallan en otros momentos.\n\n"
            "Por eso tiene sentido incluir los cuatro aunque la regresión lineal sola sea el más débil."
        )

    descargar(P, "elecprevis_predicciones_prueba.csv", "Descargar predicciones del periodo de prueba (CSV)")

# ==========================================================================
# ESCENARIOS (F12, F13, F14)
# ==========================================================================
elif pagina == "Escenarios de eficiencia":
    st.title("Escenarios de eficiencia energética")
    st.info("Los escenarios son **hipotéticos**: suponen que un grupo de carga consume cierto porcentaje menos. "
            "Los datos son observacionales y no permiten afirmar que una medida concreta produzca ese ahorro. "
            "Las tres submediciones cubren el 48,7 % de la energía; el resto no se puede desagregar.", icon="ℹ️")
    c1, c2 = st.columns([2, 1])
    rango = c1.date_input("Periodo a analizar", value=(pd.Timestamp("2010-03-01").date(), pd.Timestamp("2010-10-31").date()),
                          min_value=S.index.min().date(), max_value=S.index.max().date(), key="esc")
    if not isinstance(rango, tuple) or len(rango) != 2:
        st.stop()
    sub = S.loc[pd.Timestamp(rango[0]):pd.Timestamp(rango[1]) + pd.Timedelta(hours=23, minutes=30)]

    # F12
    d = core.desglose_kw(sub)
    energia = d.sum() * core.HORAS_INTERVALO
    nombres = {**core.SUBMEDICIONES, "Resto": "Resto del hogar (no submedido)"}
    figp = go.Figure()
    for col in ["Sub_metering_3", "Sub_metering_2", "Sub_metering_1", "Resto"]:
        figp.add_trace(go.Bar(y=["Energía"], x=[energia[col]], name=nombres[col], orientation="h",
                              marker=dict(color=core.COLOR_CARGA[col], line=dict(color="white", width=2)),
                              hovertemplate=f"{nombres[col]}: %{{x:,.0f}} kWh<extra></extra>"))
    figp.update_layout(barmode="stack", legend_traceorder="normal")
    figp.update_xaxes(title="kWh en el periodo")
    c2.metric("Energía total del periodo", fmt(energia.sum(), 0) + " kWh")
    c2.metric("Costo del periodo", cop(energia.sum() * tarifa))
    st.subheader("¿En qué se va la energía?")
    st.plotly_chart(estilo(figp, 200).update_layout(hovermode="closest"), use_container_width=True)

    perfil = d.groupby(d.index.hour).mean()
    figa = go.Figure()
    for col in ["Sub_metering_3", "Sub_metering_2", "Sub_metering_1", "Resto"]:
        figa.add_trace(go.Bar(x=perfil.index, y=perfil[col], name=nombres[col],
                              marker=dict(color=core.COLOR_CARGA[col], line=dict(color="white", width=1))))
    figa.update_layout(barmode="stack", legend_traceorder="normal")
    figa.update_xaxes(title="Hora del día", dtick=2)
    figa.update_yaxes(title="Potencia media (kW)")
    st.markdown("**Perfil horario por grupo de carga**")
    st.plotly_chart(estilo(figa, 340), use_container_width=True)

    # F13
    st.subheader("Simulador de reducción")
    base = st.radio(
        "Consumo de partida",
        ["Medido", "Previsto por el ensamble"], horizontal=True,
        help="«Previsto por el ensamble» parte de la predicción a 30 min del ensamble en lugar del consumo "
             "medido. Solo existe en el periodo de prueba (13/02/2010 – 26/11/2010).",
    )
    c1, c2, c3 = st.columns(3)
    r1 = c1.slider("Cocina: reducción (%)", 0, 100, 10, 5)
    r2 = c2.slider("Lavandería: reducción (%)", 0, 100, 10, 5)
    r3 = c3.slider("Calentador / A.A.: reducción (%)", 0, 100, 20, 5)
    red = {"Sub_metering_1": r1 / 100, "Sub_metering_2": r2 / 100, "Sub_metering_3": r3 / 100}
    if base == "Medido":
        esc = core.simular_escenario(sub, red)
    else:
        # El intervalo t+1 lo predice el ensamble; las submediciones son las de ese mismo intervalo.
        prev = P[E].reindex(sub.index).dropna()
        if prev.empty:
            st.warning("El periodo elegido no tiene predicciones: escoge fechas dentro del periodo de prueba.")
            st.stop()
        sub_p = sub.loc[prev.index].copy()
        e_med = sub_p["Global_active_power"].sum() * core.HORAS_INTERVALO
        e_prev = prev.sum() * core.HORAS_INTERVALO
        st.caption(f"En estas {fmt(len(prev), 0)} medias horas el ensamble previó {fmt(e_prev, 0)} kWh y se "
                   f"midieron {fmt(e_med, 0)} kWh (diferencia {fmt(100 * (e_prev - e_med) / e_med, 1)} %).")
        sub_p["Global_active_power"] = prev
        esc = core.simular_escenario(sub_p, red)
        esc["Escenario"] = esc["Escenario"].clip(lower=0)
        sub = sub_p
    e_real = esc["Real"].sum() * core.HORAS_INTERVALO
    e_esc = esc["Escenario"].sum() * core.HORAS_INTERVALO
    ahorro = e_real - e_esc

    # F14
    dias = max(len(sub) / 48, 1)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Energía en el escenario", fmt(e_esc, 0) + " kWh", delta="−" + fmt(ahorro, 0) + " kWh", delta_color="inverse")
    k2.metric("Reducción del consumo total", fmt(100 * ahorro / e_real, 1) + " %")
    k3.metric("Ahorro en el periodo", cop(ahorro * tarifa))
    k4.metric("Ahorro mensual equivalente", cop(ahorro * tarifa / dias * 30))
    st.caption("Fórmula del escenario: y′ₜ = yₜ − Σₖ rₖ · sₖ,ₜ, donde sₖ,ₜ es la submedición k en kW "
               "(Wh/min × 60 / 1000) y rₖ la reducción supuesta.")
    sem = esc.resample("W").mean()
    fige = go.Figure()
    fige.add_trace(go.Scatter(x=sem.index, y=sem["Real"], name="Medido" if base == "Medido" else "Previsto (ensamble)",
                              line=dict(color=core.COLOR_MODELO["Real" if base == "Medido" else E], width=2)))
    fige.add_trace(go.Scatter(x=sem.index, y=sem["Escenario"], name="Escenario", line=dict(color=core.COLOR_CARGA["Sub_metering_3"], width=2)))
    fige.update_yaxes(title="Potencia media semanal (kW)", rangemode="tozero")
    st.plotly_chart(estilo(fige, 340), use_container_width=True)
    descargar(esc, f"elecprevis_escenario_{rango[0]}_{rango[1]}.csv", "Descargar escenario (CSV)")

# ==========================================================================
# MODELO (F15, F16, F17)
# ==========================================================================
elif pagina == "Modelo":
    st.title("El modelo detrás de la app")
    st.markdown("La app no reimplementa el modelo: usa las mismas funciones del repositorio (`src/`) con las que "
                "se construyó el dataset y se entrenaron los modelos.")
    st.subheader("Problema")
    st.latex(r"\hat{y}_{t+1} = f(\mathbf{x}_t), \qquad f^{*} = \arg\min_{f}\ J(\theta) = \frac{1}{|T|}\sum_{t\in T}\big(y_{t+1}-f(\mathbf{x}_t)\big)^2")
    st.markdown("**xₜ** reúne 19 variables con información hasta el intervalo t; ŷₜ₊₁ es el consumo de los 30 minutos siguientes.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Memoria del consumo**")
        st.latex(r"y_{t-1},\ y_{t-48},\ y_{t-336},\qquad \bar{y}^{(48)}_t=\frac{1}{48}\sum_{k=1}^{48}y_{t-k}")
        st.markdown("**Calendario cíclico**")
        st.latex(r"\sin\!\left(\tfrac{2\pi h}{24}\right),\ \cos\!\left(\tfrac{2\pi h}{24}\right),\ \sin\!\left(\tfrac{2\pi (m-1)}{12}\right),\ \cos\!\left(\tfrac{2\pi (m-1)}{12}\right)")
    with c2:
        st.markdown("**Regresión lineal sobre log(1+y)**")
        st.latex(r"\log(1+y_{t+1}) = \beta_0 + \boldsymbol{\beta}^{\top}\mathbf{z}_t,\quad \mathbf{z}_t=\frac{\mathbf{x}_t-\boldsymbol{\mu}}{\boldsymbol{\sigma}}")
        st.markdown("**XGBoost (modelo aditivo)**")
        st.latex(r"\hat{y}^{(K)} = \sum_{k=1}^{K}\eta\, f_k(\mathbf{x}),\quad K=300,\ \eta=0{,}05")
    st.markdown("**LSTM** (evaluada en la app con NumPy, compuerta por compuerta)")
    st.latex(r"\begin{aligned}f_t&=\sigma(W_f x_t+U_f h_{t-1}+b_f) & i_t&=\sigma(W_i x_t+U_i h_{t-1}+b_i)\\ \tilde c_t&=\tanh(W_c x_t+U_c h_{t-1}+b_c) & c_t&=f_t\odot c_{t-1}+i_t\odot\tilde c_t\\ o_t&=\sigma(W_o x_t+U_o h_{t-1}+b_o) & h_t&=o_t\odot\tanh(c_t)\end{aligned}")
    st.markdown("**Random forest** (promedio de B árboles) y **ensamble de los cuatro modelos** (mismo principio)")
    c1, c2 = st.columns(2)
    c1.latex(r"\hat{y}_{RF} = \frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{x}),\quad B=150")
    c2.latex(r"\hat{y}_{ens} = \frac{1}{4}\left(\hat{y}_{lin}+\hat{y}_{RF}+\hat{y}_{XGB}+\hat{y}_{LSTM}\right)")
    st.markdown(
        "Si cada modelo tiene error eᵢ, el error del promedio es la media de los eᵢ. Cuando los errores no están "
        "perfectamente correlacionados se compensan en parte, y el error cuadrático del ensamble queda por debajo "
        "del promedio de los errores individuales:"
    )
    st.latex(r"\mathbb{E}\!\left[\Big(\tfrac{1}{4}\textstyle\sum_i e_i\Big)^2\right] \;\le\; \tfrac{1}{4}\textstyle\sum_i \mathbb{E}\left[e_i^2\right]")

    # F16
    st.subheader("Defensa del modelo")
    m = R["metricas_prueba"]
    mtr = R.get("metricas_entrenamiento", {})
    mejor_ind = min(core.MODELOS, key=lambda k: m[k]["RMSE"])
    caida = (mtr[E]["R2"] - m[E]["R2"]) if mtr else None
    defensa = [
        ("1. ¿Qué variables utiliza?",
         "19 variables con información hasta el intervalo t: 7 mediciones eléctricas (potencia activa y reactiva, "
         "tensión, intensidad y las 3 submediciones), 8 de calendario (hora y mes con codificación seno/coseno, "
         "día de la semana, fin de semana) y 4 de memoria del consumo (hace 30 min, hace 24 h, hace 7 días y media "
         "de 24 h). La regresión lineal excluye la intensidad, correlacionada 0,999 con la potencia."),
        ("2. ¿Qué intenta predecir?",
         "La potencia activa media (kW) de la vivienda en los siguientes 30 minutos. Es un problema de regresión: "
         "la salida es un número, no una clase."),
        ("3. ¿Cómo se entrenó?",
         "Cada uno de los cuatro modelos se entrenó por separado con el 80 % más antiguo de la serie, en orden "
         "cronológico y sin mezclar al azar, cada uno con su preprocesamiento (escalado y log(1+y) para la lineal "
         "y la LSTM; nada para los árboles). El ensamble no se entrena: promedia las cuatro predicciones, como un "
         "random forest promedia sus árboles."),
        ("4. ¿Cómo se evaluó?",
         f"Con el 20 % final ({fmt(R['dataset']['prueba'], 0)} intervalos que ningún modelo vio), con MAE, MSE, "
         f"RMSE, R² y precisión por tolerancia ±{TOL} kW, más 5 cortes de ventana expansiva y la comparación con "
         f"la persistencia. Resultado del ensamble: R² {fmt(m[E]['R2'], 3)} y precisión {fmt(m[E][ETIQ_PREC], 2)} %, "
         f"frente a R² {fmt(m[mejor_ind]['R2'], 3)} y {fmt(m[mejor_ind][ETIQ_PREC], 2)} % del mejor modelo "
         f"individual ({mejor_ind})."
         + (f" La caída de R² entre entrenamiento y prueba es {fmt(caida, 3)}." if caida is not None else "")),
        ("5. Errores principales",
         f"Los picos se subestiman: son encendidos repentinos (horno, calentador) que el pasado no anticipa. El "
         f"error que no se supera en el 95 % de los intervalos es {fmt(m[E]['Error P95'], 2)} kW, y el error es "
         f"mayor entre las 19:00 y las 21:00, la hora de la cena (ver *Comparación*)."),
        ("6. Limitaciones actuales",
         "Datos de una sola vivienda en Francia (2006–2010); el pronóstico de 24 h acumula error paso a paso; los "
         "escenarios de ahorro son hipotéticos y no causales; y la mejora del ensamble sobre el mejor modelo es "
         "moderada, porque Random Forest y XGBoost se equivocan casi igual. Su principal aporte es estabilidad."),
    ]
    for pregunta, respuesta in defensa:
        st.markdown(f"**{pregunta}**  \n{respuesta}")

    # F15
    st.subheader("¿Qué variables usa cada modelo?")
    imp = R["importancia"]
    cmod = st.selectbox("Modelo", ["XGBoost", "Random Forest", "Regresión lineal"])
    serie_imp = pd.Series(imp[cmod])
    if cmod == "Regresión lineal":
        serie_imp = serie_imp.abs()
        st.caption("Para la regresión lineal se muestra |β| sobre entradas estandarizadas (sin Global_intensity).")
    serie_imp = serie_imp.sort_values()
    figi = go.Figure(go.Bar(x=serie_imp.values, y=serie_imp.index, orientation="h",
                            marker_color=core.COLOR_MODELO[cmod], hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    figi.update_xaxes(title="Importancia" if cmod != "Regresión lineal" else "|β|")
    st.plotly_chart(estilo(figi, 520, leyenda=False).update_layout(hovermode="closest"), use_container_width=True)

    st.subheader("Hiperparámetros")
    st.dataframe(pd.DataFrame(R["hiperparametros"].items(), columns=["Modelo", "Configuración"]),
                 hide_index=True, use_container_width=True)

    # F17
    st.subheader("Trazabilidad: de la ecuación al código")
    traz = [
        ("Remuestreo a 30 min", "preprocessing.resample_30min()"),
        ("Imputación causal (forward fill)", "preprocessing.impute_missing()"),
        ("Rezagos 1, 48, 336", "features.add_lag_features()"),
        ("Media móvil de 24 h", "features.add_rolling_features()"),
        ("Calendario cíclico", "features.add_calendar_features()"),
        ("Partición 80/20 cronológica", "features.temporal_train_test_split()"),
        ("Ventana expansiva (5 cortes)", "models.walk_forward_folds()"),
        ("Regresión lineal sobre log(1+y)", "models.fit_linear_regression() → app/core.LinealOLS"),
        ("Random forest (150 árboles, ≥ 10 por hoja)", "models.fit_random_forest(**core.RF_PARAMS)"),
        ("XGBoost (300 árboles)", "models.fit_xgboost()"),
        ("LSTM (64 unidades, ventana 48)", "models.fit_lstm() → app/core.LSTMNumpy"),
        ("Ensamble: promedio de los 4 modelos", "app/core.ensamble_promedio() · app/export_ensamble.py"),
        ("Ponderado y stacking (comparación)", "app/core.combinar()"),
        ("Métricas MAE, MSE, RMSE, R², sMAPE", "app/core.metricas()"),
        ("Escenarios de reducción", "app/core.simular_escenario()"),
        ("Pronóstico encadenado 24 h", "app/core.pronostico_encadenado()"),
    ]
    st.dataframe(pd.DataFrame(traz, columns=["Concepto del modelo matemático", "Implementación"]),
                 hide_index=True, use_container_width=True)
