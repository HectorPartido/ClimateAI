"""Pantalla 03: Ficha de país (PR 3).

Responsable: Hector Martínez Sánchez.
Rama: feat/ficha-grupos
Requerimientos: RF-07, RF-10, RF-11 y gráficas G6a, G6b, G7 y G8.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from comun import (AVISO_PERIODO, INDICADOR, PAIS_EJEMPLO, SISTEMA, fila_pais, grafica_escala, grafica_reglas,
                   insignia, ir_a, nombre, opciones_paises, percentil, resultado_pais, resumen, series)
from src.config import IND, SISTEMAS

# País que llega desde Panorama, Ranking o Grupos; si no, el último que se vio (o México)
if "pais_ficha" in st.session_state:
    st.session_state.fi_pais = st.session_state.pop("pais_ficha")
st.session_state.setdefault("fi_pais", PAIS_EJEMPLO)

lista, fmt = opciones_paises()
st.title("Ficha de país")
c1, c2 = st.columns([4, 2], vertical_alignment="bottom")
pais = c1.selectbox("País", lista, format_func=lambda p: fmt[p], key="fi_pais", persist_state="session",
                    help="Escribe para buscar")
if c2.button("Abrir en simulador", icon=":material/tune:", width="stretch"):        # RF-11
    ir_a("simulador", pais_simulador=pais)

f = fila_pais(pais)
st.markdown(f"### {nombre(pais)}")
st.caption(f"Grupo de países: **{f['grupo']}**")

# ---- prioridades con la escala de 4 tramos (G8), explicación y reglas (G7)
res = {s: resultado_pais(pais, s) for s in SISTEMAS}
cols = st.columns(2, gap="large")
for col, s in zip(cols, SISTEMAS):
    r = res[s]
    with col.container(border=True):
        a, b = st.columns([3, 2], vertical_alignment="center")
        a.markdown(f"**Prioridad de {SISTEMA[s].lower()}**")
        b.markdown(f"<div style='text-align:right'>{insignia(r['clase_etiqueta'])}</div>", unsafe_allow_html=True)
        st.plotly_chart(grafica_escala(r["prediccion"]), key=f"g8_{s}", config={"displayModeBar": False})
        st.markdown(r["explicacion"])
        st.markdown("**Por qué** (las 3 reglas más fuertes)")
        grafica_reglas(r["reglas_activadas"], 3)
        if r["estado"] == "no_concluyente":
            st.warning("Resultado no concluyente: ninguna regla se activa con fuerza suficiente.")

# ---- tabla de indicadores con percentil y promedio del grupo (RF-10)
st.subheader("Indicadores")
centro = resumen()["grupos"]["grupos"][str(int(f["grupo_id"]))]["centroide"]
tabla = pd.DataFrame([{
    "Indicador": f"{c}. {INDICADOR[c][0]}",
    "Qué mide": INDICADOR[c][2],
    "Valor": f"{f[c]:,.3f} {INDICADOR[c][1]}",
    "Más alto que": percentil(c, f[c]),
    "Típico de su grupo": f"{centro[c]:,.3f} {INDICADOR[c][1]}",
} for c in IND])
st.dataframe(tabla, hide_index=True, column_config={
    "Más alto que": st.column_config.ProgressColumn("Más alto que (% de países)", min_value=0, max_value=100,
                                                    format="%d %%"),
    "Típico de su grupo": st.column_config.TextColumn("Típico de su grupo", help="Centro del grupo calculado por el modelo")})

# ---- series históricas (G6a y G6b)
s = series()
s = s[(s["pais"] == pais) & (s["anio"] <= 2012)]
g1, g2 = st.columns(2, gap="large")
with g1:
    st.markdown("**Anomalía de temperatura** (°C frente a 1951 a 1980)")
    t = s.dropna(subset=["anomalia"])
    if t.empty:
        st.info("Sin datos de temperatura para este país.")
    else:
        fig = go.Figure()
        fig.add_vrect(x0=1951, x1=1980, fillcolor="#9ca3af", opacity=0.18, line_width=0,
                      annotation_text="Periodo de referencia", annotation_position="top left")
        fig.add_trace(go.Scatter(x=t["anio"], y=t["anomalia"], mode="lines", name="Anomalía anual",
                                 line=dict(color="#e76f51", width=1.2),
                                 hovertemplate="%{x}: %{y:+.2f} °C<extra></extra>"))
        tr = t[t["anio"] >= 1970]
        if len(tr) >= 10:
            m, b0 = np.polyfit(tr["anio"], tr["anomalia"], 1)
            fig.add_trace(go.Scatter(x=[1970, 2012], y=[m * 1970 + b0, m * 2012 + b0], mode="lines",
                                     name=f"Tendencia 1970 a 2012 ({10 * m:+.2f} °C por década)",
                                     line=dict(color="#111827", width=2.5, dash="dash")))
        fig.add_hline(y=0, line=dict(color="#6b7280", width=1))
        fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.2),
                          xaxis=dict(range=[1900, 2012]), yaxis=dict(title="°C"))
        st.plotly_chart(fig, key="g6a")
with g2:
    st.markdown("**Emisiones de CO₂** (millones de toneladas por año)")
    e = s[(s["anio"] >= 1900)].dropna(subset=["co2_mt"])
    if e.empty:
        st.info("Sin datos de emisiones para este país.")
    else:
        fig = go.Figure(go.Scatter(x=e["anio"], y=e["co2_mt"], mode="lines", fill="tozeroy",
                                   line=dict(color="#457b9d", width=1.5), fillcolor="rgba(69,123,157,0.3)",
                                   hovertemplate="%{x}: %{y:,.1f} Mt<extra></extra>"))
        fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(range=[1900, 2012]),
                          yaxis=dict(title="Mt de CO₂"))
        st.plotly_chart(fig, key="g6b")
st.caption(AVISO_PERIODO)
