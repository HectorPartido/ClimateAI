"""ClimateAI: piezas compartidas del tablero (PR 1).

Responsable: Gildardo Emmanuel González García (Líder de Pipeline e Integración).
Rama: feat/tablero-base

Todas las pantallas importan de aquí los datos (con caché), la barra lateral, los
colores y las gráficas que se repiten. El tablero no entrena nada: lee los archivos que
genera src/models/train.py y llama a predecir() (RNF-03).
"""
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP = Path(__file__).resolve().parent
RAIZ = APP.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.config import (CORTES, DATA_PROCESSED, DIFUSO, INDICADORES, LOG_ENTRENAMIENTO,  # noqa: E402
                        NIVELES, NIVELES_EXPERTO, RANGOS, REPORTS, RESUMEN)
from src.models.predict import _artefactos, predecir  # noqa: E402

NIVELES_PAIS = REPORTS / "niveles_pais.csv"
NOMBRES_ES = RAIZ / "data" / "reference" / "nombres_es.csv"
REPORTES = Path(os.environ.get("CLIMATEAI_REPORTES", RAIZ / "data" / "reportes.csv"))
PAIS_EJEMPLO = "Mexico"

# Pantallas: clave -> (archivo, título, ícono, sección). El menú solo muestra las que existen.
PANTALLAS = {
    "panorama": ("paginas/1_panorama.py", "Panorama", ":material/public:", "Análisis"),
    "ranking": ("paginas/2_ranking.py", "Ranking de países", ":material/format_list_numbered:", "Análisis"),
    "ficha": ("paginas/3_ficha.py", "Ficha de país", ":material/flag:", "Análisis"),
    "grupos": ("paginas/4_grupos.py", "Grupos de países", ":material/bubble_chart:", "Modelo de IA"),
    "simulador": ("paginas/5_simulador.py", "Simulador de prioridad", ":material/tune:", "Modelo de IA"),
    "datos": ("paginas/6_datos_modelo.py", "Datos y modelo", ":material/dataset:", "Modelo de IA"),
}

SISTEMA = {"adaptacion": "Adaptación", "mitigacion": "Mitigación"}
COLOR_NIVEL = {"Baja": "#2a9d8f", "Media": "#e9c46a", "Alta": "#f4a261", "Crítica": "#d62828"}
COLOR_GRUPO = {1: "#457b9d", 2: "#2a9d8f", 3: "#d62828", 4: "#e9a23b", 5: "#7b2cbf", 6: "#6c757d"}
INDICADOR = {
    "I1": ("Calentamiento", "°C por década", "tendencia de 1970 a 2012"),
    "I2": ("Anomalía", "°C", "2003 a 2012 frente a 1951 a 1980"),
    "I3": ("Variabilidad", "°C", "de un año a otro, 1970 a 2012"),
    "I4": ("Emisiones acumuladas", "Gt de CO₂", "1850 a 2012"),
    "I5": ("Parte de las emisiones mundiales", "%", "2003 a 2012"),
    "I6": ("Crecimiento de emisiones", "%", "2003 a 2012 frente a 1993 a 2002"),
}
AVISO_PERIODO = "Periodo histórico 1850 a 2012; no es un pronóstico."
AVISO_IA = ("Las prioridades las calcula un modelo de inteligencia artificial a partir de datos históricos. "
            "Puede equivocarse: la decisión final es del analista.")


# ---------------------------------------------------------------- datos (con caché)
@st.cache_data(show_spinner=False)
def paises() -> pd.DataFrame:
    """Una fila por país: indicadores, grupo, puntajes y niveles, con el nombre en español."""
    ind = pd.read_csv(INDICADORES)
    niv = pd.read_csv(NIVELES_PAIS).drop(columns=["iso3"])
    nom = pd.read_csv(NOMBRES_ES)[["pais", "pais_es"]]
    d = ind.merge(niv, on="pais", how="left").merge(nom, on="pais", how="left")
    d["pais_es"] = d["pais_es"].fillna(d["pais"])
    return d.sort_values("pais_es").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def resumen() -> dict:
    return json.loads(RESUMEN.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def series() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "series_pais.csv")


@st.cache_data(show_spinner=False)
def parametros_difusos() -> dict:
    return json.loads(DIFUSO.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def experto() -> pd.DataFrame:
    return pd.read_csv(NIVELES_EXPERTO, dtype=str).fillna("")


@st.cache_resource(show_spinner=False)
def modelos() -> dict:
    """scaler, pca, kmeans, difuso y grupos, cargados una sola vez (RNF-01)."""
    return _artefactos()


def limpiar_cache() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
    _artefactos.cache_clear()


def nombre(pais: str) -> str:
    d = paises()
    fila = d.loc[d["pais"] == pais, "pais_es"]
    return fila.iloc[0] if len(fila) else pais


def opciones_paises() -> tuple[list, dict]:
    """Lista de países (OWID) ordenada por nombre en español y función de formato."""
    d = paises()
    return d["pais"].tolist(), dict(zip(d["pais"], d["pais_es"]))


def fila_pais(pais: str) -> pd.Series:
    return paises().set_index("pais").loc[pais]


def entrada_pais(pais: str) -> dict:
    f = fila_pais(pais)
    return {c: float(f[c]) for c in RANGOS}


def percentil(c: str, valor: float) -> int:
    """Porcentaje de países con un valor menor (la misma referencia que usa la explicación del modelo)."""
    from src.models.fuzzy import percentil as percentil_modelo
    return int(percentil_modelo(c, float(valor), parametros_difusos()))


@st.cache_data(show_spinner=False)
def resultado_pais(pais: str, sistema: str) -> dict:
    return predecir({**entrada_pais(pais), "sistema": sistema})


# ---------------------------------------------------------------- navegación
def disponible(clave: str) -> bool:
    return (APP / PANTALLAS[clave][0]).exists()


def ir_a(clave: str, **estado) -> None:
    """Guarda el estado que necesita la otra pantalla y cambia de pantalla."""
    for k, v in estado.items():
        st.session_state[k] = v
    if disponible(clave):
        st.switch_page(PANTALLAS[clave][0])
    else:
        st.info("Esa pantalla todavía no está disponible.")


def boton_ir(texto: str, clave: str, key: str, **estado) -> None:
    if disponible(clave) and st.button(texto, key=key):
        ir_a(clave, **estado)


# ---------------------------------------------------------------- barra lateral (RF-17)
def guardar_reporte(pais: str, pantalla: str, problema: str) -> None:
    REPORTES.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not REPORTES.exists()
    with REPORTES.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha", "pais", "pantalla", "problema"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), pais, pantalla, problema.strip()])


def barra_lateral() -> None:
    with st.sidebar:
        st.markdown("### ClimateAI")
        st.caption("Área de Análisis Climático")
        st.warning(AVISO_PERIODO, icon=":material/history:")
        st.info(AVISO_IA, icon=":material/smart_toy:")
        with st.popover("Reportar un error", icon=":material/flag:", width="stretch"):
            with st.form("reporte", clear_on_submit=True):
                lista, fmt = opciones_paises()
                pais = st.selectbox("País", ["(ninguno)"] + lista,
                                    format_func=lambda p: fmt.get(p, p))
                pantalla = st.selectbox("Pantalla", [v[1] for v in PANTALLAS.values()])
                problema = st.text_area("¿Qué está mal?", max_chars=500)
                if st.form_submit_button("Enviar reporte"):
                    if problema.strip():
                        guardar_reporte(pais, pantalla, problema)
                        st.success("Gracias, el reporte quedó guardado.")
                    else:
                        st.error("Escribe qué está mal antes de enviar.")
        st.markdown("**Fuentes de datos**")
        st.caption("Temperatura: [Berkeley Earth](https://berkeleyearth.org/data/) (licencia CC BY-NC-SA 4.0).  \n"
                   "Emisiones de CO₂: [Our World in Data](https://ourworldindata.org/co2-emissions) con datos del "
                   "Global Carbon Project (licencia CC BY 4.0).")


# ---------------------------------------------------------------- piezas visuales
def insignia(nivel: str) -> str:
    color = COLOR_NIVEL.get(nivel, "#6c757d")
    texto = "#1f2937" if nivel == "Media" else "white"
    return (f"<span style='background:{color};color:{texto};padding:2px 10px;border-radius:12px;"
            f"font-weight:600;font-size:0.9rem'>{nivel}</span>")


def tarjeta_resultado(sistema: str, r: dict, real: dict | None = None) -> None:
    """Puntaje, nivel y comparación con el valor real del país (RF-03)."""
    with st.container(border=True):
        st.markdown(f"**Prioridad de {SISTEMA[sistema].lower()}**")
        c1, c2 = st.columns([1, 1])
        delta = None
        if real is not None:
            dif = r["prediccion"] - real["prediccion"]
            delta = "sin cambios" if abs(dif) < 0.05 else f"{dif:+.1f} frente al valor real"
        c1.metric("Puntaje (0 a 100)", f"{r['prediccion']:.1f}", delta=delta,
                  delta_color="off" if delta == "sin cambios" else "inverse",
                  delta_arrow="off" if delta == "sin cambios" else "auto")
        c2.markdown("Nivel")
        c2.markdown(insignia(r["clase_etiqueta"]), unsafe_allow_html=True)
        if real is not None:
            st.caption(f"Valor real: {real['prediccion']:.1f} ({real['clase_etiqueta']})")
        if r.get("estado") == "no_concluyente":
            st.warning("Resultado no concluyente: ninguna regla se activa con fuerza suficiente.")


def grafica_escala(puntaje: float, altura: int = 130) -> go.Figure:
    """G8: barra de 4 tramos con el puntaje marcado."""
    fig = go.Figure()
    bordes = [0, *CORTES, 100]
    for (a, b), nivel in zip(zip(bordes[:-1], bordes[1:]), NIVELES):
        fig.add_trace(go.Bar(x=[b - a], y=[""], base=a, orientation="h", marker_color=COLOR_NIVEL[nivel],
                             name=nivel, hovertemplate=f"{nivel}: {a} a {b}<extra></extra>", showlegend=False,
                             text=nivel, textposition="inside", insidetextanchor="middle"))
    fig.add_annotation(x=puntaje, y=1, yref="paper", yanchor="bottom", text=f"<b>{puntaje:.1f}</b>",
                       showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#111827", ax=0, ay=-24,
                       font=dict(size=13, color="#111827"))
    fig.update_layout(barmode="overlay", height=altura, margin=dict(l=8, r=16, t=52, b=26),
                      xaxis=dict(range=[0, 100], tickvals=[0, *CORTES, 100], fixedrange=True),
                      yaxis=dict(visible=False, fixedrange=True))
    return fig


def grafica_reglas(reglas: list, n: int) -> None:
    """G7: fuerza de activación (0 a 1) de las n reglas más fuertes, como barras horizontales."""
    for x in reglas[:n]:
        st.progress(min(max(float(x["fuerza"]), 0.0), 1.0), text=f"{x['regla']}  \n**Fuerza {x['fuerza']:.2f}**")


def grafica_conjunto(r: dict, sistema: str) -> go.Figure:
    """G5: funciones de salida, conjunto agregado y centroide."""
    from src.models.fuzzy import SALIDA, U
    fig = go.Figure()
    for nivel, mu in SALIDA.items():
        fig.add_trace(go.Scatter(x=U, y=mu, mode="lines", name=nivel,
                                 line=dict(color=COLOR_NIVEL[nivel], dash="dot", width=1.5)))
    ca = r["conjunto_agregado"]
    fig.add_trace(go.Scatter(x=ca["x"], y=ca["mu"], mode="lines", fill="tozeroy", name="Conjunto agregado",
                             line=dict(color="#264653", width=1), fillcolor="rgba(38,70,83,0.35)"))
    fig.add_vline(x=ca["centroide"], line=dict(color="#111827", width=2),
                  annotation_text=f"Centroide {ca['centroide']:.1f}", annotation_position="top")
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), yaxis=dict(range=[0, 1.08], title="Pertenencia"),
                      xaxis=dict(title=f"Prioridad de {SISTEMA[sistema].lower()} (0 a 100)"),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def tabla_descarga(df: pd.DataFrame, nombre_archivo: str, texto: str, key: str) -> None:
    st.download_button(texto, df.to_csv(index=False).encode("utf-8-sig"), file_name=nombre_archivo,
                       mime="text/csv", key=key, icon=":material/download:")


def ultima_ejecucion() -> str:
    """Fecha de la última vez que terminó src/models/train.py (se lee del registro, no de la fecha del archivo)."""
    if not LOG_ENTRENAMIENTO.exists():
        return "sin registro"
    for linea in reversed(LOG_ENTRENAMIENTO.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if "Entrenamiento terminado" in linea:
            try:
                return datetime.strptime(linea[:19], "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y %H:%M")
            except ValueError:
                break
    return datetime.fromtimestamp(LOG_ENTRENAMIENTO.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
