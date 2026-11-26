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
    """Descarga los datos como Excel: hoja "Datos" con formato y hoja "Gráficas" con gráficas automáticas."""
    from io import BytesIO

    from openpyxl import Workbook
    from openpyxl.chart import BarChart, PieChart, Reference
    from openpyxl.chart.series import DataPoint
    from openpyxl.formatting.rule import DataBarRule
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    AZUL = PatternFill("solid", start_color="1F4E79")
    BLANCO = Font(bold=True, color="FFFFFF")
    CEBRA = PatternFill("solid", start_color="EEF3F8")          # filas alternadas
    BORDE = Border(*(Side(style="thin", color="C9D3DE"),) * 4)
    CENTRO = Alignment(horizontal="center", vertical="center")

    # Encabezados en español y formato de número según la columna original
    nombres = {"Lugar": "Lugar", "pais": "País (inglés)" if "pais_es" in df else "País", "pais_es": "País",
               "iso3": "Código", "grupo": "Grupo de países",
               "puntaje_adaptacion": "Prioridad de adaptación (0 a 100)", "nivel_adaptacion": "Nivel de adaptación",
               "puntaje_mitigacion": "Prioridad de mitigación (0 a 100)", "nivel_mitigacion": "Nivel de mitigación",
               "pais_berkeley": "País (Berkeley Earth)", "motivo": "Motivo", "fecha": "Fecha",
               "pantalla": "Pantalla", "problema": "Problema",
               **{c: f"{INDICADOR[c][0]} ({INDICADOR[c][1]})" for c in INDICADOR}}
    formatos = {"Lugar": "0", "puntaje_adaptacion": "0.0", "puntaje_mitigacion": "0.0", "I1": "0.000",
                "I2": "+0.000;-0.000;0.000", "I3": "0.000", "I4": "#,##0.000", "I5": "0.000",
                "I6": "+0.0;-0.0;0.0"}

    wb = Workbook()

    # ---- hoja 1: título, aviso y la tabla con formato
    ws = wb.active
    ws.title = "Datos"
    ws.sheet_view.showGridLines = False
    ws["A1"] = "ClimateAI: " + nombre_archivo.rsplit(".", 1)[0].replace("climateai_", "").replace("_", " ").capitalize()
    ws["A1"].font = Font(bold=True, size=16, color="1F4E79")
    ws["A2"] = f"{AVISO_PERIODO} Generado el {datetime.now():%d/%m/%Y %H:%M}. {len(df)} filas."
    ws["A2"].font = Font(italic=True, size=9, color="6B7280")
    ENC = 4                                                       # fila del encabezado
    for j, col in enumerate(df.columns, start=1):
        c = ws.cell(ENC, j, nombres.get(col, str(col)))
        c.fill, c.font, c.border = AZUL, BLANCO, BORDE
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[ENC].height = 36
    for i, registro in enumerate(df.itertuples(index=False), start=ENC + 1):
        for j, (col, v) in enumerate(zip(df.columns, registro), start=1):
            v = None if pd.isna(v) else v.item() if hasattr(v, "item") else v
            c = ws.cell(i, j, v)
            c.border = BORDE
            if (i - ENC) % 2 == 0:
                c.fill = CEBRA
            if col in formatos:
                c.number_format, c.alignment = formatos[col], CENTRO
            elif str(col).startswith("nivel_") or col in ("iso3", "Lugar") or isinstance(v, bool):
                c.alignment = CENTRO
            if str(col).startswith("nivel_") and v in COLOR_NIVEL:   # nivel con el color del tablero
                c.fill = PatternFill("solid", start_color=COLOR_NIVEL[v].lstrip("#"))
                c.font = Font(bold=True, color="1F2937" if v == "Media" else "FFFFFF")
    ultima = ENC + len(df)
    for j, col in enumerate(df.columns, start=1):
        letra = get_column_letter(j)
        largo = max([len(nombres.get(col, str(col))) // 2] + [len(str(v)) for v in df[col].head(300)])
        ws.column_dimensions[letra].width = min(max(largo + 3, 9), 45)
        if str(col).startswith("puntaje_") and len(df):          # barra de color dentro de la celda
            color = "F4A261" if col.endswith("adaptacion") else "7FA8C9"
            ws.conditional_formatting.add(f"{letra}{ENC + 1}:{letra}{ultima}",
                                          DataBarRule(start_type="num", start_value=0, end_type="num",
                                                      end_value=100, color=color))
    ws.freeze_panes = ws.cell(ENC + 1, 1)
    ws.auto_filter.ref = f"A{ENC}:{get_column_letter(len(df.columns))}{max(ultima, ENC)}"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_title_rows = f"{ENC}:{ENC}"                          # el encabezado se repite al imprimir

    # ---- hoja 2: gráficas (solo las que se pueden hacer con las columnas que trae la tabla)
    g = wb.create_sheet("Gráficas")
    g.sheet_view.showGridLines = False
    g["A1"] = "ClimateAI: resumen gráfico"
    g["A1"].font = Font(bold=True, size=16, color="1F4E79")
    g["A2"] = AVISO_PERIODO
    g["A2"].font = Font(italic=True, size=9, color="6B7280")
    g.column_dimensions["A"].width = 42
    g.column_dimensions["B"].width = 22
    col_nombre = "pais_es" if "pais_es" in df else "pais" if "pais" in df else None
    fila = 4                                   # fila donde empieza la siguiente tabla auxiliar + gráfica

    def tabla_aux(titulo: str, encabezados: list, filas: list, formato: str = "0") -> int:
        nonlocal fila
        g.cell(fila, 1, titulo).font = Font(bold=True, size=12, color="1F4E79")
        for j, e in enumerate(encabezados, start=1):
            c = g.cell(fila + 1, j, e)
            c.fill, c.font, c.border, c.alignment = AZUL, BLANCO, BORDE, CENTRO
        for k, f in enumerate(filas):
            for j, v in enumerate(f, start=1):
                c = g.cell(fila + 2 + k, j, v.item() if hasattr(v, "item") else v)
                c.border = BORDE
                if k % 2:
                    c.fill = CEBRA
                if j == 2:
                    c.number_format, c.alignment = formato, CENTRO
                if v in COLOR_NIVEL:                               # el nombre del nivel con su color
                    c.fill = PatternFill("solid", start_color=COLOR_NIVEL[v].lstrip("#"))
                    c.font = Font(bold=True, color="1F2937" if v == "Media" else "FFFFFF")
        inicio = fila + 1
        fila += max(len(filas) + 3, 20)        # deja espacio para que no se encimen las gráficas
        return inicio

    # 1) Distribución de niveles (pastel con los colores del tablero)
    for s in ("adaptacion", "mitigacion"):
        col = f"nivel_{s}"
        if col not in df:
            continue
        conteo = df[col].value_counts()
        filas = [[n, int(conteo.get(n, 0))] for n in COLOR_NIVEL]
        ini = tabla_aux(f"Países por nivel de {SISTEMA[s].lower()}", ["Nivel", "Países"], filas)
        pie = PieChart()
        pie.title = f"Nivel de {SISTEMA[s].lower()}"
        pie.add_data(Reference(g, min_col=2, min_row=ini, max_row=ini + 4), titles_from_data=True)
        pie.set_categories(Reference(g, min_col=1, min_row=ini + 1, max_row=ini + 4))
        for i, n in enumerate(COLOR_NIVEL):
            p = DataPoint(idx=i)
            p.graphicalProperties.solidFill = COLOR_NIVEL[n].lstrip("#")
            pie.series[0].dPt.append(p)
        pie.height, pie.width = 7.5, 11
        g.add_chart(pie, f"D{ini}")

    # 2) Los 15 países con mayor puntaje (barras horizontales)
    for s in ("adaptacion", "mitigacion"):
        col = f"puntaje_{s}"
        if col not in df or col_nombre is None:
            continue
        top = df.nlargest(15, col)[[col_nombre, col]]
        ini = tabla_aux(f"Mayor prioridad de {SISTEMA[s].lower()}", ["País", "Puntaje (0 a 100)"],
                        top.values.tolist(), "0.0")
        bar = BarChart()
        bar.type, bar.title, bar.legend = "bar", f"Top 15: prioridad de {SISTEMA[s].lower()}", None
        bar.add_data(Reference(g, min_col=2, min_row=ini, max_row=ini + len(top)), titles_from_data=True)
        bar.set_categories(Reference(g, min_col=1, min_row=ini + 1, max_row=ini + len(top)))
        bar.series[0].graphicalProperties.solidFill = "D62828" if s == "adaptacion" else "457B9D"
        bar.y_axis.scaling.min, bar.y_axis.scaling.max = 0, 100
        bar.y_axis.number_format = "0"
        bar.x_axis.scaling.orientation = "maxMin"      # el puntaje más alto queda arriba
        bar.x_axis.tickLblSkip = 1
        bar.height, bar.width = 9.5, 16
        g.add_chart(bar, f"D{ini}")

    # 3) Países por grupo
    if "grupo" in df:
        conteo = df["grupo"].value_counts()
        ini = tabla_aux("Países por grupo", ["Grupo", "Países"], [[k, int(v)] for k, v in conteo.items()])
        bar = BarChart()
        bar.type, bar.title, bar.legend = "col", "Países por grupo", None
        bar.add_data(Reference(g, min_col=2, min_row=ini, max_row=ini + len(conteo)), titles_from_data=True)
        bar.set_categories(Reference(g, min_col=1, min_row=ini + 1, max_row=ini + len(conteo)))
        bar.series[0].graphicalProperties.solidFill = "2A9D8F"
        bar.height, bar.width = 8, 16
        g.add_chart(bar, f"D{ini}")

    # 4) Mayores emisores históricos
    if "I4" in df and col_nombre is not None:
        top = df.nlargest(10, "I4")[[col_nombre, "I4"]].round(2)
        ini = tabla_aux("Mayores emisores históricos", ["País", "Gt de CO₂ (1850 a 2012)"],
                        top.values.tolist(), "#,##0.00")
        bar = BarChart()
        bar.type, bar.title, bar.legend = "bar", "Top 10: emisiones acumuladas (Gt de CO₂)", None
        bar.add_data(Reference(g, min_col=2, min_row=ini, max_row=ini + len(top)), titles_from_data=True)
        bar.set_categories(Reference(g, min_col=1, min_row=ini + 1, max_row=ini + len(top)))
        bar.series[0].graphicalProperties.solidFill = "264653"
        bar.x_axis.scaling.orientation = "maxMin"
        bar.x_axis.tickLblSkip = 1
        bar.height, bar.width = 8, 16
        g.add_chart(bar, f"D{ini}")

    from openpyxl.chart.layout import Layout, ManualLayout
    for grafica in g._charts:
        grafica.title.overlay = False          # el título va arriba, no encima de la gráfica
        ancho = 0.62 if isinstance(grafica, PieChart) else 0.94   # en el pastel deja lugar a la leyenda
        grafica.layout = Layout(manualLayout=ManualLayout(xMode="edge", yMode="edge", x=0.03, y=0.16, w=ancho, h=0.80))
        if hasattr(grafica, "x_axis"):         # sin esto, Excel 365 a veces oculta los ejes
            grafica.x_axis.delete = grafica.y_axis.delete = False

    if fila == 4:                              # la tabla no tiene columnas para graficar (p. ej. reportes)
        g["A4"] = "Esta tabla no tiene columnas numéricas para graficar; los datos están en la hoja Datos."
        wb.active = 0
    else:
        wb.active = 1                          # el archivo abre directo en las gráficas

    buffer = BytesIO()
    wb.save(buffer)
    st.download_button(texto, buffer.getvalue(), file_name=nombre_archivo.rsplit(".", 1)[0] + ".xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key=key, icon=":material/download:")


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
