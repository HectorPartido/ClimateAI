"""Pantalla 01: Panorama (PR 2).

Responsable: Ismael Alexander Hidalgo Aguilando.
Rama: feat/panorama-ranking
Requerimientos: RF-03, RF-04, RF-07, RF-09, RF-16 y gráficas G1 y G2.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from comun import (COLOR_NIVEL, SISTEMA, boton_ir, ir_a, opciones_paises, paises, resumen, tabla_descarga)
from src.config import DATA_PROCESSED, NIVELES, SISTEMAS

d = paises()
r = resumen()

st.title("Panorama")
st.caption(f"Prioridad climática de {len(d)} países, calculada con datos históricos de temperatura y de emisiones de CO₂.")

# ---- barra superior: búsqueda (RF-07) y exportar informe (RF-09)
lista, fmt = opciones_paises()
c1, c2, c3 = st.columns([4, 1.4, 1.6], vertical_alignment="bottom")
buscado = c1.selectbox("Buscar país", lista, index=None, format_func=lambda p: fmt[p],
                       placeholder="Escribe el nombre de un país", key="pan_busca")
if c2.button("Abrir ficha", icon=":material/flag:", disabled=buscado is None, width="stretch"):
    ir_a("ficha", pais_ficha=buscado)
informe = d[["pais_es", "iso3", "grupo", "puntaje_adaptacion", "nivel_adaptacion", "puntaje_mitigacion",
             "nivel_mitigacion", "I1", "I2", "I3", "I4", "I5", "I6"]].rename(columns={"pais_es": "pais"})
with c3:
    tabla_descarga(informe, "climateai_informe_prioridades.csv", "Exportar informe", "pan_informe")

# ---- KPIs (RF-03)
emparejados = len(pd.read_csv(DATA_PROCESSED / "equivalencias_paises.csv"))
sin_par = len(pd.read_csv(DATA_PROCESSED / "sin_emparejar.csv"))
silueta = next(e["silueta_train"] for e in r["experimentos"] if e.get("experimento", "").startswith("Exp 2"))
k1, k2, k3, k4 = st.columns(4)
k1.metric("Países analizados", len(d), border=True, delta_arrow="off", delta_color="off",
          delta=f"{100 * emparejados / (emparejados + sin_par):.0f} % de los países emparejados",
          help=f"{emparejados} de {emparejados + sin_par} países de la fuente de temperatura se emparejaron con la "
               f"fuente de emisiones; {len(d)} tienen los 6 indicadores completos. La lista de los que no se "
               f"emparejaron está en Datos y modelo.")
k2.metric("Prioridad crítica de adaptación", int((d["nivel_adaptacion"] == "Crítica").sum()), border=True)
k3.metric("Prioridad crítica de mitigación", int((d["nivel_mitigacion"] == "Crítica").sum()), border=True)
k4.metric("Grupos de países", 4, border=True, delta_arrow="off", delta_color="off",
          delta=f"separación {silueta:.2f} (meta 0.35)",
          help="Qué tan separados quedan los grupos, de 0 a 1: la meta es 0.35 o más.")

# ---- mapa y top 8 (RF-04, G1 y G2)
sistema = st.segmented_control("Prioridad que se muestra", list(SISTEMAS), format_func=SISTEMA.get,
                               default="adaptacion", key="pan_sistema", required=True)
col_p, col_n = f"puntaje_{sistema}", f"nivel_{sistema}"
izq, der = st.columns([3, 2], gap="large")

with izq:
    mapa = px.choropleth(d, locations="iso3", color=col_n, hover_name="pais_es", custom_data=["pais"],
                         hover_data={col_p: ":.1f", "grupo": True, "iso3": False, col_n: False},
                         category_orders={col_n: NIVELES}, color_discrete_map=COLOR_NIVEL,
                         labels={col_n: "Nivel", col_p: "Puntaje", "grupo": "Grupo"})
    mapa.update_geos(showcountries=True, countrycolor="white", showland=True, landcolor="#d1d5db",
                     showframe=False, projection_type="natural earth", showocean=True, oceancolor="#eef6fb",
                     projection_rotation=dict(lon=10), lataxis_range=[-58, 85], lonaxis_range=[-170, 190])
    mapa.update_layout(height=360, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.02),
                       clickmode="event+select")
    evento = st.plotly_chart(mapa, key="g1", on_select="rerun", selection_mode="points",
                             config={"topojsonURL": "app/static/"})   # mapa local (app/static/world_110m.json)
    st.caption("En gris, territorios sin datos suficientes. Da clic en un país para abrir su ficha.")
    puntos = evento.selection.points if evento and evento.selection else []
    if puntos:                                             # RF-16: clic en el mapa abre la ficha
        p = puntos[0]
        pais = (p.get("customdata") or [None])[0]
        if pais is None and p.get("location"):
            pais = d.loc[d["iso3"] == p["location"], "pais"].iloc[0]
        if pais:
            ir_a("ficha", pais_ficha=pais)

with der:
    top = d.nlargest(8, col_p).iloc[::-1]
    barras = px.bar(top, x=col_p, y="pais_es", orientation="h", color=col_n, color_discrete_map=COLOR_NIVEL,
                    category_orders={col_n: NIVELES}, text=col_p,
                    labels={col_p: "Puntaje (0 a 100)", "pais_es": "", col_n: "Nivel"})
    barras.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    barras.update_layout(height=400, margin=dict(l=0, r=10, t=30, b=0), xaxis_range=[0, 105], showlegend=False,
                         title=f"Mayor prioridad de {SISTEMA[sistema].lower()}")
    st.plotly_chart(barras, key="g2")
    boton_ir("Ver ranking completo", "ranking", key="pan_ranking", ranking_orden=sistema)

# ---- resumen de los 4 grupos (RF-03)
st.subheader("Grupos de países")
grupos = r["grupos"]["grupos"]
cols = st.columns(4)
for col, (gid, g) in zip(cols, sorted(grupos.items(), key=lambda x: int(x[0]))):
    with col.container(border=True):
        st.markdown(f"**{g['nombre']}**")
        st.caption(f"{g['n']} países. El país típico del grupo se calienta {g['centroide']['I1']:.2f} °C por "
                   f"década y ha emitido {g['centroide']['I4']:,.2f} Gt de CO₂.")
        boton_ir("Ver grupo", "grupos", key=f"pan_grupo_{gid}", grupo_sel=int(gid))
