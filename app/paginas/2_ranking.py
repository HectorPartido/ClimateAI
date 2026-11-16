"""Pantalla 02: Ranking de países (PR 2).

Responsable: Ismael Alexander Hidalgo Aguilando.
Rama: feat/panorama-ranking
Requerimientos: RF-04, RF-07, RF-08, RF-09 y RF-14 (estado sin datos).
"""
import math
import unicodedata

import streamlit as st

from comun import SISTEMA, ir_a, paises, resumen, tabla_descarga
from src.config import NIVELES, SISTEMAS

POR_PAGINA = 12


def simple(texto: str) -> str:
    """Minúsculas y sin acentos, para que "mexico" encuentre "México"."""
    return "".join(c for c in unicodedata.normalize("NFD", texto.casefold()) if unicodedata.category(c) != "Mn")


def primera_pagina() -> None:
    st.session_state.rk_pagina = 1


d = paises()
grupos = {int(k): v["nombre"] for k, v in resumen()["grupos"]["grupos"].items()}

# Valores que llegan desde otras pantallas (Panorama y Grupos)
if "ranking_orden" in st.session_state:
    st.session_state.rk_orden = st.session_state.pop("ranking_orden")
if "ranking_grupo" in st.session_state:
    st.session_state.rk_grupos = [st.session_state.pop("ranking_grupo")]
    st.session_state.rk_pagina = 1
st.session_state.setdefault("rk_orden", "adaptacion")
st.session_state.setdefault("rk_niveles", [])
st.session_state.setdefault("rk_grupos", [])

st.title("Ranking de países")
st.caption("Ordena y filtra los países por su prioridad de adaptación o de mitigación. "
           "Selecciona una fila para abrir la ficha del país.")

c1, c2 = st.columns([2, 3], vertical_alignment="bottom")
orden = c1.segmented_control("Prioridad a mostrar", list(SISTEMAS), format_func=SISTEMA.get, key="rk_orden",
                             on_change=primera_pagina, required=True, persist_state="session")
busca = c2.text_input("Buscar país", key="rk_busca", on_change=primera_pagina, placeholder="Ej. México",
                      persist_state="session")
c3, c4 = st.columns([2, 3], vertical_alignment="bottom")
niveles = c3.pills("Nivel (sin elegir = todos)", NIVELES, selection_mode="multi", key="rk_niveles",
                   on_change=primera_pagina, persist_state="session")
sel_grupos = c4.multiselect("Grupo", sorted(grupos), format_func=grupos.get, key="rk_grupos",
                            on_change=primera_pagina, persist_state="session", placeholder="Todos los grupos")

col_p, col_n = f"puntaje_{orden}", f"nivel_{orden}"
f = d[d[col_n].isin(niveles or NIVELES) & d["grupo_id"].isin(sel_grupos or grupos)].copy()
f = f.sort_values(col_p, ascending=False)
f.insert(0, "Lugar", range(1, len(f) + 1))
if busca.strip():                                              # RF-07: búsqueda sin acentos ni mayúsculas
    clave = simple(busca.strip())
    f = f[f["pais_es"].map(simple).str.contains(clave, regex=False) | f["pais"].map(simple).str.contains(clave, regex=False)]

tabla = f[["Lugar", "pais", "pais_es", "puntaje_adaptacion", "nivel_adaptacion", "puntaje_mitigacion",
           "nivel_mitigacion", "grupo", "I4", "I5", "I6"]]

if tabla.empty:                                                  # RF-14: estado sin datos
    st.info("Sin coincidencias: ningún país cumple con la búsqueda y los filtros elegidos.", icon=":material/search_off:")
    st.stop()

paginas = math.ceil(len(tabla) / POR_PAGINA)
st.session_state.setdefault("rk_pagina", 1)
if st.session_state.rk_pagina > paginas:
    st.session_state.rk_pagina = 1
pagina = st.session_state.rk_pagina
vista = tabla.iloc[(pagina - 1) * POR_PAGINA: pagina * POR_PAGINA]

evento = st.dataframe(
    vista.drop(columns=["pais"]), hide_index=True, on_select="rerun", selection_mode="single-row", key="rk_tabla",
    height=35 * (len(vista) + 1) + 3,
    column_config={
        "Lugar": st.column_config.NumberColumn("Lugar", width="small"),
        "pais_es": st.column_config.TextColumn("País", width="medium"),
        "puntaje_adaptacion": st.column_config.ProgressColumn("Adaptación", min_value=0, max_value=100, format="%.1f",
                                                            width="small"),
        "nivel_adaptacion": st.column_config.TextColumn("Nivel ad."),
        "puntaje_mitigacion": st.column_config.ProgressColumn("Mitigación", min_value=0, max_value=100, format="%.1f",
                                                            width="small"),
        "nivel_mitigacion": st.column_config.TextColumn("Nivel mit."),
        "grupo": st.column_config.TextColumn("Grupo", width="medium"),
        "I4": st.column_config.NumberColumn("Emisiones (Gt)", format="%.2f", help="Emisiones acumuladas 1850 a 2012"),
        "I5": st.column_config.NumberColumn("Parte mundial (%)", format="%.2f"),
        "I6": st.column_config.NumberColumn("Crecimiento (%)", format="%.1f", help="2003 a 2012 frente a 1993 a 2002"),
    })
filas = evento.selection.rows if evento and evento.selection else []
if filas:                                                         # RF-08: la fila abre la ficha
    ir_a("ficha", pais_ficha=vista.iloc[filas[0]]["pais"])

p1, p2, p3, p4 = st.columns([1, 2, 1, 4], vertical_alignment="center")
p1.button("Anterior", icon=":material/chevron_left:", disabled=pagina <= 1, key="rk_ant",
          on_click=lambda: st.session_state.update(rk_pagina=pagina - 1))
p2.markdown(f"<div style='text-align:center'>Página {pagina} de {paginas} "
            f"({len(tabla)} {'país' if len(tabla) == 1 else 'países'})</div>",
            unsafe_allow_html=True)
p3.button("Siguiente", icon=":material/chevron_right:", icon_position="right", disabled=pagina >= paginas,
          key="rk_sig", on_click=lambda: st.session_state.update(rk_pagina=pagina + 1))
with p4:                                                          # RF-09: exportar las filas filtradas
    tabla_descarga(tabla.drop(columns=["pais"]).rename(columns={"pais_es": "pais"}),
                   f"climateai_ranking_{orden}.csv", "Exportar CSV", "rk_csv")
