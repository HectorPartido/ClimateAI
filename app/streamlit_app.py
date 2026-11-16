"""ClimateAI: punto de entrada del tablero (PR 1).

Responsable: Gildardo Emmanuel González García (Líder de Pipeline e Integración).
Rama: feat/tablero-base

Arranque:
    streamlit run app/streamlit_app.py

El menú lateral muestra las pantallas que ya existen en app/paginas/. Cada integrante
agrega la suya en su PR y aparece sola en el menú.
"""
import streamlit as st

st.set_page_config(page_title="ClimateAI", page_icon=":material/public:", layout="wide")

from comun import APP, PANTALLAS, barra_lateral  # noqa: E402

secciones: dict[str, list] = {}
for clave, (archivo, titulo, icono, seccion) in PANTALLAS.items():
    if (APP / archivo).exists():
        secciones.setdefault(seccion, []).append(
            st.Page(archivo, title=titulo, icon=icono, url_path=clave, default=not secciones and clave == "panorama"))

if not secciones:
    st.title("ClimateAI")
    st.info("Todavía no hay pantallas en app/paginas/.")
    st.stop()

navegacion = st.navigation(secciones)
barra_lateral()
navegacion.run()
