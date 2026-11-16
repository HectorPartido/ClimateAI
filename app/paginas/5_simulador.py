"""Pantalla 05: Simulador de prioridad (PR 1).

Responsable: Gildardo Emmanuel González García (Líder de Pipeline e Integración).
Rama: feat/tablero-base
Requerimientos: RF-01, RF-02, RF-03, RF-05, RF-06, RF-13, RF-14 y gráficas G5 y G7.
"""
import streamlit as st

from comun import (PAIS_EJEMPLO, SISTEMA, entrada_pais, grafica_conjunto, grafica_reglas, nombre,
                   opciones_paises, parametros_difusos, percentil, predecir, resultado_pais, tarjeta_resultado)
from src.config import SISTEMAS
from src.models.fuzzy import fuzzificar

# Controles del RF-01: (mínimo, máximo, paso, formato). I2 no tiene control: se toma del país cargado.
CONTROLES = {
    "I1": (0.0, 0.60, 0.001, "%.3f", "Calentamiento (°C por década)"),
    "I3": (0.0, 1.20, 0.001, "%.3f", "Variabilidad de un año a otro (°C)"),
    "I4": (0.0, 400.0, 0.001, "%.3f", "Emisiones acumuladas 1850 a 2012 (Gt de CO₂)"),
    "I5": (0.0, 30.0, 0.001, "%.3f", "Parte de las emisiones mundiales 2003 a 2012 (%)"),
    "I6": (-100.0, 500.0, 0.1, "%.1f", "Crecimiento de emisiones, 2003 a 2012 frente a 1993 a 2002 (%)"),
}
USA_EN = {"I1": ["adaptacion"], "I3": ["adaptacion"], "I4": ["adaptacion", "mitigacion"],
          "I5": ["mitigacion"], "I6": ["mitigacion"]}


def cargar(pais: str) -> None:
    """Llena los controles con los valores reales del país (RF-05, RF-11 y RF-13)."""
    v = entrada_pais(pais)
    st.session_state.sim_base = pais
    st.session_state.sim_I2 = v["I2"]
    for c in CONTROLES:
        st.session_state[f"sl_{c}"] = v[c]
    st.session_state.pop("sim_resultado", None)


def al_elegir_pais() -> None:
    if st.session_state.get("sim_sel"):
        cargar(st.session_state.sim_sel)


# Primera visita: México (RF-05). Si se viene de la Ficha, el país elegido allá (RF-11).
if "pais_simulador" in st.session_state:
    cargar(st.session_state.pop("pais_simulador"))
elif "sim_base" not in st.session_state:
    cargar(PAIS_EJEMPLO)

st.title("Simulador de prioridad")
st.caption("Cambia los indicadores de un país y mira cómo cambian sus prioridades de adaptación y de mitigación.")

lista, fmt = opciones_paises()
c1, c2, c3, c4 = st.columns([3, 2, 1.6, 2.6], vertical_alignment="bottom")
c1.selectbox("Cargar país", lista, index=None, placeholder="Elige un país para cargar sus valores reales",
             format_func=lambda p: fmt[p], key="sim_sel", on_change=al_elegir_pais)
c2.button("Cargar ejemplo (México)", on_click=cargar, args=(PAIS_EJEMPLO,), icon=":material/science:", width="stretch")
c3.button("Restablecer", on_click=lambda: cargar(st.session_state.sim_base), icon=":material/restart_alt:",
          width="stretch", help="Regresa los controles a los valores reales del país cargado")
st.session_state.setdefault("sim_auto", True)
auto = c4.toggle("Recalcular automáticamente", key="sim_auto", persist_state="session")
st.markdown(f"País cargado: **{nombre(st.session_state.sim_base)}**")

params = parametros_difusos()
izq, der = st.columns([5, 6], gap="large")

with izq:
    st.subheader("Indicadores")
    for c, (lo, hi, paso, formato, etiqueta) in CONTROLES.items():
        st.slider(etiqueta, lo, hi, step=paso, format=formato, key=f"sl_{c}", persist_state="session")
    entrada = {c: float(st.session_state[f"sl_{c}"]) for c in CONTROLES}
    entrada["I2"] = float(st.session_state.sim_I2)
    with st.expander("Cómo se ve cada valor para el modelo", icon=":material/insights:"):
        mu = {s: fuzzificar(entrada, params, s) for s in SISTEMAS}
        for c in CONTROLES:
            partes = []
            for s in USA_EN[c]:
                m = mu[s][c]
                terminos = ", ".join(f"{t} {m[t]:.2f}" for t in m if not t.startswith("no_"))
                partes.append(f"{SISTEMA[s].lower()}: {terminos}" if len(USA_EN[c]) > 1 else terminos)
            st.markdown(f"**{c}** = {entrada[c]:,.3f}, más alto que el {percentil(c, entrada[c])} % de los países  \n"
                        f"<small>{' | '.join(partes)}</small>", unsafe_allow_html=True)
    calcular = st.button("Calcular prioridad", type="primary", icon=":material/play_arrow:", width="stretch")

# RF-02: con el interruptor encendido se recalcula en cada cambio; apagado, solo con el botón.
pendiente = st.session_state.get("sim_entrada") != entrada
if auto or calcular or "sim_resultado" not in st.session_state:
    with st.spinner("Calculando prioridad..."):
        try:
            st.session_state.sim_resultado = {s: predecir({**entrada, "sistema": s}) for s in SISTEMAS}
            st.session_state.sim_entrada = entrada
            pendiente = False
        except ValueError as e:      # RF-14: se muestra la advertencia y se conserva el último resultado válido
            st.warning(f"Entrada no válida: {e}")

res = st.session_state.sim_resultado
reales = {s: resultado_pais(st.session_state.sim_base, s) for s in SISTEMAS}

with der:
    st.subheader("Resultado")
    if pendiente:
        st.info("Hay cambios sin calcular: presiona **Calcular prioridad**.", icon=":material/pending:")
    t1, t2 = st.columns(2)
    with t1:
        tarjeta_resultado("adaptacion", res["adaptacion"], reales["adaptacion"])
    with t2:
        tarjeta_resultado("mitigacion", res["mitigacion"], reales["mitigacion"])
    st.caption(f"Grupo de países: **{res['adaptacion']['grupo_kmeans']['nombre']}**. "
               f"Calculado en {max(r['tiempo_ms'] for r in res.values()):.1f} ms.")

    sistema = st.segmented_control("Ver el detalle de", list(SISTEMAS), format_func=SISTEMA.get,
                                   default="adaptacion", key="sim_sistema", required=True)   # RF-06
    r = res[sistema]
    st.markdown(r["explicacion"])
    st.plotly_chart(grafica_conjunto(r, sistema), key="g5")                                    # G5
    st.markdown("**Reglas que se activaron** (las 5 más fuertes)")
    grafica_reglas(r["reglas_activadas"], 5)                                               # G7

with st.expander("Cómo funciona", icon=":material/help:"):
    st.markdown(
        "1. Cada indicador se compara con los demás países y se describe como **bajo**, **medio** o **alto** "
        "(un valor puede ser en parte medio y en parte alto).\n"
        "2. Se aplican reglas del tipo *SI el calentamiento es alto Y la variabilidad es alta ENTONCES la "
        "prioridad de adaptación es crítica*. Cada regla se activa con una fuerza de 0 a 1.\n"
        "3. Las reglas activadas se combinan en una sola figura (la zona sombreada de la gráfica) y su punto de "
        "equilibrio es el puntaje de 0 a 100.\n"
        "4. El puntaje se traduce a nivel: Baja (menos de 25), Media (25 a 50), Alta (50 a 75) y Crítica (75 o más).\n\n"
        "Adaptación usa calentamiento, variabilidad y emisiones acumuladas (como señal de capacidad para adaptarse). "
        "Mitigación usa emisiones acumuladas, parte de las emisiones mundiales y crecimiento de las emisiones.")

with st.expander("Probar una entrada escrita a mano", icon=":material/keyboard:"):
    st.caption("Sirve para revisar qué pasa con valores fuera de rango o que no son números: el simulador avisa "
               "y conserva el último resultado válido.")
    with st.form("manual"):
        cols = st.columns(6)
        textos = {c: cols[i].text_input(c, value=f"{entrada[c]:.3f}")
                  for i, c in enumerate(["I1", "I2", "I3", "I4", "I5", "I6"])}
        if st.form_submit_button("Probar entrada"):
            valores = {}
            for c, t in textos.items():
                try:
                    valores[c] = float(t.replace(",", "."))
                except ValueError:
                    valores[c] = t
            try:
                prueba = {s: predecir({**valores, "sistema": s}) for s in SISTEMAS}
                st.success(" | ".join(f"{SISTEMA[s]}: {prueba[s]['prediccion']:.1f} ({prueba[s]['clase_etiqueta']})"
                                      for s in SISTEMAS))
            except ValueError as e:
                st.warning(f"Entrada no válida: {e}. Se conserva el último resultado válido.")
