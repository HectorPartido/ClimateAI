"""Pantalla 04: Grupos de países (PR 3).

Responsable: Hector Martínez Sánchez.
Rama: feat/ficha-grupos
Requerimientos: RF-07, RF-12, RF-16 y gráficas G3 y G4.

Con k = 4 se muestran los grupos oficiales del modelo (models/kmeans_k4.joblib). Con otro k
se entrena un K-means de exploración sobre los mismos 2 componentes de Train (no se guarda).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from comun import COLOR_GRUPO, PAIS_EJEMPLO, boton_ir, ir_a, modelos, opciones_paises, paises, resumen
from src.config import IND, META_SILUETA_TRAIN, SEMILLA
from src.data.preprocess import preparar_matriz


@st.cache_data(show_spinner=False)
def componentes() -> pd.DataFrame:
    """PC1 y PC2 de los 201 países con el escalador y el PCA guardados (ajustados solo con Train)."""
    m = modelos()
    d = paises()
    pc = m["pca"].transform(m["scaler"].transform(preparar_matriz(d)))
    return d.assign(PC1=pc[:, 0], PC2=pc[:, 1])


def a_indicadores(centros_pc: np.ndarray) -> np.ndarray:
    """Regresa centros del plano PC1 y PC2 a la escala original (deshace PCA, z-score y log10 de I4 e I5)."""
    m = modelos()
    x = m["scaler"].inverse_transform(m["pca"].inverse_transform(centros_pc))
    x[:, 3:5] = 10 ** x[:, 3:5] - 0.001
    return x


@st.cache_data(show_spinner=False)
def agrupar(k: int) -> tuple[pd.DataFrame, dict, dict]:
    """Grupo de cada país, nombres y centro de cada grupo (I1 a I6) para el k elegido."""
    d = componentes().copy()
    if k == 4:                                                   # grupos oficiales (models/kmeans_k4.joblib)
        info = resumen()["grupos"]["grupos"]
        nombres = {int(i): g["nombre"] for i, g in info.items()}
        centros = {int(i): g["centroide"] for i, g in info.items()}
        d["g"] = d["grupo_id"].astype(int)
    else:                                                        # exploración (RF-12)
        tr = d["conjunto"] == "Train"
        km = KMeans(k, n_init=10, random_state=SEMILLA).fit(d.loc[tr, ["PC1", "PC2"]].to_numpy())
        etiquetas = km.predict(d[["PC1", "PC2"]].to_numpy())
        x = a_indicadores(km.cluster_centers_)
        # se numeran de mayor a menor emisión acumulada del centro para que los colores sean estables
        orden = np.argsort(-x[:, 3])
        mapa = {int(e): i + 1 for i, e in enumerate(orden)}
        d["g"] = [mapa[int(e)] for e in etiquetas]
        nombres = {i: f"Grupo {i}" for i in range(1, k + 1)}
        centros = {mapa[int(e)]: dict(zip(IND, map(float, x[e]))) for e in range(k)}
    return d, nombres, centros


@st.cache_data(show_spinner=False)
def siluetas() -> pd.DataFrame:
    """G4: separación de los grupos en Train para k = 2 a 6."""
    tr = componentes().query("conjunto == 'Train'")[["PC1", "PC2"]].to_numpy()
    filas = []
    for k in range(2, 7):
        e = KMeans(k, n_init=10, random_state=SEMILLA).fit_predict(tr)
        filas.append({"k": k, "silueta": silhouette_score(tr, e)})
    return pd.DataFrame(filas)


# ---- estado que llega desde Panorama (Ver grupo)
if "grupo_sel" in st.session_state:
    st.session_state.gr_k = 4
    st.session_state.gr_visibles = [st.session_state.pop("grupo_sel")]
st.session_state.setdefault("gr_k", 4)
st.session_state.setdefault("gr_pais", PAIS_EJEMPLO)

st.title("Grupos de países")
st.caption("El modelo junta a los países que se parecen en calentamiento y en emisiones. "
           "Los grupos oficiales son 4; puedes explorar otras cantidades.")

k = st.segmented_control("Número de grupos (k)", [3, 4, 5, 6], key="gr_k", required=True, persist_state="session",
                         format_func=lambda x: f"{x} (oficial)" if x == 4 else str(x))
d, nombres, centros = agrupar(k)
if st.session_state.get("gr_visibles") and not set(st.session_state.gr_visibles) <= set(nombres):
    del st.session_state["gr_visibles"]
st.session_state.setdefault("gr_visibles", list(nombres))
visibles = st.pills("Grupos visibles", list(nombres), selection_mode="multi", key="gr_visibles",
                    format_func=nombres.get)
if k != 4:
    st.info(f"Exploración con k = {k}: estos grupos no se guardan; las prioridades siguen usando los 4 oficiales.",
            icon=":material/science:")

lista, fmt = opciones_paises()
pais = st.selectbox("Resaltar un país", lista, format_func=lambda p: fmt[p], key="gr_pais", persist_state="session")

# ---- G3: dispersión I1 frente a I4 (escala logarítmica) con centroides
izq, der = st.columns([3, 2], gap="large")
with izq:
    fig = go.Figure()
    for g in visibles or []:
        sub = d[d["g"] == g]
        fig.add_trace(go.Scatter(
            x=sub["I1"], y=sub["I4"] + 0.001, mode="markers", name=nombres[g], customdata=sub[["pais", "pais_es"]],
            marker=dict(size=8, color=COLOR_GRUPO[g], opacity=0.75, line=dict(width=0.5, color="white")),
            hovertemplate="<b>%{customdata[1]}</b><br>Calentamiento: %{x:.3f} °C por década<br>"
                          "Emisiones acumuladas: %{y:,.2f} Gt<extra>" + nombres[g] + "</extra>"))
        c = centros[g]
        fig.add_trace(go.Scatter(x=[c["I1"]], y=[c["I4"] + 0.001], mode="markers", showlegend=False,
                                 marker=dict(symbol="diamond", size=16, color=COLOR_GRUPO[g],
                                             line=dict(width=2, color="#111827")),
                                 hovertemplate=f"Centro de {nombres[g]}<extra></extra>"))
    p = d[d["pais"] == pais].iloc[0]
    fig.add_trace(go.Scatter(x=[p["I1"]], y=[p["I4"] + 0.001], mode="markers+text", text=[p["pais_es"]],
                             textposition="top center", showlegend=False, hoverinfo="skip",
                             marker=dict(symbol="circle-open", size=20, color="#111827", line=dict(width=3))))
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.18),
                      xaxis=dict(title="Calentamiento 1970 a 2012 (°C por década)"),
                      yaxis=dict(title="Emisiones acumuladas 1850 a 2012 (Gt de CO₂, escala log)", type="log"))
    evento = st.plotly_chart(fig, key="g3", on_select="rerun", selection_mode="points")
    st.caption("Los rombos son el centro de cada grupo. Da clic en un punto para abrir la ficha del país.")
    puntos = evento.selection.points if evento and evento.selection else []
    if puntos and puntos[0].get("customdata"):                   # RF-16
        ir_a("ficha", pais_ficha=puntos[0]["customdata"][0])

with der:
    st.markdown(f"**Los 4 países más parecidos a {p['pais_es']}**")
    dist = np.hypot(d["PC1"] - p["PC1"], d["PC2"] - p["PC2"])
    cerca = d.assign(dist=dist).query("pais != @pais").nsmallest(4, "dist")
    for _, f in cerca.iterrows():
        a, b = st.columns([3, 2], vertical_alignment="center")
        a.markdown(f"{f['pais_es']}  \n<small>{nombres[int(f['g'])]}</small>", unsafe_allow_html=True)
        if b.button("Ver ficha", key=f"gr_ficha_{f['pais']}"):
            ir_a("ficha", pais_ficha=f["pais"])
    st.caption("Parecido medido como la distancia en el mapa de los 2 componentes principales (PC1 y PC2).")

    st.markdown("**Separación según el número de grupos**")
    s = siluetas()
    g4 = go.Figure(go.Bar(x=s["k"], y=s["silueta"], text=s["silueta"].round(3), textposition="outside",
                          marker_color=["#2563eb" if x == 4 else "#9ca3af" for x in s["k"]],
                          hovertemplate="k = %{x}: %{y:.3f}<extra></extra>"))
    g4.add_hline(y=META_SILUETA_TRAIN, line=dict(color="#d62828", dash="dash"))
    g4.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(range=[0, 0.6], title="Silueta (Train)"),
                     xaxis=dict(title="k", tickvals=list(s["k"])))
    st.plotly_chart(g4, key="g4", config={"displayModeBar": False})
    st.caption(f"Línea roja: meta de {META_SILUETA_TRAIN}. En azul, los 4 grupos oficiales.")

# ---- tarjetas de grupo con acceso al ranking filtrado
st.subheader("Resumen de los grupos")
cols = st.columns(len(nombres))
for col, g in zip(cols, nombres):
    sub = d[d["g"] == g]
    with col.container(border=True):
        st.markdown(f"<span style='color:{COLOR_GRUPO[g]};font-size:1.2rem'>●</span> **{nombres[g]}**",
                    unsafe_allow_html=True)
        st.caption(f"{len(sub)} países. El país típico del grupo se calienta {centros[g]['I1']:.2f} °C por década "
                   f"y ha emitido {centros[g]['I4']:,.2f} Gt de CO₂.")
        st.caption("Ejemplos: " + ", ".join(sub.nlargest(3, "I4")["pais_es"]))
        if k == 4:
            boton_ir("Ver en ranking", "ranking", key=f"gr_rk_{g}", ranking_grupo=g)
