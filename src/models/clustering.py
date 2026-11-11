"""ClimateAI: experimentos de agrupamiento (PR 2).

Responsable: Hector Martínez Sánchez (Líder de Modelado: grupos de países).
Rama: feat/clustering

Los 3 experimentos usan la misma partición de data/processed/particion.csv y los
mismos 6 indicadores (I4 e I5 con log10), estandarizados con la media y la
desviación de Train (el escalador nunca ve Test):

  * Exp 1, modelo base:  K-means k = 4 sobre los 6 indicadores en z-score (sin PCA).
  * Exp 2, candidato:    z-score, PCA de 2 componentes y K-means k = 4.
  * Exp 3, alternativo:  mezcla gaussiana (GaussianMixture) de 4 componentes sobre el mismo PCA.

Métrica principal: silueta en Test (meta 0.30; en Train la meta es 0.35).
Métrica secundaria: índice de Davies-Bouldin en Test (menor es mejor).
Diagnóstico: se compara la silueta de Train con la de Test.

Guarda en models/: scaler.joblib, pca.joblib, kmeans_k4.joblib, gmm_k4.joblib y grupos.json.
Guarda en reports/figuras/: silueta_por_k.png, comparacion_agrupamiento.png,
convergencia_kmeans.png y grupos_pca.png.

Uso (después del PR 1):
    python src/models/clustering.py
"""
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import KMeans, kmeans_plusplus  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (FIGURAS, GMM, GRUPOS, IND, IND_LOG, K, KMEANS, META_SILUETA_TEST,  # noqa: E402
                        META_SILUETA_TRAIN, MODELS, PCA_MODELO, SCALER, SEMILLA, consola_utf8)
from src.data.preprocess import cargar, preparar_matriz  # noqa: E402

log = logging.getLogger("climateai")

# Nombre de cada grupo según su centroide: (calentamiento I1, emisiones acumuladas I4).
# El número de grupo es fijo para que el tablero y los documentos siempre coincidan.
PERFILES = {
    ("bajo", "alto"): (1, "Grandes emisores con calentamiento bajo"),
    ("bajo", "bajo"): (2, "Bajo impacto con calentamiento leve"),
    ("alto", "alto"): (3, "Emisores históricos con calentamiento alto"),
    ("alto", "bajo"): (4, "Bajo impacto con calentamiento alto"),
}


def diagnostico(s_train: float, s_test: float) -> str:
    if s_train < META_SILUETA_TRAIN:
        return f"Subajuste: los grupos no quedan bien separados (silueta en Train < {META_SILUETA_TRAIN})"
    if s_train - s_test > 0.10:
        return "Sobreajuste: la separación cae más de 0.10 al pasar de Train a Test"
    if s_test < META_SILUETA_TEST:
        return f"Generaliza poco: silueta en Test < {META_SILUETA_TEST}"
    return "Óptimo: cumple la meta y la separación se mantiene en Test"


def _fila(exp, modelo, hiper, s_tr, s_te, db_te, t_fit, t_inf, ari):
    return {
        "experimento": exp, "modelo": modelo, "hiperparametros": hiper,
        "metrica_principal": f"Silueta Test: {s_te:.3f}",
        "metrica_secundaria": f"Davies-Bouldin Test: {db_te:.3f}",
        "metrica_train": f"Silueta Train: {s_tr:.3f}",
        "tiempo": f"{t_fit:.3f} s / {t_inf:.2f} ms",
        "diagnostico": diagnostico(s_tr, s_te),
        "silueta_train": round(s_tr, 3), "silueta_test": round(s_te, 3),
        "davies_bouldin_test": round(db_te, 3), "ari_estabilidad": round(ari, 3),
        "tiempo_entrenamiento_s": round(t_fit, 4), "tiempo_inferencia_ms": round(t_inf, 3),
    }


def _ari(modelo_cls, kwargs, x_train, x_test, etiquetas_test):
    """Estabilidad: grupos de Test con el modelo de Train frente a un modelo entrenado con todos."""
    todo = modelo_cls(**kwargs).fit(np.vstack([x_train, x_test]))
    return adjusted_rand_score(todo.predict(x_test), etiquetas_test)


def nombrar_grupos(km: KMeans, scaler: StandardScaler, pca: PCA) -> dict:
    """Asigna id (1 a 4) y nombre a cada etiqueta de K-means según su centroide."""
    centros = scaler.inverse_transform(pca.inverse_transform(km.cluster_centers_))
    centros = pd.DataFrame(centros, columns=IND)
    for c in IND_LOG:
        centros[c] = 10 ** centros[c] - 0.001
    alto_i1 = set(centros["I1"].nlargest(2).index)
    alto_i4 = set(centros["I4"].nlargest(2).index)
    mapa = {}
    for etq in range(K):
        clave = ("alto" if etq in alto_i1 else "bajo", "alto" if etq in alto_i4 else "bajo")
        gid, nombre = PERFILES[clave]
        mapa[etq] = {"id": gid, "nombre": nombre,
                     "centroide": {c: round(float(centros.loc[etq, c]), 4) for c in IND}}
    if len({v["id"] for v in mapa.values()}) != K:
        raise RuntimeError("Dos grupos quedaron con el mismo perfil; revisar el agrupamiento")
    return mapa


def _figuras(ztr, ptr, pte, km, mapa, sil, base, ind, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    ks = list(range(2, 9))
    s6 = [silhouette_score(ztr, KMeans(k, n_init=10, random_state=SEMILLA).fit_predict(ztr)) for k in ks]
    s2 = [silhouette_score(ptr, KMeans(k, n_init=10, random_state=SEMILLA).fit_predict(ptr)) for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ks, s6, "o-", color="#8C8C8C", label="6 indicadores (Exp 1)")
    ax.plot(ks, s2, "o-", color="#1F5FAD", label="PCA de 2 componentes (Exp 2)")
    ax.axhline(META_SILUETA_TRAIN, color="#C0392B", ls="--", lw=1, label="Meta en Train (0.35)")
    ax.axvline(K, color="#555555", ls=":", lw=1)
    ax.set_xlabel("Número de grupos (k)")
    ax.set_ylabel("Silueta en Train")
    ax.set_title("Silueta de K-means según k")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "silueta_por_k.png", dpi=150)
    plt.close(fig)

    # comparación de los 3 experimentos y las 2 líneas base
    etiquetas = ["Exp 1\nK-means 6D", "Exp 2\nK-means PCA", "Exp 3\nGMM PCA"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(len(etiquetas))
    ax.bar(xs - 0.18, [s["train"] for s in sil], 0.36, color="#9DB9DE", label="Train")
    ax.bar(xs + 0.18, [s["test"] for s in sil], 0.36, color="#1F5FAD", label="Test")
    for x, s in zip(xs, sil):
        ax.text(x - 0.18, s["train"] + 0.01, f"{s['train']:.3f}", ha="center", fontsize=8)
        ax.text(x + 0.18, s["test"] + 0.01, f"{s['test']:.3f}", ha="center", fontsize=8)
    ax.axhline(META_SILUETA_TRAIN, color="#C0392B", ls="--", lw=1, label="Meta en Train (0.35)")
    ax.axhline(base["cuartiles_I4"], color="#8C8C8C", ls=":", lw=1,
               label=f"Línea base: cuartiles de I4 ({base['cuartiles_I4']:.3f})")
    ax.set_xticks(xs, etiquetas)
    ax.set_ylabel("Silueta")
    ax.set_ylim(min(0, base["aleatoria"]) - 0.02, 0.6)
    ax.set_title("Comparación de modelos de agrupamiento (mismo Test)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "comparacion_agrupamiento.png", dpi=150)
    plt.close(fig)

    # curva de convergencia: inercia por iteración (algoritmo de Lloyd, mismo inicio k-means++)
    centros, _ = kmeans_plusplus(ptr, K, random_state=SEMILLA)
    inercias = []
    for _ in range(100):
        d = ((ptr[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2)
        etq = d.argmin(axis=1)
        inercias.append(float(d[np.arange(len(ptr)), etq].sum()))
        nuevos = np.array([ptr[etq == j].mean(axis=0) for j in range(K)])
        if np.allclose(nuevos, centros):
            break
        centros = nuevos
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(range(1, len(inercias) + 1), inercias, "o-", color="#1F5FAD")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Inercia (suma de distancias al cuadrado)")
    ax.set_title(f"Convergencia de K-means (Exp 2): {len(inercias)} iteraciones")
    fig.tight_layout()
    fig.savefig(fig_dir / "convergencia_kmeans.png", dpi=150)
    plt.close(fig)

    # países en el plano PC1-PC2 por grupo
    colores = {1: "#1F5FAD", 2: "#2E9E5B", 3: "#C0392B", 4: "#E08E0B"}
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for gid in (1, 2, 3, 4):
        etq = [e for e, v in mapa.items() if v["id"] == gid][0]
        mtr = km.labels_ == etq
        mte = km.predict(pte) == etq
        nombre = mapa[etq]["nombre"]
        ax.scatter(ptr[mtr, 0], ptr[mtr, 1], s=22, color=colores[gid], alpha=0.75, label=f"{gid}. {nombre}")
        ax.scatter(pte[mte, 0], pte[mte, 1], s=40, color=colores[gid], marker="^", edgecolor="black", lw=0.5)
        cx, cy = km.cluster_centers_[etq]
        ax.scatter([cx], [cy], s=160, color=colores[gid], marker="X", edgecolor="black")
    ax.scatter([], [], marker="^", color="white", edgecolor="black", label="Países de Test")
    ax.set_xlabel("PC1 (magnitud)")
    ax.set_ylabel("PC2 (emisiones frente a calentamiento)")
    ax.set_title("Grupos de K-means (Exp 2) en el plano de PCA")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "grupos_pca.png", dpi=150)
    plt.close(fig)


def entrenar(indicadores: pd.DataFrame, out_dir: Path = MODELS, fig_dir: Path = FIGURAS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = indicadores[indicadores["conjunto"] == "Train"]
    te = indicadores[indicadores["conjunto"] == "Test"]
    scaler = StandardScaler().fit(preparar_matriz(tr))
    ztr, zte = scaler.transform(preparar_matriz(tr)), scaler.transform(preparar_matriz(te))
    filas, sil = [], []
    km_kw = dict(n_clusters=K, n_init=10, random_state=SEMILLA)

    # ---- Exp 1: K-means sobre los 6 indicadores
    log.info("Exp 1 (modelo base): K-means k=%d sobre los 6 indicadores en z-score...", K)
    t = time.perf_counter()
    km6 = KMeans(**km_kw).fit(ztr)
    t_fit = time.perf_counter() - t
    t = time.perf_counter()
    e_te = km6.predict(zte)
    t_inf = (time.perf_counter() - t) * 1000
    s_tr, s_te = silhouette_score(ztr, km6.labels_), silhouette_score(zte, e_te)
    filas.append(_fila("Exp 1: Baseline", "K-means (6 indicadores)", f"k = {K}, n_init = 10, distancia euclidiana",
                       s_tr, s_te, davies_bouldin_score(zte, e_te), t_fit, t_inf,
                       _ari(KMeans, km_kw, ztr, zte, e_te)))
    sil.append({"train": s_tr, "test": s_te})

    # ---- Exp 2: PCA de 2 componentes + K-means (candidato)
    log.info("Exp 2 (candidato): z-score, PCA de 2 componentes y K-means k=%d...", K)
    t = time.perf_counter()
    pca = PCA(n_components=2, random_state=SEMILLA).fit(ztr)
    ptr = pca.transform(ztr)
    km = KMeans(**km_kw).fit(ptr)
    t_fit = time.perf_counter() - t
    t = time.perf_counter()
    pte = pca.transform(zte)
    e_te = km.predict(pte)
    t_inf = (time.perf_counter() - t) * 1000
    s_tr, s_te = silhouette_score(ptr, km.labels_), silhouette_score(pte, e_te)
    var = pca.explained_variance_ratio_
    filas.append(_fila("Exp 2: Candidato", "PCA + K-means",
                       f"PCA 2 componentes ({var.sum() * 100:.1f} % de varianza), k = {K}, n_init = 10",
                       s_tr, s_te, davies_bouldin_score(pte, e_te), t_fit, t_inf,
                       _ari(KMeans, km_kw, ptr, pte, e_te)))
    sil.append({"train": s_tr, "test": s_te})

    # ---- Exp 3: mezcla gaussiana sobre el mismo PCA
    log.info("Exp 3 (alternativo): mezcla gaussiana de %d componentes sobre el mismo PCA...", K)
    gmm_kw = dict(n_components=K, covariance_type="full", n_init=10, random_state=SEMILLA)
    t = time.perf_counter()
    gmm = GaussianMixture(**gmm_kw).fit(ptr)
    t_fit = time.perf_counter() - t
    t = time.perf_counter()
    g_te = gmm.predict(pte)
    t_inf = (time.perf_counter() - t) * 1000
    g_tr = gmm.predict(ptr)
    s_tr, s_te = silhouette_score(ptr, g_tr), silhouette_score(pte, g_te)
    filas.append(_fila("Exp 3: Alternativo", "PCA + mezcla gaussiana (GMM)",
                       f"{K} componentes, covarianza completa, n_init = 10",
                       s_tr, s_te, davies_bouldin_score(pte, g_te), t_fit, t_inf,
                       _ari(GaussianMixture, gmm_kw, ptr, pte, g_te)))
    sil.append({"train": s_tr, "test": s_te})

    # ---- líneas base de la Entrega 2 (sin modelo)
    rng = np.random.default_rng(SEMILLA)
    base = {
        "aleatoria": round(float(silhouette_score(ptr, rng.integers(0, K, size=len(ptr)))), 3),
        "cuartiles_I4": round(float(silhouette_score(ptr, pd.qcut(tr["I4"], 4, labels=False).values)), 3),
    }

    # ---- grupos con nombre (modelo candidato) y artefactos
    mapa = nombrar_grupos(km, scaler, pca)
    etq_todos = km.predict(pca.transform(scaler.transform(preparar_matriz(indicadores))))
    ids = pd.Series([mapa[e]["id"] for e in etq_todos], index=indicadores.index)
    grupos = {"etiqueta_a_grupo": {str(e): v["id"] for e, v in mapa.items()}, "grupos": {}}
    for e, v in sorted(mapa.items(), key=lambda kv: kv[1]["id"]):
        m = ids == v["id"]
        grupos["grupos"][str(v["id"])] = {
            "nombre": v["nombre"], "n": int(m.sum()),
            "n_train": int((m & (indicadores["conjunto"] == "Train")).sum()),
            "n_test": int((m & (indicadores["conjunto"] == "Test")).sum()),
            "centroide": v["centroide"],
        }
    grupos["pca_varianza"] = [round(float(v), 4) for v in var]
    grupos["pca_cargas"] = {c: [round(float(pca.components_[0, i]), 3), round(float(pca.components_[1, i]), 3)]
                            for i, c in enumerate(IND)}
    joblib.dump(scaler, out_dir / SCALER.name)
    joblib.dump(pca, out_dir / PCA_MODELO.name)
    joblib.dump(km, out_dir / KMEANS.name)
    joblib.dump(gmm, out_dir / GMM.name)
    (out_dir / GRUPOS.name).write_text(json.dumps(grupos, ensure_ascii=False, indent=2), encoding="utf-8")
    _figuras(ztr, ptr, pte, km, mapa, sil, base, indicadores, fig_dir)

    for f in filas:
        log.info("%s | %s | %s | %s | %s", f["experimento"], f["metrica_train"], f["metrica_principal"],
                 f["metrica_secundaria"], f["diagnostico"])
    log.info("Líneas base (silueta en Train): partición aleatoria %.3f, cuartiles de I4 %.3f",
             base["aleatoria"], base["cuartiles_I4"])
    for gid, g in grupos["grupos"].items():
        log.info("Grupo %s, %s: %d países (%d en Test)", gid, g["nombre"], g["n"], g["n_test"])
    return {"experimentos": filas, "lineas_base": base, "grupos": grupos,
            "grupo_por_pais": dict(zip(indicadores["pais"], ids.astype(int)))}


if __name__ == "__main__":
    consola_utf8()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    entrenar(cargar())
