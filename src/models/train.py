"""ClimateAI: entrenamiento completo en una sola línea (PR 4).

Responsable: Gildardo Emmanuel González García (Líder de Pipeline e Integración).
Rama: feat/pipeline-predict

Pasos:
  1. src.data.preprocess.main(): indicadores I1 a I6 y partición Train/Test.
  2. src.models.clustering.entrenar(): Exp 1 a 3 (grupos de países).
  3. src.models.fuzzy.ajustar() y evaluar(): Exp 4 y 5 (prioridades).
  4. Matriz de experimentación en reports/experimentos.csv y números en reports/resumen.json.
  5. Prueba final de predecir() con los 201 países y los 2 sistemas (reports/niveles_pais.csv).
El registro sale en consola y en logs/entrenamiento.log con "[INFO]" y "[SUCCESS]".

Comando de reproducibilidad (el que va en el documento):
    python src/models/train.py --raw data/raw --output models/
"""
import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (DATA_PROCESSED, DATA_RAW, DIFUSO, EXPERIMENTOS, FIGURAS, LOG_ENTRENAMIENTO,  # noqa: E402
                        MODELS, REPORTS, RESUMEN, SEMILLA, SISTEMAS, asegurar_carpetas, consola_utf8)

SUCCESS = 25
logging.addLevelName(SUCCESS, "SUCCESS")
log = logging.getLogger("climateai")

NIVELES_PAIS = REPORTS / "niveles_pais.csv"
COLUMNAS = {  # nombre interno -> encabezado de la matriz de la plantilla
    "experimento": "Variante", "modelo": "Modelo", "hiperparametros": "Hiperparámetros",
    "metrica_principal": "Métrica principal (Test)", "metrica_secundaria": "Métrica secundaria (Test)",
    "metrica_train": "Métrica en Train", "tiempo": "Tiempo (entrenamiento / inferencia)",
    "diagnostico": "Diagnóstico", "verificaciones": "Verificaciones",
}


def configurar_registro(archivo: Path = LOG_ENTRENAMIENTO) -> None:
    archivo.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    consola = logging.StreamHandler(sys.stdout)
    consola.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    a_archivo = logging.FileHandler(archivo, mode="w", encoding="utf-8")
    a_archivo.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(consola)
    log.addHandler(a_archivo)


def versiones() -> dict:
    import matplotlib
    import numpy
    import sklearn
    import skfuzzy
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "scikit-fuzzy": skfuzzy.__version__,
            "matplotlib": matplotlib.__version__}


def relativa(ruta: Path) -> str:
    try:
        return ruta.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return ruta.as_posix()


def verificar_predecir(indicadores: pd.DataFrame, grupo_por_pais: dict) -> dict:
    """Llama a predecir() como lo hará el tablero: 201 países x 2 sistemas."""
    from src.models.predict import EJEMPLO_MEXICO, _artefactos, _resumen, predecir
    _artefactos.cache_clear()
    filas, tiempos, no_conc, grupo_distinto = [], [], 0, 0
    for _, f in indicadores.iterrows():
        fila = {"pais": f["pais"], "iso3": f["iso3"], "conjunto": f["conjunto"]}
        for s in SISTEMAS:
            r = predecir({**{c: float(f[c]) for c in ("I1", "I2", "I3", "I4", "I5", "I6")}, "sistema": s})
            tiempos.append(r["tiempo_ms"])
            no_conc += r["estado"] == "no_concluyente"
            fila.update({"grupo_id": r["grupo_kmeans"]["id"], "grupo": r["grupo_kmeans"]["nombre"],
                         f"puntaje_{s}": r["prediccion"], f"nivel_{s}": r["clase_etiqueta"],
                         f"fuerza_{s}": r["confianza_o_score"]})
        grupo_distinto += fila["grupo_id"] != grupo_por_pais[f["pais"]]
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(NIVELES_PAIS, index=False, encoding="utf-8")
    mexico = {s: _resumen(predecir({**EJEMPLO_MEXICO, "sistema": s})) for s in SISTEMAS}
    tiempos = pd.Series(tiempos[2:])  # sin las 2 primeras llamadas (cargan los modelos)
    return {
        "llamadas": 2 * len(indicadores), "no_concluyentes": int(no_conc), "grupo_distinto": int(grupo_distinto),
        "tiempo_ms_promedio": round(float(tiempos.mean()), 2), "tiempo_ms_p95": round(float(tiempos.quantile(0.95)), 2),
        "tiempo_ms_maximo": round(float(tiempos.max()), 2),
        "distribucion": {s: tabla[f"nivel_{s}"].value_counts().to_dict() for s in SISTEMAS},
        "mexico": mexico,
    }


def main(raw_dir: Path = DATA_RAW, out_dir: Path = MODELS) -> dict:
    from src.data import preprocess
    from src.models import clustering, fuzzy

    inicio = time.perf_counter()
    asegurar_carpetas()
    configurar_registro()
    log.info("ClimateAI, Entrega 3: entrenamiento y experimentación (random_state = %d)", SEMILLA)
    log.info("Versiones: %s", ", ".join(f"{k} {v}" for k, v in versiones().items()))

    log.info("Cargando datasets desde %s...", relativa(raw_dir))
    indicadores = preprocess.main(raw_dir, DATA_PROCESSED)
    log.log(SUCCESS, "Indicadores y partición guardados en %s", relativa(DATA_PROCESSED))

    log.info("Entrenando agrupamiento (Exp 1 a 3)...")
    agrup = clustering.entrenar(indicadores, out_dir, FIGURAS)
    log.log(SUCCESS, "Modelos guardados en %s: scaler.joblib, pca.joblib, kmeans_k4.joblib, gmm_k4.joblib y grupos.json",
            relativa(out_dir))

    log.info("Ajustando los sistemas difusos con los percentiles de Train y evaluando en Test (Exp 4 a 6)...")
    params = fuzzy.ajustar(indicadores, out_dir / DIFUSO.name)
    dif = fuzzy.evaluar(indicadores, params, FIGURAS)
    for f in dif["experimentos"]:
        log.info("%s | %s | %s | %s", f["experimento"], f["metrica_principal"], f["metrica_secundaria"], f["diagnostico"])
    for exp in ("Exp 5", "Exp 6"):
        v = dif["verificaciones"][exp]
        for s in SISTEMAS:
            c, co = v["cobertura"][s], v["coherencia"][s]
            log.info("%s %s: cobertura %d combinaciones (fuerza dominante mínima %.2f, %d no concluyentes) | "
                     "coherencia %d retrocesos en %d pasos (%d bajan de nivel) | acuerdo con Exp 4 %.0f %%",
                     exp, s, c["combinaciones"], c["fuerza_dominante_minima"], c["no_concluyentes"],
                     co["retrocesos"], co["pasos"], co["bajadas_de_nivel"], 100 * v["acuerdo_con_exp4"][s])
    if not dif["experto_disponible"]:
        log.warning("Sin niveles del experto en data/expert/niveles_experto.csv: concordancia y F1 quedan pendientes")
    log.log(SUCCESS, "Sistemas difusos guardados en %s", relativa(out_dir / DIFUSO.name))

    filas = agrup["experimentos"] + dif["experimentos"]
    matriz = pd.DataFrame(filas).reindex(columns=list(COLUMNAS)).rename(columns=COLUMNAS)
    matriz["Verificaciones"] = matriz["Verificaciones"].fillna(
        pd.Series([f"ARI de estabilidad en Test: {f['ari_estabilidad']:.3f}" for f in agrup["experimentos"]]))
    matriz.to_csv(EXPERIMENTOS, index=False, encoding="utf-8-sig")
    log.log(SUCCESS, "Matriz de experimentación guardada en %s", relativa(EXPERIMENTOS))

    if out_dir.resolve() == MODELS.resolve():
        log.info("Probando predecir() con los %d países y los 2 sistemas...", len(indicadores))
        pred = verificar_predecir(indicadores, agrup["grupo_por_pais"])
        log.info("predecir(): %d llamadas, %d no concluyentes, tiempo promedio %.2f ms (máximo %.2f ms; meta <= 1000 ms)",
                 pred["llamadas"], pred["no_concluyentes"], pred["tiempo_ms_promedio"], pred["tiempo_ms_maximo"])
        log.log(SUCCESS, "Niveles de los 201 países guardados en %s", relativa(NIVELES_PAIS))
    else:
        pred = None
        log.warning("predecir() lee models/; se omitió la prueba final porque --output es %s", out_dir)

    resumen = {
        "semilla": SEMILLA, "versiones": versiones(),
        "paises": len(indicadores), "train": int((indicadores["conjunto"] == "Train").sum()),
        "test": int((indicadores["conjunto"] == "Test").sum()),
        "experimentos": filas, "lineas_base_agrupamiento": agrup["lineas_base"], "grupos": agrup["grupos"],
        "difuso": {k: dif[k] for k in ("verificaciones", "experto_disponible", "metricas_experto", "niveles_test")},
        "percentiles_train": params["percentiles_train"], "cuartiles_linea_base": params["cuartiles_linea_base"],
        "predecir": pred, "segundos_totales": round(time.perf_counter() - inicio, 1),
    }
    RESUMEN.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    log.log(SUCCESS, "Entrenamiento terminado en %.1f s. Figuras en %s y registro en %s",
            resumen["segundos_totales"], relativa(FIGURAS), relativa(LOG_ENTRENAMIENTO))
    return resumen


if __name__ == "__main__":
    consola_utf8()
    ap = argparse.ArgumentParser(description="Entrenamiento completo de ClimateAI")
    ap.add_argument("--raw", type=Path, default=DATA_RAW, help="carpeta con los 3 CSV originales")
    ap.add_argument("--output", type=Path, default=MODELS, help="carpeta donde se guardan los modelos")
    a = ap.parse_args()
    main(a.raw, a.output)
