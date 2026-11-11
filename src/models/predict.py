"""ClimateAI: función de inferencia desacoplada predecir() (PR 4).

Responsable: Gildardo Emmanuel González García (Líder de Pipeline e Integración).
Rama: feat/pipeline-predict

Es el contrato entre el motor de IA y el tablero de la Entrega 4: el tablero solo
llama a predecir() y nunca entrena nada. La Entrega 2 la ubicaba en src/modelo.py;
aquí vive en src/models/predict.py, que es la estructura que pide la plantilla.

Entrada (sección 5 de la Entrega 2): dict con I1 a I6 en unidades originales y
"sistema" ("adaptacion" o "mitigacion"); "pais" es opcional. Si falta una llave, un
valor no es número, está fuera de rango o el sistema no es válido: ValueError.

Salida: prediccion (puntaje 0 a 100), clase_etiqueta (Baja, Media, Alta o Crítica),
confianza_o_score (fuerza de la regla dominante, 0 a 1) y tiempo_ms, más las llaves
extra grupo_kmeans, explicacion, reglas_activadas, conjunto_agregado y estado
("ok" o "no_concluyente" si la fuerza dominante es menor que 0.30).

Prueba rápida (después de entrenar):
    python src/models/predict.py
"""
import json
import sys
import time
from functools import lru_cache
from numbers import Real
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (DIFUSO, GRUPOS, KMEANS, MODELS, PCA_MODELO, RANGOS, SCALER, SISTEMAS,  # noqa: E402
                        consola_utf8)
from src.data.preprocess import preparar_matriz  # noqa: E402
from src.models.fuzzy import cargar_params, explicar, inferir  # noqa: E402

# Ejemplo de la Entrega 2 (México, corte 2012)
EJEMPLO_MEXICO = {"pais": "Mexico", "I1": 0.216, "I2": 0.551, "I3": 0.233,
                  "I4": 16.428, "I5": 1.483, "I6": 25.9, "sistema": "mitigacion"}


@lru_cache(maxsize=1)
def _artefactos(carpeta: str = str(MODELS)) -> dict:
    """Carga los modelos una sola vez (la primera llamada) y los reutiliza."""
    c = Path(carpeta)
    archivos = (SCALER.name, PCA_MODELO.name, KMEANS.name, DIFUSO.name, GRUPOS.name)
    faltan = [n for n in archivos if not (c / n).exists()]
    if faltan:
        raise FileNotFoundError(f"Faltan artefactos en {c}: {', '.join(faltan)}. "
                                "Corre primero: python src/models/train.py --raw data/raw --output models/")
    return {
        "scaler": joblib.load(c / SCALER.name),
        "pca": joblib.load(c / PCA_MODELO.name),
        "kmeans": joblib.load(c / KMEANS.name),
        "difuso": cargar_params(c / DIFUSO.name),
        "grupos": json.loads((c / GRUPOS.name).read_text(encoding="utf-8")),
    }


def validar(datos_entrada) -> None:
    if not isinstance(datos_entrada, dict):
        raise ValueError("La entrada debe ser un diccionario con I1 a I6 y sistema")
    for llave, (minimo, maximo) in RANGOS.items():
        if llave not in datos_entrada:
            raise ValueError(f"Falta la llave {llave}")
        valor = datos_entrada[llave]
        if isinstance(valor, bool) or not isinstance(valor, Real) or not np.isfinite(valor):
            raise ValueError(f"{llave} debe ser un número (se recibió {valor!r})")
        if not minimo <= valor <= maximo:
            raise ValueError(f"{llave} = {valor} fuera de rango [{minimo}, {maximo}]")
    if datos_entrada.get("sistema") not in SISTEMAS:
        raise ValueError("sistema debe ser 'adaptacion' o 'mitigacion'")


def predecir(datos_entrada: dict) -> dict:
    """Prioridad climática (0 a 100) con lógica difusa Mamdani y grupo con PCA + K-means."""
    inicio = time.perf_counter()
    validar(datos_entrada)
    a = _artefactos()
    valores = {c: float(datos_entrada[c]) for c in RANGOS}
    sistema = datos_entrada["sistema"]

    # grupo: mismo preprocesamiento que en el entrenamiento (log10, z-score de Train y PCA)
    etiqueta = int(a["kmeans"].predict(a["pca"].transform(a["scaler"].transform(preparar_matriz(valores))))[0])
    gid = a["grupos"]["etiqueta_a_grupo"][str(etiqueta)]
    grupo = {"id": int(gid), "nombre": a["grupos"]["grupos"][str(gid)]["nombre"]}

    # prioridad: sistema difuso del tipo pedido
    r = inferir(valores, sistema, a["difuso"])
    texto = explicar(valores, sistema, r, a["difuso"])
    estado = "no_concluyente" if r["no_concluyente"] else "ok"
    if estado == "no_concluyente":
        texto += " El resultado es no concluyente: ninguna regla se activa con fuerza de al menos 0.30."

    return {
        "prediccion": r["puntaje"],
        "clase_etiqueta": r["nivel"],
        "confianza_o_score": r["fuerza_maxima"],
        "tiempo_ms": round((time.perf_counter() - inicio) * 1000, 2),
        "grupo_kmeans": grupo,
        "explicacion": texto,
        "reglas_activadas": [{"regla": x["regla"], "fuerza": x["fuerza"]} for x in r["reglas_activadas"]],
        "conjunto_agregado": r["conjunto_agregado"],
        "estado": estado,
    }


def _resumen(salida: dict) -> dict:
    """Copia para imprimir: recorta las 1,001 posiciones del conjunto agregado."""
    s = dict(salida)
    ca = s["conjunto_agregado"]
    s["conjunto_agregado"] = {"x": ca["x"][:3] + ["...", ca["x"][-1]], "mu": ca["mu"][:3] + ["...", ca["mu"][-1]],
                              "centroide": ca["centroide"]}
    return s


if __name__ == "__main__":
    consola_utf8()
    for sistema in SISTEMAS:
        entrada = {**EJEMPLO_MEXICO, "sistema": sistema}
        print(f"\n>>> Entrada ({sistema}):")
        print(json.dumps(entrada, ensure_ascii=False))
        print(">>> Salida:")
        print(json.dumps(_resumen(predecir(entrada)), ensure_ascii=False, indent=2))
