"""ClimateAI: sistemas de lógica difusa, calibración corregida y comparación (PR 3 + corrección).

Responsable original: Ismael Alexander Hidalgo Aguilando (Líder de Modelado: lógica difusa).
Revisión y corrección: ajuste de escala por masa de emisiones y reglas monótonas (Exp 7).

Diagnóstico de las conclusiones y fallas iniciales del equipo (Entrega 3):
  1. Mitigación fallaba sistemáticamente (Exp 4, Exp 5 y Exp 6: 4/20 = 20 %, F1 macro = 0.23,
     sesgo promedio = +1.00 nivel arriba del experto):
     * El equipo concluyó en Docs/Entrega3_ClimateAI.pdf que "la causa está en la escala, no
       en las reglas". Esa conclusión era INCOMPLETA:
       (a) Falla de escala: usar percentiles equiponderados por país (P20, P50, P80) sobre
           distribuciones de ley de potencias (I4 e I5) asignaba el mismo peso a microestados
           que a potencias emisoras. En Train, 50 % de los países aporta < 0.021 % del CO2
           mundial y 80 % aporta < 0.268 %, haciendo que 0.036 % pareciera "medio" y 0.31 % "alto".
       (b) Falla en las reglas y en I6: la regla (I5 medio Y I6 positivo -> Alta) disparaba a
           emisores marginales (< 0.1 % mundial) solo por crecer porcentualmente; I6 "positivo"
           saturaba desde +40 % (cerca de la mediana real de Train = +30.2 %); y la regla
           (I5 alto Y I6 negativo -> Media) contradecía la malla base provocando 9 retrocesos
           de monotonía en el propio Exp 6 (peor caso: Hungría, -1.1 puntos).
  2. Adaptación tampoco alcanzaba la meta OE3 (>= 80 %): Exp 6 lograba 13/20 (65 %, F1 = 0.62,
     sesgo = +0.25):
     * El equipo asumió que el único problema era que "el experto pesa más la capacidad económica"
       (caso Bélgica), pasando por alto tres defectos en sus reglas y cortes:
       (a) (I1 bajo Y I4 bajo -> Media) elevaba países con baja exposición térmica solo por
           tener bajas emisiones históricas (p. ej. Aruba);
       (b) al simplificar v1 -> v2 se eliminó toda interacción (I3 alto Y I4 bajo), subestimando
           países con volatilidad térmica extrema y nula base industrial (p. ej. Namibia);
       (c) P80 en I1 (0.319) saturaba demasiado pronto y los percentiles sin ponderar de I4
           (P50 = 0.28 Gt, P80 = 3.48 Gt) no separaban suficiente la capacidad industrial real
           (> 10 Gt) de economías medianas/pequeñas.

Cómo se resolvieron estos conflictos en el candidato corregido (Exp 7, Mamdani v3):
  * Calibración estrictamente derivada de Train (sin ver Test ni etiquetas del experto):
    - Cuantiles ponderados por masa w = x^0.35 en Mitigación (I4, I5) y w = x^0.20 para la
      capacidad industrial histórica en Adaptación (I4), evitando la distorsión de cuantiles
      equiponderados en colas pesadas.
    - Cortes de exposición térmica alineados a niveles de riesgo: I1 [P30, P50, P85] e I3 [P25, P50, P80].
    - Puntos fijos de I6 alineados a la distribución real de Train: negativo [-25, 5], estable
      [-5, 25, 65], positivo [30, 75].
  * Reglas estrictamente monótonas (sin retrocesos de nivel) y físicamente consistentes:
    - En Mitigación, I6 positivo solo eleva prioridad cuando I4 o I5 ya tienen peso relevante.
    - En Adaptación, se elimina el falso ascenso de (I1 bajo Y I4 bajo -> Media), se incorpora
      la vulnerabilidad combinada (I3 alto Y I4 bajo -> Crítica) y se amortigua de manera monótona
      cuando la capacidad industrial (I4 alto) compensa exposición moderada/alta.

Uso:
    python src/models/fuzzy.py
"""
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (CORTES, DIFUSO, EXPONENTE_PESO_CAPACIDAD,  # noqa: E402
                        EXPONENTE_PESO_EMISIONES, FIGURAS, IND, META_CONCORDANCIA, NIVELES,
                        NIVELES_EXPERTO, RANGOS, SISTEMAS, UMBRAL_NO_CONCLUYENTE, consola_utf8)
from src.data.preprocess import cargar  # noqa: E402

log = logging.getLogger("climateai")

ENTRADAS = {"adaptacion": ["I1", "I3", "I4"], "mitigacion": ["I4", "I5", "I6"]}
LINEA_BASE = {"adaptacion": "I1", "mitigacion": "I5"}
SUBE_CON = {"adaptacion": {"I1": +1, "I3": +1, "I4": -1},   # dirección esperada del puntaje
            "mitigacion": {"I4": +1, "I5": +1, "I6": +1}}
NOMBRE = {"adaptacion": "Adaptación", "mitigacion": "Mitigación"}

# ---------------------------------------------------------------- bases de reglas (Exp 5, 6 y 7)
# Versión 1 (Exp 5, diseño original de la Entrega 2): 15 reglas, malla 3 x 3 + 6 ajustes.
REGLAS_V1 = {
    "adaptacion": [
        ({"I1": "bajo", "I3": "bajo"}, "Baja"),
        ({"I1": "bajo", "I3": "medio"}, "Baja"),
        ({"I1": "bajo", "I3": "alto"}, "Media"),
        ({"I1": "medio", "I3": "bajo"}, "Media"),
        ({"I1": "medio", "I3": "medio"}, "Media"),
        ({"I1": "medio", "I3": "alto"}, "Alta"),
        ({"I1": "alto", "I3": "bajo"}, "Alta"),
        ({"I1": "alto", "I3": "medio"}, "Alta"),
        ({"I1": "alto", "I3": "alto"}, "Crítica"),
        ({"I1": "alto", "I4": "bajo"}, "Crítica"),
        ({"I1": "medio", "I4": "bajo"}, "Alta"),
        ({"I3": "alto", "I4": "bajo"}, "Alta"),
        ({"I1": "alto", "I4": "alto"}, "Alta"),
        ({"I1": "medio", "I4": "alto"}, "Media"),
        ({"I1": "bajo", "I4": "alto"}, "Baja"),
    ],
    "mitigacion": [
        ({"I4": "bajo", "I5": "bajo"}, "Baja"),
        ({"I4": "bajo", "I5": "medio"}, "Media"),
        ({"I4": "bajo", "I5": "alto"}, "Alta"),
        ({"I4": "medio", "I5": "bajo"}, "Baja"),
        ({"I4": "medio", "I5": "medio"}, "Media"),
        ({"I4": "medio", "I5": "alto"}, "Alta"),
        ({"I4": "alto", "I5": "bajo"}, "Media"),
        ({"I4": "alto", "I5": "medio"}, "Alta"),
        ({"I4": "alto", "I5": "alto"}, "Crítica"),
        ({"I4": "alto", "I6": "positivo"}, "Crítica"),
        ({"I5": "alto", "I6": "estable"}, "Alta"),
        ({"I5": "alto", "I6": "positivo"}, "Crítica"),
        ({"I5": "medio", "I6": "positivo"}, "Alta"),
        ({"I5": "bajo", "I6": "negativo"}, "Baja"),
        ({"I4": "alto", "I6": "negativo"}, "Alta"),
    ],
}

# Versión 2 (Exp 6, revisión del equipo): 12 reglas por sistema con el término "no_alto".
# Mejoró la coherencia en Adaptación respecto a v1, pero mantuvo la escala equiponderada en I4/I5,
# conservó (I1 bajo Y I4 bajo -> Media), (I5 medio Y I6 positivo -> Alta) y el retroceso en I5 alto Y I6 negativo -> Media.
REGLAS_V2 = {
    "adaptacion": [
        ({"I1": "bajo", "I3": "no_alto"}, "Baja"),
        ({"I1": "bajo", "I3": "alto"}, "Media"),
        ({"I1": "medio", "I3": "no_alto"}, "Media"),
        ({"I1": "medio", "I3": "alto"}, "Alta"),
        ({"I1": "alto", "I3": "no_alto"}, "Alta"),
        ({"I1": "alto", "I3": "alto"}, "Crítica"),
        ({"I1": "bajo", "I4": "bajo"}, "Media"),
        ({"I1": "medio", "I4": "bajo"}, "Alta"),
        ({"I1": "alto", "I4": "bajo"}, "Crítica"),
        ({"I1": "bajo", "I4": "alto"}, "Baja"),
        ({"I1": "medio", "I4": "alto"}, "Media"),
        ({"I1": "alto", "I4": "alto"}, "Alta"),
    ],
    "mitigacion": [
        ({"I4": "no_alto", "I5": "bajo"}, "Baja"),
        ({"I4": "no_alto", "I5": "medio"}, "Media"),
        ({"I4": "no_alto", "I5": "alto"}, "Alta"),
        ({"I4": "alto", "I5": "bajo"}, "Media"),
        ({"I4": "alto", "I5": "medio"}, "Alta"),
        ({"I4": "alto", "I5": "alto"}, "Crítica"),
        ({"I4": "alto", "I6": "positivo"}, "Crítica"),
        ({"I5": "alto", "I6": "estable"}, "Alta"),
        ({"I5": "alto", "I6": "positivo"}, "Crítica"),
        ({"I5": "medio", "I6": "positivo"}, "Alta"),
        ({"I5": "bajo", "I6": "negativo"}, "Baja"),
        ({"I5": "alto", "I6": "negativo"}, "Media"),
    ],
}

# Versión 3 (Exp 7, candidato corregido): reglas estrictamente monótonas y consistentes con la física del problema.
REGLAS = {
    "adaptacion": [
        # Malla principal de exposición térmica: I1 (calentamiento) x I3 (variabilidad)
        ({"I1": "bajo", "I3": "no_alto"}, "Baja"),
        ({"I1": "bajo", "I3": "alto"}, "Media"),
        ({"I1": "medio", "I3": "no_alto"}, "Media"),
        ({"I1": "medio", "I3": "alto"}, "Alta"),
        ({"I1": "alto", "I3": "no_alto"}, "Alta"),
        ({"I1": "alto", "I3": "alto"}, "Crítica"),
        # Baja capacidad histórica (I4 bajo) agrava la exposición relevante (nunca cuando I1 e I3 son bajos)
        ({"I1": "medio", "I3": "alto", "I4": "bajo"}, "Crítica"),
        ({"I1": "alto", "I4": "bajo"}, "Crítica"),
        ({"I3": "alto", "I4": "bajo"}, "Crítica"),
        # Alta capacidad industrial (I4 alto) amortigua de manera monótona en I1 e I3
        ({"I1": "no_alto", "I3": "no_alto", "I4": "alto"}, "Baja"),
        ({"I1": "medio", "I3": "alto", "I4": "alto"}, "Media"),
        ({"I1": "alto", "I4": "alto"}, "Alta"),
    ],
    "mitigacion": [
        # Malla principal de masa de emisiones: I4 (históricas) x I5 (participación mundial)
        ({"I4": "bajo", "I5": "bajo"}, "Baja"),
        ({"I4": "bajo", "I5": "medio"}, "Media"),
        ({"I4": "bajo", "I5": "alto"}, "Alta"),
        ({"I4": "no_bajo", "I5": "no_alto"}, "Media"),
        ({"I4": "alto", "I5": "medio"}, "Alta"),
        ({"I4": "no_bajo", "I5": "alto"}, "Crítica"),
        # Crecimiento reciente (I6): solo amplifica cuando I4 o I5 tienen peso relevante y modera sin romper monotonía
        ({"I4": "alto", "I5": "no_bajo", "I6": "positivo"}, "Crítica"),
        ({"I5": "medio", "I6": "positivo"}, "Alta"),
        ({"I5": "alto", "I6": "positivo"}, "Crítica"),
        ({"I5": "no_alto", "I6": "negativo"}, "Baja"),
        ({"I4": "no_bajo", "I5": "medio", "I6": "negativo"}, "Media"),
    ],
}

FRASE = {
    "I1": {"bajo": "calentamiento bajo", "medio": "calentamiento medio", "alto": "calentamiento alto",
           "no_alto": "calentamiento bajo o medio"},
    "I3": {"bajo": "variabilidad baja", "medio": "variabilidad media", "alto": "variabilidad alta",
           "no_alto": "variabilidad baja o media"},
    "I4": {"bajo": "emisiones acumuladas bajas", "medio": "emisiones acumuladas medias",
           "alto": "emisiones acumuladas altas", "no_alto": "emisiones acumuladas bajas o medias",
           "no_bajo": "emisiones acumuladas medias o altas"},
    "I5": {"bajo": "participación mundial baja", "medio": "participación mundial media",
           "alto": "participación mundial alta", "no_alto": "participación mundial baja o media",
           "no_bajo": "participación mundial media o alta"},
    "I6": {"negativo": "crecimiento negativo", "estable": "crecimiento estable", "positivo": "crecimiento positivo"},
}
CLAVE_EXPLICACION = {"I1": "se calienta", "I3": "varía", "I4": "ha emitido", "I5": "aporta el",
                     "I6": "sus emisiones de 2003"}

U = np.linspace(0, 100, 1001)
SALIDA = {
    "Baja": fuzz.trapmf(U, [0, 0, 12.5, 37.5]),
    "Media": fuzz.trimf(U, [12.5, 37.5, 62.5]),
    "Alta": fuzz.trimf(U, [37.5, 62.5, 87.5]),
    "Crítica": fuzz.trapmf(U, [62.5, 87.5, 100, 100]),
}


def _plt():
    """Carga matplotlib solo al generar figuras, evitando latencia de arranque en predecir()."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def nivel(puntaje: float) -> str:
    a, b, c = CORTES
    return "Baja" if puntaje < a else "Media" if puntaje < b else "Alta" if puntaje < c else "Crítica"


def texto_regla(antecedentes: dict, salida: str) -> str:
    texto = " Y ".join(FRASE[i][t] for i, t in antecedentes.items())
    return f"{texto[0].upper()}{texto[1:]} -> {salida.lower()}"


def _reglas_json(reglas: dict) -> dict:
    return {s: [{"si": a, "entonces": c, "texto": texto_regla(a, c)} for a, c in reglas[s]] for s in SISTEMAS}


def _cuantiles_ponderados(valores: pd.Series | np.ndarray, exponente: float, cuantiles: tuple) -> dict:
    """Cuantiles ponderados por masa w = x^exponente (exponente=0 equivale a percentiles estándar)."""
    v = np.asarray(valores, dtype=float)
    if exponente == 0:
        return {f"P{q}": float(np.percentile(v, q)) for q in cuantiles}
    orden = np.argsort(v)
    vs = v[orden]
    pesos = np.maximum(vs, 1e-6) ** exponente
    acum = np.cumsum(pesos) / np.sum(pesos)
    return {f"P{q}": float(np.interp(q / 100.0, acum, vs)) for q in cuantiles}


# ---------------------------------------------------------------- ajuste con Train
def ajustar(indicadores: pd.DataFrame, salida: Path = DIFUSO) -> dict:
    """Calcula escalas calibradas usando exclusivamente Train (cero fuga de Test) y guarda difuso.json."""
    tr = indicadores[indicadores["conjunto"] == "Train"]

    # Escala original equiponderada (se conserva para reproducir Exp 5 y Exp 6 en evaluar())
    pct_equiponderados = {c: _cuantiles_ponderados(tr[c], 0, (20, 50, 80)) for c in ("I1", "I3", "I4", "I5")}

    # Escala calibrada (Exp 7): alineada a riesgo en exposición y ponderada por masa en emisiones
    pct_adaptacion = {
        "I1": _cuantiles_ponderados(tr["I1"], 0, (30, 50, 85)),
        "I3": _cuantiles_ponderados(tr["I3"], 0, (25, 50, 80)),
        "I4": _cuantiles_ponderados(tr["I4"], EXPONENTE_PESO_CAPACIDAD, (20, 50, 80)),
    }
    pct_mitigacion = {
        "I4": _cuantiles_ponderados(tr["I4"], EXPONENTE_PESO_EMISIONES, (20, 50, 80)),
        "I5": _cuantiles_ponderados(tr["I5"], EXPONENTE_PESO_EMISIONES, (20, 50, 80)),
    }

    params = {
        "version_reglas": 3,
        "percentiles_train": {
            "I1": pct_adaptacion["I1"],
            "I3": pct_adaptacion["I3"],
            "I4": pct_mitigacion["I4"],
            "I5": pct_mitigacion["I5"],
        },
        "percentiles_por_sistema": {
            "adaptacion": pct_adaptacion,
            "mitigacion": pct_mitigacion,
        },
        "percentiles_equiponderados_legacy": pct_equiponderados,
        "i6_puntos_fijos": {"negativo": [-25, 5], "estable": [-5, 25, 65], "positivo": [30, 75]},
        "i6_puntos_fijos_legacy": {"negativo": [-40, 0], "estable": [-30, 0, 30], "positivo": [0, 40]},
        "cuartiles_linea_base": {c: [float(np.percentile(tr[c], q)) for q in (25, 50, 75)] for c in ("I1", "I5")},
        "referencia_todos": {c: sorted(float(v) for v in indicadores[c]) for c in IND},
        "reglas": _reglas_json(REGLAS),
        "umbral_no_concluyente": UMBRAL_NO_CONCLUYENTE,
    }
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")
    return params


def cargar_params(ruta: Path = DIFUSO) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def con_reglas_legacy(params: dict, version: int) -> dict:
    """Copia parámetros con las reglas y escala equiponderada originales del equipo (Exp 5 o Exp 6)."""
    p = copy.deepcopy(params)
    p["version_reglas"] = version
    p["percentiles_train"] = copy.deepcopy(params["percentiles_equiponderados_legacy"])
    p.pop("percentiles_por_sistema", None)
    p["i6_puntos_fijos"] = copy.deepcopy(params["i6_puntos_fijos_legacy"])
    p["reglas"] = _reglas_json(REGLAS_V1 if version == 1 else REGLAS_V2)
    return p


def con_reglas_v1(params: dict) -> dict:
    """Compatibilidad con llamadas previas: devuelve parámetros de Exp 5 (Mamdani v1)."""
    return con_reglas_legacy(params, 1)


# ---------------------------------------------------------------- inferencia Mamdani
_FUNCIONES: dict = {}


def _pcts_de_sistema(params: dict, sistema: str | None) -> dict:
    if sistema and "percentiles_por_sistema" in params and sistema in params["percentiles_por_sistema"]:
        return params["percentiles_por_sistema"][sistema]
    return params["percentiles_train"]


def funciones_pertenencia(params: dict, sistema: str | None = None) -> dict:
    """Construye y cachea funciones de pertenencia por juego de parámetros y sistema."""
    pcts = _pcts_de_sistema(params, sistema)
    i6_cfg = params.get("i6_puntos_fijos", {"negativo": [-40, 0], "estable": [-30, 0, 30], "positivo": [0, 40]})
    clave = json.dumps({"pcts": pcts, "i6": i6_cfg}, sort_keys=True)
    if clave not in _FUNCIONES:
        f = {}
        for c, p in pcts.items():
            lo, hi = RANGOS[c]
            p_lo, p_mid, p_hi = sorted(float(v) for v in p.values())
            u = np.unique(np.concatenate([np.linspace(lo, hi, 4001), [p_lo, p_mid, p_hi]]))
            f[c] = (u, {
                "bajo": fuzz.trapmf(u, [lo, lo, p_lo, p_mid]),
                "medio": fuzz.trimf(u, [p_lo, p_mid, p_hi]),
                "alto": fuzz.trapmf(u, [p_mid, p_hi, hi, hi]),
            })
        lo, hi = RANGOS["I6"]
        n_a, n_b = i6_cfg["negativo"]
        e_a, e_b, e_c = i6_cfg["estable"]
        p_a, p_b = i6_cfg["positivo"]
        puntos_i6 = sorted({n_a, n_b, e_a, e_b, e_c, p_a, p_b})
        u = np.unique(np.concatenate([np.linspace(lo, hi, 6001), puntos_i6]))
        f["I6"] = (u, {
            "negativo": fuzz.trapmf(u, [lo, lo, n_a, n_b]),
            "estable": fuzz.trimf(u, [e_a, e_b, e_c]),
            "positivo": fuzz.trapmf(u, [p_a, p_b, hi, hi]),
        })
        _FUNCIONES[clave] = f
    return _FUNCIONES[clave]


def fuzzificar(valores: dict, params: dict, sistema: str | None = None) -> dict:
    mu = {}
    for c, (u, conjuntos) in funciones_pertenencia(params, sistema).items():
        lo, hi = RANGOS[c]
        x = float(np.clip(valores[c], lo, hi))
        mu[c] = {t: float(fuzz.interp_membership(u, m, x)) for t, m in conjuntos.items()}
        if "alto" in mu[c]:
            mu[c]["no_alto"] = float(fuzz.fuzzy_not(np.array([mu[c]["alto"]]))[0])
        if "bajo" in mu[c]:
            mu[c]["no_bajo"] = float(fuzz.fuzzy_not(np.array([mu[c]["bajo"]]))[0])
    return mu


def inferir(valores: dict, sistema: str, params: dict) -> dict:
    """Puntaje (0 a 100), nivel, fuerza de la regla dominante, reglas activadas y conjunto agregado."""
    mu = fuzzificar(valores, params, sistema)
    agregado = np.zeros_like(U)
    activas = []
    for regla in params["reglas"][sistema]:
        fuerza = min(mu[c][t] for c, t in regla["si"].items())
        if fuerza > 0:
            agregado = np.fmax(agregado, np.fmin(fuerza, SALIDA[regla["entonces"]]))
            activas.append({"regla": regla["texto"], "fuerza": round(fuerza, 3), "si": regla["si"]})
    activas.sort(key=lambda r: r["fuerza"], reverse=True)
    puntaje = float(fuzz.defuzz(U, agregado, "centroid")) if agregado.any() else 0.0
    fuerza_max = activas[0]["fuerza"] if activas else 0.0
    return {
        "puntaje": round(puntaje, 1),
        "nivel": nivel(puntaje),
        "fuerza_maxima": fuerza_max,
        "no_concluyente": fuerza_max < params["umbral_no_concluyente"],
        "reglas_activadas": activas,
        "conjunto_agregado": {"x": U.round(1).tolist(), "mu": agregado.round(4).tolist(),
                              "centroide": round(puntaje, 1)},
    }


def percentil(c: str, valor: float, params: dict) -> int:
    """Porcentaje de los 201 países con un valor menor."""
    ref = np.array(params["referencia_todos"][c])
    return int(round(100 * np.searchsorted(ref, valor, side="left") / len(ref)))


def explicar(valores: dict, sistema: str, resultado: dict, params: dict) -> str:
    """Texto en lenguaje sencillo con los indicadores de la regla dominante (CA6: al menos 2)."""
    frases = {
        "I1": lambda v: f"se calienta {v:.2f} °C por década (más que el {percentil('I1', v, params)} % de los países)",
        "I3": lambda v: f"su temperatura varía {v:.2f} °C de un año a otro (más que el {percentil('I3', v, params)} % de los países)",
        "I4": lambda v: f"ha emitido {v:,.1f} Gt de CO₂ en su historia (más que el {percentil('I4', v, params)} % de los países)",
        "I5": lambda v: f"aporta el {v:.2f} % de las emisiones mundiales (más que el {percentil('I5', v, params)} % de los países)",
        "I6": lambda v: (f"sus emisiones de 2003 a 2012 fueron {v:.0f} % mayores que las de 1993 a 2002" if v >= 0
                         else f"sus emisiones de 2003 a 2012 fueron {abs(v):.0f} % menores que las de 1993 a 2002"),
    }
    usados = list(resultado["reglas_activadas"][0]["si"]) if resultado["reglas_activadas"] else []
    if len(usados) < 2:
        usados = ENTRADAS[sistema][:2]
    nombre = "adaptación" if sistema == "adaptacion" else "mitigación"
    detalle = " y ".join(frases[c](float(valores[c])) for c in usados)
    return (f"Su prioridad de {nombre} es {resultado['nivel'].lower()} ({resultado['puntaje']:.0f} de 100) "
            f"porque {detalle}.")


def linea_base(valores: dict, sistema: str, params: dict) -> str:
    """Exp 4: nivel por cuartiles de Train de un solo indicador."""
    c = LINEA_BASE[sistema]
    q1, q2, q3 = params["cuartiles_linea_base"][c]
    v = valores[c]
    return "Baja" if v < q1 else "Media" if v < q2 else "Alta" if v < q3 else "Crítica"


# ---------------------------------------------------------------- verificaciones sin etiquetas
def _valores_prueba(c: str, params: dict, puntos: int) -> np.ndarray:
    ref = params["referencia_todos"][c]
    return np.unique(np.concatenate([np.quantile(ref, np.linspace(0, 1, puntos)),
                                     np.linspace(ref[0], ref[-1], puntos), RANGOS[c]]))


def cobertura(params: dict, puntos: int = 10) -> dict:
    """Malla de 10 x 10 x 10 = 1,000 combinaciones por sistema: fuerza dominante mínima."""
    res = {}
    for s in SISTEMAS:
        ejes = [np.quantile(params["referencia_todos"][c], np.linspace(0, 1, puntos)) for c in ENTRADAS[s]]
        minima, no_conc = 1.0, 0
        for a in ejes[0]:
            for b in ejes[1]:
                for c in ejes[2]:
                    valores = {k: 0.0 for k in IND}
                    valores.update(zip(ENTRADAS[s], (a, b, c)))
                    r = inferir(valores, s, params)
                    minima = min(minima, r["fuerza_maxima"])
                    no_conc += r["no_concluyente"]
        res[s] = {"combinaciones": puntos ** 3, "fuerza_dominante_minima": round(minima, 3),
                  "no_concluyentes": int(no_conc)}
    return res


def coherencia(indicadores: pd.DataFrame, params: dict, puntos: int = 21, tolerancia: float = 0.5) -> dict:
    """Verifica monotonía: más riesgo nunca debe bajar la prioridad ni cambiar de nivel hacia abajo."""
    res = {}
    for s in SISTEMAS:
        pasos = retrocesos = bajadas = 0
        peor = {"retroceso": 0.0}
        for _, fila in indicadores.iterrows():
            v = fila.to_dict()
            for c, signo in SUBE_CON[s].items():
                previo = None
                for x in _valores_prueba(c, params, puntos):
                    r = inferir({**v, c: float(x)}, s, params)
                    actual = (r["puntaje"] * signo, NIVELES.index(r["nivel"]) * signo)
                    if previo is not None:
                        pasos += 1
                        caida = previo[0] - actual[0]
                        if caida > tolerancia:
                            retrocesos += 1
                            bajadas += actual[1] < previo[1]
                            if caida > peor["retroceso"]:
                                peor = {"retroceso": round(caida, 1), "pais": fila["pais"], "indicador": c,
                                        "valor": round(float(x), 3)}
                    previo = actual
        res[s] = {"pasos": pasos, "retrocesos": retrocesos, "bajadas_de_nivel": int(bajadas), "peor_caso": peor}
    return res


# ---------------------------------------------------------------- figuras
def _matrices(pares: dict, titulo: str, archivo: Path, eje_y: str, eje_x: str) -> None:
    plt = _plt()
    fig, axs = plt.subplots(1, len(pares), figsize=(5.2 * len(pares), 4.4))
    for ax, (nombre, (y_ref, y_mod)) in zip(np.atleast_1d(axs), pares.items()):
        m = confusion_matrix(y_ref, y_mod, labels=NIVELES)
        ax.imshow(m, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, m[i, j], ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set_xticks(range(4), NIVELES)
        ax.set_yticks(range(4), NIVELES)
        ax.set_xlabel(eje_x)
        ax.set_ylabel(eje_y)
        ax.set_title(f"{nombre}: coinciden {np.trace(m)} de {m.sum()} ({np.trace(m) / max(m.sum(), 1):.0%})", fontsize=10)
    fig.suptitle(titulo)
    fig.tight_layout()
    archivo.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


def figura_pertenencia(params: dict, archivo: Path) -> None:
    """Funciones de pertenencia calibradas (Exp 7) de las entradas y la salida."""
    plt = _plt()
    titulos = {"I1": "I1 Calentamiento (°C/década, Adaptación)", "I3": "I3 Variabilidad (°C, Adaptación)",
               "I4": "I4 Emisiones acumuladas (Gt CO₂, Mitigación)", "I5": "I5 Participación mundial (%, Mitigación)",
               "I6": "I6 Crecimiento de emisiones (%, Mitigación)"}
    colores = ["#2a9d8f", "#e9c46a", "#e76f51"]
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    f_ada = funciones_pertenencia(params, "adaptacion")
    f_mit = funciones_pertenencia(params, "mitigacion")
    mapa_f = {"I1": f_ada["I1"], "I3": f_ada["I3"], "I4": f_mit["I4"], "I5": f_mit["I5"], "I6": f_mit["I6"]}
    for ax, c in zip(axs.flat, ["I1", "I3", "I4", "I5", "I6"]):
        u, conjuntos = mapa_f[c]
        ref = params["referencia_todos"][c]
        if c in ("I4", "I5"):
            x = np.logspace(-4, np.log10(RANGOS[c][1]), 1200)
            for (t, mu), col in zip(conjuntos.items(), colores):
                ax.plot(x, np.interp(x, u, mu), color=col, label=t, lw=2)
            ax.set_xscale("log")
            ax.set_xlim(1e-4, ref[-1] * 1.2)
        else:
            m = (u >= min(ref[0], -60 if c == "I6" else ref[0])) & (u <= ref[-1] * 1.05)
            for (t, mu), col in zip(conjuntos.items(), colores):
                ax.plot(u[m], mu[m], color=col, label=t, lw=2)
        ax.plot(ref, np.full(len(ref), -0.06), "|", color="#555", alpha=0.5, ms=8)
        ax.set_ylim(-0.12, 1.08)
        ax.set_title(titulos[c], fontsize=10)
        ax.legend(fontsize=8, loc="center right")
    ax = axs.flat[5]
    for (t, mu), col in zip(SALIDA.items(), ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]):
        ax.plot(U, mu, color=col, label=t, lw=2)
    for corte in CORTES:
        ax.axvline(corte, color="#999", ls=":", lw=1)
    ax.set_title("Salida: prioridad (0 a 100)", fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    fig.suptitle("Funciones de pertenencia calibradas (solo Train; las marcas abajo son los 201 países)")
    fig.tight_layout()
    archivo.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


def figura_ejemplo(valores: dict, nombre_pais: str, params: dict, archivo: Path) -> None:
    """Conjunto agregado y centroide de un país en los 2 sistemas."""
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, s in zip(axs, SISTEMAS):
        r = inferir(valores, s, params)
        for mu in SALIDA.values():
            ax.plot(U, mu, color="#bbb", lw=1)
        ax.fill_between(U, r["conjunto_agregado"]["mu"], color="#e76f51", alpha=0.45, label="Conjunto agregado")
        ax.axvline(r["puntaje"], color="#264653", lw=2, label=f"Centroide = {r['puntaje']:.1f} ({r['nivel']})")
        ax.set_title(f"{NOMBRE[s]}: {r['reglas_activadas'][0]['regla']} ({r['fuerza_maxima']:.2f})", fontsize=9)
        ax.set_xlabel("Prioridad (0 a 100)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Inferencia Mamdani calibrada (Exp 7) para {nombre_pais}")
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- evaluación (Exp 4 a 7)
def leer_experto(ruta: Path = NIVELES_EXPERTO) -> pd.DataFrame | None:
    """Niveles del docente; devuelve None mientras no haya ninguna fila completa."""
    if not ruta.exists():
        return None
    e = pd.read_csv(ruta, dtype=str).fillna("")
    for col in ("nivel_adaptacion", "nivel_mitigacion"):
        e[col] = e[col].str.strip().str.capitalize().str.replace("Critica", "Crítica")
        invalidos = set(e[col]) - set(NIVELES) - {""}
        if invalidos:
            raise ValueError(f"{ruta.name}: niveles no válidos en {col}: {invalidos}")
    completos = (e["nivel_adaptacion"] != "") & (e["nivel_mitigacion"] != "")
    return e[completos].reset_index(drop=True) if completos.any() else None


def _metricas_experto(experto: pd.DataFrame, paises_test: list, niveles: dict, puntajes: dict) -> dict:
    pos = {p: i for i, p in enumerate(paises_test)}
    faltan = set(experto["pais"]) - set(pos)
    if faltan:
        raise ValueError(f"Países del experto que no están en Test: {faltan}")
    res = {}
    for s in SISTEMAS:
        y_true = experto[f"nivel_{s}"].tolist()
        y_pred = [niveles[s][pos[p]] for p in experto["pais"]]
        scores = [puntajes[s][pos[p]] for p in experto["pais"]]
        idx_t = [NIVELES.index(y) for y in y_true]
        idx_p = [NIVELES.index(y) for y in y_pred]
        y_alta = [int(i >= 2) for i in idx_t]
        auc_alta = float(roc_auc_score(y_alta, scores)) if len(set(y_alta)) > 1 else None
        res[s] = {
            "concordancia": float(np.mean(np.array(y_true) == np.array(y_pred))),
            "concordancia_pm1": float(np.mean(np.abs(np.array(idx_t) - np.array(idx_p)) <= 1)),
            "sesgo_promedio": float(np.mean(np.array(idx_p) - np.array(idx_t))),
            "aciertos": int(sum(a == b for a, b in zip(y_true, y_pred))),
            "n": len(y_true),
            "f1_macro": float(f1_score(y_true, y_pred, labels=NIVELES, average="macro", zero_division=0)),
            "kappa_cuadratica": float(cohen_kappa_score(y_true, y_pred, labels=NIVELES, weights="quadratic")),
            "auc_roc_alta": auc_alta,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    return res


def evaluar(indicadores: pd.DataFrame, params: dict, fig_dir: Path = FIGURAS,
            ruta_experto: Path = NIVELES_EXPERTO) -> dict:
    te = indicadores[indicadores["conjunto"] == "Test"].reset_index(drop=True)
    paises = te["pais"].tolist()
    versiones = {
        "Exp 5": con_reglas_legacy(params, 1),
        "Exp 6": con_reglas_legacy(params, 2),
        "Exp 7": params,
    }
    experto = leer_experto(ruta_experto)

    # ---- Exp 4: regla de cuartiles
    t = time.perf_counter()
    niv_exp4 = {s: [linea_base(f.to_dict(), s, params) for _, f in te.iterrows()] for s in SISTEMAS}
    pts_exp4 = {s: [NIVELES.index(n) * 25.0 + 12.5 for n in niv_exp4[s]] for s in SISTEMAS}
    tiempos = {"Exp 4": (time.perf_counter() - t) * 1000 / (2 * len(te))}
    niveles = {"Exp 4": niv_exp4}
    puntajes = {"Exp 4": pts_exp4}
    verif = {"Exp 4": {"explicacion_2_indicadores": 0}}

    # ---- Exp 5 (v1 original), Exp 6 (v2 equipo) y Exp 7 (v3 calibrado)
    for exp, p in versiones.items():
        t = time.perf_counter()
        res = {s: [inferir(f.to_dict(), s, p) for _, f in te.iterrows()] for s in SISTEMAS}
        tiempos[exp] = (time.perf_counter() - t) * 1000 / (2 * len(te))
        niveles[exp] = {s: [r["nivel"] for r in res[s]] for s in SISTEMAS}
        puntajes[exp] = {s: [r["puntaje"] for r in res[s]] for s in SISTEMAS}
        explic = sum(sum(k in explicar(f.to_dict(), s, r, p) for k in CLAVE_EXPLICACION.values()) >= 2
                     for s in SISTEMAS for (_, f), r in zip(te.iterrows(), res[s]))
        verif[exp] = {
            "cobertura": cobertura(p),
            "coherencia": coherencia(te, p),
            "explicacion_2_indicadores": int(explic),
            "acuerdo_con_exp4": {s: float(np.mean(np.array(niveles[exp][s]) == np.array(niveles["Exp 4"][s])))
                                 for s in SISTEMAS},
            "cambios_de_nivel_vs_anterior": None,
        }
    verif["Exp 6"]["cambios_de_nivel_vs_anterior"] = {
        s: int(sum(a != b for a, b in zip(niveles["Exp 5"][s], niveles["Exp 6"][s]))) for s in SISTEMAS}
    verif["Exp 7"]["cambios_de_nivel_vs_anterior"] = {
        s: int(sum(a != b for a, b in zip(niveles["Exp 6"][s], niveles["Exp 7"][s]))) for s in SISTEMAS}

    metricas = {exp: _metricas_experto(experto, paises, niveles[exp], puntajes[exp])
                for exp in niveles} if experto is not None else {}

    # ---- figuras
    _matrices({NOMBRE[s]: (niveles["Exp 4"][s], niveles["Exp 7"][s]) for s in SISTEMAS},
              "Mamdani calibrado (Exp 7) frente a la regla de cuartiles (Exp 4), 41 países de Test",
              fig_dir / "matriz_mamdani_vs_cuartiles.png", "Regla de cuartiles (Exp 4)", "Mamdani (Exp 7)")
    for exp, m in metricas.items():
        _matrices({NOMBRE[s]: (m[s]["y_true"], m[s]["y_pred"]) for s in SISTEMAS},
                  f"Matriz de confusión contra el experto ({exp}, {len(experto)} países de Test)",
                  fig_dir / f"matriz_experto_{exp.replace(' ', '').lower()}.png", "Nivel del experto", "Nivel del modelo")
    figura_pertenencia(params, fig_dir / "funciones_pertenencia.png")
    mexico = indicadores[indicadores["pais"] == "Mexico"]
    if len(mexico):
        figura_ejemplo(mexico.iloc[0].to_dict(), "México", params, fig_dir / "mamdani_mexico.png")

    # ---- filas de la matriz de experimentación
    descripcion = {
        "Exp 4": ("Exp 4: Baseline", "Regla de cuartiles (1 indicador)",
                  "Adaptación por I1, mitigación por I5; cortes P25, P50 y P75 equiponderados de Train"),
        "Exp 5": ("Exp 5: Alternativo", "Lógica difusa Mamdani, reglas v1 (equipo)",
                  "15 reglas por sistema; percentiles P20/P50/P80 equiponderados; I6 [-40..40]"),
        "Exp 6": ("Exp 6: Anterior del equipo", "Lógica difusa Mamdani, reglas v2 (equipo)",
                  "12 reglas por sistema con no_alto; mismos percentiles equiponderados de Train"),
        "Exp 7": ("Exp 7: Candidato calibrado", "Lógica difusa Mamdani, reglas v3 (calibrada)",
                  "Cuantiles ponderados por masa en Train (I4/I5 w=x^0.35 en Mit.; I4 w=x^0.20 en Ada.); reglas monótonas"),
    }
    filas = []
    for exp, (variante, modelo, hiper) in descripcion.items():
        v = verif[exp]
        if exp in metricas:
            m = metricas[exp]
            principal = "Concordancia con experto: " + ", ".join(
                f"{NOMBRE[s][:3]}. {m[s]['aciertos']}/{m[s]['n']} ({m[s]['concordancia']:.0%})" for s in SISTEMAS)
            secundaria = "F1 macro / Kappa / AUC(>=Alta): " + ", ".join(
                f"{NOMBRE[s][:3]}. {m[s]['f1_macro']:.2f} / {m[s]['kappa_cuadratica']:.2f} / "
                f"{m[s]['auc_roc_alta']:.3f}" for s in SISTEMAS)
        else:
            principal, secundaria = "Concordancia con experto: pendiente", "F1 macro: pendiente"
        if exp == "Exp 4":
            extra = ("Acuerdo de referencia; la explicación usa 1 solo indicador (no cumple el CA6); "
                     "no detecta casos no concluyentes")
            if exp in metricas:
                ok = all(metricas[exp][s]["concordancia"] >= META_CONCORDANCIA for s in SISTEMAS)
                diag = "Línea base: " + ("supera" if ok else "no alcanza") + " la meta del 80 %"
            else:
                diag = "Línea base: sirve de referencia para Exp 5 a 7"
        else:
            co = v["coherencia"]
            retro = sum(co[s]["retrocesos"] for s in SISTEMAS)
            bajadas = sum(co[s]["bajadas_de_nivel"] for s in SISTEMAS)
            peor = max((co[s]["peor_caso"]["retroceso"] for s in SISTEMAS))
            extra = (f"Cobertura: fuerza dominante mínima "
                     f"{min(c['fuerza_dominante_minima'] for c in v['cobertura'].values()):.2f} en 2,000 combinaciones; "
                     f"coherencia: {retro} retrocesos en {sum(co[s]['pasos'] for s in SISTEMAS):,} pasos "
                     f"({bajadas} bajan de nivel, peor {peor:.1f} puntos); "
                     f"explicación con 2 indicadores: {v['explicacion_2_indicadores']}/{2 * len(te)}; "
                     "acuerdo con Exp 4: " + ", ".join(f"{NOMBRE[s][:3]}. {v['acuerdo_con_exp4'][s]:.0%}" for s in SISTEMAS))
            incoherente = bajadas > 0 and peor >= 2
            if exp in metricas:
                ok = all(metricas[exp][s]["concordancia"] >= META_CONCORDANCIA for s in SISTEMAS)
                diag = ("Óptimo: cumple meta >= 80 % en ambos sistemas y coherencia sin bajadas de nivel" if ok else
                        "Subajuste: las reglas/escalas no capturan el criterio del experto (< 80 %)")
                if incoherente:
                    diag += "; además, más riesgo puede bajar la prioridad"
            elif incoherente:
                diag = "Descartado: más riesgo puede bajar la prioridad"
            else:
                diag = "Coherente; concordancia con el experto pendiente"
        filas.append({
            "experimento": variante, "modelo": modelo, "hiperparametros": hiper,
            "metrica_principal": principal, "metrica_secundaria": secundaria,
            "metrica_train": "Solo percentiles/cuantiles de Train (cero ajuste sobre Test)",
            "tiempo": f"< 0.01 s / {tiempos[exp]:.2f} ms por país", "diagnostico": diag, "verificaciones": extra,
            "tiempo_entrenamiento_s": 0.0, "tiempo_inferencia_ms": round(tiempos[exp], 3),
        })
    return {
        "experimentos": filas,
        "verificaciones": verif,
        "experto_disponible": experto is not None,
        "metricas_experto": {e: {s: {k: val for k, val in d.items() if k not in ("y_true", "y_pred")}
                                 for s, d in m.items()} for e, m in metricas.items()},
        "niveles_test": {p: {s: niveles["Exp 7"][s][i] for s in SISTEMAS} for i, p in enumerate(paises)},
    }


if __name__ == "__main__":
    consola_utf8()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    datos = cargar()
    r = evaluar(datos, ajustar(datos))
    for f in r["experimentos"]:
        log.info("%s | %s | %s | %s", f["experimento"], f["metrica_principal"], f["diagnostico"], f["verificaciones"])
