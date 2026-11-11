"""ClimateAI: sistemas de lógica difusa y su comparación (PR 3).

Responsable: Ismael Alexander Hidalgo Aguilando (Líder de Modelado: lógica difusa).
Rama: feat/difuso

Definición (Entrega 2, sección 5):
  * Adaptación usa I1 (calentamiento), I3 (variabilidad) e I4 (emisiones acumuladas).
    Mitigación usa I4, I5 (participación mundial) e I6 (crecimiento de emisiones).
  * Entradas: bajo vale 1 hasta P20 y 0 desde P50; medio es un triángulo P20, P50, P80;
    alto vale 0 hasta P50 y 1 desde P80 (percentiles de Train). I6 usa puntos fijos:
    negativo (1 hasta -40, 0 desde 0), estable (-30, 0, +30) y positivo (0 hasta 0, 1 desde +40).
  * Salida 0 a 100: Baja, Media, Alta y Crítica, centradas en 12.5, 37.5, 62.5 y 87.5,
    que se cruzan en los cortes 25, 50 y 75.
  * Mamdani: Y = mínimo, agregación = máximo, defuzzificación = centroide.
  * Versión 1 de las reglas (Entrega 2): 15 por sistema, una malla de 3 x 3 con los 2
    indicadores principales y 6 ajustes con el tercero. Versión 2 (esta entrega): 12 por
    sistema, se fusionan las celdas vecinas con el mismo nivel (ver REGLAS).
  * "No concluyente" si la fuerza de la regla dominante es menor que 0.30.

Experimentos (mismos 41 países de Test):
  * Exp 4, modelo base: regla de un solo indicador por cuartiles de Train
    (adaptación con I1, mitigación con I5), la línea base de la Entrega 2.
  * Exp 5: Mamdani con la versión 1 de las reglas.
  * Exp 6, candidato: Mamdani con la versión 2 de las reglas.
Métrica principal: concordancia de nivel con el experto en 20 países de Test (meta 80 %).
Métrica secundaria: F1 macro contra el experto. Mientras no estén los niveles del
experto en data/expert/niveles_experto.csv, se reportan como pendientes y se mide lo
que no necesita etiquetas: cobertura de las reglas, coherencia (más riesgo nunca debe
bajar la prioridad), explicación con al menos 2 indicadores y acuerdo con la línea base.

Uso (después del PR 1):
    python src/models/fuzzy.py
"""
import copy
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import skfuzzy as fuzz

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (CORTES, DIFUSO, FIGURAS, IND, META_CONCORDANCIA, NIVELES,  # noqa: E402
                        NIVELES_EXPERTO, RANGOS, SISTEMAS, UMBRAL_NO_CONCLUYENTE, consola_utf8)
from src.data.preprocess import cargar  # noqa: E402

log = logging.getLogger("climateai")

ENTRADAS = {"adaptacion": ["I1", "I3", "I4"], "mitigacion": ["I4", "I5", "I6"]}
LINEA_BASE = {"adaptacion": "I1", "mitigacion": "I5"}
SUBE_CON = {"adaptacion": {"I1": +1, "I3": +1, "I4": -1},   # dirección esperada del puntaje
            "mitigacion": {"I4": +1, "I5": +1, "I6": +1}}
NOMBRE = {"adaptacion": "Adaptación", "mitigacion": "Mitigación"}

# Reglas (antecedentes, consecuente). Cada sistema tiene una malla principal con sus 2
# indicadores más importantes, así siempre se activa al menos una regla con fuerza >= 0.5.
#
# Versión 1 (Exp 5, diseño de la Entrega 2): 15 reglas, malla completa de 3 x 3 + 6 ajustes.
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
        ({"I4": "alto", "I6": "positivo"}, "Crítica"),     # ejemplo de la Entrega 1
        ({"I5": "alto", "I6": "estable"}, "Alta"),         # ejemplo de la Entrega 1
        ({"I5": "alto", "I6": "positivo"}, "Crítica"),
        ({"I5": "medio", "I6": "positivo"}, "Alta"),
        ({"I5": "bajo", "I6": "negativo"}, "Baja"),
        ({"I4": "alto", "I6": "negativo"}, "Alta"),
    ],
}

# Versión 2 (Exp 6, candidato): 12 reglas. Hallazgo del Exp 5: en la malla de 3 x 3 hay
# celdas vecinas con el mismo nivel (por ejemplo, variabilidad baja y media dan lo mismo).
# Con agregación por máximo, a medio camino entre "bajo" y "medio" esa regla pesa solo 0.5
# y el puntaje se mueve hacia los ajustes, así que más riesgo podía bajar la prioridad.
# Se fusionaron esas celdas con el término "no alto" (1 - alto, con fuzzy_not), que vale
# exactamente bajo + medio, y se corrigieron los 2 ajustes que contradecían a la malla.
REGLAS = {
    "adaptacion": [
        # malla principal: calentamiento (I1) x variabilidad (I3)
        ({"I1": "bajo", "I3": "no_alto"}, "Baja"),
        ({"I1": "bajo", "I3": "alto"}, "Media"),
        ({"I1": "medio", "I3": "no_alto"}, "Media"),
        ({"I1": "medio", "I3": "alto"}, "Alta"),
        ({"I1": "alto", "I3": "no_alto"}, "Alta"),
        ({"I1": "alto", "I3": "alto"}, "Crítica"),
        # países que emitieron poco tienen menos capacidad para adaptarse: suben un nivel
        ({"I1": "bajo", "I4": "bajo"}, "Media"),
        ({"I1": "medio", "I4": "bajo"}, "Alta"),
        ({"I1": "alto", "I4": "bajo"}, "Crítica"),
        # grandes emisores históricos tienen más capacidad para adaptarse: se quedan abajo
        ({"I1": "bajo", "I4": "alto"}, "Baja"),
        ({"I1": "medio", "I4": "alto"}, "Media"),
        ({"I1": "alto", "I4": "alto"}, "Alta"),
    ],
    "mitigacion": [
        # malla principal: emisiones acumuladas (I4) x participación mundial (I5)
        ({"I4": "no_alto", "I5": "bajo"}, "Baja"),
        ({"I4": "no_alto", "I5": "medio"}, "Media"),
        ({"I4": "no_alto", "I5": "alto"}, "Alta"),
        ({"I4": "alto", "I5": "bajo"}, "Media"),
        ({"I4": "alto", "I5": "medio"}, "Alta"),
        ({"I4": "alto", "I5": "alto"}, "Crítica"),
        # el crecimiento de las emisiones (I6) ajusta el nivel
        ({"I4": "alto", "I6": "positivo"}, "Crítica"),     # ejemplo de la Entrega 1
        ({"I5": "alto", "I6": "estable"}, "Alta"),         # ejemplo de la Entrega 1
        ({"I5": "alto", "I6": "positivo"}, "Crítica"),
        ({"I5": "medio", "I6": "positivo"}, "Alta"),
        ({"I5": "bajo", "I6": "negativo"}, "Baja"),
        ({"I5": "alto", "I6": "negativo"}, "Media"),
    ],
}

# Frases para las reglas y la explicación (sin tecnicismos, para el tablero)
FRASE = {
    "I1": {"bajo": "calentamiento bajo", "medio": "calentamiento medio", "alto": "calentamiento alto",
           "no_alto": "calentamiento bajo o medio"},
    "I3": {"bajo": "variabilidad baja", "medio": "variabilidad media", "alto": "variabilidad alta",
           "no_alto": "variabilidad baja o media"},
    "I4": {"bajo": "emisiones acumuladas bajas", "medio": "emisiones acumuladas medias",
           "alto": "emisiones acumuladas altas", "no_alto": "emisiones acumuladas bajas o medias"},
    "I5": {"bajo": "participación mundial baja", "medio": "participación mundial media",
           "alto": "participación mundial alta", "no_alto": "participación mundial baja o media"},
    "I6": {"negativo": "crecimiento negativo", "estable": "crecimiento estable", "positivo": "crecimiento positivo"},
}
# Inicio de la frase de cada indicador en la explicación (las pruebas cuentan cuántos aparecen)
CLAVE_EXPLICACION = {"I1": "se calienta", "I3": "varía", "I4": "ha emitido", "I5": "aporta el",
                     "I6": "sus emisiones de 2003"}

U = np.linspace(0, 100, 1001)
SALIDA = {
    "Baja": fuzz.trapmf(U, [0, 0, 12.5, 37.5]),
    "Media": fuzz.trimf(U, [12.5, 37.5, 62.5]),
    "Alta": fuzz.trimf(U, [37.5, 62.5, 87.5]),
    "Crítica": fuzz.trapmf(U, [62.5, 87.5, 100, 100]),
}


def nivel(puntaje: float) -> str:
    a, b, c = CORTES
    return "Baja" if puntaje < a else "Media" if puntaje < b else "Alta" if puntaje < c else "Crítica"


def texto_regla(antecedentes: dict, salida: str) -> str:
    texto = " Y ".join(FRASE[i][t] for i, t in antecedentes.items())
    return f"{texto[0].upper()}{texto[1:]} -> {salida.lower()}"


def _reglas_json(reglas: dict) -> dict:
    return {s: [{"si": a, "entonces": c, "texto": texto_regla(a, c)} for a, c in reglas[s]] for s in SISTEMAS}


# ---------------------------------------------------------------- ajuste con Train
def ajustar(indicadores: pd.DataFrame, salida: Path = DIFUSO) -> dict:
    """Percentiles de Train para las funciones de pertenencia y referencias para explicar."""
    tr = indicadores[indicadores["conjunto"] == "Train"]
    pct = {c: {f"P{q}": float(np.percentile(tr[c], q)) for q in (20, 50, 80)} for c in ("I1", "I3", "I4", "I5")}
    params = {
        "version_reglas": 2,
        "percentiles_train": pct,
        "i6_puntos_fijos": {"negativo": [-40, 0], "estable": [-30, 0, 30], "positivo": [0, 40]},
        "cuartiles_linea_base": {c: [float(np.percentile(tr[c], q)) for q in (25, 50, 75)] for c in ("I1", "I5")},
        # valores de los 201 países: solo para decir en qué percentil está un país en la explicación
        "referencia_todos": {c: sorted(float(v) for v in indicadores[c]) for c in IND},
        "reglas": _reglas_json(REGLAS),
        "umbral_no_concluyente": UMBRAL_NO_CONCLUYENTE,
    }
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")
    return params


def cargar_params(ruta: Path = DIFUSO) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def con_reglas_v1(params: dict) -> dict:
    """Copia de los parámetros con la versión 1 de las reglas (para el Exp 5)."""
    p = copy.deepcopy(params)
    p["version_reglas"], p["reglas"] = 1, _reglas_json(REGLAS_V1)
    return p


# ---------------------------------------------------------------- inferencia Mamdani
_FUNCIONES: dict = {}


def funciones_pertenencia(params: dict) -> dict:
    """Arreglos de pertenencia de cada entrada (se calculan una vez por juego de percentiles).

    El universo incluye exactamente P20, P50 y P80, así interp_membership da el valor exacto
    aunque I4 e I5 tengan percentiles muy pequeños frente a su rango (0 a 400 y 0 a 30).
    """
    clave = json.dumps(params["percentiles_train"], sort_keys=True)
    if clave not in _FUNCIONES:
        f = {}
        for c, p in params["percentiles_train"].items():
            lo, hi = RANGOS[c]
            p20, p50, p80 = p["P20"], p["P50"], p["P80"]
            u = np.unique(np.concatenate([np.linspace(lo, hi, 4001), [p20, p50, p80]]))
            f[c] = (u, {"bajo": fuzz.trapmf(u, [lo, lo, p20, p50]), "medio": fuzz.trimf(u, [p20, p50, p80]),
                        "alto": fuzz.trapmf(u, [p50, p80, hi, hi])})
        lo, hi = RANGOS["I6"]
        u = np.unique(np.concatenate([np.linspace(lo, hi, 6001), [-40, -30, 0, 30, 40]]))
        f["I6"] = (u, {"negativo": fuzz.trapmf(u, [lo, lo, -40, 0]), "estable": fuzz.trimf(u, [-30, 0, 30]),
                       "positivo": fuzz.trapmf(u, [0, 40, hi, hi])})
        _FUNCIONES[clave] = f
    return _FUNCIONES[clave]


def fuzzificar(valores: dict, params: dict) -> dict:
    mu = {}
    for c, (u, conjuntos) in funciones_pertenencia(params).items():
        lo, hi = RANGOS[c]
        x = float(np.clip(valores[c], lo, hi))
        mu[c] = {t: float(fuzz.interp_membership(u, m, x)) for t, m in conjuntos.items()}
        if "alto" in mu[c]:
            mu[c]["no_alto"] = float(fuzz.fuzzy_not(np.array([mu[c]["alto"]]))[0])   # 1 - alto = bajo + medio
    return mu


def inferir(valores: dict, sistema: str, params: dict) -> dict:
    """Puntaje (0 a 100), nivel, fuerza de la regla dominante, reglas activadas y conjunto agregado."""
    mu = fuzzificar(valores, params)
    agregado = np.zeros_like(U)
    activas = []
    for regla in params["reglas"][sistema]:
        fuerza = min(mu[c][t] for c, t in regla["si"].items())      # Y = mínimo
        if fuerza > 0:
            agregado = np.fmax(agregado, np.fmin(fuerza, SALIDA[regla["entonces"]]))  # agregación = máximo
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
    """Percentiles de los 201 países, puntos equidistantes en su rango y los extremos del contrato."""
    ref = params["referencia_todos"][c]
    return np.unique(np.concatenate([np.quantile(ref, np.linspace(0, 1, puntos)),
                                     np.linspace(ref[0], ref[-1], puntos), RANGOS[c]]))


def cobertura(params: dict, puntos: int = 10) -> dict:
    """Malla de 10 x 10 x 10 = 1,000 combinaciones por sistema: fuerza dominante mínima.

    Los 10 valores de cada eje son los percentiles 0, 11, 22, ..., 100 de los 201 países
    (no puntos equidistantes, porque I4 e I5 están muy sesgados y casi todo caería en "alto").
    """
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
    """Más riesgo nunca debe bajar la prioridad.

    Para cada país, recorre cada entrada de su sistema (las otras 2 fijas) y cuenta los pasos
    en que el puntaje retrocede más de medio punto (la salida está discretizada cada 0.1)
    y los que además bajan de nivel. En adaptación, I4 alto debe bajar la prioridad.
    """
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
    """Funciones de pertenencia de las 5 entradas y de la salida."""
    titulos = {"I1": "I1 Calentamiento (°C por década)", "I3": "I3 Variabilidad (°C)",
               "I4": "I4 Emisiones acumuladas (Gt CO₂, escala log)", "I5": "I5 Participación mundial (%, escala log)",
               "I6": "I6 Crecimiento de emisiones (%)"}
    colores = ["#2a9d8f", "#e9c46a", "#e76f51"]
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    for ax, c in zip(axs.flat, ["I1", "I3", "I4", "I5", "I6"]):
        u, conjuntos = funciones_pertenencia(params)[c]
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
    fig.suptitle("Funciones de pertenencia (percentiles de Train; las marcas abajo son los 201 países)")
    fig.tight_layout()
    archivo.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


def figura_ejemplo(valores: dict, nombre_pais: str, params: dict, archivo: Path) -> None:
    """Conjunto agregado y centroide de un país en los 2 sistemas (gráfica G5 del tablero)."""
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
    fig.suptitle(f"Inferencia Mamdani para {nombre_pais}")
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- evaluación (Exp 4 a 6)
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


def _metricas_experto(experto: pd.DataFrame, paises_test: list, niveles: dict) -> dict:
    pos = {p: i for i, p in enumerate(paises_test)}
    faltan = set(experto["pais"]) - set(pos)
    if faltan:
        raise ValueError(f"Países del experto que no están en Test: {faltan}")
    res = {}
    for s in SISTEMAS:
        y_true = experto[f"nivel_{s}"].tolist()
        y_pred = [niveles[s][pos[p]] for p in experto["pais"]]
        res[s] = {"concordancia": float(np.mean(np.array(y_true) == np.array(y_pred))),
                  "aciertos": int(sum(a == b for a, b in zip(y_true, y_pred))), "n": len(y_true),
                  "f1_macro": float(f1_score(y_true, y_pred, labels=NIVELES, average="macro", zero_division=0)),
                  "y_true": y_true, "y_pred": y_pred}
    return res


def evaluar(indicadores: pd.DataFrame, params: dict, fig_dir: Path = FIGURAS,
            ruta_experto: Path = NIVELES_EXPERTO) -> dict:
    te = indicadores[indicadores["conjunto"] == "Test"].reset_index(drop=True)
    paises = te["pais"].tolist()
    versiones = {"Exp 5": con_reglas_v1(params), "Exp 6": params}
    experto = leer_experto(ruta_experto)

    # ---- Exp 4: regla de cuartiles
    t = time.perf_counter()
    niveles = {"Exp 4": {s: [linea_base(f.to_dict(), s, params) for _, f in te.iterrows()] for s in SISTEMAS}}
    tiempos = {"Exp 4": (time.perf_counter() - t) * 1000 / (2 * len(te))}
    verif = {"Exp 4": {"explicacion_2_indicadores": 0}}

    # ---- Exp 5 y 6: Mamdani con cada versión de reglas
    for exp, p in versiones.items():
        t = time.perf_counter()
        res = {s: [inferir(f.to_dict(), s, p) for _, f in te.iterrows()] for s in SISTEMAS}
        tiempos[exp] = (time.perf_counter() - t) * 1000 / (2 * len(te))
        niveles[exp] = {s: [r["nivel"] for r in res[s]] for s in SISTEMAS}
        explic = sum(sum(k in explicar(f.to_dict(), s, r, p) for k in CLAVE_EXPLICACION.values()) >= 2
                     for s in SISTEMAS for (_, f), r in zip(te.iterrows(), res[s]))
        verif[exp] = {"cobertura": cobertura(p), "coherencia": coherencia(te, p),
                      "explicacion_2_indicadores": int(explic),
                      "acuerdo_con_exp4": {s: float(np.mean(np.array(niveles[exp][s]) == np.array(niveles["Exp 4"][s])))
                                           for s in SISTEMAS},
                      "cambios_de_nivel_vs_exp5": None}
    verif["Exp 6"]["cambios_de_nivel_vs_exp5"] = {
        s: int(sum(a != b for a, b in zip(niveles["Exp 5"][s], niveles["Exp 6"][s]))) for s in SISTEMAS}

    metricas = {exp: _metricas_experto(experto, paises, niveles[exp]) for exp in niveles} if experto is not None else {}

    # ---- figuras
    _matrices({NOMBRE[s]: (niveles["Exp 4"][s], niveles["Exp 6"][s]) for s in SISTEMAS},
              "Mamdani (Exp 6) frente a la regla de cuartiles (Exp 4), 41 países de Test",
              fig_dir / "matriz_mamdani_vs_cuartiles.png", "Regla de cuartiles (Exp 4)", "Mamdani (Exp 6)")
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
                  "Adaptación por I1, mitigación por I5; cortes P25, P50 y P75 de Train"),
        "Exp 5": ("Exp 5: Alternativo", "Lógica difusa Mamdani, reglas versión 1",
                  "15 reglas por sistema (malla 3 x 3 + 6 ajustes); P20, P50 y P80 de Train; mínimo, máximo y centroide"),
        "Exp 6": ("Exp 6: Candidato", "Lógica difusa Mamdani, reglas versión 2",
                  "12 reglas por sistema (malla 3 x 2 con el término no alto + 6 ajustes); mismas funciones y operadores"),
    }
    filas = []
    for exp, (variante, modelo, hiper) in descripcion.items():
        v = verif[exp]
        if exp in metricas:
            m = metricas[exp]
            principal = "Concordancia con experto: " + ", ".join(
                f"{NOMBRE[s][:3]}. {m[s]['aciertos']}/{m[s]['n']} ({m[s]['concordancia']:.0%})" for s in SISTEMAS)
            secundaria = "F1 macro: " + ", ".join(f"{NOMBRE[s][:3]}. {m[s]['f1_macro']:.2f}" for s in SISTEMAS)
        else:
            principal, secundaria = "Concordancia con experto: pendiente", "F1 macro: pendiente"
        if exp == "Exp 4":
            extra = ("Acuerdo de referencia; la explicación usa 1 solo indicador (no cumple el CA6); "
                     "no detecta casos no concluyentes")
            if exp in metricas:
                ok = all(metricas[exp][s]["concordancia"] >= META_CONCORDANCIA for s in SISTEMAS)
                diag = "Línea base: " + ("supera" if ok else "no alcanza") + " la meta del 80 %"
            else:
                diag = "Línea base: sirve de referencia para Exp 5 y 6"
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
                diag = ("Óptimo: coincide con el experto en al menos 80 % de los países" if ok else
                        "Subajuste: las reglas no capturan el criterio del experto (menos de 80 %)")
                if incoherente:
                    diag += "; además, más riesgo puede bajar la prioridad"
            elif incoherente:
                diag = "Descartado: más riesgo puede bajar la prioridad (reglas que contradicen la malla)"
            else:
                diag = "Coherente; concordancia con el experto pendiente"
        filas.append({
            "experimento": variante, "modelo": modelo, "hiperparametros": hiper,
            "metrica_principal": principal, "metrica_secundaria": secundaria,
            "metrica_train": "No aplica: no se entrena, solo se calculan percentiles de Train",
            "tiempo": f"< 0.01 s / {tiempos[exp]:.2f} ms por país", "diagnostico": diag, "verificaciones": extra,
            "tiempo_entrenamiento_s": 0.0, "tiempo_inferencia_ms": round(tiempos[exp], 3),
        })
    return {"experimentos": filas, "verificaciones": verif, "experto_disponible": experto is not None,
            "metricas_experto": {e: {s: {k: v for k, v in d.items() if k not in ("y_true", "y_pred")}
                                     for s, d in m.items()} for e, m in metricas.items()},
            "niveles_test": {p: {s: niveles["Exp 6"][s][i] for s in SISTEMAS} for i, p in enumerate(paises)}}


if __name__ == "__main__":
    consola_utf8()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    datos = cargar()
    r = evaluar(datos, ajustar(datos))
    for f in r["experimentos"]:
        log.info("%s | %s | %s | %s", f["experimento"], f["metrica_principal"], f["diagnostico"], f["verificaciones"])
