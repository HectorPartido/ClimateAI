"""Pruebas de predecir() (PR 4, Gildardo Emmanuel González García).

Antes, entrenar:  python src/models/train.py --raw data/raw --output models/
Correr con:       python -m pytest -q
"""
import math

import numpy as np
import pandas as pd
import pytest

from src.config import INDICADORES, KMEANS, NIVELES, RANGOS, SISTEMAS
from src.models.predict import EJEMPLO_MEXICO, predecir

if not KMEANS.exists() or not INDICADORES.exists():
    pytest.skip("Faltan los modelos: corre primero python src/models/train.py --raw data/raw --output models/",
                allow_module_level=True)

LLAVES = {"prediccion", "clase_etiqueta", "confianza_o_score", "tiempo_ms"}
EXTRA = {"grupo_kmeans", "explicacion", "reglas_activadas", "conjunto_agregado", "estado"}
FRASES = {"I1": "se calienta", "I3": "varía", "I4": "ha emitido", "I5": "aporta el", "I6": "sus emisiones de 2003"}
PAISES = pd.read_csv(INDICADORES)


def entrada(base=EJEMPLO_MEXICO, **cambios):
    return {**base, **cambios}


@pytest.mark.parametrize("sistema", SISTEMAS)
def test_contrato_completo(sistema):
    s = predecir(entrada(sistema=sistema))
    assert LLAVES | EXTRA <= s.keys()
    assert 0 <= s["prediccion"] <= 100
    assert s["clase_etiqueta"] in NIVELES
    assert 0 <= s["confianza_o_score"] <= 1
    assert s["tiempo_ms"] <= 3000
    assert set(s["grupo_kmeans"]) == {"id", "nombre"} and s["grupo_kmeans"]["id"] in (1, 2, 3, 4)
    fuerzas = [r["fuerza"] for r in s["reglas_activadas"]]
    assert fuerzas == sorted(fuerzas, reverse=True) and fuerzas[0] == s["confianza_o_score"]
    ca = s["conjunto_agregado"]
    assert len(ca["x"]) == len(ca["mu"]) and ca["centroide"] == s["prediccion"]


def test_mexico_como_en_la_entrega_2():
    mit = predecir(entrada(sistema="mitigacion"))
    ada = predecir(entrada(sistema="adaptacion"))
    assert mit["grupo_kmeans"]["id"] == 1                  # Grandes emisores con calentamiento bajo
    assert mit["clase_etiqueta"] == "Crítica"              # la Entrega 2 esperaba crítica (79.8 con 2 reglas)
    assert mit["reglas_activadas"][0]["fuerza"] == 1.0
    assert ada["clase_etiqueta"] == "Media"


@pytest.mark.parametrize("mala", [
    {k: v for k, v in EJEMPLO_MEXICO.items() if k != "I3"},      # falta una llave
    entrada(I4=401.0),                                           # fuera de rango
    entrada(I6=-150),                                            # fuera de rango
    entrada(I1="0.2"),                                           # texto en vez de número
    entrada(I2=True),                                            # booleano
    entrada(I5=math.nan),                                        # NaN
    entrada(sistema="otro"),                                     # sistema no válido
    {k: v for k, v in EJEMPLO_MEXICO.items() if k != "sistema"},  # falta el sistema
    [0.2, 0.5, 0.2, 16, 1.4, 25],                                # no es dict
])
def test_entrada_invalida_lanza_valueerror(mala):
    with pytest.raises(ValueError):
        predecir(mala)


@pytest.mark.parametrize("sistema", SISTEMAS)
def test_los_201_paises_tienen_nivel_y_explicacion(sistema):
    for _, f in PAISES.iterrows():
        s = predecir({**{c: float(f[c]) for c in RANGOS}, "sistema": sistema})
        assert s["estado"] == "ok", f["pais"]
        usados = sum(frase in s["explicacion"] for frase in FRASES.values())
        assert usados >= 2, (f["pais"], s["explicacion"])        # CA6: al menos 2 indicadores


@pytest.mark.parametrize("valores", [{c: lo for c, (lo, hi) in RANGOS.items()},
                                     {c: hi for c, (lo, hi) in RANGOS.items()}])
@pytest.mark.parametrize("sistema", SISTEMAS)
def test_extremos_del_rango(valores, sistema):
    s = predecir({**valores, "sistema": sistema})
    assert s["clase_etiqueta"] in NIVELES


@pytest.mark.parametrize("sistema,indicador", [("adaptacion", "I1"), ("mitigacion", "I5"), ("mitigacion", "I6")])
def test_mas_riesgo_no_baja_la_prioridad(sistema, indicador):
    lo, hi = RANGOS[indicador]
    puntajes = [predecir(entrada(sistema=sistema, **{indicador: float(v)}))["prediccion"]
                for v in np.linspace(lo, hi, 25)]
    assert all(b >= a - 0.5 for a, b in zip(puntajes, puntajes[1:])), puntajes


def test_misma_entrada_misma_salida():
    a, b = predecir(EJEMPLO_MEXICO), predecir(EJEMPLO_MEXICO)
    a.pop("tiempo_ms"), b.pop("tiempo_ms")
    assert a == b
