"""Regeneración completa, reproducibilidad y hoja del experto (PR 4).

Responsable: Anthony Oziel Ramos López.
Rama: feat/datos-modelo-pruebas

Corre el mismo comando que el botón "Ejecutar análisis" (src/models/train.py) y revisa que,
con random_state = 42, salgan exactamente los mismos indicadores, grupos y niveles que están
guardados en el repositorio (RNF-04). Antes de correr se respaldan data/processed/, models/,
reports/ y logs/, y al terminar se restauran, así la prueba no deja cambios en git.

Tarda alrededor de 30 a 60 segundos.
"""
import json
import shutil

import pandas as pd
import pytest

from src.config import (DATA_EXPERT, DATA_PROCESSED, INDICADORES, LOGS, META_CONCORDANCIA, META_SILUETA_TEST,
                        META_SILUETA_TRAIN, MODELS, NIVELES, NIVELES_EXPERTO, PARTICION, REPORTS, RESUMEN)

CARPETAS = [DATA_PROCESSED, MODELS, REPORTS, LOGS]
NIVELES_PAIS = REPORTS / "niveles_pais.csv"


@pytest.fixture(scope="module")
def regenerado(tmp_path_factory):
    """Respalda las salidas, corre el entrenamiento completo y restaura todo al final."""
    from src.models import train
    from src.models.predict import _artefactos

    respaldo = tmp_path_factory.mktemp("respaldo")
    for c in CARPETAS:
        if c.exists():
            shutil.copytree(c, respaldo / c.name)
    antes = {"indicadores": pd.read_csv(INDICADORES), "particion": pd.read_csv(PARTICION),
             "niveles": pd.read_csv(NIVELES_PAIS), "resumen": json.loads(RESUMEN.read_text(encoding="utf-8"))}
    try:
        resumen = train.main()
        _artefactos.cache_clear()
        yield antes, resumen
    finally:
        import logging
        logging.shutdown()  # en Windows no se puede borrar un log abierto
        for c in CARPETAS:
            shutil.rmtree(c, ignore_errors=True)
            if (respaldo / c.name).exists():
                shutil.copytree(respaldo / c.name, c, dirs_exist_ok=True)
        _artefactos.cache_clear()


def test_mismos_indicadores(regenerado):
    antes, _ = regenerado
    pd.testing.assert_frame_equal(antes["indicadores"], pd.read_csv(INDICADORES), rtol=1e-9)


def test_misma_particion(regenerado):
    antes, _ = regenerado
    pd.testing.assert_frame_equal(antes["particion"], pd.read_csv(PARTICION))


def test_mismos_grupos_y_niveles(regenerado):
    antes, _ = regenerado
    nuevo = pd.read_csv(NIVELES_PAIS)
    columnas = ["pais", "conjunto", "grupo_id", "nivel_adaptacion", "nivel_mitigacion"]
    pd.testing.assert_frame_equal(antes["niveles"][columnas], nuevo[columnas])
    for c in ("puntaje_adaptacion", "puntaje_mitigacion"):
        assert (antes["niveles"][c] - nuevo[c]).abs().max() < 0.05


def test_metas_de_la_entrega_2(regenerado):
    _, r = regenerado
    exp2 = next(e for e in r["experimentos"] if e["experimento"].startswith("Exp 2"))
    assert exp2["silueta_train"] >= META_SILUETA_TRAIN
    assert exp2["silueta_test"] >= META_SILUETA_TEST
    for s, m in r["difuso"]["metricas_experto"]["Exp 7"].items():
        assert m["concordancia"] >= META_CONCORDANCIA, s


def test_predecir_responde_rapido_y_sin_errores(regenerado):
    _, r = regenerado
    p = r["predecir"]
    assert p["llamadas"] == 2 * r["paises"]
    assert p["no_concluyentes"] == 0 and p["grupo_distinto"] == 0
    assert p["tiempo_ms_maximo"] < 1000          # RNF-01: meta de 1 s
    assert set(p["distribucion"]["adaptacion"]) <= set(NIVELES)


def test_hoja_del_experto_ida_y_vuelta(tmp_path):
    """La hoja se genera y las respuestas del docente se importan igual que en el repositorio."""
    from src.data import hoja_experto

    hoja = hoja_experto.crear(tmp_path / "hoja.xlsx")
    assert hoja.exists() and hoja.stat().st_size > 5000
    salida = tmp_path / "niveles_experto.csv"
    shutil.copy(NIVELES_EXPERTO, salida)
    out = hoja_experto.importar(DATA_EXPERT / "respuestas_docente.xlsx", salida)
    original = pd.read_csv(NIVELES_EXPERTO, dtype=str).fillna("")
    cols = ["pais", "nivel_adaptacion", "nivel_mitigacion"]
    pd.testing.assert_frame_equal(original[cols].reset_index(drop=True), out[cols].reset_index(drop=True))
