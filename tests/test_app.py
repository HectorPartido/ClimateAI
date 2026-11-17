"""Pruebas del tablero Streamlit con AppTest (PR 4).

Responsable: Anthony Oziel Ramos López.
Rama: feat/datos-modelo-pruebas

Cada prueba abre el tablero sin navegador (streamlit.testing), entra a una pantalla y revisa
que no haya errores y que muestre lo que pide la Entrega 2 (RF-01 a RF-17).
"""
import os
import tempfile
from pathlib import Path

import pytest

# Los reportes de error de estas pruebas van a una carpeta temporal, no a data/reportes.csv
os.environ["CLIMATEAI_REPORTES"] = str(Path(tempfile.mkdtemp()) / "reportes.csv")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PANTALLAS = ["paginas/1_panorama.py", "paginas/2_ranking.py", "paginas/3_ficha.py", "paginas/4_grupos.py",
             "paginas/5_simulador.py", "paginas/6_datos_modelo.py"]


def abrir(pagina: str | None = None) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    if pagina:
        at.switch_page(pagina)
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("pagina", PANTALLAS)
def test_cada_pantalla_abre_sin_errores_y_con_avisos(pagina):
    at = abrir(pagina)
    assert len(at.title) == 1
    avisos = " ".join(w.value for w in at.sidebar.warning) + " ".join(i.value for i in at.sidebar.info)
    assert "1850 a 2012" in avisos and "inteligencia artificial" in avisos       # RNF-05


def test_panorama_kpis():
    at = abrir()
    kpis = {m.label: m.value for m in at.metric}
    assert kpis["Países analizados"] == "201"
    assert kpis["Grupos de países"] == "4"
    assert int(kpis["Prioridad crítica de adaptación"]) > 0


def test_simulador_carga_mexico_por_defecto():
    at = abrir("paginas/5_simulador.py")
    valores = [m.value for m in at.metric]
    assert valores[:2] == ["25.6", "84.1"]                      # RF-05 y contrato de la Entrega 2
    assert len(at.slider) == 5                                  # RF-01: I1, I3, I4, I5 e I6


def test_simulador_recalcula_al_mover_un_control():
    at = abrir("paginas/5_simulador.py")
    at.slider(key="sl_I5").set_value(10.0).run()                # RF-02: recalcular automáticamente
    assert at.metric[1].value != "84.1" or at.metric[1].delta != "sin cambios"
    assert not at.exception


def test_simulador_sin_recalculo_automatico_espera_el_boton():
    at = abrir("paginas/5_simulador.py")
    at.toggle(key="sim_auto").set_value(False).run()
    at.slider(key="sl_I1").set_value(0.55).run()
    assert any("sin calcular" in i.value for i in at.info)
    at.button[[b.label for b in at.button].index("Calcular prioridad")].click().run()
    assert not any("sin calcular" in i.value for i in at.info)


def test_simulador_entrada_invalida_muestra_advertencia():
    at = abrir("paginas/5_simulador.py")
    caja = next(t for t in at.text_input if t.label == "I4")
    caja.set_value("mil")
    next(b for b in at.button if b.label == "Probar entrada").click().run()
    assert any("no válida" in w.value for w in at.warning)      # RF-14
    assert at.metric[0].value == "25.6"                         # conserva el último resultado válido


def test_ranking_busqueda_sin_acentos_y_sin_coincidencias():
    at = abrir("paginas/2_ranking.py")
    at.text_input(key="rk_busca").set_value("mexico").run()     # RF-07
    assert "(1 país)" in " ".join(m.value for m in at.markdown)
    at.text_input(key="rk_busca").set_value("zzzz").run()
    assert any("Sin coincidencias" in i.value for i in at.info)


def test_grupos_permite_explorar_otro_k():
    at = abrir("paginas/4_grupos.py")
    at.segmented_control(key="gr_k").set_value(5).run()         # RF-12
    assert not at.exception
    assert any("Exploración con k = 5" in i.value for i in at.info)


def test_ficha_muestra_mexico_y_sus_indicadores():
    at = abrir("paginas/3_ficha.py")
    assert at.selectbox(key="fi_pais").value == "Mexico"
    assert len(at.dataframe) >= 1


def test_reportar_un_error_guarda_el_reporte():
    at = abrir()
    at.sidebar.text_area[0].set_value("El mapa no carga")
    next(b for b in at.sidebar.button if b.label == "Enviar reporte").click().run()
    archivo = Path(os.environ["CLIMATEAI_REPORTES"])
    assert archivo.exists() and "El mapa no carga" in archivo.read_text(encoding="utf-8")
