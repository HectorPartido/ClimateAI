"""ClimateAI: configuración compartida por todo el motor de IA.

Todas las rutas se calculan a partir de la raíz del repositorio, así el proyecto
corre igual en cualquier computadora después de clonarlo (nada de C:\\Users\\...).
Nadie debe escribir rutas absolutas: importen las constantes de este archivo.

    from src.config import DATA_RAW, DATA_PROCESSED, MODELS, SEMILLA
"""
import sys
from pathlib import Path

# ---------------------------------------------------------------- rutas
RAIZ = Path(__file__).resolve().parents[1]
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROCESSED = RAIZ / "data" / "processed"
DATA_EXPERT = RAIZ / "data" / "expert"
MODELS = RAIZ / "models"
REPORTS = RAIZ / "reports"
FIGURAS = REPORTS / "figuras"
LOGS = RAIZ / "logs"

# ---------------------------------------------------------------- archivos de entrada (data/raw)
ARCH_TEMP = "GlobalLandTemperaturesByCountry.csv"
ARCH_GLOBAL = "GlobalTemperatures.csv"
ARCH_CO2 = "annual-co2-emissions-per-country.csv"

# ---------------------------------------------------------------- archivos que se generan
INDICADORES = DATA_PROCESSED / "indicadores_pais.csv"   # una fila por país con I1 a I6
PARTICION = DATA_PROCESSED / "particion.csv"            # pais, conjunto (Train o Test) y orden del split
NIVELES_EXPERTO = DATA_EXPERT / "niveles_experto.csv"   # niveles del docente (20 países de Test)
EXPERIMENTOS = REPORTS / "experimentos.csv"             # matriz de experimentación (Exp 1 a 5)
LOG_ENTRENAMIENTO = LOGS / "entrenamiento.log"

# Artefactos serializados (models/)
SCALER = MODELS / "scaler.joblib"
PCA_MODELO = MODELS / "pca.joblib"
KMEANS = MODELS / "kmeans_k4.joblib"        # modelo candidato (Exp 2)
GMM = MODELS / "gmm_k4.joblib"              # modelo comparativo (Exp 3)
DIFUSO = MODELS / "difuso.json"             # percentiles de Train y reglas de los 2 sistemas
GRUPOS = MODELS / "grupos.json"             # nombre, tamaño y centroide de cada grupo
RESUMEN = REPORTS / "resumen.json"          # números de la última ejecución (para el documento)

# Rango válido de cada entrada de predecir() (contrato de la sección 5 de la Entrega 2)
RANGOS = {"I1": (0.00, 0.60), "I2": (-1.00, 2.50), "I3": (0.00, 1.20),
          "I4": (0.0, 400.0), "I5": (0.00, 30.00), "I6": (-100.0, 500.0)}
SISTEMAS = ("adaptacion", "mitigacion")

# ---------------------------------------------------------------- decisiones de la Entrega 2 (no cambiar sin avisar al equipo)
SEMILLA = 42
TEST_SIZE = 0.20                 # 160 países en Train y 41 en Test
ULTIMO_ANIO = 2012
K = 4
IND = ["I1", "I2", "I3", "I4", "I5", "I6"]
IND_LOG = ["I4", "I5"]           # se transforman con log10(x + 0.001) antes del z-score
NIVELES = ["Baja", "Media", "Alta", "Crítica"]
CORTES = (25, 50, 75)            # Baja < 25 <= Media < 50 <= Alta < 75 <= Crítica
UMBRAL_NO_CONCLUYENTE = 0.30     # fuerza máxima de las reglas por debajo de esto: "no concluyente"
META_CONCORDANCIA = 0.80         # OE3: al menos 16 de 20 países por sistema
META_SILUETA_TRAIN = 0.35        # OE2
META_SILUETA_TEST = 0.30


def asegurar_carpetas() -> None:
    """Crea las carpetas de salida si no existen."""
    for carpeta in (DATA_PROCESSED, DATA_EXPERT, MODELS, REPORTS, FIGURAS, LOGS):
        carpeta.mkdir(parents=True, exist_ok=True)


def consola_utf8() -> None:
    """Evita errores con "°C" o "CO₂" en consolas de Windows que no usan UTF-8."""
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
