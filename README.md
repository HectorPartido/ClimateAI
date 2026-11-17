# ClimateAI: sistema funcional

Proyecto final de **Fundamentos de Inteligencia Artificial**, Entrega 4 (liberación del sistema funcional). Grupo LITIID71, Equipo ClimateAI.

ClimateAI asigna a cada país una prioridad climática de **adaptación** y otra de **mitigación** (0 a 100, con cuatro niveles: Baja, Media, Alta y Crítica) a partir de 6 indicadores históricos de temperatura (Berkeley Earth) y de emisiones de CO₂ (Our World in Data), con corte en 2012. Todo se consulta en un tablero web de 6 pantallas hecho con Streamlit.

| | |
|---|---|
| **Repositorio** | https://github.com/HectorPartido/ClimateAI |
| **Aplicación desplegada** | https://climateai-litiid71.streamlit.app |
| **Rama final congelada** | `release/v1.0` (etiqueta `v1.0`) |
| **Video de respaldo** | https://drive.google.com/file/d/1WN2sV4ohw_3IYaLF0PJ4MaX_68iABJb1/view?usp=sharing |
| **Comando de arranque** | `streamlit run app/streamlit_app.py` |
| **Credenciales** | No aplica: el tablero no tiene inicio de sesión |
| **Variables de entorno** | No aplica: no usa claves ni servicios externos |

## Guía rápida de ejecución

### 1. Requisitos

| Componente | Versión |
|---|---|
| Python | **3.13** (probado también con 3.12 y 3.14; mínimo 3.12) |
| Streamlit | 1.67.0 |
| Plotly | 7.1.0 |
| scikit-learn | 1.9.1 (debe ser esta: los modelos de `models/` se guardaron con ella) |
| numpy / pandas / scipy | 2.5.5 / 3.1.1 / 1.18.1 |
| scikit-fuzzy / joblib / matplotlib / openpyxl | 0.5.0 / 1.6.0 / 3.11.2 / 3.1.5 |
| pytest / pytest-cov | 9.2.0 / 7.1.0 |

Todas las versiones están fijas en `requirements.txt`. Navegadores probados: Chrome, Edge y Firefox recientes, en Windows y macOS.

### 2. Instalación

```bash
git clone https://github.com/HectorPartido/ClimateAI.git
cd ClimateAI
git checkout release/v1.0           # versión entregada

python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Arranque del tablero

```bash
streamlit run app/streamlit_app.py
```

Se abre en http://localhost:8501. No hace falta entrenar antes: los modelos, los indicadores y los niveles de los 201 países ya están en el repositorio (`models/`, `data/processed/` y `reports/`).

### 4. Variables de entorno y credenciales

No se necesitan. El tablero lee solo archivos del repositorio, no llama a ninguna API y no tiene inicio de sesión, por eso no hay `.env.example`. La única variable opcional es `CLIMATEAI_REPORTES`, que cambia dónde se guardan los reportes de error (por defecto `data/reportes.csv`); las pruebas la usan para no ensuciar el repositorio.

### 5. Regenerar todo y correr las pruebas

```bash
python src/models/train.py          # indicadores, grupos, sistemas difusos, figuras y reportes (~30 s)
python -m pytest -q --cov=src       # 48 pruebas: predecir(), tablero y regeneración completa
```

El botón **Ejecutar análisis** de la pantalla *Datos y modelo* corre el mismo `train.py`. Con `random_state = 42` el resultado es idéntico al guardado (lo verifica `tests/test_pipeline.py`).

## Despliegue en Streamlit Community Cloud

1. Entrar a https://share.streamlit.io con la cuenta de GitHub de HectorPartido y elegir **Create app**, opción *Deploy a public app from GitHub*.
2. Repository: `HectorPartido/ClimateAI`. Branch: `release/v1.0`. Main file path: `app/streamlit_app.py`.
3. App URL: `climateai-litiid71`.
4. **Advanced settings**: Python 3.13. Secrets: vacío.
5. **Deploy**. La primera instalación tarda unos minutos; después la aplicación responde en https://climateai-litiid71.streamlit.app.

Notas del despliegue:

- El mapa del mundo se sirve desde `app/static/world_110m.json` (opción `enableStaticServing` de `.streamlit/config.toml`), así no depende de un CDN externo durante la presentación. La ruta es relativa (`app/static/`) porque Streamlit Cloud corre la aplicación dentro de un marco en `/~/+/`; con `/app/static/` el mapa no carga en la nube.
- En la nube los reportes de error se guardan en `data/reportes.csv` del servidor y se pierden cuando la aplicación se reinicia; se pueden consultar y descargar en *Datos y modelo* antes de eso.
- Si la aplicación se duerme por inactividad, basta con abrirla y presionar el botón para despertarla (tarda menos de un minuto).

## Pantallas

| Menú | Pantalla | Qué hace |
|---|---|---|
| Análisis | Panorama | Mapa de prioridades, los 8 países con mayor prioridad, indicadores clave, búsqueda y exportar informe |
| Análisis | Ranking de países | Tabla ordenada y filtrable (nivel, grupo, búsqueda), 12 filas por página, exportar CSV; una fila abre la ficha |
| Análisis | Ficha de país | Prioridades con escala de 4 tramos, explicación, 3 reglas más fuertes, indicadores con percentil y promedio del grupo, series históricas |
| Modelo de IA | Grupos de países | Dispersión de calentamiento frente a emisiones con centros, otros valores de k, los 4 países más parecidos y separación por k |
| Modelo de IA | Simulador de prioridad | Controles de 5 indicadores, cálculo automático o con botón, conjunto difuso y reglas activadas |
| Modelo de IA | Datos y modelo | Fuentes y licencias, pasos y parámetros, calidad del modelo, validación con el docente, experimentos, países sin emparejar, reportes y botón **Ejecutar análisis** |

La barra lateral muestra en todas las pantallas el aviso del periodo histórico, el aviso de IA, las fuentes con su licencia y el botón **Reportar un error**.

## Motor de IA

1. **Grupos de países (Exp 2):** log10 en I4 e I5, z-score, PCA de 2 componentes y K-means con k = 4 (silueta Train 0.407, Test 0.382, Davies-Bouldin 0.814).
2. **Prioridad (Exp 7):** dos sistemas de lógica difusa Mamdani calibrados solo con Train (cuantiles ponderados por masa de emisiones y reglas monótonas). Frente a los niveles del docente en 20 países de Test: **adaptación 16/20 (80 %)** y **mitigación 19/20 (95 %)**.

La calibración del Exp 7 la propuso el profesor en su fork del repositorio ([Carloscastro16/ClimateAI-test](https://github.com/Carloscastro16/ClimateAI-test)); se integró sin cambios en el PR 0 de esta entrega y conserva su autoría en el historial. El diagnóstico completo de los Exp 1 a 7 está en `docs/Entrega3_ClimateAI.pdf` y en `reports/experimentos.csv`.

## Estructura

| Ruta | Contenido | PR |
|---|---|---|
| `app/streamlit_app.py` | Punto de entrada: menú en dos secciones y barra lateral | 1 |
| `app/comun.py` | Datos con caché, navegación, barra lateral, colores y gráficas compartidas | 1 |
| `app/paginas/5_simulador.py` | Simulador de prioridad | 1 |
| `app/paginas/1_panorama.py`, `2_ranking.py` | Panorama y Ranking de países | 2 |
| `app/static/world_110m.json` | Mapa del mundo para Plotly (Natural Earth, vía `sane-topojson`, licencia MIT) | 2 |
| `app/paginas/3_ficha.py`, `4_grupos.py` | Ficha de país y Grupos de países | 3 |
| `app/paginas/6_datos_modelo.py` | Datos y modelo | 4 |
| `tests/test_app.py`, `tests/test_pipeline.py` | Pruebas del tablero (AppTest) y de regeneración completa | 4 |
| `data/reference/nombres_es.csv` | Nombre en español de los 201 países | 1 |
| `.streamlit/config.toml` | Tema, servidor y archivos estáticos | 1 |
| `src/` | Motor de IA de la Entrega 3 (`config.py`, `data/preprocess.py`, `models/clustering.py`, `models/fuzzy.py`, `models/train.py`, `models/predict.py`) | E3 |
| `data/raw/`, `data/processed/`, `data/expert/` | Datos originales, indicadores y niveles del docente | E3 |
| `models/`, `reports/`, `logs/` | Modelos, resultados, figuras y registro del último entrenamiento | E3 |
| `docs/` | Documentos de las Entregas 3 y 4 | 5 |

## Equipo y PR de la Entrega 4

| PR | Integrante | Rama | Entrega |
|---|---|---|---|
| 0 | Hector Partido Flores | `integra/fork-profesor` | Integra el Exp 7 del fork del profesor |
| 1 | Gildardo Emmanuel González García | `feat/tablero-base` | Base del tablero, barra lateral y Simulador |
| 2 | Ismael Alexander Hidalgo Aguilando | `feat/panorama-ranking` | Panorama y Ranking |
| 3 | Hector Martínez Sánchez | `feat/ficha-grupos` | Ficha de país y Grupos |
| 4 | Anthony Oziel Ramos López | `feat/datos-modelo-pruebas` | Datos y modelo, pruebas y cobertura |
| Fix | Hector Partido Flores | `fix/mapa-nube` | Ruta relativa del mapa para Streamlit Cloud |
| 5 | Hector Partido Flores | `docs/entrega4` | README técnico, documento, despliegue y rama congelada |

## Reglas del equipo

- Nada de rutas absolutas: las rutas salen de `src/config.py` (motor) y de `app/comun.py` (tablero).
- `random_state = 42` en todo lo que tenga aleatoriedad.
- El tablero no entrena nada: lee `models/` y `reports/` y llama a `predecir()`.
- Después de la fecha de entrega no se suben cambios a `release/v1.0`.

## Licencias de los datos

- Temperatura: Berkeley Earth, licencia CC BY-NC-SA 4.0.
- Emisiones de CO₂: Our World in Data con datos del Global Carbon Project, licencia CC BY 4.0.
