# ClimateAI: motor de IA

Proyecto final de **Fundamentos de Inteligencia Artificial**, Entrega 3 (desarrollo, entrenamiento y experimentación del modelo). Grupo LITIID71, Equipo ClimateAI.

ClimateAI asigna a cada país una prioridad climática de **adaptación** y otra de **mitigación** (0 a 100, con cuatro niveles: Baja, Media, Alta y Crítica) a partir de 6 indicadores calculados con datos de temperatura de Berkeley Earth y de emisiones de CO₂ de Our World in Data (corte en 2012). El motor tiene dos etapas:

1. **Grupos de países:** z-score, PCA de 2 componentes y K-means con k = 4.
2. **Prioridad:** dos sistemas de lógica difusa Mamdani (adaptación y mitigación), con 12 reglas cada uno en su versión final.

En esta entrega no hay interfaz gráfica: solo el motor de IA, los modelos guardados, la comparación de modelos y la función `predecir()`.

## Cómo reproducir el entrenamiento

Requiere Python 3.10 o más reciente.

```bash
git clone https://github.com/HectorPartido/ClimateAI.git
cd ClimateAI
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/models/train.py --raw data/raw --output models/
python src/models/predict.py       # ejemplo de inferencia con México (adaptación y mitigación)
python -m pytest -q                # 22 pruebas del contrato de predecir()
```

`train.py` corre todo desde los CSV originales en unos 30 segundos: indicadores, partición, los 6 experimentos, los modelos, las figuras, la matriz de experimentación y el registro.

## Experimentos

Todos usan la partición de la Entrega 2 (`train_test_split`, 20 % de prueba, `random_state = 42`): 160 países en Train y 41 en Test.

| Variante | Modelo | Métrica principal (Test) |
|---|---|---|
| Exp 1: Baseline | K-means sobre los 6 indicadores | Silueta |
| Exp 2: Candidato | PCA de 2 componentes + K-means | Silueta |
| Exp 3: Alternativo | PCA + mezcla gaussiana | Silueta |
| Exp 4: Baseline | Regla de cuartiles de un solo indicador | Concordancia con el experto |
| Exp 5: Alternativo | Mamdani, reglas versión 1 (15 por sistema) | Concordancia con el experto |
| Exp 6: Candidato | Mamdani, reglas versión 2 (12 por sistema) | Concordancia con el experto |

Los resultados de la última ejecución están en `reports/experimentos.csv` y `reports/resumen.json`.

## Niveles del experto

El docente asigna niveles a 20 países de Test sin ver el modelo. Mientras no estén, la concordancia y el F1 aparecen como pendientes y se reportan las verificaciones que no necesitan etiquetas (cobertura de reglas, coherencia, explicación con 2 indicadores y acuerdo con la línea base).

```bash
python src/data/hoja_experto.py crear                         # hoja para el docente
python src/data/hoja_experto.py importar RUTA/DEL/EXCEL.xlsx  # llena data/expert/niveles_experto.csv
python src/models/train.py --raw data/raw --output models/    # recalcula con el experto
```

## Estructura

| Ruta | Contenido | PR |
|---|---|---|
| `data/raw/` | Los 3 CSV originales | 1 |
| `data/processed/` | `indicadores_pais.csv` (201 países, I1 a I6), `particion.csv` (160 Train, 41 Test) y tablas auxiliares | 1 |
| `data/expert/niveles_experto.csv` | Los 20 países de Test que evalúa el docente y sus niveles | 0 |
| `data/expert/hoja_experto_ClimateAI.xlsx` | Hoja para el docente | 5 |
| `src/config.py` | Rutas y decisiones compartidas (semilla 42, k = 4, cortes, metas, rangos) | 0 |
| `src/data/preprocess.py` | Limpieza, emparejamiento de países, indicadores y partición | 1 |
| `src/models/clustering.py` | Exp 1 a 3 | 2 |
| `src/models/fuzzy.py` | Exp 4 a 6 y verificaciones de los sistemas difusos | 3 |
| `src/models/train.py` | Entrenamiento completo con un solo comando y registro | 4 |
| `src/models/predict.py` | Función `predecir()` para el tablero de la Entrega 4 | 4 |
| `tests/test_predict.py` | Pruebas de `predecir()` | 4 |
| `src/data/hoja_experto.py` | Crea la hoja del docente e importa sus respuestas | 5 |
| `models/` | `scaler.joblib`, `pca.joblib`, `kmeans_k4.joblib`, `gmm_k4.joblib`, `grupos.json`, `difuso.json` | 2, 3 |
| `reports/figuras/` | Silueta, convergencia, grupos, funciones de pertenencia, inferencia de México y matrices | 2, 3 |
| `reports/` | `experimentos.csv`, `resumen.json` y `niveles_pais.csv` (201 países) | 4 |
| `logs/entrenamiento.log` | Registro de la última ejecución | 4 |
| `docs/` | Documento de la Entrega 3 | 5 |

## Equipo, roles y PR

Cada integrante abre **su propio PR desde su cuenta de GitHub**. Hector Partido revisa y fusiona en el orden de la tabla.

| PR | Integrante | Rol | Rama | Listo cuando |
|---|---|---|---|---|
| 0 | Hector Partido Flores | Coordinación (Project Manager) | `main` (estructura inicial) | Todos pueden clonar e instalar |
| 1 | Anthony Oziel Ramos López | Líder de Datos y Preprocesamiento | `feat/preprocesamiento` | `preprocess.py` genera los 201 países y la partición 160/41 |
| 2 | Hector Martínez Sánchez | Líder de Modelado (grupos de países) | `feat/clustering` | Exp 1 a 3 con silueta en Train y Test, tiempos y modelos guardados |
| 3 | Ismael Alexander Hidalgo Aguilando | Líder de Modelado (lógica difusa) | `feat/difuso` | Exp 4 a 6 con cobertura, coherencia y comparación contra el experto |
| 4 | Gildardo Emmanuel González García | Líder de Pipeline e Integración | `feat/pipeline-predict` | El comando de una línea corre desde cero, `predecir()` devuelve el JSON y pasan las pruebas |
| 5 | Hector Partido Flores | Coordinación (Project Manager) | `docs/entrega3` | Hoja del experto y documento con los números reales y las evidencias |

**Calendario:**

- **Lunes 9 en la noche:** estructura inicial (PR 0).
- **Martes 10 en la mañana:** PR 1.
- **Martes 10 en la tarde:** PR 2 y PR 3 (necesitan el PR 1 ya fusionado).
- **Martes 10 en la noche:** PR 4.
- **Miércoles 11:** PR 5 y entrega.

## Cómo trabajar tu PR

```bash
git clone https://github.com/HectorPartido/ClimateAI.git
cd ClimateAI
git checkout -b feat/tu-rama             # la rama de la tabla

# ... trabajas solo en tus archivos ...

git add src/models/tu_archivo.py models/ reports/
git commit -m "feat: descripción corta de lo que hiciste"
git push -u origin feat/tu-rama
```

Después, en GitHub: **Compare & pull request**, base `main`, y llenas la plantilla del PR.

Si `main` cambió mientras trabajabas:

```bash
git checkout main
git pull
git checkout feat/tu-rama
git merge main
```

## Reglas del equipo

- Nada de rutas absolutas: importa las rutas de `src/config.py`.
- `random_state = 42` en todo lo que tenga aleatoriedad, y la misma partición de la Entrega 2 para todos los experimentos.
- Solo modifica los archivos de tu tarea. Los contratos entre archivos están escritos al inicio de cada archivo; si necesitas cambiar uno, avísale al equipo antes.
- Los datos, los modelos, los reportes y el registro sí se suben al repositorio: son parte de la entrega.
