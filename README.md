# ClimateAI: motor de IA

Proyecto final de **Fundamentos de Inteligencia Artificial**, Entrega 3 (desarrollo, entrenamiento y experimentación del modelo). Grupo LITIID71, Equipo ClimateAI.

ClimateAI asigna a cada país una prioridad climática de **adaptación** y otra de **mitigación** (0 a 100, con cuatro niveles: Baja, Media, Alta y Crítica) a partir de 6 indicadores calculados con datos de temperatura de Berkeley Earth y de emisiones de CO₂ de Our World in Data (corte en 2012). El motor tiene dos etapas:

1. **Grupos de países (`Exp 2`):** z-score, PCA de 2 componentes y K-means con k = 4 (`Silueta Train = 0.407`, `Silueta Test = 0.382`, `Davies-Bouldin = 0.814`).
2. **Prioridad (`Exp 7`):** dos sistemas de lógica difusa Mamdani (adaptación y mitigación) calibrados exclusivamente sobre `Train` con cuantiles ponderados por masa y reglas estrictamente monótonas (**Adaptación `16/20 = 80 %`**, `F1 = 0.77`, `Kappa = 0.92`, `AUC(>=Alta) = 0.989`; **Mitigación `19/20 = 95 %`**, `F1 = 0.97`, `Kappa = 0.97`, `AUC(>=Alta) = 1.000`).

## Diagnóstico y corrección de los conflictos iniciales (Exp 1–6 → Exp 7)

Las conclusiones iniciales del equipo (ver `docs/Entrega3_ClimateAI.pdf`) dejaban abierta la brecha de desempeño frente a la meta OE3 (`>= 80 %`):

1. **Agrupamiento (`Exp 1` y `Exp 3` con silueta < 0.35):** K-means sobre los 6 indicadores (`Exp 1`) subajustó (`Silueta Train = 0.318`) porque las parejas colineales `I1–I2` (`r = 0.91`) e `I4–I5` (`r = 0.87`) duplicaban peso en distancia euclidiana 6D mientras `I6` agregaba ruido ortogonal; la mezcla gaussiana (`Exp 3`) maximizó verosimilitud generando grupos traslapados (`DB = 1.645`). El PCA de 2 componentes (`Exp 2`) resolvió esto (`Silueta Train = 0.407`, `Test = 0.382`).
2. **Mitigación (`Exp 4–6`: `4/20 = 20 %`, `F1 = 0.23`, sesgo = `+1.00`):** El equipo concluyó que *"la causa está en la escala, no en las reglas"*. Esa conclusión era incompleta porque coexistían dos fallas:
   - *Escala:* los percentiles equiponderados (`P20/P50/P80`) sobre distribuciones de cola pesada (`I4`, `I5`) equiparaban microestados con potencias emisoras (`P50` de `I5 = 0.021 %`, `P80 = 0.268 %`).
   - *Reglas:* la regla `I5 medio Y I6 positivo -> Alta` promocionaba emisores marginales (`< 0.1 %`) solo por crecer porcentualmente; `I6 positivo` saturaba desde `+40 %`; y `I5 alto Y I6 negativo -> Media` chocaba con `I4 alto Y I5 medio -> Alta` provocando 9 retrocesos de monotonía en `Exp 6`.
   - *Solución (`Exp 7`):* cuantiles ponderados por masa (`w = x^0.35` solo sobre Train: `I4 = [0.47, 3.81, 25.93] Gt`, `I5 = [0.025 %, 0.331 %, 1.512 %]`), umbrales realistas de `I6` (`[-25, 5] / [-5, 25, 65] / [30, 75]`) y reglas monótonas sin promociones de emisores marginales (`19/20 = 95 %`, `0` bajadas de nivel).
3. **Adaptación (`Exp 6`: `13/20 = 65 %`, `F1 = 0.62`):** El equipo asumió que bastaba el peso económico de casos como Bélgica, omitiendo que `I1 bajo Y I4 bajo -> Media` inflaba países de baja exposición (p. ej. Aruba), que al pasar de v1 a v2 se perdió la interacción `I3 alto Y I4 bajo` (p. ej. Namibia) y que `I1 P80` saturaba demasiado pronto mientras `I4` equiponderado no distinguía capacidad industrial real. En `Exp 7` se calibró `I1` con `[P30, P50, P85]`, `I3` con `[P25, P50, P80]`, `I4` con `w = x^0.20` (`[0.125, 1.908, 12.317] Gt`) y reglas 100 % monótonas (`0/5,043` retrocesos, `16/20 = 80 %`).
4. **Latencia de arranque en `predecir()`:** `fuzzy.py` importaba `matplotlib.pyplot` al cargar el módulo, disparando reconstrucción de caché de fuentes en entornos restringidos (`6.97 s > 3 s` en la primera llamada de `pytest`). Ahora `matplotlib` se carga de forma perezosa solo al generar gráficas y `_artefactos()` pre-calcula las mallas difusas (`~1.2 ms` en inferencia; `< 250 ms` en arranque frío).

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
python -m pytest -q                # 27 pruebas del contrato, monotonía y meta OE3
```

`train.py` corre todo desde los CSV originales en ~26 segundos: indicadores, partición, los 7 experimentos, los modelos, las figuras, la matriz de experimentación y el registro.

## Experimentos

Todos usan la partición de la Entrega 2 (`train_test_split`, 20 % de prueba, `random_state = 42`): 160 países en Train y 41 en Test.

| Variante | Modelo | Métrica principal (Test) |
|---|---|---|
| Exp 1: Baseline | K-means sobre los 6 indicadores | Silueta: 0.368 (Train 0.318 < 0.35) |
| Exp 2: Candidato | PCA de 2 componentes + K-means | Silueta: 0.382 (Train 0.407, DB 0.814) |
| Exp 3: Alternativo | PCA + mezcla gaussiana | Silueta: 0.328 (DB 1.645) |
| Exp 4: Baseline | Regla de cuartiles de un solo indicador | Ada. 9/20 (45 %), Mit. 4/20 (20 %) |
| Exp 5: Alternativo | Mamdani v1 original (15 reglas, equiponderado) | Ada. 12/20 (60 %), Mit. 4/20 (20 %) |
| Exp 6: Anterior del equipo | Mamdani v2 original (12 reglas, equiponderado) | Ada. 13/20 (65 %), Mit. 4/20 (20 %) |
| Exp 7: Candidato calibrado | Mamdani v3 (cuantiles por masa en Train + reglas monótonas) | **Ada. 16/20 (80 %), Mit. 19/20 (95 %)** |

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
