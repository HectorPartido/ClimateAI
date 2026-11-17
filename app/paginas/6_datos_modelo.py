"""Pantalla 06: Datos y modelo (PR 4).

Responsable: Anthony Oziel Ramos López.
Rama: feat/datos-modelo-pruebas
Requerimientos: RF-09, RF-15 y RF-17 (consulta de reportes) y la sección de validación con el docente.

"Ejecutar análisis" corre src/models/train.py en un proceso aparte (el mismo comando de la
terminal), espera a que termine y limpia la caché para que todas las pantallas lean los
archivos nuevos.
"""
import subprocess
import sys

import pandas as pd
import streamlit as st

from comun import (INDICADOR, RAIZ, REPORTES, experto, limpiar_cache, paises, parametros_difusos, resumen,
                   tabla_descarga, ultima_ejecucion)
from src.config import (CORTES, DATA_PROCESSED, DATA_RAW, EXPERIMENTOS, FIGURAS, IND, K, META_CONCORDANCIA,
                        META_SILUETA_TEST, META_SILUETA_TRAIN, SEMILLA, SISTEMAS, TEST_SIZE, UMBRAL_NO_CONCLUYENTE)
from src.models.fuzzy import linea_base

r = resumen()
st.title("Datos y modelo")
st.caption("De dónde salen los datos, qué pasos sigue el modelo, qué tan bien funciona y cómo volver a generarlo todo.")

# ---- RF-15: regenerar todo
c1, c2 = st.columns([3, 2], vertical_alignment="center")
c1.markdown(f"Última ejecución del análisis: **{ultima_ejecucion()}** "
            f"({r['paises']} países, {r['train']} de entrenamiento y {r['test']} de prueba, "
            f"{r['segundos_totales']} s).")
if c2.button("Ejecutar análisis", type="primary", icon=":material/refresh:", width="stretch",
             help="Vuelve a calcular indicadores, grupos y prioridades desde los datos originales (tarda menos de un minuto)."):
    with st.spinner("Ejecutando el análisis completo (python src/models/train.py)..."):
        try:
            p = subprocess.run([sys.executable, "src/models/train.py"], cwd=RAIZ, capture_output=True, text=True,
                               timeout=600, encoding="utf-8", errors="replace")
            ok, salida = p.returncode == 0, (p.stdout + p.stderr)
        except subprocess.TimeoutExpired:
            ok, salida = False, "Se agotó el tiempo de espera (10 minutos)."
    if ok:
        limpiar_cache()          # todas las pantallas leen los archivos nuevos
    st.session_state.dm_resultado = (ok, "\n".join(salida.strip().splitlines()[-12:]) or "(sin salida)")
    st.rerun()

if "dm_resultado" in st.session_state:
    ok, cola = st.session_state.dm_resultado
    if ok:
        st.success("Análisis terminado: todas las pantallas ya usan los resultados nuevos.", icon=":material/check_circle:")
    else:
        st.error("El análisis falló; se conservan los resultados anteriores.", icon=":material/error:")
    with st.expander("Últimas líneas del registro", expanded=not ok):
        st.code(cola, language="text")

t_fuentes, t_pasos, t_calidad, t_validacion, t_exp, t_sin, t_rep = st.tabs(
    ["Fuentes", "Pasos y parámetros", "Calidad del modelo", "Validación con el docente", "Experimentos",
     "Países sin emparejar", "Reportes de error"])

# ---- fuentes
with t_fuentes:
    fuentes = pd.DataFrame([
        {"Archivo": "GlobalLandTemperaturesByCountry.csv", "Contenido": "Temperatura mensual por país, 1743 a 2013",
         "Fuente": "Berkeley Earth (vía Kaggle)", "Licencia": "CC BY-NC-SA 4.0"},
        {"Archivo": "GlobalTemperatures.csv", "Contenido": "Temperatura mensual del planeta",
         "Fuente": "Berkeley Earth (vía Kaggle)", "Licencia": "CC BY-NC-SA 4.0"},
        {"Archivo": "annual-co2-emissions-per-country.csv", "Contenido": "Emisiones anuales de CO₂ por país",
         "Fuente": "Our World in Data y Global Carbon Project", "Licencia": "CC BY 4.0"},
    ])
    fuentes["Tamaño"] = [f"{(DATA_RAW / a).stat().st_size / 1e6:.1f} MB" if (DATA_RAW / a).exists() else "no está"
                         for a in fuentes["Archivo"]]
    st.dataframe(fuentes, hide_index=True)
    st.markdown("**Indicadores que se calculan por país** (periodo histórico 1850 a 2012)")
    st.dataframe(pd.DataFrame([{"Clave": c, "Indicador": INDICADOR[c][0], "Unidad": INDICADOR[c][1],
                                "Cómo se calcula": INDICADOR[c][2]} for c in IND]), hide_index=True)
    tabla_descarga(paises()[["pais", "pais_es", "iso3", *IND]], "climateai_indicadores.csv",
                   "Descargar indicadores", "dm_ind")

# ---- pasos y parámetros
with t_pasos:
    pca = r["grupos"]["pca_varianza"]
    n_reglas = {s: len(v) for s, v in parametros_difusos()["reglas"].items()}
    st.markdown(
        f"1. **Emparejar países** entre las dos fuentes por nombre y con una tabla de equivalencias.\n"
        f"2. **Calcular los 6 indicadores** de cada país; quedan {r['paises']} países completos.\n"
        f"3. **Separar** {r['train']} países para calibrar (Train) y {r['test']} para probar (Test), "
        f"con semilla {SEMILLA} ({TEST_SIZE:.0%} de prueba).\n"
        f"4. **Agrupar**: log10 en I4 e I5, z-score con la media de Train, PCA de 2 componentes "
        f"({100 * sum(pca):.1f} % de la varianza) y K-means con k = {K}.\n"
        f"5. **Prioridad** con lógica difusa Mamdani: percentiles de Train como cortes, "
        f"{n_reglas['adaptacion']} reglas de adaptación y {n_reglas['mitigacion']} de mitigación, centroide como puntaje.\n"
        f"6. **Nivel**: Baja < {CORTES[0]} ≤ Media < {CORTES[1]} ≤ Alta < {CORTES[2]} ≤ Crítica. Si ninguna regla "
        f"llega a {UMBRAL_NO_CONCLUYENTE}, el resultado se marca como no concluyente.")
    st.markdown("**Cortes de las funciones de pertenencia** (calculados solo con Train)")
    params = parametros_difusos()
    nombre_sis = {"adaptacion": "Adaptación", "mitigacion": "Mitigación"}
    filas = []
    for sis, ind in params["percentiles_por_sistema"].items():
        for c, cortes in ind.items():
            filas.append({"Sistema": nombre_sis[sis], "Indicador": f"{c}. {INDICADOR[c][0]}",
                          **{f"Corte {i}": f"{k} = {v:,.4g}" for i, (k, v) in enumerate(cortes.items(), 1)}})
    filas.append({"Sistema": "Mitigación", "Indicador": "I6. Crecimiento de emisiones",
                  **{f"Corte {i}": f"{t}: {pts}" for i, (t, pts) in enumerate(params["i6_puntos_fijos"].items(), 1)}})
    st.dataframe(pd.DataFrame(filas).fillna(""), hide_index=True)
    st.caption("I4 e I5 usan cuantiles ponderados por el tamaño de las emisiones, para que los emisores muy pequeños "
               "no cuenten como medios o altos. I6 usa puntos fijos en %.")
    with st.expander("Reglas de los dos sistemas", icon=":material/rule:"):
        for sis, reglas in params["reglas"].items():
            st.markdown(f"**{nombre_sis[sis]}** ({len(reglas)} reglas)")
            st.markdown("\n".join(f"{i}. {rg['texto']}" for i, rg in enumerate(reglas, 1)))

# ---- calidad
with t_calidad:
    exp = {e["experimento"].split(":")[0]: e for e in r["experimentos"]}
    m7 = r["difuso"]["metricas_experto"]["Exp 7"]
    st.markdown("**Agrupamiento (Exp 2: PCA + K-means)**")
    a, b, c, d_ = st.columns(4)
    a.metric("Silueta en Train", f"{exp['Exp 2']['silueta_train']:.3f}", f"meta {META_SILUETA_TRAIN}",
             delta_arrow="off", delta_color="off", border=True)
    b.metric("Silueta en Test", f"{exp['Exp 2']['silueta_test']:.3f}", f"meta {META_SILUETA_TEST}",
             delta_arrow="off", delta_color="off", border=True)
    c.metric("Davies-Bouldin en Test", f"{exp['Exp 2']['davies_bouldin_test']:.3f}", "menor es mejor",
             delta_arrow="off", delta_color="off", border=True)
    d_.metric("Estabilidad (ARI)", f"{exp['Exp 2']['ari_estabilidad']:.3f}", "1 es idéntico",
              delta_arrow="off", delta_color="off", border=True)
    st.markdown("**Prioridad (Exp 7: Mamdani calibrado) frente a los niveles del docente en 20 países de Test**")
    cols = st.columns(4)
    for i, s in enumerate(SISTEMAS):
        m = m7[s]
        cols[2 * i].metric(f"Coincidencia en {s.replace('cion', 'ción')}", f"{m['aciertos']}/{m['n']} ({m['concordancia']:.0%})",
                           f"meta {META_CONCORDANCIA:.0%}", delta_arrow="off", delta_color="off", border=True)
        cols[2 * i + 1].metric(f"F1 / Kappa / AUC ({s[:3]}.)",
                               f"{m['f1_macro']:.2f} / {m['kappa_cuadratica']:.2f} / {m['auc_roc_alta']:.3f}",
                               f"a un nivel o menos: {m['concordancia_pm1']:.0%}", delta_arrow="off",
                               delta_color="off", border=True)
    pr = r["predecir"]
    st.markdown("**Respuesta de predecir()**")
    a, b, c, d_ = st.columns(4)
    a.metric("Llamadas de prueba", pr["llamadas"], border=True)
    b.metric("Tiempo promedio", f"{pr['tiempo_ms_promedio']:.2f} ms", "meta 1,000 ms", delta_arrow="off",
             delta_color="off", border=True)
    c.metric("Tiempo máximo", f"{pr['tiempo_ms_maximo']:.2f} ms", border=True)
    d_.metric("No concluyentes", pr["no_concluyentes"], border=True)
    figs = [("matriz_experto_exp7.png", "Exp 7 frente al docente"), ("silueta_por_k.png", "Silueta por k"),
            ("grupos_pca.png", "Grupos en el plano PCA"), ("funciones_pertenencia.png", "Funciones de pertenencia")]
    with st.expander("Figuras generadas por el entrenamiento", icon=":material/image:"):
        cols = st.columns(2)
        for i, (arch, titulo) in enumerate(figs):
            if (FIGURAS / arch).exists():
                cols[i % 2].image(str(FIGURAS / arch), caption=titulo)

# ---- validación con el docente (20 países de Test)
with t_validacion:
    ex = experto()
    if ex.empty:
        st.info("Todavía no hay niveles del docente en data/expert/niveles_experto.csv.")
    else:
        d = paises().set_index("pais")
        params = parametros_difusos()
        filas = []
        for _, e in ex.iterrows():
            v = d.loc[e["pais"], IND].astype(float).to_dict()
            fila = {"País": e.get("pais_es") or d.loc[e["pais"], "pais_es"]}
            for s, corto in (("adaptacion", "Ad."), ("mitigacion", "Mit.")):
                fila[f"Docente {corto}"] = e[f"nivel_{s}"]
                fila[f"Modelo {corto}"] = d.loc[e["pais"], f"nivel_{s}"]
                fila[f"Cuartiles {corto}"] = linea_base(v, s, params)
                fila[f"Coincide {corto}"] = fila[f"Modelo {corto}"] == fila[f"Docente {corto}"]
            filas.append(fila)
        val = pd.DataFrame(filas)
        a, b = st.columns(2)
        for col, corto, s in ((a, "Ad.", "adaptación"), (b, "Mit.", "mitigación")):
            mod = val[f"Coincide {corto}"].mean()
            cua = (val[f"Cuartiles {corto}"] == val[f"Docente {corto}"]).mean()
            col.metric(f"Coincidencia en {s}", f"{val[f'Coincide {corto}'].sum()}/{len(val)} ({mod:.0%})",
                       f"{(mod - META_CONCORDANCIA) * 100:+.0f} puntos frente a la meta del 80 %",
                       delta_color="normal" if mod >= META_CONCORDANCIA else "inverse", border=True)
            col.caption(f"La regla simple de cuartiles coincide en {cua:.0%}.")
        st.dataframe(val, hide_index=True, column_config={
            "Coincide Ad.": st.column_config.CheckboxColumn("✓ Ad."),
            "Coincide Mit.": st.column_config.CheckboxColumn("✓ Mit.")})
        st.caption("Docente: niveles asignados por el profesor sin ver el modelo. Cuartiles: línea base de un solo "
                   "indicador (Exp 4).")
        tabla_descarga(val, "climateai_validacion_docente.csv", "Exportar validación", "dm_val")

# ---- experimentos
with t_exp:
    if EXPERIMENTOS.exists():
        m = pd.read_csv(EXPERIMENTOS)
        st.dataframe(m, hide_index=True)
        tabla_descarga(m, "experimentos.csv", "Descargar matriz de experimentos", "dm_exp")
    else:
        st.info("Sin matriz de experimentos: presiona Ejecutar análisis.")

# ---- países sin emparejar (RF-15)
with t_sin:
    sin = pd.read_csv(DATA_PROCESSED / "sin_emparejar.csv")
    eq = pd.read_csv(DATA_PROCESSED / "equivalencias_paises.csv")
    st.markdown(f"{len(eq)} países se emparejaron ({(eq['metodo'] == 'nombre igual').sum()} por nombre igual y "
                f"{(eq['metodo'] != 'nombre igual').sum()} con la tabla de equivalencias); "
                f"{len(sin)} no tienen pareja en la fuente de emisiones.")
    st.dataframe(sin.rename(columns={"pais_berkeley": "País (Berkeley Earth)", "motivo": "Motivo"}), hide_index=True)
    tabla_descarga(sin, "sin_emparejar.csv", "Descargar sin_emparejar.csv", "dm_sin")

# ---- reportes de error (RF-17)
with t_rep:
    if REPORTES.exists():
        rep = pd.read_csv(REPORTES)
        st.dataframe(rep.iloc[::-1], hide_index=True)
        tabla_descarga(rep, "reportes.csv", "Descargar reportes", "dm_rep")
    else:
        st.info("Todavía no hay reportes. Se envían desde el botón Reportar un error de la barra lateral.")
