"""ClimateAI: pipeline de preprocesamiento (PR 1).

Responsable: Anthony Oziel Ramos López (Líder de Datos y Preprocesamiento).
Rama: feat/preprocesamiento

Parte de src/datos.py de la Entrega 2 (mismas reglas y mismos resultados) y agrega
la partición Train/Test y la transformación que usan los modelos.

Lee los 3 archivos de data/raw/ y genera en data/processed/:
  - indicadores_pais.csv      6 indicadores por país (entrada del modelo)
  - particion.csv             pais, conjunto (Train o Test) y orden
  - equivalencias_paises.csv  Berkeley Earth y Our World in Data (tabla editable, RF03)
  - sin_emparejar.csv         entidades sin serie en OWID o sin datos suficientes
  - series_pais.csv           series anuales por país (gráficas de la Entrega 4)
  - temperatura_global.csv    referencia mundial para la ficha de país

Reglas (Entrega 1, Informe de diseño de la Unidad I y Entrega 2, con corte a 2012):
  * RF01: se validan las columnas y los tipos de los 3 archivos antes de usarlos.
  * RF02: se descartan meses sin temperatura y meses con incertidumbre > 1 °C;
    la media anual de un país solo vale si tiene al menos 10 meses.
  * Último año efectivo: 2012 (2013 solo tiene 9 meses).
  * I1 tendencia (°C/década): pendiente OLS de la media anual 1970-2012 x 10 (al menos 25 de 43 años).
  * I2 anomalía (°C): media 2003-2012 menos media 1951-1980 (al menos 5 de 10 y 20 de 30 años).
  * I3 variabilidad (°C): desviación estándar (ddof = 1) de los residuos de la tendencia 1970-2012.
  * I4 emisiones acumuladas (Gt CO2): suma 1850-2012.
  * I5 participación mundial (%): emisiones 2003-2012 del país / emisiones 2003-2012 de "World" x 100.
  * I6 crecimiento (%): (media 2003-2012 / media 1993-2002 - 1) x 100.
  * Emisiones: 2003-2012 completo (10 de 10 años) y 1993-2002 con al menos 8 de 10 años.
  * Partición: train_test_split(test_size=0.20, random_state=42) sobre la tabla ordenada
    por país, igual que en la Entrega 2 (160 países en Train y 41 en Test).

Uso:
    python src/data/preprocess.py --raw data/raw --output data/processed
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (ARCH_CO2, ARCH_GLOBAL, ARCH_TEMP, DATA_PROCESSED, DATA_RAW,  # noqa: E402
                        IND, IND_LOG, SEMILLA, TEST_SIZE, ULTIMO_ANIO, consola_utf8)

log = logging.getLogger("climateai")

MESES_MIN = 10
INCERT_MAX = 1.0

# Entidades de Berkeley que no son países: continentes, Antártida y las 4 versiones
# que incluyen territorios de ultramar ("Denmark" incluye Groenlandia, etc.).
EXCLUIR_BERKELEY = {
    "Africa", "Asia", "Europe", "North America", "South America", "Oceania", "Antarctica",
    "Denmark", "France", "Netherlands", "United Kingdom",
}
# Tabla por defecto: nombre en Berkeley -> nombre en Our World in Data (los demás coinciden tal cual).
EQUIVALENCIAS = {
    "Antigua And Barbuda": "Antigua and Barbuda",
    "Bonaire, Saint Eustatius And Saba": "Bonaire Sint Eustatius and Saba",
    "Bosnia And Herzegovina": "Bosnia and Herzegovina",
    "Burma": "Myanmar",
    "Congo (Democratic Republic Of The)": "Democratic Republic of Congo",
    "Curaçao": "Curacao",
    "Czech Republic": "Czechia",
    "Côte D'Ivoire": "Cote d'Ivoire",
    "Denmark (Europe)": "Denmark",
    "Federated States Of Micronesia": "Micronesia (country)",
    "France (Europe)": "France",
    "Guinea Bissau": "Guinea-Bissau",
    "Macau": "Macao",
    "Macedonia": "North Macedonia",
    "Netherlands (Europe)": "Netherlands",
    "Palestina": "Palestine",
    "Saint Kitts And Nevis": "Saint Kitts and Nevis",
    "Saint Pierre And Miquelon": "Saint Pierre and Miquelon",
    "Saint Vincent And The Grenadines": "Saint Vincent and the Grenadines",
    "Sao Tome And Principe": "Sao Tome and Principe",
    "Sint Maarten": "Sint Maarten (Dutch part)",
    "Swaziland": "Eswatini",
    "Timor Leste": "East Timor",
    "Trinidad And Tobago": "Trinidad and Tobago",
    "Turks And Caicas Islands": "Turks and Caicos Islands",
    "United Kingdom (Europe)": "United Kingdom",
}


# ---------------------------------------------------------------- RF01: validación
def validar(df: pd.DataFrame, archivo: str, columnas: dict) -> None:
    """Lanza ValueError si falta una columna o si su tipo no es el esperado."""
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"{archivo}: faltan las columnas {faltan}")
    for col, tipo in columnas.items():
        if tipo == "numero" and not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{archivo}: la columna {col} debe ser numérica")
        if tipo == "fecha" and not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise ValueError(f"{archivo}: la columna {col} debe tener fechas AAAA-MM-DD")


# ---------------------------------------------------------------- RF02: limpieza
def cargar_temperatura(raw: Path):
    t = pd.read_csv(raw / ARCH_TEMP, parse_dates=["dt"])
    validar(t, ARCH_TEMP, {"dt": "fecha", "AverageTemperature": "numero",
                           "AverageTemperatureUncertainty": "numero", "Country": "texto"})
    nombres = t["Country"].unique()
    filas = len(t)
    t = t[t["AverageTemperature"].notna() & (t["AverageTemperatureUncertainty"] <= INCERT_MAX)]
    log.info("Temperatura por país: %s filas leídas, %s descartadas (sin dato o incertidumbre > 1 °C)",
             f"{filas:,}", f"{filas - len(t):,}")
    t = t.assign(anio=t["dt"].dt.year)
    t = t[(t["anio"] >= 1850) & (t["anio"] <= ULTIMO_ANIO)]
    g = t.groupby(["Country", "anio"])["AverageTemperature"].agg(["mean", "count"]).reset_index()
    anual = g[g["count"] >= MESES_MIN].rename(columns={"mean": "temp"})[["Country", "anio", "temp"]]
    return anual, nombres


def temperatura_global(raw: Path) -> pd.DataFrame:
    g = pd.read_csv(raw / ARCH_GLOBAL, parse_dates=["dt"])
    validar(g, ARCH_GLOBAL, {"dt": "fecha", "LandAverageTemperature": "numero",
                             "LandAverageTemperatureUncertainty": "numero"})
    g = g[g["LandAverageTemperature"].notna() & (g["LandAverageTemperatureUncertainty"] <= INCERT_MAX)]
    g = g.assign(anio=g["dt"].dt.year)
    g = g[(g["anio"] >= 1850) & (g["anio"] <= ULTIMO_ANIO)]
    a = g.groupby("anio")["LandAverageTemperature"].agg(["mean", "count"])
    a = a[a["count"] >= MESES_MIN]["mean"]
    return pd.DataFrame({"anio": a.index, "temp_tierra": a.values,
                         "anomalia": a.values - a.loc[1951:1980].mean()})


def cargar_co2(raw: Path) -> pd.DataFrame:
    e = pd.read_csv(raw / ARCH_CO2)
    col = [c for c in e.columns if c.lower().startswith("annual co")]
    if not col:
        raise ValueError(f'{ARCH_CO2}: falta la columna "Annual CO₂ emissions"')
    e = e.rename(columns={col[0]: "co2_t"})
    validar(e, ARCH_CO2, {"Entity": "texto", "Code": "texto", "Year": "numero", "co2_t": "numero"})
    log.info("Emisiones de CO2: %s filas leídas", f"{len(e):,}")
    return e[["Entity", "Code", "Year", "co2_t"]]


# ---------------------------------------------------------------- RF03: emparejamiento
def tabla_editable(processed: Path) -> dict:
    """Si equivalencias_paises.csv ya existe, sus filas mandan (correcciones manuales incluidas)."""
    f = processed / "equivalencias_paises.csv"
    if not f.exists():
        return {}
    e = pd.read_csv(f)
    return dict(zip(e["pais_berkeley"], e["pais_owid"]))


def equivalencias(nombres_berkeley, co2: pd.DataFrame, editadas: dict):
    paises_owid = co2[co2["Code"].notna() & ~co2["Code"].astype(str).str.startswith("OWID")
                      & (co2["Entity"] != "Antarctica")]
    codigos = paises_owid.drop_duplicates("Entity").set_index("Entity")["Code"]
    filas, sin = [], []
    for nombre in sorted(nombres_berkeley):
        if nombre in EXCLUIR_BERKELEY:
            continue
        owid = editadas.get(nombre, EQUIVALENCIAS.get(nombre, nombre))
        if owid in codigos.index:
            filas.append({"pais_berkeley": nombre, "pais_owid": owid, "iso3": codigos[owid],
                          "metodo": "nombre igual" if owid == nombre else "tabla"})
        else:
            sin.append({"pais_berkeley": nombre, "motivo": "sin serie propia en Our World in Data"})
    return pd.DataFrame(filas), pd.DataFrame(sin)


# ---------------------------------------------------------------- indicadores
def indicadores_temperatura(serie: pd.DataFrame) -> dict:
    s = serie.set_index("anio")["temp"]
    tramo = s.loc[1970:ULTIMO_ANIO]
    base, reciente = s.loc[1951:1980], s.loc[ULTIMO_ANIO - 9:ULTIMO_ANIO]
    if len(tramo) < 25 or len(base) < 20 or len(reciente) < 5:
        return {}
    x = tramo.index.values.astype(float)
    pendiente, intercepto = np.polyfit(x, tramo.values, 1)
    residuos = tramo.values - (pendiente * x + intercepto)
    return {"I1": pendiente * 10, "I2": reciente.mean() - base.mean(), "I3": residuos.std(ddof=1)}


def indicadores_emisiones(serie: pd.Series, mundo: pd.Series) -> dict:
    reciente = serie.reindex(range(ULTIMO_ANIO - 9, ULTIMO_ANIO + 1))
    previo = serie.reindex(range(ULTIMO_ANIO - 19, ULTIMO_ANIO - 9))
    if reciente.isna().any() or previo.notna().sum() < 8 or previo.mean() <= 0:
        return {}
    return {
        "I4": serie.loc[1850:ULTIMO_ANIO].sum() / 1e9,
        "I5": reciente.sum() / mundo.loc[ULTIMO_ANIO - 9:ULTIMO_ANIO].sum() * 100,
        "I6": (reciente.mean() / previo.mean() - 1) * 100,
    }


# ---------------------------------------------------------------- partición y transformación
def particion(ind: pd.DataFrame) -> pd.DataFrame:
    """80 % Train y 20 % Test por país, con la misma semilla de la Entrega 2.

    La columna "orden" guarda el orden en que train_test_split devuelve los países.
    K-means depende del orden de las filas (su inicialización elige filas al azar),
    así que se conserva para reproducir exactamente los resultados de la Entrega 2.
    """
    train, test = train_test_split(ind, test_size=TEST_SIZE, random_state=SEMILLA)
    return pd.concat([
        pd.DataFrame({"pais": train["pais"].values, "conjunto": "Train", "orden": range(len(train))}),
        pd.DataFrame({"pais": test["pais"].values, "conjunto": "Test", "orden": range(len(test))}),
    ], ignore_index=True)


def preparar_matriz(datos) -> np.ndarray:
    """I1 a I6 como matriz numérica, con log10(x + 0.001) en I4 e I5.

    Recibe un DataFrame (varios países) o un dict (un país, desde predecir()).
    Es el primer paso del modelo; después vienen el z-score y el PCA.
    """
    df = pd.DataFrame([datos]) if isinstance(datos, dict) else datos
    x = df[IND].astype(float).copy()
    for c in IND_LOG:
        x[c] = np.log10(x[c] + 0.001)
    return x.values


def cargar(processed: Path = DATA_PROCESSED) -> pd.DataFrame:
    """Indicadores con la columna "conjunto" (lo que usan clustering.py y fuzzy.py)."""
    ind = pd.read_csv(processed / "indicadores_pais.csv")
    part = pd.read_csv(processed / "particion.csv")
    df = ind.merge(part, on="pais", how="left", validate="one_to_one")
    # primero Train y luego Test, cada uno en el orden de train_test_split
    df["_test"] = df["conjunto"] == "Test"
    return df.sort_values(["_test", "orden"]).drop(columns=["_test", "orden"]).reset_index(drop=True)


# ---------------------------------------------------------------- programa principal
def main(raw_dir: Path = DATA_RAW, out_dir: Path = DATA_PROCESSED) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    (temp, nombres), co2 = cargar_temperatura(raw_dir), cargar_co2(raw_dir)
    eq, sin = equivalencias(nombres, co2, tabla_editable(out_dir))
    log.info("Emparejamiento: %d entidades de Berkeley Earth con serie en Our World in Data", len(eq))
    mundo = co2[co2["Entity"] == "World"].set_index("Year")["co2_t"]
    filas, series = [], []
    for _, r in eq.iterrows():
        st = temp[temp["Country"] == r.pais_berkeley]
        se = co2[co2["Entity"] == r.pais_owid].set_index("Year")["co2_t"]
        it, ie = indicadores_temperatura(st), indicadores_emisiones(se, mundo)
        if it and ie:
            filas.append({"pais": r.pais_owid, "iso3": r.iso3, **it, **ie})
            s = st.set_index("anio")["temp"]
            base = s.loc[1951:1980].mean()
            anios = range(1850, ULTIMO_ANIO + 1)
            series.append(pd.DataFrame({"pais": r.pais_owid, "iso3": r.iso3, "anio": list(anios),
                                        "temp": s.reindex(anios).values,
                                        "anomalia": (s.reindex(anios) - base).values,
                                        "co2_mt": (se.reindex(anios) / 1e6).values}))
        else:
            sin = pd.concat([sin, pd.DataFrame([{"pais_berkeley": r.pais_berkeley,
                             "motivo": "datos insuficientes en las ventanas"}])], ignore_index=True)
    ind = pd.DataFrame(filas).sort_values("pais").reset_index(drop=True)
    part = particion(ind)
    eq.to_csv(out_dir / "equivalencias_paises.csv", index=False)
    sin.to_csv(out_dir / "sin_emparejar.csv", index=False)
    ind.to_csv(out_dir / "indicadores_pais.csv", index=False, float_format="%.6f")
    part.to_csv(out_dir / "particion.csv", index=False)
    pd.concat(series).to_csv(out_dir / "series_pais.csv", index=False, float_format="%.4f")
    temperatura_global(raw_dir).to_csv(out_dir / "temperatura_global.csv", index=False, float_format="%.4f")
    n_tr = int((part["conjunto"] == "Train").sum())
    log.info("Indicadores I1 a I6 para %d países | Train: %d | Test: %d (random_state=%d)",
             len(ind), n_tr, len(ind) - n_tr, SEMILLA)
    # se vuelve a leer del CSV para que todos los modelos usen exactamente los mismos valores
    return cargar(out_dir)


if __name__ == "__main__":
    consola_utf8()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Preprocesamiento de ClimateAI")
    ap.add_argument("--raw", type=Path, default=DATA_RAW)
    ap.add_argument("--output", type=Path, default=DATA_PROCESSED)
    a = ap.parse_args()
    main(a.raw, a.output)
