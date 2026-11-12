"""ClimateAI: hoja para el experto y su importación (PR 5).

Responsable: Hector Partido Flores (Coordinación, Project Manager).
Rama: docs/entrega3

El experto (docente de Fundamentos de IA) asigna a 20 países de Test un nivel de prioridad
de adaptación y otro de mitigación, a ciegas: solo ve los indicadores en lenguaje sencillo y
en qué lugar quedan frente a los 201 países; no ve las reglas ni los resultados del modelo.

    python src/data/hoja_experto.py crear                      # genera data/expert/hoja_experto_ClimateAI.xlsx
    python src/data/hoja_experto.py importar RUTA_DEL_EXCEL    # llena data/expert/niveles_experto.csv

Después de importar, volver a correr train.py: fuzzy.py calcula la concordancia, el F1 macro
y las matrices de confusión contra el experto.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_EXPERT, INDICADORES, NIVELES, NIVELES_EXPERTO, consola_utf8  # noqa: E402

HOJA = DATA_EXPERT / "hoja_experto_ClimateAI.xlsx"
FILA_ENCABEZADO, EJEMPLO, PRIMERA = 6, 7, 8   # encabezado, fila de ejemplo y primer país
COL_ADA, COL_MIT, COL_ID = 13, 14, 16    # M, N y P (oculta, nombre del país en los datos)

INDICADORES_HOJA = [  # (columna de datos, encabezado, formato)
    ("I1", "Calentamiento\n(°C por década, 1970 a 2012)", "0.00"),
    ("I3", "Variabilidad de un año a otro\n(°C, 1970 a 2012)", "0.00"),
    ("I4", "Emisiones acumuladas\n(Gt de CO₂, 1850 a 2012)", "#,##0.000"),
    ("I5", "Parte de las emisiones mundiales\n(%, 2003 a 2012)", "0.000"),
    ("I6", "Crecimiento de sus emisiones\n(%, 2003 a 2012 frente a 1993 a 2002)", "+0;-0;0"),
]

FUENTE = "Arial"
AZUL = PatternFill("solid", start_color="1F4E79")
GRIS = PatternFill("solid", start_color="F2F2F2")
AMARILLO = PatternFill("solid", start_color="FFF2CC")
BORDE = Border(*(Side(style="thin", color="BFBFBF"),) * 4)

CAJA = PatternFill("solid", start_color="EAF1F8")

# Hoja "Cómo llenarla": explicación en lenguaje sencillo, con un ejemplo leído en voz alta.
# Tipos: titulo, subtitulo, texto, paso (numerado), lista (viñeta) y caja (recuadro del ejemplo).
INSTRUCCIONES = [
    ("titulo", "ClimateAI: cómo llenar esta hoja"),
    ("texto", "Profesor, gracias por apoyarnos. Le toma unos 15 minutos."),
    ("subtitulo", "Para qué lo necesitamos"),
    ("texto", "Nuestro sistema le da a cada país una prioridad de adaptación y otra de mitigación, en cuatro niveles: "
              "Baja, Media, Alta y Crítica. No existe una respuesta oficial con qué comparar, así que no tenemos cómo "
              "medir si el sistema acierta. Por eso, desde la Entrega 2, definimos que lo validaríamos con el criterio "
              "de un experto: usted."),
    ("texto", "Con sus respuestas medimos el objetivo específico 3 y el criterio de aceptación 5 del proyecto: que el "
              "sistema coincida con el experto en al menos 16 de los 20 países (80 %) en cada tipo de prioridad. Es "
              "también la métrica principal de la Entrega 3 (concordancia, F1 y matriz de confusión)."),
    ("subtitulo", "Qué tiene que hacer"),
    ("paso", "Abra la hoja \"Países\". Ahí están los 20 países, que el sistema nunca vio al entrenar."),
    ("paso", "Lea los 5 datos de cada país. Junto a cada dato dice qué tan alto es frente a los 201 países analizados."),
    ("paso", "En las columnas amarillas M y N, haga clic y elija de la lista el nivel de adaptación y el de "
             "mitigación. El comentario de la columna O es opcional."),
    ("paso", "Guarde el archivo y regréselo por correo. Arriba de la tabla aparece cuántos países lleva completos."),
    ("subtitulo", "Un ejemplo"),
    ("caja", "En la fila gris de la hoja \"Países\" viene un país de ejemplo. Su fila se lee así:"),
    ("caja", "     Se calienta 0.25 °C por década: más que el 60 % de los países."),
    ("caja", "     Su temperatura varía 0.31 °C de un año a otro: más que el 40 %."),
    ("caja", "     Ha emitido 2.5 Gt de CO₂ en su historia: más que el 75 %."),
    ("caja", "     Aporta el 0.10 % de las emisiones mundiales: más que el 70 %."),
    ("caja", "     Sus emisiones crecieron 18 % entre 1993 a 2002 y 2003 a 2012: más que el 55 %."),
    ("caja", "Con esos datos usted decide, según su criterio, qué tan urgente es que ese país se prepare para el "
             "calentamiento (adaptación) y qué tan urgente es que reduzca sus emisiones (mitigación). En el ejemplo "
             "aparece Alta y Media solo para mostrar el formato."),
    ("subtitulo", "Tres cosas importantes"),
    ("lista", "Es a ciegas: no le mostramos nuestras reglas ni nuestros resultados, para no influir en su criterio."),
    ("lista", "Base su decisión solo en los datos de la tabla, no en lo que sepa de cada país; así la comparación es justa."),
    ("lista", "No hay respuestas correctas ni cuotas: varios países pueden tener el mismo nivel."),
    ("subtitulo", "Qué significa cada prioridad"),
    ("lista", "Adaptación: qué tan urgente es que el país se prepare para los efectos del calentamiento. Pesa más "
              "si se calienta rápido, si su temperatura es muy variable y si ha emitido poco en su historia "
              "(señal de menos recursos industriales para adaptarse)."),
    ("lista", "Mitigación: qué tan urgente es que el país reduzca sus emisiones. Pesa más si ha emitido mucho en su "
              "historia, si hoy aporta una parte grande de las emisiones mundiales y si sus emisiones siguen creciendo."),
    ("subtitulo", "Qué significa cada nivel"),
    ("lista", "Baja: no es prioridad frente a la mayoría de los países."),
    ("lista", "Media: requiere seguimiento, sin urgencia."),
    ("lista", "Alta: debe atenderse antes que la mayoría de los países."),
    ("lista", "Crítica: está entre los casos más urgentes."),
    ("subtitulo", "Qué significa cada dato"),
    ("texto", "\"Más alto que el 93 %\" quiere decir que solo el 7 % de los 201 países tiene un valor mayor."),
    ("lista", "Calentamiento: cuánto sube la temperatura del país por década (tendencia de 1970 a 2012)."),
    ("lista", "Variabilidad: cuánto se aleja su temperatura de la tendencia de un año a otro (1970 a 2012)."),
    ("lista", "Emisiones acumuladas: todo el CO₂ que ha emitido de 1850 a 2012, en miles de millones de toneladas (Gt)."),
    ("lista", "Parte de las emisiones mundiales: su porcentaje de todo el CO₂ emitido en el mundo de 2003 a 2012."),
    ("lista", "Crecimiento de sus emisiones: cuánto cambiaron sus emisiones de 2003 a 2012 frente a 1993 a 2002."),
    ("texto", "Datos: Berkeley Earth (temperatura) y Our World in Data (emisiones de CO₂), con corte en 2012."),
]


def _tabla_paises() -> pd.DataFrame:
    ind = pd.read_csv(INDICADORES)
    expertos = pd.read_csv(NIVELES_EXPERTO, dtype=str)
    t = expertos[["pais", "pais_es"]].merge(ind, on="pais", how="left", validate="one_to_one")
    if t["I1"].isna().any():
        raise ValueError(f"Países sin indicadores: {t.loc[t['I1'].isna(), 'pais'].tolist()}")
    for c, _, _ in INDICADORES_HOJA:
        ref = np.sort(ind[c].values)
        t[f"{c}_lugar"] = [np.searchsorted(ref, v, side="left") / len(ref) for v in t[c]]
    return t.sort_values("pais_es").reset_index(drop=True)


def _celda(ws, fila, col, valor, negrita=False, tam=10, color="000000", relleno=None, formato=None,
           alinear="left", ajustar=True, cursiva=False):
    c = ws.cell(fila, col, valor)
    c.font = Font(name=FUENTE, size=tam, bold=negrita, italic=cursiva, color=color)
    c.alignment = Alignment(horizontal=alinear, vertical="center", wrap_text=ajustar)
    if relleno:
        c.fill = relleno
    if formato:
        c.number_format = formato
    return c


def crear(salida: Path = HOJA) -> Path:
    t = _tabla_paises()
    wb = Workbook()

    # ---------------------------------------------------------------- Instrucciones
    wi = wb.active
    wi.title = "Cómo llenarla"
    wi.sheet_view.showGridLines = False
    wi.column_dimensions["A"].width = 3
    wi.column_dimensions["B"].width = 100
    fila, paso = 2, 0
    for tipo, texto in INSTRUCCIONES:
        if tipo == "titulo":
            _celda(wi, fila, 2, texto, negrita=True, tam=16, color="1F4E79")
            wi.row_dimensions[fila].height = 26
        elif tipo == "subtitulo":
            fila += 1
            _celda(wi, fila, 2, texto, negrita=True, tam=12, color="1F4E79")
        else:
            if tipo == "paso":
                paso += 1
                texto = f"{paso}.  {texto}"
            elif tipo == "lista":
                texto = f"•  {texto}"
            _celda(wi, fila, 2, texto, tam=11, relleno=CAJA if tipo == "caja" else None)
            wi.row_dimensions[fila].height = 15 * max(1, -(-len(texto) // 95)) + (3 if tipo == "caja" else 0)
        fila += 1

    # ---------------------------------------------------------------- Países
    ws = wb.create_sheet("Países")
    ws.sheet_view.showGridLines = False
    _celda(ws, 1, 1, "ClimateAI: niveles de prioridad del experto (20 países)", negrita=True, tam=14, color="1F4E79",
           ajustar=False)
    _celda(ws, 2, 1, "Llene solo las celdas amarillas (columnas M y N): haga clic en la celda y elija Baja, Media, Alta "
                     "o Crítica. El comentario de la columna O es opcional. La fila gris es solo un ejemplo "
                     "(se explica en la hoja \"Cómo llenarla\").", tam=10, ajustar=False)
    _celda(ws, 3, 1, "Países completos:", negrita=True, ajustar=False)
    _celda(ws, 3, 3, f'=COUNTIFS(M{PRIMERA}:M{PRIMERA + 19},"<>",N{PRIMERA}:N{PRIMERA + 19},"<>")&" de 20"',
           negrita=True, ajustar=False)

    grupos = [(3, 6, "Exposición al calentamiento"), (7, 8, "Responsabilidad histórica"),
              (9, 12, "Emisiones actuales"), (13, 15, "Su evaluación")]
    for c1, c2, texto in grupos:
        ws.merge_cells(start_row=FILA_ENCABEZADO - 1, start_column=c1, end_row=FILA_ENCABEZADO - 1, end_column=c2)
        _celda(ws, FILA_ENCABEZADO - 1, c1, texto, negrita=True, color="FFFFFF", relleno=AZUL, alinear="center")
    encabezados = ["N.º", "País"]
    for _, nombre, _ in INDICADORES_HOJA:
        encabezados += [nombre, "Más alto que\n(% de países)"]
    encabezados += ["Prioridad de\nadaptación", "Prioridad de\nmitigación", "Comentario (opcional)", "pais_datos"]
    for j, texto in enumerate(encabezados, start=1):
        c = _celda(ws, FILA_ENCABEZADO, j, texto, negrita=True, color="FFFFFF", relleno=AZUL, alinear="center")
        c.border = BORDE
    ws.row_dimensions[FILA_ENCABEZADO].height = 80

    # fila de ejemplo (no se importa): muestra el formato esperado
    ejemplo = ["Ej.", "País de ejemplo (no contestar)", 0.25, 0.60, 0.31, 0.40, 2.5, 0.75, 0.10, 0.70, 18, 0.55,
               "Alta", "Media", "Ejemplo del formato", ""]
    for j, v in enumerate(ejemplo, start=1):
        fmt = None
        if 3 <= j <= 12:
            fmt = "0%" if j % 2 == 0 else INDICADORES_HOJA[(j - 3) // 2][2]
        c = _celda(ws, EJEMPLO, j, v, cursiva=True, color="7F7F7F", formato=fmt,
                   alinear="left" if j in (2, 15) else "center")
        c.border = BORDE

    for i, r in t.iterrows():
        f = PRIMERA + i
        relleno = GRIS if i % 2 else None
        _celda(ws, f, 1, i + 1, relleno=relleno, alinear="center")
        _celda(ws, f, 2, r["pais_es"], negrita=True, relleno=relleno)
        for k, (c, _, fmt) in enumerate(INDICADORES_HOJA):
            _celda(ws, f, 3 + 2 * k, float(r[c]), formato=fmt, relleno=relleno, alinear="center")
            _celda(ws, f, 4 + 2 * k, float(r[f"{c}_lugar"]), formato="0%", relleno=relleno, alinear="center",
                   color="595959")
        for col in (COL_ADA, COL_MIT, COL_MIT + 1):
            _celda(ws, f, col, None, relleno=AMARILLO, alinear="left" if col == COL_MIT + 1 else "center")
        _celda(ws, f, COL_ID, r["pais"], color="7F7F7F")
        for col in range(1, COL_MIT + 2):
            ws.cell(f, col).border = BORDE
        ws.row_dimensions[f].height = 20

    dv = DataValidation(type="list", formula1='"' + ",".join(NIVELES) + '"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Nivel no válido",
                        error="Elija Baja, Media, Alta o Crítica.", showInputMessage=True, promptTitle="Elija un nivel",
                        prompt="Baja: no es prioridad frente a la mayoría. Media: requiere seguimiento. "
                               "Alta: antes que la mayoría. Crítica: entre los más urgentes.")
    ws.add_data_validation(dv)
    dv.add(f"M{PRIMERA}:N{PRIMERA + 19}")

    anchos = [5, 22] + [14, 10] * 4 + [17, 10] + [12, 12, 30, 18]
    for j, w in enumerate(anchos, start=1):
        ws.column_dimensions[ws.cell(1, j).column_letter].width = w
    ws.column_dimensions["P"].hidden = True
    ws.freeze_panes = ws.cell(PRIMERA, 3)
    for hoja in (wi, ws):
        hoja.page_setup.orientation = "landscape"
        hoja.page_setup.fitToWidth = 1
        hoja.page_setup.fitToHeight = 0
        hoja.sheet_properties.pageSetUpPr.fitToPage = True
    wb.active = 0   # abre en "Cómo llenarla"

    salida.parent.mkdir(parents=True, exist_ok=True)
    wb.save(salida)
    return salida


def importar(ruta_excel: Path, salida: Path = NIVELES_EXPERTO) -> pd.DataFrame:
    """Lee las respuestas del experto y actualiza data/expert/niveles_experto.csv."""
    ws = load_workbook(ruta_excel, data_only=True)["Países"]
    filas = []
    for f in range(PRIMERA, PRIMERA + 20):
        pais = ws.cell(f, COL_ID).value
        if not pais:
            continue
        niveles = [str(ws.cell(f, c).value or "").strip().capitalize().replace("Critica", "Crítica")
                   for c in (COL_ADA, COL_MIT)]
        for n in niveles:
            if n and n not in NIVELES:
                raise ValueError(f"Fila {f} ({pais}): nivel no válido {n!r}")
        filas.append({"pais": pais, "nivel_adaptacion": niveles[0], "nivel_mitigacion": niveles[1],
                      "comentario": ws.cell(f, COL_MIT + 1).value or ""})
    resp = pd.DataFrame(filas)
    base = pd.read_csv(salida, dtype=str).fillna("")
    base = base.drop(columns=[c for c in ("nivel_adaptacion", "nivel_mitigacion", "comentario") if c in base])
    out = base.merge(resp, on="pais", how="left").fillna("")
    out.to_csv(salida, index=False, encoding="utf-8")
    completos = ((out["nivel_adaptacion"] != "") & (out["nivel_mitigacion"] != "")).sum()
    print(f"{completos} de {len(out)} países con los 2 niveles; guardado en {salida}")
    return out


if __name__ == "__main__":
    consola_utf8()
    ap = argparse.ArgumentParser(description="Hoja del experto de ClimateAI")
    sub = ap.add_subparsers(dest="accion", required=True)
    sub.add_parser("crear")
    imp = sub.add_parser("importar")
    imp.add_argument("excel", type=Path)
    a = ap.parse_args()
    if a.accion == "crear":
        print(f"Hoja creada: {crear()}")
    else:
        importar(a.excel)
