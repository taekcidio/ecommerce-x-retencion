"""💬 Motor determinista de consultas en lenguaje natural (español).

Dos pasos separados:
  interpretar(pregunta, df) → Interpretacion | Respuesta(aclaracion/no_soportada)
  ejecutar(df, interpretacion) → Respuesta con cálculo, columnas y registros de respaldo
Las columnas se resuelven por nombre, sinónimos y valores presentes: nada está fijado a un dataset.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ingesta.tipos import normalize_name

from .kpis import MUESTRA_MINIMA
from .perfil import FRECUENCIAS, diferencias_por_grupo, frecuencia_auto, perfil, serie_temporal

# concepto -> (raíces que lo activan en la pregunta, fragmentos a buscar en columnas/valores, en orden de prioridad)
CONCEPTOS = {
    "abandono": (("abandon", "churn", "baja", "fuga", "desercion", "perdid"), ("churn", "abandon", "baja", "fuga")),
    "retencion": (("retenc", "retenid", "retien", "permanen", "fideliz"), ("churn", "retenc", "retenid", "abandon")),
    "ingresos": (("ingreso", "venta", "factur", "revenue", "sales"), ("ingreso", "venta", "factur", "revenue", "sale", "monto", "importe", "amount")),
    "devoluciones": (("devol", "devuel", "reembols", "return"), ("devol", "devuel", "return", "reembols")),
    "cancelaciones": (("cancel",), ("cancel",)),
}
STOP = set("""que cual cuales como cuanto cuantos cuanta cuantas donde quien el la los las un una unos unas de del
al a en y o por para con sin se su sus es son fue fueron presenta presentan tiene tienen hay mayor menor mas menos
tasa porcentaje proporcion nivel cantidad numero top principal principales durante periodo entre existen existe
diferencias diferencia evoluciono evolucionaron evolucion tendencia ha han muestra muestran registra registran
alta baja alto bajo peor mejor segun cada total suma promedio media mediana no si lo le valor valores datos""".split())
_FREQ_WORDS = {"dia": "D", "diario": "D", "semana": "W", "semanal": "W", "mes": "M", "mensual": "M",
               "trimestre": "Q", "trimestral": "Q", "ano": "Y", "anual": "Y"}
EJEMPLOS = ["¿Qué segmento presenta la mayor tasa de abandono?",
            "¿Cómo evolucionaron los ingresos durante el periodo?",
            "¿Qué categorías presentan más devoluciones?",
            "¿Qué diferencias existen entre clientes retenidos y no retenidos?",
            "¿Cuál es el promedio de monto_neto por ciudad?"]


@dataclass
class Interpretacion:
    intencion: str
    grupo: str | None = None
    evento: dict | None = None  # {"tipo": booleano|valor|suma, "columna", "valores"}
    medida: str | None = None
    agregacion: str = "suma"
    fecha: str | None = None
    frecuencia: str | None = None
    direccion: str = "mayor"
    criterio: str = "tasa"  # tasa | conteo
    notas: list[str] = field(default_factory=list)


@dataclass
class Respuesta:
    estado: str  # ok | aclaracion | no_soportada
    texto: str
    interpretacion: dict | None = None
    columnas: list[str] = field(default_factory=list)
    calculo: str = ""
    tabla: pd.DataFrame | None = None
    registros: pd.DataFrame | None = None
    grafico: dict | None = None
    opciones: dict[str, list[str]] = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)


# ---------------- utilidades de lenguaje ----------------
def _palabras(q: str) -> list[str]:
    return [w for w in normalize_name(q).split("_") if w]


def _raiz(w: str) -> str:
    for suf in ("es", "s"):
        if len(w) > 5 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _concepto(w: str) -> str | None:
    for c, (raices, _) in CONCEPTOS.items():
        if any(w.startswith(r) for r in raices):
            return c
    return None


def _match_nombre(col: str, frag: str) -> bool:
    n = normalize_name(col)
    if frag in n:
        return True
    return any(len(t) >= 4 and (t.startswith(frag[:max(4, len(frag) - 2)]) or frag.startswith(t))
               for t in n.split("_"))


def _fragmentos(w: str) -> tuple[str, ...]:
    c = _concepto(w)
    return CONCEPTOS[c][1] if c else (_raiz(w),)


def _cols(df, tipos, w, kinds) -> list[str]:
    for frag in _fragmentos(w):
        found = [c for c, t in tipos.items() if t in kinds and _match_nombre(c, frag)]
        if found:
            return found
    return []


def resolver_evento(df, tipos, w) -> tuple[list[dict], str]:
    """Candidatos de evento para la palabra w: columna booleana > valor categórico > columna numérica."""
    frags = _fragmentos(w)
    for frag in frags:
        b = [c for c, t in tipos.items() if t == "booleano" and _match_nombre(c, frag)]
        if b:
            return [{"tipo": "booleano", "columna": c} for c in b], frag
    for frag in frags:
        cands = []
        for c, t in tipos.items():
            if t in ("categorica", "texto"):
                vals = [v for v in df[c].dropna().astype(str).unique()[:500] if frag in normalize_name(v)]
                if vals and len(vals) < df[c].nunique():
                    cands.append({"tipo": "valor", "columna": c, "valores": sorted(vals)})
        if cands:
            return cands, frag
    for frag in frags:
        n = [c for c, t in tipos.items() if t == "numerica" and _match_nombre(c, frag)]
        if n:
            return [{"tipo": "suma", "columna": c} for c in n], frag
    return [], ""


def _evento_de_columna(df, tipos, col) -> dict:
    t = tipos.get(col)
    if t == "booleano":
        return {"tipo": "booleano", "columna": col}
    if t == "numerica":
        return {"tipo": "suma", "columna": col}
    raise ValueError(f"«{col}» no sirve como evento sin indicar un valor.")


def _mask_evento(df, ev) -> pd.Series:
    s = df[ev["columna"]]
    if ev["tipo"] == "booleano":
        return s.astype("Float64").eq(1).fillna(False).astype(bool)
    return s.astype(str).isin(ev["valores"]) & s.notna()


def _pide(rol: str, cands: list[str], texto: str, interp=None) -> Respuesta:
    return Respuesta("aclaracion", texto, interp, opciones={rol: cands})


def _no(texto: str, df) -> Respuesta:
    return Respuesta("no_soportada", texto + "\n\nColumnas disponibles: " + ", ".join(
        f"{c} ({t})" for c, t in perfil(df).items()))


# ---------------- interpretación ----------------
def interpretar(pregunta: str, df: pd.DataFrame, aclaraciones: dict[str, str] | None = None):
    acl = aclaraciones or {}
    tipos = perfil(df)
    if not pregunta or not pregunta.strip():
        return _no("Escribe una pregunta.", df)
    if df.empty:
        return Respuesta("no_soportada", "No hay registros con los filtros actuales: no se puede responder.")
    words = _palabras(pregunta)
    t = " ".join(words)
    contenido = [w for w in words if w not in STOP and len(w) > 2]
    qn = normalize_name(pregunta)
    exactas = [c for c in sorted(df.columns, key=lambda c: -len(str(c)))
               if len(normalize_name(c)) > 3 and re.search(rf"(^|_){re.escape(normalize_name(c))}(_|$)", qn)]
    direccion = "menor" if re.search(r"\b(menor|menos|mas baja|mas bajo|peor)\b", t) else "mayor"

    def grupo(kinds=("categorica", "booleano")):
        if "grupo" in acl:
            return acl["grupo"], None
        ex = [c for c in exactas if tipos.get(c) in kinds]
        if ex:
            return ex[0], normalize_name(ex[0])
        after = re.findall(r"\b(?:que|por|segun|cada|cual|cuales)\s+(\w+)", t)
        for w in after + contenido:
            if w in STOP:
                continue
            cs = _cols(df, tipos, w, kinds)
            if len(cs) == 1:
                return cs[0], w
            if len(cs) > 1:
                return None, _pide("grupo", cs, f"«{w}» coincide con varias columnas. ¿Cuál quieres usar?")
        return None, None

    def medida(excluir=()):
        if "medida" in acl:
            return acl["medida"], None
        ex = [c for c in exactas if tipos.get(c) == "numerica"]
        if ex:
            return ex[0], None
        for w in contenido:
            if w in excluir:
                continue
            cs = _cols(df, tipos, w, ("numerica",))
            if len(cs) == 1:
                return cs[0], None
            if len(cs) > 1:
                return None, _pide("medida", cs, f"«{w}» coincide con varias columnas numéricas. ¿Cuál mide lo que buscas?")
        return None, None

    agg = ("promedio" if re.search(r"\b(promedio|media)\b", t) else
           "mediana" if "mediana" in t else "suma")

    # 1) evolución temporal
    if re.search(r"\bevoluc|\btendencia|a lo largo|en el tiempo|durante el periodo|mensual|trimestral|"
                 r"por (mes|trimestre|semana|ano|dia)", t):
        fechas = [c for c, k in tipos.items() if k == "fecha"]
        if not fechas:
            return _no("La pregunta pide una evolución temporal, pero el dataset no tiene columnas de fecha.", df)
        f = acl.get("fecha")
        if not f:
            named = [c for c in fechas if c in exactas]
            if len(fechas) == 1 or len(named) == 1:
                f = fechas[0] if len(fechas) == 1 else named[0]
            else:
                return _pide("fecha", fechas, "Hay varias columnas de fecha. ¿Cuál define el periodo?")
        m, err = medida()
        if err:
            return err
        freq = next((v for w, v in _FREQ_WORDS.items() if w in words), None)
        notas = [] if freq else ["Frecuencia elegida automáticamente según el rango de fechas."]
        return Interpretacion("evolucion", medida=m, agregacion=agg if m else "conteo", fecha=f,
                              frecuencia=freq or frecuencia_auto(df[f]), notas=notas)

    # 2) comparación entre dos grupos
    if re.search(r"diferenc|compar|versus|\bvs\b|frente a", t):
        g = acl.get("grupo")
        if not g:
            for w in contenido:
                cs = [c for c in _cols(df, tipos, w, ("booleano", "categorica")) if df[c].nunique() == 2]
                if len(cs) > 1:
                    return _pide("grupo", cs, f"«{w}» coincide con varias columnas de dos grupos. ¿Cuál separa los grupos?")
                if cs:
                    g = cs[0]
                    break
        if not g:
            return _no("Para comparar grupos necesito una columna con exactamente dos valores "
                       "(p. ej. retenido/no retenido) que coincida con la pregunta.", df)
        notas = []
        if any(_concepto(w) == "retencion" for w in words) and re.search(r"churn|abandon|baja", normalize_name(g)):
            notas.append(f"Interpretación: «{g}» = 0 → retenidos; «{g}» = 1 → no retenidos (abandono).")
        return Interpretacion("comparar_grupos", grupo=g, notas=notas)

    # 3) ranking de un evento por grupo
    if re.search(r"\b(mayor|menor|mas|menos|top|principal|principales|peor|mejor)\b", t):
        g, err = grupo()
        if isinstance(err, Respuesta):
            return err
        if g:
            if "evento" in acl:
                ev = _evento_de_columna(df, tipos, acl["evento"])
            else:
                ev = None
                for w in contenido:
                    if w == err or g and _match_nombre(g, _raiz(w)):
                        continue
                    cands, _ = resolver_evento(df, tipos.copy() | {g: "_grupo"}, w)
                    if len(cands) > 1:
                        return _pide("evento", [c["columna"] for c in cands],
                                     f"«{w}» puede medirse con varias columnas. ¿Cuál representa el evento?")
                    if cands:
                        ev = cands[0]
                        break
                if ev is None:
                    return _no("Identifiqué el grupo pero no qué evento o medida comparar con las columnas disponibles.", df)
            crit = "tasa" if re.search(r"\b(tasa|porcentaje|proporcion|ratio)\b", t) else "conteo"
            return Interpretacion("ranking_evento", grupo=g, evento=ev, direccion=direccion, criterio=crit)

    # 4) agregado por grupo / global
    if re.search(r"\b(promedio|media|mediana|total|suma|cuanto|cuantos|cuantas)\b", t):
        m, err = medida()
        if err:
            return err
        g, gerr = grupo() if re.search(r"\b(por|segun|cada)\b", t) else (None, None)
        if isinstance(gerr, Respuesta):
            return gerr
        if m is None and not re.search(r"cuant", t):
            return _no("No encontré una columna numérica que coincida con la pregunta.", df)
        return Interpretacion("agregado", grupo=g, medida=m, agregacion=agg if m else "conteo",
                              direccion=direccion)

    return _no("No reconozco el tipo de pregunta. Formas compatibles:\n- " + "\n- ".join(EJEMPLOS), df)


# ---------------- ejecución ----------------
def _fmt(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if pct:
        return f"{x:.1%}"
    return f"{x:,.0f}" if float(x).is_integer() or abs(x) >= 1000 else f"{x:,.2f}"


def ejecutar(df: pd.DataFrame, it: Interpretacion) -> Respuesta:
    meta = asdict(it)
    for c in [it.grupo, it.medida, it.fecha, (it.evento or {}).get("columna")]:
        if c and c not in df.columns:
            return Respuesta("no_soportada", f"La columna «{c}» no existe en el dataset seleccionado.", meta)
    if df.empty:
        return Respuesta("no_soportada", "No hay registros con los filtros actuales.", meta)
    fn = {"ranking_evento": _ranking, "evolucion": _evolucion, "comparar_grupos": _comparar,
          "agregado": _agregado}[it.intencion]
    r = fn(df, it)
    r.interpretacion = meta
    r.avisos = it.notas + r.avisos
    return r


def _grupo_series(df, g):
    return df[g].astype(object).where(df[g].notna(), "(vacío)").astype(str)


def _ranking(df, it: Interpretacion) -> Respuesta:
    ev, g = it.evento, it.grupo
    col = ev["columna"]
    base = df[df[col].notna()]
    if base.empty:
        return Respuesta("no_soportada", f"«{col}» no tiene valores tras los filtros.")
    gs = _grupo_series(base, g)
    if ev["tipo"] == "suma":
        v = pd.to_numeric(base[col], errors="coerce")
        tab = v.groupby(gs).agg(registros="count", total="sum", promedio="mean").reset_index()
        tab.columns = [g, "registros", "total", "promedio"]
        key = "promedio" if it.criterio == "tasa" else "total"
        calc = f"{'promedio' if key == 'promedio' else 'suma'} de «{col}» por «{g}»"
        mask = v > 0
    else:
        mask = _mask_evento(base, ev)
        tab = mask.groupby(gs).agg(registros="size", eventos="sum").reset_index()
        tab.columns = [g, "registros", "eventos"]
        tab["tasa"] = tab.eventos / tab.registros.replace(0, np.nan)
        key = "tasa" if it.criterio == "tasa" else "eventos"
        cond = (f"«{col}» = 1" if ev["tipo"] == "booleano" else f"«{col}» ∈ {ev['valores']}")
        calc = (f"tasa = registros con {cond} ÷ registros con «{col}» no vacío, por «{g}»"
                if key == "tasa" else f"conteo de registros con {cond}, por «{g}» (también se muestra la tasa)")
    tab["muestra_pequena"] = tab.registros < MUESTRA_MINIMA
    tab = tab.sort_values(key, ascending=it.direccion == "menor", ignore_index=True)
    elegibles = tab[~tab.muestra_pequena & (tab[g] != "(vacío)")]
    avisos = []
    if elegibles.empty:
        elegibles = tab
        avisos.append(f"Todos los grupos tienen menos de {MUESTRA_MINIMA} registros: resultado poco robusto.")
    elif len(elegibles) < len(tab):
        avisos.append(f"Se excluyen del primer puesto los grupos vacíos o con menos de {MUESTRA_MINIMA} registros.")
    top = elegibles.iloc[0]
    pct = key == "tasa"
    etiqueta = {"tasa": "tasa", "eventos": "número de casos", "total": "total", "promedio": "promedio"}[key]
    texto = (f"El grupo de «{g}» con {it.direccion} {etiqueta} es **{top[g]}** ({_fmt(top[key], pct)}, "
             f"{int(top['registros']):,} registros).")
    if len(elegibles) > 1:
        sec = elegibles.iloc[1]
        texto += f" Le sigue **{sec[g]}** ({_fmt(sec[key], pct)})."
        if pct:
            texto += f" Diferencia: {abs(top[key] - sec[key]) * 100:.1f} puntos porcentuales."
    if key == "eventos":
        texto += " ⚠️ El conteo depende del tamaño de cada grupo; compara también la columna «tasa»."
    sel = base[(gs == top[g]) & mask]
    return Respuesta("ok", texto, columnas=[g, col], calculo=calc, tabla=tab, registros=sel.head(50),
                     grafico={"tipo": "barras", "x": g, "y": key, "porcentaje": pct}, avisos=avisos)


def _evolucion(df, it: Interpretacion) -> Respuesta:
    f, freq = it.fecha, it.frecuencia
    d = df[df[f].notna()]
    if d.empty:
        return Respuesta("no_soportada", f"«{f}» no tiene fechas tras los filtros.")
    val = d[it.medida] if it.medida else None
    ser = serie_temporal(d, f, freq, val, it.agregacion)
    avisos = []
    if len(ser) < 2:
        return Respuesta("no_soportada", "Solo hay un periodo con datos: no se puede describir una evolución.",
                         tabla=ser)
    last_end = pd.Period(ser.periodo.iloc[-1], freq=freq).end_time.normalize()
    first_start = pd.Period(ser.periodo.iloc[0], freq=freq).start_time
    if d[f].max().normalize() < last_end:
        avisos.append(f"El último periodo ({ser.periodo.iloc[-1]}) está incompleto: datos hasta {d[f].max():%Y-%m-%d}.")
    if d[f].min().normalize() > first_start:
        avisos.append(f"El primer periodo ({ser.periodo.iloc[0]}) está incompleto: datos desde {d[f].min():%Y-%m-%d}.")
    a, b = ser.iloc[0], ser.iloc[-1]
    hi, lo = ser.loc[ser.valor.idxmax()], ser.loc[ser.valor.idxmin()]
    var = (b.valor / a.valor - 1) if a.valor else np.nan
    nombre = f"{it.agregacion} de «{it.medida}»" if it.medida else "número de registros"
    texto = (f"{nombre.capitalize()} por {FRECUENCIAS[freq].lower()}: pasó de {_fmt(a.valor)} ({a.periodo}) a "
             f"{_fmt(b.valor)} ({b.periodo})"
             + (f", una variación de {var:+.1%}." if not np.isnan(var) else " (variación no calculable: base cero).")
             + f" Máximo en {hi.periodo} ({_fmt(hi.valor)}); mínimo en {lo.periodo} ({_fmt(lo.valor)}).")
    if it.medida:
        avisos.append("Se agregan todas las filas filtradas; aplica filtros (p. ej. estado) si deben excluirse algunas.")
    return Respuesta("ok", texto, columnas=[f] + ([it.medida] if it.medida else []),
                     calculo=f"{nombre} agrupado por {FRECUENCIAS[freq].lower()} de «{f}»", tabla=ser,
                     registros=d.sort_values(f).tail(50),
                     grafico={"tipo": "linea", "x": "inicio", "y": "valor"}, avisos=avisos)


def _comparar(df, it: Interpretacion) -> Respuesta:
    g = it.grupo
    d = df[df[g].notna()]
    niveles = sorted(d[g].unique(), key=str)
    if len(niveles) != 2:
        return Respuesta("no_soportada", f"«{g}» tiene {len(niveles)} valores tras los filtros; se necesitan 2.")
    tipos = perfil(d)
    if not [c for c, t in tipos.items() if t in ("numerica", "booleano") and c != g]:
        return Respuesta("no_soportada", "No hay columnas numéricas para comparar entre los grupos.")
    A, B = (d[d[g] == v] for v in niveles)
    tab = diferencias_por_grupo(d, g)
    if tab.empty:
        return Respuesta("no_soportada", "Los grupos tienen muy pocos registros para comparar.")
    top = tab.head(3)
    texto = (f"Grupos: «{g}» = {niveles[0]} ({len(A):,} registros) y «{g}» = {niveles[1]} ({len(B):,} registros). "
             "Mayores diferencias (en desviaciones estándar): " +
             "; ".join(f"**{r.columna}** ({r.diferencia_estandarizada:+.2f})" if not np.isnan(r.diferencia_estandarizada)
                      else f"**{r.columna}** (no estandarizable)" for r in top.itertuples()) +
             ". Son diferencias descriptivas, no prueban causalidad.")
    avisos = [f"Grupo con menos de {MUESTRA_MINIMA} registros: comparación poco robusta."] \
        if min(len(A), len(B)) < MUESTRA_MINIMA else []
    return Respuesta("ok", texto, columnas=[g] + list(tab.columna), tabla=tab,
                     calculo=("por columna numérica: promedio y mediana en cada grupo; diferencia estandarizada = "
                              "(media₂ − media₁) ÷ desviación estándar combinada"),
                     registros=d.head(50), grafico={"tipo": "barras_h", "x": "diferencia_estandarizada", "y": "columna"},
                     avisos=avisos)


def _agregado(df, it: Interpretacion) -> Respuesta:
    m, g, agg = it.medida, it.grupo, it.agregacion
    fn = {"suma": "sum", "promedio": "mean", "mediana": "median", "conteo": "size"}[agg]
    nombre = f"{agg} de «{m}»" if m else "número de registros"
    if g:
        gs = _grupo_series(df, g)
        v = pd.to_numeric(df[m], errors="coerce") if m else pd.Series(1, index=df.index)
        tab = v.groupby(gs).agg([fn, "count"]).reset_index()
        tab.columns = [g, "valor", "registros"]
        tab = tab.sort_values("valor", ascending=it.direccion == "menor", ignore_index=True)
        top = tab.iloc[0]
        texto = f"{nombre.capitalize()} por «{g}»: el valor {it.direccion} es **{top[g]}** ({_fmt(top['valor'])})."
        return Respuesta("ok", texto, columnas=[g] + ([m] if m else []), calculo=f"{nombre} agrupado por «{g}»",
                         tabla=tab, registros=df.head(50), grafico={"tipo": "barras", "x": g, "y": "valor"})
    v = pd.to_numeric(df[m], errors="coerce").dropna() if m else df.index.to_series()
    if v.empty:
        return Respuesta("no_soportada", f"«{m}» no tiene valores numéricos tras los filtros.")
    val = getattr(v, fn)() if fn != "size" else len(v)
    return Respuesta("ok", f"{nombre.capitalize()}: **{_fmt(float(val))}** (sobre {len(v):,} registros).",
                     columnas=[m] if m else [], calculo=nombre, registros=df.head(50))


def responder(pregunta: str, df: pd.DataFrame, aclaraciones: dict[str, str] | None = None,
              interprete=None) -> Respuesta:
    """Interpreta (con el motor determinista o un intérprete opcional validado) y ejecuta."""
    it = interprete(pregunta, df) if interprete else interpretar(pregunta, df, aclaraciones)
    if isinstance(it, Respuesta):
        return it
    return ejecutar(df, it)
