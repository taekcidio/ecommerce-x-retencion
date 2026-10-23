"""📐 Clasificación de columnas y estadísticas descriptivas genéricas (cualquier dataset)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ingesta.tipos import normalize_name

FRECUENCIAS = {"D": "Día", "W": "Semana", "M": "Mes", "Q": "Trimestre", "Y": "Año"}
_PANDAS_FREQ = {"D": "D", "W": "W-MON", "M": "M", "Q": "Q", "Y": "Y"}


def es_id(name: str, s: pd.Series) -> bool:
    n = normalize_name(name)
    return n == "id" or n.endswith("_id") or n.startswith("id_")


def tipo_columna(s: pd.Series, name: str = "") -> str:
    """fecha | booleano | numerica | categorica | id | texto"""
    if pd.api.types.is_datetime64_any_dtype(s):
        return "fecha"
    if pd.api.types.is_bool_dtype(s):
        return "booleano"
    v = s.dropna()
    if pd.api.types.is_numeric_dtype(s):
        if len(v) and set(pd.unique(v)) <= {0, 1}:
            return "booleano"
        return "id" if es_id(name, s) else "numerica"
    if es_id(name, s):
        return "id"
    n_uni = v.nunique()
    if n_uni == len(v) and len(v) > 1:  # todos distintos: texto libre, no categoría
        return "texto"
    if n_uni and n_uni <= max(30, 0.05 * len(v)):
        return "categorica"
    return "texto"


def perfil(df: pd.DataFrame) -> dict[str, str]:
    return {c: tipo_columna(df[c], c) for c in df.columns if not str(c).startswith("_")}


def cols_de(df: pd.DataFrame, *tipos: str) -> list[str]:
    return [c for c, t in perfil(df).items() if t in tipos]


def resumen_numerico(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols if cols is not None else cols_de(df, "numerica")
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").astype(float)
        v = s.dropna()
        rows.append({"columna": c, "registros": len(s), "con_valor": len(v), "faltantes": int(s.isna().sum()),
                     "suma": v.sum() if len(v) else np.nan, "promedio": v.mean(), "mediana": v.median(),
                     "desv_est": v.std(), "minimo": v.min(), "p25": v.quantile(.25), "p75": v.quantile(.75),
                     "maximo": v.max()})
    return pd.DataFrame(rows)


def resumen_categorico(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols if cols is not None else cols_de(df, "categorica", "booleano")
    rows = []
    for c in cols:
        vc = df[c].value_counts(dropna=True)
        rows.append({"columna": c, "registros": len(df), "faltantes": int(df[c].isna().sum()),
                     "categorias": int(vc.size), "moda": str(vc.index[0]) if vc.size else None,
                     "frecuencia_moda": int(vc.iloc[0]) if vc.size else 0,
                     "pct_moda": vc.iloc[0] / vc.sum() if vc.size else np.nan})
    return pd.DataFrame(rows)


def faltantes(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    out = pd.DataFrame({"columna": df.columns, "faltantes": df.isna().sum().values})
    out["pct"] = out.faltantes / n if n else np.nan
    return out.sort_values("faltantes", ascending=False, ignore_index=True)


def distribucion(df: pd.DataFrame, col: str, top: int = 20, bins: int = 20) -> pd.DataFrame:
    """Frecuencias (categóricas) o histograma (numéricas). Incluye vacíos como categoría explícita."""
    s = df[col]
    if tipo_columna(s, col) == "numerica":
        v = pd.to_numeric(s, errors="coerce").dropna()
        if v.empty:
            return pd.DataFrame(columns=["intervalo", "desde", "hasta", "registros", "pct"])
        cut = pd.cut(v, bins=min(bins, max(1, v.nunique())), include_lowest=True)
        vc = cut.value_counts(sort=False)
        return pd.DataFrame({"intervalo": vc.index.astype(str), "desde": [i.left for i in vc.index],
                             "hasta": [i.right for i in vc.index], "registros": vc.values,
                             "pct": vc.values / len(v)})
    vc = s.astype(object).where(s.notna(), "(vacío)").astype(str).value_counts()
    out = vc.head(top).rename_axis(col).reset_index(name="registros")
    if vc.size > top:
        out.loc[len(out)] = ["(otros)", int(vc.iloc[top:].sum())]
    out["pct"] = out.registros / len(s) if len(s) else np.nan
    return out


def frecuencia_auto(fechas: pd.Series) -> str:
    v = fechas.dropna()
    if v.empty:
        return "M"
    days = (v.max() - v.min()).days
    return "D" if days <= 62 else "W" if days <= 180 else "M" if days <= 1100 else "Q"


def periodo(fechas: pd.Series, freq: str) -> pd.Series:
    return fechas.dt.to_period(_PANDAS_FREQ[freq].split("-")[0])


def serie_temporal(df: pd.DataFrame, col_fecha: str, freq: str, valor: pd.Series | None = None,
                   agg: str = "conteo") -> pd.DataFrame:
    """Agrega por periodo. `valor`: serie numérica alineada con df (o None para contar registros)."""
    if not pd.api.types.is_datetime64_any_dtype(df[col_fecha]):
        raise ValueError(f"«{col_fecha}» no es una columna de fecha.")
    mask = df[col_fecha].notna()
    p = periodo(df.loc[mask, col_fecha], freq)
    if valor is None or agg == "conteo":
        g = p.groupby(p).size().rename("valor")
        n = g
    else:
        v = pd.to_numeric(valor[mask], errors="coerce")
        grp = v.groupby(p)
        fn = {"suma": "sum", "promedio": "mean", "mediana": "median", "minimo": "min", "maximo": "max"}[agg]
        g, n = grp.agg(fn).rename("valor"), grp.count()
    out = pd.DataFrame({"periodo": g.index.astype(str), "valor": g.values, "registros": n.values})
    out["inicio"] = g.index.to_timestamp()
    return out


def retencion_periodos(df: pd.DataFrame, col_id: str, col_fecha: str, freq: str = "Q",
                       mascara: pd.Series | None = None) -> pd.DataFrame:
    """Tasa de repetición: entidades activas en t−1 que vuelven a estar activas en t ÷ activas en t−1."""
    d = df if mascara is None else df[mascara]
    d = d[d[col_id].notna() & d[col_fecha].notna()]
    if d.empty:
        return pd.DataFrame(columns=["periodo", "base_anterior", "retenidos", "tasa"])
    p = periodo(d[col_fecha], freq)
    sets = d[col_id].groupby(p).apply(set).sort_index()
    rows = []
    idx = list(sets.index)
    for prev, cur in zip(idx, idx[1:]):
        if cur != prev + 1:  # periodo intermedio sin actividad: no hay base comparable
            continue
        base, ret = len(sets[prev]), len(sets[prev] & sets[cur])
        rows.append({"periodo": str(cur), "base_anterior": base, "retenidos": ret,
                     "tasa": ret / base if base else np.nan})
    return pd.DataFrame(rows)


def diferencias_por_grupo(df: pd.DataFrame, g: str) -> pd.DataFrame:
    """Compara columnas numéricas/booleanas entre los dos valores de «g» (descriptivo, sin causalidad).

    diferencia_estandarizada = (media₂ − media₁) ÷ desviación estándar combinada; ordenada por |valor|.
    """
    d = df[df[g].notna()]
    niveles = sorted(d[g].unique(), key=str)
    if len(niveles) != 2:
        return pd.DataFrame()
    tipos = perfil(d)
    nums = [c for c, t in tipos.items() if t in ("numerica", "booleano") and c != g]
    A, B = (d[d[g] == v] for v in niveles)
    rows = []
    for c in nums:
        a, b = (pd.to_numeric(X[c].astype("Float64") if tipos[c] == "booleano" else X[c], errors="coerce")
                .dropna().astype(float) for X in (A, B))
        if a.empty or b.empty:
            continue
        sd = (np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
              if len(a) >= 2 and len(b) >= 2 else np.nan)
        rows.append({"columna": c, f"promedio {g}={niveles[0]}": a.mean(), f"promedio {g}={niveles[1]}": b.mean(),
                     f"mediana {g}={niveles[0]}": a.median(), f"mediana {g}={niveles[1]}": b.median(),
                     "diferencia": b.mean() - a.mean(),
                     "diferencia_estandarizada": (b.mean() - a.mean()) / sd if sd and not np.isnan(sd) else np.nan})
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab
    return tab.reindex(tab.diferencia_estandarizada.abs().sort_values(ascending=False).index).reset_index(drop=True)
