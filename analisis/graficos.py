"""📊 Gráficos Altair según tipo de datos (incluido con Streamlit). Exportables a HTML."""
from __future__ import annotations

import altair as alt
import pandas as pd

from .perfil import distribucion, tipo_columna

TIPOS_GRAFICO = {"fecha": ["linea", "barras"], "categorica": ["barras", "composicion", "tabla"],
                 "booleano": ["barras", "composicion", "tabla"], "numerica": ["histograma", "caja", "tabla"],
                 "id": ["tabla"], "texto": ["tabla"]}


def linea(d: pd.DataFrame, x: str, y: str, titulo: str = "", porcentaje: bool = False) -> alt.Chart:
    return alt.Chart(d, title=titulo).mark_line(point=True).encode(
        x=alt.X(x, title=None), y=alt.Y(y, title=y, axis=alt.Axis(format="%" if porcentaje else ",")),
        tooltip=list(d.columns)).properties(height=320)


def barras(d: pd.DataFrame, x: str, y: str, titulo: str = "", porcentaje: bool = False,
           horizontal: bool = True) -> alt.Chart:
    fmt = "%" if porcentaje else ","
    enc_cat = alt.Y(x, sort="-x", title=None) if horizontal else alt.X(x, sort="-y", title=None)
    enc_val = (alt.X if horizontal else alt.Y)(y, title=y, axis=alt.Axis(format=fmt))
    return alt.Chart(d, title=titulo).mark_bar().encode(
        enc_cat, enc_val, tooltip=list(d.columns)).properties(height=max(200, min(28 * len(d), 600)))


def composicion(d: pd.DataFrame, cat: str, val: str, titulo: str = "", top: int = 8) -> alt.Chart:
    d = d.sort_values(val, ascending=False)
    if len(d) > top:
        resto = pd.DataFrame({cat: ["(otros)"], val: [d[val].iloc[top:].sum()]})
        d = pd.concat([d.head(top)[[cat, val]], resto], ignore_index=True)
    return alt.Chart(d, title=titulo).mark_arc(innerRadius=60).encode(
        theta=alt.Theta(val, stack=True), color=alt.Color(cat, sort=None),
        tooltip=[cat, val]).properties(height=320)


def histograma(df: pd.DataFrame, col: str, titulo: str = "") -> alt.Chart:
    d = distribucion(df, col)
    return alt.Chart(d, title=titulo or f"Distribución de {col}").mark_bar().encode(
        x=alt.X("desde:Q", bin="binned", title=col), x2="hasta:Q", y=alt.Y("registros:Q"),
        tooltip=["intervalo", "registros", alt.Tooltip("pct:Q", format=".1%")]).properties(height=300)


def caja(df: pd.DataFrame, col: str, por: str | None = None) -> alt.Chart:
    d = df[[col] + ([por] if por else [])].dropna()
    enc = {"y": alt.Y(f"{col}:Q")}
    if por:
        enc["x"] = alt.X(f"{por}:N")
    return alt.Chart(d).mark_boxplot().encode(**enc).properties(height=300)


def grafico_columna(df: pd.DataFrame, col: str, tipo: str) -> alt.Chart | None:
    kind = tipo_columna(df[col], col)
    if df[col].dropna().empty:
        return None
    if tipo == "histograma" and kind == "numerica":
        return histograma(df, col)
    if tipo == "caja" and kind == "numerica":
        return caja(df, col)
    if kind in ("categorica", "booleano"):
        d = distribucion(df, col)
        return composicion(d, col, "registros") if tipo == "composicion" else barras(d, col, "registros")
    return None


def desde_respuesta(tabla: pd.DataFrame, spec: dict | None) -> alt.Chart | None:
    if not spec or tabla is None or tabla.empty:
        return None
    if spec["tipo"] == "linea":
        return linea(tabla, spec["x"], spec["y"])
    if spec["tipo"] == "barras_h":
        return barras(tabla, spec["y"], spec["x"])
    return barras(tabla, spec["x"], spec["y"], porcentaje=spec.get("porcentaje", False))


def a_html(chart: alt.Chart) -> bytes:
    return chart.to_html().encode("utf-8")
