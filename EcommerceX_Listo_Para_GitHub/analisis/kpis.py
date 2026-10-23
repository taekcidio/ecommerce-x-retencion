"""🎯 KPIs configurables: métricas declarativas (lista blanca de agregaciones y operadores)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ingesta.tipos import normalize_name

from .perfil import perfil

AGREGACIONES = ["conteo", "conteo_distinto", "suma", "promedio", "mediana", "minimo", "maximo", "tasa"]
OPERADORES = ["==", "!=", ">", ">=", "<", "<=", "contiene"]
FORMATOS = ["numero", "moneda", "porcentaje", "decimal"]
MUESTRA_MINIMA = 30
_MONEY = ("monto", "ingreso", "venta", "total", "precio", "valor", "importe", "revenue", "costo", "reembolso")


@dataclass
class Condicion:
    columna: str
    operador: str
    valor: object


@dataclass
class Metrica:
    agregacion: str
    columna: str | None = None
    condiciones: list[Condicion] = field(default_factory=list)  # filtran filas antes de agregar
    evento: Condicion | None = None  # solo para 'tasa': filas que cumplen ÷ filas con valor


@dataclass
class KPI:
    nombre: str
    numerador: Metrica
    denominador: Metrica | None = None
    formato: str = "numero"
    descripcion: str = ""


@dataclass
class Valor:
    valor: float | None
    calculable: bool
    motivo: str = ""
    n: int = 0
    aviso: str = ""


def _coerce(s: pd.Series, v):
    if pd.api.types.is_bool_dtype(s):
        return str(v).strip().lower() in ("1", "true", "verdadero", "si", "sí")
    if pd.api.types.is_numeric_dtype(s):
        return float(v)
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.Timestamp(v)
    return str(v)


def mascara(df: pd.DataFrame, c: Condicion) -> pd.Series:
    if c.columna not in df.columns:
        raise ValueError(f"La columna «{c.columna}» no existe.")
    if c.operador not in OPERADORES:
        raise ValueError(f"Operador no permitido: {c.operador}")
    s = df[c.columna]
    if c.operador == "contiene":
        return s.astype("string").map(normalize_name, na_action="ignore").str.contains(
            normalize_name(str(c.valor)), regex=False).fillna(False).astype(bool)
    try:
        v = _coerce(s, c.valor)
    except (ValueError, TypeError):
        raise ValueError(f"El valor {c.valor!r} no es compatible con la columna «{c.columna}».")
    ops = {"==": s.eq, "!=": s.ne, ">": s.gt, ">=": s.ge, "<": s.lt, "<=": s.le}
    return ops[c.operador](v).fillna(False).astype(bool) & s.notna()


def calcular_metrica(df: pd.DataFrame, m: Metrica) -> Valor:
    if m.agregacion not in AGREGACIONES:
        return Valor(None, False, f"Agregación no permitida: {m.agregacion}")
    try:
        d = df
        for c in m.condiciones:
            d = d[mascara(d, c)]
        if m.agregacion == "conteo" and m.columna is None:
            return Valor(float(len(d)), True, n=len(d))
        if m.columna is None or m.columna not in d.columns:
            return Valor(None, False, f"La columna «{m.columna}» no está disponible en el dataset.")
        s = d[m.columna]
        if m.agregacion == "conteo":
            return Valor(float(s.notna().sum()), True, n=len(d))
        if m.agregacion == "conteo_distinto":
            return Valor(float(s.nunique()), True, n=len(d))
        if m.agregacion == "tasa":
            if m.evento is None:
                return Valor(None, False, "Una tasa requiere definir el evento (condición).")
            base = d[s.notna()]
            if base.empty:
                return Valor(None, False, "No hay registros con valor para calcular la tasa (división por cero).")
            v = mascara(base, m.evento).mean()
            return Valor(float(v), True, n=len(base), aviso=_aviso(len(base)))
        if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)):
            return Valor(None, False, f"«{m.columna}» no es numérica: no se puede calcular {m.agregacion}.")
        v = pd.to_numeric(s.astype("Float64") if pd.api.types.is_bool_dtype(s) else s,
                          errors="coerce").astype(float).dropna()
        if v.empty:
            return Valor(None, False, "No hay valores numéricos tras los filtros.")
        fn = {"suma": v.sum, "promedio": v.mean, "mediana": v.median, "minimo": v.min, "maximo": v.max}
        return Valor(float(fn[m.agregacion]()), True, n=len(v),
                     aviso=_aviso(len(v)) if m.agregacion in ("promedio", "mediana") else "")
    except ValueError as e:
        return Valor(None, False, str(e))


def _aviso(n: int) -> str:
    return f"Muestra pequeña (n={n}): interpretar con cautela." if 0 < n < MUESTRA_MINIMA else ""


def calcular_kpi(df: pd.DataFrame, k: KPI) -> Valor:
    if df.empty:
        return Valor(None, False, "No hay registros con los filtros actuales.")
    num = calcular_metrica(df, k.numerador)
    if not num.calculable or k.denominador is None:
        return num
    den = calcular_metrica(df, k.denominador)
    if not den.calculable:
        return Valor(None, False, f"Denominador: {den.motivo}")
    if den.valor == 0 or den.valor is None or math.isnan(den.valor):
        return Valor(None, False, "El denominador es cero: el cociente no está definido.")
    return Valor(num.valor / den.valor, True, n=num.n, aviso=num.aviso or den.aviso)


def formatear(v: Valor, formato: str) -> str:
    if not v.calculable or v.valor is None or (isinstance(v.valor, float) and np.isnan(v.valor)):
        return "No calculable"
    x = v.valor
    if formato == "porcentaje":
        return f"{x:.1%}"
    if formato == "moneda":
        return f"${x:,.0f}".replace(",", ".")
    if formato == "decimal":
        return f"{x:,.2f}"
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"


def sugerir_kpis(df: pd.DataFrame, max_kpis: int = 8) -> list[KPI]:
    """KPIs según las columnas disponibles (sin nombres fijos de un dataset concreto)."""
    tipos = perfil(df)
    out = [KPI("Registros", Metrica("conteo"), descripcion="Número de filas tras los filtros")]
    for c in [c for c, t in tipos.items() if t == "id"][:2]:
        out.append(KPI(f"{c} distintos", Metrica("conteo_distinto", c)))
    nums = [c for c, t in tipos.items() if t == "numerica"]
    money = [c for c in nums if any(h in normalize_name(c) for h in _MONEY)]
    for c in money[:2]:
        out.append(KPI(f"Total {c}", Metrica("suma", c), formato="moneda"))
        out.append(KPI(f"Promedio {c}", Metrica("promedio", c), formato="moneda"))
    for c in [c for c in nums if c not in money][:2]:
        out.append(KPI(f"Promedio {c}", Metrica("promedio", c), formato="decimal"))
    for c in [c for c, t in tipos.items() if t == "booleano"][:3]:
        positivo = True if pd.api.types.is_bool_dtype(df[c]) else 1
        out.append(KPI(f"Tasa {c}", Metrica("tasa", c, evento=Condicion(c, "==", positivo)), formato="porcentaje",
                       descripcion=f"% de registros con {c} = {positivo} (sobre filas con valor)"))
    return out[:max_kpis]
