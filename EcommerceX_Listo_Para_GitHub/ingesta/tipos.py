"""🔢 Inferencia y conversión explícita de tipos (sin inventar valores).

Una conversión fallida nunca se rellena: el valor queda nulo y se marca como inválido.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

TIPOS = ["texto", "entero", "decimal", "fecha", "fecha_hora", "booleano"]
_TRUE = {"1", "true", "verdadero", "si", "sí", "s", "yes", "y", "x"}
_FALSE = {"0", "false", "falso", "no", "n"}
_THOUSANDS_COMMA = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_THOUSANDS_DOT = re.compile(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$")
_DATE_HINT = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _nonnull(s: pd.Series) -> pd.Series:
    return s[s.notna()].astype(str).str.strip().replace("", np.nan).dropna()


def parse_number(s: pd.Series) -> pd.Series:
    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        t = re.sub(r"[\s$€£]|COP|USD", "", str(v).strip())
        if not t:
            return np.nan
        if _THOUSANDS_COMMA.match(t):
            t = t.replace(",", "")
        elif _THOUSANDS_DOT.match(t) and ("," in t or t.count(".") > 1):
            t = t.replace(".", "").replace(",", ".")
        elif "," in t and "." not in t and t.count(",") == 1:
            t = t.replace(",", ".")
        try:
            x = float(t)
            return x if np.isfinite(x) else np.nan
        except ValueError:
            return np.nan
    return s.map(one).astype(float)


def parse_date(s: pd.Series, dayfirst: bool = True) -> pd.Series:
    txt = s.where(s.notna(), None).astype(object)
    hinted = txt.map(lambda v: bool(v) and bool(_DATE_HINT.search(str(v))))
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if not hinted.any():
        return out
    sub = txt[hinted].astype(str).str.strip()
    iso = pd.to_datetime(sub, format="ISO8601", errors="coerce")
    rest = iso.isna() & ~sub.str.match(r"^\d{4}[-/.]")  # un ISO inválido no se reinterpreta
    if rest.any():
        iso[rest] = pd.to_datetime(sub[rest], format="mixed", dayfirst=dayfirst, errors="coerce")
    out[hinted] = iso.values
    return out


def parse_bool(s: pd.Series) -> pd.Series:
    low = s.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip().lower())
    return low.map(lambda v: True if v in _TRUE else (False if v in _FALSE else pd.NA)).astype("boolean")


def infer_type(s: pd.Series, threshold: float = 0.8) -> str:
    v = _nonnull(s)
    if v.empty:
        return "texto"
    low = set(v.str.lower().unique())
    if low <= (_TRUE | _FALSE) and len(low) <= 2 and not low <= {"1"} and not low <= {"0"}:
        return "booleano"
    if v.str.match(r"^0\d+$").any():  # códigos con ceros a la izquierda → texto
        return "texto"
    nums = parse_number(v)
    if nums.notna().mean() >= threshold:
        ok = nums.dropna()
        return "entero" if (ok == ok.round()).all() and not v.str.contains(r"[.,]\d*[1-9]").any() else "decimal"
    dates = parse_date(v)
    if dates.notna().mean() >= threshold:
        d = dates.dropna()
        return "fecha_hora" if (d != d.dt.normalize()).any() else "fecha"
    return "texto"


def convert(s: pd.Series, tipo: str) -> tuple[pd.Series, pd.Series]:
    """Devuelve (serie convertida, máscara de valores inválidos)."""
    present = s.notna() & (s.astype(str).str.strip() != "")
    if tipo == "texto":
        out = s.where(present, None)
    elif tipo in ("entero", "decimal"):
        out = parse_number(s)
        if tipo == "entero":
            frac = out.notna() & (out != out.round())
            out = out.where(~frac).round().astype("Int64")
    elif tipo in ("fecha", "fecha_hora"):
        out = parse_date(s)
        if tipo == "fecha":
            out = out.dt.normalize()
    elif tipo == "booleano":
        out = parse_bool(s)
    else:
        raise ValueError(f"Tipo desconocido: {tipo}")
    invalid = present & pd.Series(out, index=s.index).isna()
    return out, invalid
