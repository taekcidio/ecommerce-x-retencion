"""🔎 Filtros declarativos (sin evaluar código del usuario)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

VACIO = "(vacío)"


@dataclass
class Filtro:
    columna: str
    tipo: str  # rango_fecha | rango_num | valores
    valores: tuple | list


def aplicar_filtros(df: pd.DataFrame, filtros: list[Filtro]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for f in filtros:
        if f.columna not in df.columns:
            raise ValueError(f"El filtro usa la columna «{f.columna}», que no existe en el dataset.")
        s = df[f.columna]
        if f.tipo == "rango_fecha":
            lo, hi = (pd.Timestamp(x) if x is not None else None for x in f.valores)
            m = s.notna()
            if lo is not None:
                m &= s >= lo
            if hi is not None:
                m &= s < hi + pd.Timedelta(days=1)  # fin de rango inclusivo (día completo)
        elif f.tipo == "rango_num":
            lo, hi = f.valores
            v = pd.to_numeric(s, errors="coerce")
            m = v.between(lo, hi)
        elif f.tipo == "valores":
            sel = list(f.valores)
            m = s.astype(object).where(s.notna(), VACIO).astype(str).isin([str(x) for x in sel])
        else:
            raise ValueError(f"Tipo de filtro desconocido: {f.tipo}")
        mask &= m.fillna(False)
    return df[mask]
