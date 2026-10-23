"""🔗 Relación de tablas mediante claves comunes (sugerencia, verificación y unión)."""
from __future__ import annotations

import pandas as pd

from .tipos import normalize_name


def _keyvals(s: pd.Series) -> pd.Series:
    return s.dropna().astype(str).str.strip()


def sugerir_relaciones(tablas: dict[str, pd.DataFrame], min_cobertura: float = 0.5) -> list[dict]:
    """Pares (hija.col → padre.col) donde el padre tiene valores únicos y la hija los referencia."""
    out = []
    for padre, dp in tablas.items():
        for pc in dp.columns:
            pv = _keyvals(dp[pc])
            if pv.empty or pv.duplicated().any():
                continue
            pset = set(pv)
            for hija, dh in tablas.items():
                if hija == padre:
                    continue
                for hc in dh.columns:
                    if normalize_name(hc) != normalize_name(pc):
                        continue
                    hv = _keyvals(dh[hc])
                    cob = hv.isin(pset).mean() if len(hv) else 0
                    if cob >= min_cobertura:
                        out.append({"hija": hija, "columna_hija": hc, "padre": padre,
                                    "columna_padre": pc, "cobertura": round(float(cob), 4)})
    return sorted(out, key=lambda r: -r["cobertura"])


def verificar(hija: pd.DataFrame, fk: str, padre: pd.DataFrame, pk: str) -> dict:
    pv, hv = _keyvals(padre[pk]), hija[fk].astype("string").str.strip()
    pset = set(pv)
    present = hv.notna()
    orphan = present & ~hv.isin(pset)
    return {
        "registros_hija": int(len(hija)),
        "con_clave": int(present.sum()),
        "sin_clave": int((~present).sum()),
        "coinciden": int((present & ~orphan).sum()),
        "huerfanos": int(orphan.sum()),
        "ejemplos_huerfanos": hv[orphan].unique()[:10].tolist(),
        "clave_padre_unica": not pv.duplicated().any(),
        "duplicados_clave_padre": int(pv.duplicated().sum()),
        "cardinalidad": "1:N" if hv[present].duplicated().any() else "1:1",
    }


def unir(hija: pd.DataFrame, fk: str, padre: pd.DataFrame, pk: str, tipo: str = "left",
         sufijo_padre: str = "_padre") -> pd.DataFrame:
    """Une sin multiplicar filas de la hija: exige clave única en el padre."""
    if padre[pk].dropna().astype(str).str.strip().duplicated().any():
        raise ValueError(f"«{pk}» tiene valores repetidos en la tabla padre: la unión duplicaría registros.")
    if tipo not in ("left", "inner"):
        raise ValueError("Tipo de unión no soportado (usa left o inner).")
    h, p = hija.copy(), padre.copy()
    h["__k"], p["__k"] = h[fk].astype("string").str.strip(), p[pk].astype("string").str.strip()
    p = p.drop(columns=[pk])
    out = h.merge(p, on="__k", how=tipo, suffixes=("", sufijo_padre), validate="many_to_one")
    return out.drop(columns="__k")
