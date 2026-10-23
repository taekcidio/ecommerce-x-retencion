"""🩺 Diagnóstico de calidad: detecta problemas, no los corrige."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .esquemas import Esquema
from .tipos import convert, normalize_name

FORMULA_PREFIX = ("=", "+", "-", "@", "\t", "\r")
NEG_HINTS = ("monto", "precio", "total", "cantidad", "valor", "importe", "edad", "dias", "unidades",
             "stock", "ingreso", "costo", "articulos", "paginas", "duracion")

# acción por defecto y acciones permitidas por tipo de problema
ACCIONES = {
    "tipo_invalido": ("corregir", ["corregir", "excluir", "conservar"]),
    "fecha_invalida": ("corregir", ["corregir", "excluir", "conservar"]),
    "nulo_requerido": ("excluir", ["excluir", "conservar"]),
    "clave_duplicada": ("excluir", ["excluir", "conservar"]),
    "fila_duplicada": ("excluir", ["excluir", "conservar"]),
    "columna_vacia": ("excluir", ["excluir", "conservar"]),
    "espacios": ("corregir", ["corregir", "conservar"]),
    "variantes_categoria": ("conservar", ["corregir", "conservar"]),
    "negativo": ("conservar", ["excluir", "conservar"]),
    "fecha_fuera_rango": ("conservar", ["excluir", "conservar"]),
    "formula": ("conservar", ["excluir", "conservar"]),
}
DESCRIPCION_ACCION = {
    "tipo_invalido": "corregir = dejar vacío el valor inválido · conservar = vacío y anotado en _observaciones",
    "fecha_invalida": "corregir = dejar vacía la fecha inválida · conservar = vacía y anotada en _observaciones",
    "espacios": "corregir = recortar espacios al inicio/fin",
    "variantes_categoria": "corregir = unificar a la variante más frecuente ya presente",
    "fila_duplicada": "excluir = conservar solo la primera aparición",
    "clave_duplicada": "excluir = conservar solo la primera aparición de cada clave",
    "columna_vacia": "excluir = eliminar la columna",
    "formula": "siempre se neutraliza al exportar (prefijo ')",
}


@dataclass
class Issue:
    severidad: str  # error | advertencia | info
    codigo: str
    columna: str | None
    mensaje: str
    filas: list[int] = field(default_factory=list)  # índices del DataFrame

    def as_dict(self, max_rows: int = 20) -> dict:
        return {"severidad": self.severidad, "codigo": self.codigo, "columna": self.columna or "—",
                "afectados": len(self.filas), "mensaje": self.mensaje,
                "ejemplos_registro": ", ".join(str(i + 1) for i in self.filas[:max_rows])}


def variant_groups(s: pd.Series) -> dict[str, list[str]]:
    v = s.dropna().astype(str)
    if v.empty or v.nunique() > 200 or v.nunique() == len(v):
        return {}
    norm = v.map(normalize_name)
    groups = {}
    for key, vals in v.groupby(norm):
        uniq = vals.value_counts()
        if len(uniq) > 1:
            groups[key] = list(uniq.index)  # orden: más frecuente primero
    return groups


def diagnose(df: pd.DataFrame, tipos: dict[str, str], clave: list[str] | None = None,
             requeridas: list[str] | None = None, esquema: Esquema | None = None,
             hoy: pd.Timestamp | None = None) -> list[Issue]:
    clave, requeridas = clave or [], requeridas or []
    hoy = hoy or pd.Timestamp.now().normalize()
    data_cols = [c for c in df.columns if not c.startswith("_")]
    issues: list[Issue] = []
    add = issues.append

    if esquema:
        norm = {normalize_name(c): c for c in data_cols}
        for c in data_cols:
            if normalize_name(c) not in esquema.columnas:
                add(Issue("info", "columna_desconocida", c,
                          f"No pertenece al esquema «{esquema.nombre}»; se conserva tal cual."))
        for c in esquema.columnas:
            if c not in norm:
                add(Issue("advertencia", "columna_faltante", c,
                          f"Columna esperada en «{esquema.nombre}» no está en el archivo (no se inventa)."))

    for c in data_cols:
        s = df[c]
        txt = s.where(s.notna(), None)
        present = txt.notna() & (txt.astype(str).str.strip() != "")
        if not present.any():
            add(Issue("advertencia", "columna_vacia", c, "La columna no tiene ningún valor."))
            continue
        nulls = list(df.index[~present])
        if nulls:
            if c in requeridas or c in clave:
                add(Issue("error", "nulo_requerido", c, f"{len(nulls)} valor(es) vacío(s) en columna obligatoria.", nulls))
            else:
                add(Issue("info", "nulos", c, f"{len(nulls)} valor(es) vacío(s) ({len(nulls) / len(df):.1%}).", nulls))
        tipo = tipos.get(c, "texto")
        conv, invalid = convert(s, tipo)
        bad = list(df.index[invalid])
        if bad:
            code = "fecha_invalida" if tipo.startswith("fecha") else "tipo_invalido"
            ej = ", ".join(repr(x) for x in s[invalid].astype(str).unique()[:3])
            add(Issue("error", code, c, f"{len(bad)} valor(es) no son «{tipo}» (p. ej. {ej}).", bad))
        if tipo == "texto":
            st = s[present].astype(str)
            sp = list(st.index[st != st.str.strip()])
            if sp:
                add(Issue("advertencia", "espacios", c, f"{len(sp)} valor(es) con espacios al inicio o final.", sp))
            numeric_like = st.str.match(r"^[-+]?\d+([.,]\d+)?$")
            fm = list(st.index[st.str.startswith(FORMULA_PREFIX) & ~numeric_like])
            if fm:
                add(Issue("advertencia", "formula", c,
                          f"{len(fm)} valor(es) parecen fórmulas (riesgo de inyección en hojas de cálculo).", fm))
            groups = variant_groups(st.str.strip())
            if groups:
                rows = [i for vals in groups.values() for i in st.index[st.str.strip().isin(vals[1:])]]
                ej = "; ".join(" / ".join(v) for v in list(groups.values())[:3])
                add(Issue("advertencia", "variantes_categoria", c,
                          f"Valores que solo difieren en mayúsculas/tildes/espacios: {ej}.", rows))
        elif tipo in ("entero", "decimal") and any(h in normalize_name(c) for h in NEG_HINTS):
            neg = list(df.index[conv.fillna(0) < 0])
            if neg:
                add(Issue("advertencia", "negativo", c, f"{len(neg)} valor(es) negativos en una magnitud.", neg))
        elif tipo.startswith("fecha"):
            out = list(df.index[(conv > hoy + pd.Timedelta(days=1)) | (conv < pd.Timestamp("1900-01-01"))])
            if out:
                add(Issue("advertencia", "fecha_fuera_rango", c,
                          f"{len(out)} fecha(s) futuras o anteriores a 1900.", out))

    body = df[data_cols].astype(str).apply(lambda col: col.str.strip())
    dup = list(df.index[body.duplicated(keep="first")])
    if dup:
        add(Issue("advertencia", "fila_duplicada", None, f"{len(dup)} registro(s) idénticos a otro anterior.", dup))
    if clave and all(k in df.columns for k in clave):
        k = df[clave].astype(str).apply(lambda col: col.str.strip())
        kd = list(df.index[k.duplicated(keep="first") & df[clave].notna().all(axis=1) & ~body.duplicated()])
        if kd:
            add(Issue("error", "clave_duplicada", "+".join(clave),
                      f"{len(kd)} registro(s) repiten la clave {clave} con contenido distinto.", kd))
    return issues
