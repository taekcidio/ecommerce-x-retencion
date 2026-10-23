"""🧹 Aplicación explícita de un plan de limpieza, informe de calidad y exportación segura.

Nunca modifica el DataFrame de entrada ni el archivo original: trabaja sobre copias y
registra cada cambio. No rellena valores faltantes ni crea registros.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .calidad import ACCIONES, FORMULA_PREFIX, Issue, diagnose, variant_groups
from .esquemas import match_schema
from .tipos import convert, infer_type, normalize_name


@dataclass
class Plan:
    columnas: list[str] | None = None          # columnas a importar (None = todas)
    renombrar: dict[str, str] = field(default_factory=dict)
    tipos: dict[str, str] = field(default_factory=dict)  # sobre nombres finales
    clave: list[str] = field(default_factory=list)
    requeridas: list[str] = field(default_factory=list)
    acciones: dict[str, str] = field(default_factory=dict)  # código de problema -> acción


@dataclass
class Resultado:
    limpio: pd.DataFrame
    excluidos: pd.DataFrame
    issues: list[Issue]
    cambios: list[str]
    informe: dict


def sugerir_plan(df: pd.DataFrame) -> tuple[Plan, str | None]:
    """Plan inicial: tipos inferidos y, si coincide un esquema conocido, su clave y tipos."""
    esq, _ = match_schema(list(df.columns))
    tipos = {c: infer_type(df[c]) for c in df.columns}
    plan = Plan(tipos=tipos)
    if esq:
        by_norm = {normalize_name(c): c for c in df.columns}
        for c, t in esq.columnas.items():
            if c in by_norm:
                plan.tipos[by_norm[c]] = t
        plan.clave = [by_norm[k] for k in esq.clave if k in by_norm]
        plan.requeridas = [by_norm[k] for k in esq.requeridas if k in by_norm]
    return plan, (esq.nombre if esq else None)


def preparar(df: pd.DataFrame, plan: Plan) -> pd.DataFrame:
    cols = [c for c in (plan.columnas or list(df.columns)) if c in df.columns]
    if not cols:
        raise ValueError("Selecciona al menos una columna para importar.")
    out = df[cols].copy().rename(columns={k: v for k, v in plan.renombrar.items() if v and v.strip()})
    if out.columns.duplicated().any():
        raise ValueError("Hay columnas con el mismo nombre después de renombrar.")
    return out


def aplicar(df: pd.DataFrame, plan: Plan, fuente: str = "datos") -> Resultado:
    work = preparar(df, plan)
    work.insert(0, "_registro", range(1, len(work) + 1))  # número de registro en el origen
    esquema, _ = match_schema([c for c in work.columns if c != "_registro"])
    act = {k: plan.acciones.get(k, v[0]) for k, v in ACCIONES.items()}
    cambios: list[str] = []
    obs = pd.Series("", index=work.index)
    excl = pd.Series("", index=work.index)

    def mark(idx, txt, target):
        for i in idx:
            target[i] = (target[i] + "; " if target[i] else "") + txt

    issues = diagnose(work, plan.tipos, plan.clave, plan.requeridas, esquema)

    # 1) correcciones de texto que afectan a otras validaciones
    for iss in issues:
        a = act.get(iss.codigo)
        if iss.codigo == "espacios" and a == "corregir":
            work.loc[iss.filas, iss.columna] = work.loc[iss.filas, iss.columna].astype(str).str.strip()
            cambios.append(f"{iss.columna}: espacios recortados en {len(iss.filas)} valor(es).")
        elif iss.codigo == "variantes_categoria" and a == "corregir":
            s = work[iss.columna]
            for vals in variant_groups(s.dropna().astype(str).str.strip()).values():
                m = s.astype(str).str.strip().isin(vals[1:])
                work.loc[m, iss.columna] = vals[0]
                cambios.append(f"{iss.columna}: {vals[1:]} unificado(s) a «{vals[0]}» ({int(m.sum())} valor(es)).")

    # 2) acciones por registro / columna
    drop_cols = []
    for iss in issues:
        a, c = act.get(iss.codigo), iss.columna
        if iss.codigo in ("tipo_invalido", "fecha_invalida"):
            if a == "excluir":
                mark(iss.filas, f"{c}: valor inválido", excl)
            else:
                raw = work.loc[iss.filas, c].astype(str)
                if a == "conservar":
                    for i, v in raw.items():
                        mark([i], f"{c} inválido: {v!r}", obs)
                cambios.append(f"{c}: {len(iss.filas)} valor(es) inválido(s) dejados vacíos.")
        elif iss.codigo == "columna_vacia" and a == "excluir":
            drop_cols.append(c)
        elif iss.codigo in ACCIONES and a == "excluir" and iss.filas:
            mark(iss.filas, f"{iss.codigo}{' en ' + c if c else ''}", excl)
        elif iss.codigo in ("negativo", "fecha_fuera_rango", "nulo_requerido", "clave_duplicada") and a == "conservar":
            mark(iss.filas, f"{iss.codigo}{' en ' + c if c else ''}", obs)

    # 3) conversión de tipos sobre las filas finales
    typed = work.drop(columns=drop_cols)
    if drop_cols:
        cambios.append(f"Columnas vacías eliminadas: {', '.join(drop_cols)}.")
    for c in typed.columns:
        if c != "_registro":
            typed[c] = convert(typed[c], plan.tipos.get(c, "texto"))[0]
    is_ex = excl != ""
    excluidos = df.loc[work.index[is_ex]].copy()
    excluidos.insert(0, "_registro", work.loc[is_ex, "_registro"].values)
    excluidos["_motivo_exclusion"] = excl[is_ex].values
    limpio = typed[~is_ex].copy()
    if (obs[~is_ex] != "").any():
        limpio["_observaciones"] = obs[~is_ex].replace("", None)
    if is_ex.any():
        cambios.append(f"{int(is_ex.sum())} registro(s) excluidos (ver archivo de excluidos).")

    err_rows = {i for x in issues if x.severidad == "error" for i in x.filas}
    warn_rows = {i for x in issues if x.severidad == "advertencia" for i in x.filas}
    informe = {
        "fuente": fuente,
        "generado": datetime.now().isoformat(timespec="seconds"),
        "esquema_detectado": esquema.nombre if esquema else "genérico",
        "registros_leidos": int(len(work)),
        "registros_validos": int(len(limpio)),
        "registros_excluidos": int(is_ex.sum()),
        "registros_con_errores": len(err_rows),
        "registros_con_advertencias": len(warn_rows - err_rows),
        "registros_sin_problemas": int(len(work) - len(err_rows | warn_rows)),
        "errores": sum(x.severidad == "error" for x in issues),
        "advertencias": sum(x.severidad == "advertencia" for x in issues),
        "columnas": {c: plan.tipos.get(c, "texto") for c in limpio.columns if not c.startswith("_")},
        "clave": plan.clave,
        "acciones": {k: act[k] for k in {x.codigo for x in issues} if k in act},
        "problemas": [x.as_dict() for x in issues],
        "cambios": cambios,
    }
    return Resultado(limpio.drop(columns="_registro"), excluidos, issues, cambios, informe)


# ---------- exportación ----------
def _neutralize(df: pd.DataFrame) -> pd.DataFrame:
    """Evita inyección de fórmulas en Excel/Sheets anteponiendo un apóstrofo."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].map(lambda v: "'" + v if isinstance(v, str) and v.startswith(FORMULA_PREFIX)
                                and not v.lstrip("+-").replace(".", "", 1).isdigit() else v)
    return out


def _fmt_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes("datetime").columns:
        has_time = (out[c].dropna() != out[c].dropna().dt.normalize()).any()
        out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M:%S" if has_time else "%Y-%m-%d")
    return out


def to_csv_bytes(df: pd.DataFrame, sep: str = ",") -> bytes:
    return _neutralize(_fmt_dates(df)).to_csv(index=False, sep=sep).encode("utf-8-sig")


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            _neutralize(_fmt_dates(d)).to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


def informe_json(informe: dict) -> bytes:
    return json.dumps(informe, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def informe_markdown(inf: dict) -> str:
    L = [f"# Informe de calidad · {inf['fuente']}", "", f"Generado: {inf['generado']} · "
         f"Esquema: **{inf['esquema_detectado']}**", "", "| Métrica | Valor |", "|---|---:|"]
    for k in ["registros_leidos", "registros_validos", "registros_excluidos", "registros_con_errores",
              "registros_con_advertencias", "registros_sin_problemas", "errores", "advertencias"]:
        L.append(f"| {k.replace('_', ' ').capitalize()} | {inf[k]:,} |")
    L += ["", "## Problemas detectados", "", "| Severidad | Código | Columna | Afectados | Detalle |",
          "|---|---|---|---:|---|"]
    L += [f"| {p['severidad']} | {p['codigo']} | {p['columna']} | {p['afectados']} | {p['mensaje']} |"
          for p in inf["problemas"]] or ["| — | — | — | 0 | Sin problemas |"]
    L += ["", "## Acciones aplicadas", ""] + [f"- {k}: **{v}**" for k, v in inf["acciones"].items()]
    L += ["", "## Cambios", ""] + ([f"- {c}" for c in inf["cambios"]] or ["- Ningún cambio."])
    L += ["", "## Tipos de columna", "", "| Columna | Tipo |", "|---|---|"]
    L += [f"| {c} | {t} |" for c, t in inf["columnas"].items()]
    return "\n".join(L) + "\n"


__all__ = ["Plan", "Resultado", "sugerir_plan", "preparar", "aplicar", "to_csv_bytes", "to_excel_bytes",
           "informe_json", "informe_markdown"]
