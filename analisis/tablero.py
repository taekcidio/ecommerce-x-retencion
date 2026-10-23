"""📈 Cálculos del dashboard de E-Commerce X. Una sola fuente de fórmulas para KPIs, tendencias y segmentos.

Fórmulas (coinciden con README · Definiciones y con la validación de la fase 1):
- Pedido válido: estado_pedido ≠ «Cancelado».
- Ingresos brutos = Σ monto_neto de pedidos válidos.
- Reembolsos = Σ monto_reembolso de las devoluciones de esos pedidos (asignadas al mes del pedido).
- Ingresos netos = brutos − reembolsos.
- Retención trimestral T = compradores de T−1 que vuelven a comprar en T ÷ compradores de T−1.
- Churn = % de clientes elegibles con churn = 1 (ventana fija jul–sep 2026).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .kpis import MUESTRA_MINIMA, Valor
from .perfil import diferencias_por_grupo, retencion_periodos

DIMENSIONES = {"segmento": "Segmento", "canal_adquisicion": "Canal de adquisición", "ciudad": "Ciudad",
               "miembro_fidelidad": "Programa de fidelidad"}
ENTREGADOS = ("Entregado", "Devuelto", "Devuelto parcial")


@dataclass
class Filtros:
    desde: pd.Timestamp | None = None
    hasta: pd.Timestamp | None = None
    atributos: dict[str, list] = field(default_factory=dict)  # dimensión -> valores permitidos


def _attr_mask(df: pd.DataFrame, f: Filtros) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for col, vals in f.atributos.items():
        if vals is not None and col in df.columns:
            m &= df[col].isin(vals)
    return m


def filtrar(t: dict[str, pd.DataFrame], f: Filtros) -> dict[str, pd.DataFrame]:
    """Aplica filtros de atributos de cliente y rango de fechas de pedido."""
    cli = t["clientes"][_attr_mask(t["clientes"], f)]
    tx_all = t["transacciones_con_cliente"]
    tx_attr = tx_all[_attr_mask(tx_all, f)]
    m = pd.Series(True, index=tx_attr.index)
    if f.desde is not None:
        m &= tx_attr.fecha_pedido >= pd.Timestamp(f.desde)
    if f.hasta is not None:
        m &= tx_attr.fecha_pedido < pd.Timestamp(f.hasta) + pd.Timedelta(days=1)
    tx = tx_attr[m]
    dev = t["devoluciones"][t["devoluciones"].pedido_id.isin(tx.pedido_id)]
    out = {"clientes": cli, "tx": tx, "tx_atributos": tx_attr, "devoluciones": dev,
           "devoluciones_todas": t["devoluciones"], "navegacion": t.get("navegacion_web")}
    if "features" in t:
        out["features"] = t["features"][_attr_mask(t["features"], f)]
    return out


def _div(a, b) -> float:
    return a / b if b else np.nan


def _v(x, n=0, motivo="") -> Valor:
    ok = x is not None and not (isinstance(x, float) and np.isnan(x))
    return Valor(float(x) if ok else None, ok, motivo if not ok else "", n,
                 f"Muestra pequeña (n={n})." if ok and 0 < n < MUESTRA_MINIMA else "")


def retencion(tx_atributos: pd.DataFrame) -> pd.DataFrame:
    """Serie trimestral con la definición oficial (pedidos válidos, trimestres completos)."""
    valid = tx_atributos.estado_pedido.ne("Cancelado")
    return retencion_periodos(tx_atributos, "cliente_id", "fecha_pedido", "Q", valid)


def kpis(d: dict[str, pd.DataFrame], f: Filtros) -> dict[str, Valor | dict]:
    tx, dev, cli = d["tx"], d["devoluciones"], d["clientes"]
    val = tx[tx.estado_pedido.ne("Cancelado")]
    brutos, reemb = float(val.monto_neto.sum()), float(dev.monto_reembolso.sum())
    entregados = tx[tx.estado_pedido.isin(ENTREGADOS)]
    sin = "Sin pedidos con los filtros actuales."
    out = {
        "ingresos_brutos": _v(brutos if len(val) else np.nan, len(val), sin),
        "reembolsos": _v(reemb if len(val) else np.nan, len(dev), sin),
        "ingresos_netos": _v(brutos - reemb if len(val) else np.nan, len(val), sin),
        "pedidos": _v(float(len(val)), len(val)),
        "compradores": _v(float(val.cliente_id.nunique()), len(val)),
        "ticket_promedio": _v(_div(brutos, len(val)), len(val), sin),
        "tasa_devolucion": _v(_div(entregados.estado_pedido.isin(ENTREGADOS[1:]).sum(), len(entregados)),
                              len(entregados), "Sin pedidos entregados."),
        "tasa_cancelacion": _v(_div((tx.estado_pedido == "Cancelado").sum(), len(tx)), len(tx), sin),
    }
    elig = cli[cli.churn.notna()]
    out["churn"] = _v(_div(elig.churn.sum(), len(elig)), len(elig), "Sin clientes elegibles con los filtros.")
    ret = retencion(d["tx_atributos"])
    if f.desde is not None or f.hasta is not None:
        q = pd.PeriodIndex(ret.periodo, freq="Q")
        lo = pd.Period(f.desde or "1900-01-01", "Q")
        hi = pd.Period(f.hasta or "2100-01-01", "Q")
        ret_rango = ret[(q >= lo) & (q <= hi)]
    else:
        ret_rango = ret
    if ret_rango.empty:
        out["retencion"] = _v(np.nan, motivo="No hay trimestres consecutivos en el rango.")
        out["retencion_info"] = {}
    else:
        last = ret_rango.iloc[-1]
        prev_y = ret[ret.periodo == str(pd.Period(last.periodo, "Q") - 4)]
        ref = prev_y.tasa.iloc[0] if len(prev_y) else np.nan
        out["retencion"] = _v(last.tasa, int(last.base_anterior))
        out["retencion_info"] = {"periodo": last.periodo, "ref_periodo": str(pd.Period(last.periodo, "Q") - 4),
                                 "ref": ref, "cambio_relativo": _div(last.tasa - ref, ref) if ref == ref else np.nan}
    return out


def tendencias(d: dict[str, pd.DataFrame], freq: str = "M") -> pd.DataFrame:
    tx, dev = d["tx"], d["devoluciones"]
    if tx.empty:
        return pd.DataFrame()
    t = tx.assign(periodo=tx.fecha_pedido.dt.to_period(freq))
    val = t[t.estado_pedido.ne("Cancelado")]
    reemb = dev.merge(t[["pedido_id", "periodo"]], on="pedido_id").groupby("periodo").monto_reembolso.sum()
    ent = t[t.estado_pedido.isin(ENTREGADOS)]
    late = ent.dias_entrega_reales > ent.dias_entrega_prometidos + 1
    g = pd.DataFrame({
        "ingresos_brutos": val.groupby("periodo").monto_neto.sum(),
        "pedidos": val.groupby("periodo").size(),
        "compradores": val.groupby("periodo").cliente_id.nunique(),
        "tasa_devolucion": ent.estado_pedido.isin(ENTREGADOS[1:]).groupby(ent.periodo).mean(),
        "pct_entregas_tardias": late.groupby(ent.periodo).mean(),
        "tasa_cancelacion": (t.estado_pedido == "Cancelado").groupby(t.periodo).mean(),
    }).sort_index()
    g["reembolsos"] = reemb.reindex(g.index).fillna(0)
    g["ingresos_netos"] = g.ingresos_brutos.fillna(0) - g.reembolsos
    g["ticket_promedio"] = g.ingresos_brutos / g.pedidos.replace(0, np.nan)
    g = g.reset_index()
    g["inicio"] = g.periodo.dt.to_timestamp()
    g["periodo"] = g.periodo.astype(str)
    return g


def segmentacion(d: dict[str, pd.DataFrame], dim: str) -> pd.DataFrame:
    cli, tx, dev = d["clientes"], d["tx"], d["devoluciones"]
    elig = cli[cli.churn.notna()]
    val = tx[tx.estado_pedido.ne("Cancelado")]
    reemb = dev.merge(val[["pedido_id", dim]], on="pedido_id").groupby(dim).monto_reembolso.sum()
    s = pd.DataFrame({
        "clientes": cli.groupby(dim).size(),
        "clientes_elegibles": elig.groupby(dim).size(),
        "churn": elig.groupby(dim).churn.mean(),
        "compradores": val.groupby(dim).cliente_id.nunique(),
        "pedidos": val.groupby(dim).size(),
        "ingresos_brutos": val.groupby(dim).monto_neto.sum(),
    })
    s["ingresos_netos"] = s.ingresos_brutos.fillna(0) - reemb.reindex(s.index).fillna(0)
    s["ticket_promedio"] = s.ingresos_brutos / s.pedidos.replace(0, np.nan)
    s["pedidos_por_comprador"] = s.pedidos / s.compradores.replace(0, np.nan)
    s["pct_ingresos"] = s.ingresos_netos / s.ingresos_netos.sum() if s.ingresos_netos.sum() else np.nan
    s["muestra_pequena"] = s.clientes_elegibles.fillna(0) < MUESTRA_MINIMA
    s.index = s.index.map(lambda v: {1: "Sí", 0: "No", True: "Sí", False: "No"}.get(v, v)
                          if dim == "miembro_fidelidad" else v)
    return s.rename_axis(dim).reset_index().sort_values("ingresos_netos", ascending=False, ignore_index=True)


# ---------- cohorte trimestral: experiencia en T−1 → ¿volvió a comprar en T? ----------
def cohorte(tx_atributos: pd.DataFrame, clientes: pd.DataFrame, devoluciones: pd.DataFrame,
            navegacion: pd.DataFrame | None, T: str) -> pd.DataFrame:
    """Una fila por comprador válido de T−1. churn = 1 si no tiene pedidos válidos en T.

    Es la misma base que la retención de T (retención = 1 − churn medio de esta tabla).
    Las variables describen la experiencia del cliente durante T−1 (mismos nombres que features_modelo).
    """
    t = pd.Period(T, "Q")
    q = tx_atributos.fecha_pedido.dt.to_period("Q")
    prev_all = tx_atributos[q == t - 1]
    prev = prev_all[prev_all.estado_pedido.ne("Cancelado")]
    if prev.empty:
        return pd.DataFrame()
    cur = tx_atributos[(q == t) & tx_atributos.estado_pedido.ne("Cancelado")]
    g = prev.groupby("cliente_id")
    ent = prev[prev.estado_pedido.isin(ENTREGADOS)]
    late = (ent.dias_entrega_reales > ent.dias_entrega_prometidos + 1).groupby(ent.cliente_id).mean()
    dv = devoluciones[devoluciones.pedido_id.isin(prev.pedido_id)]
    out = pd.DataFrame({
        "pedidos_trimestre": g.size(), "valor_monetario": g.monto_neto.sum(),
        "descuento_promedio_pct": g.descuento_pct.mean(), "pct_entregas_tardias": late,
        "dias_entrega_promedio": ent.groupby("cliente_id").dias_entrega_reales.mean(),
        "num_devoluciones": dv.groupby("cliente_id").size(),
        "dias_reembolso_promedio": dv.groupby("cliente_id").dias_reembolso.mean(),
        "num_cancelaciones": prev_all[prev_all.estado_pedido == "Cancelado"].groupby("cliente_id").size(),
    }).reindex(g.size().index)  # solo compradores válidos de T−1 (misma base que la retención)
    if navegacion is not None and not navegacion.empty:
        nv = navegacion[navegacion.inicio_sesion.dt.to_period("Q") == t - 1]
        out["tiempo_carga_ms_prom_90d"] = nv.groupby("cliente_id").tiempo_carga_ms.mean()
        out["sesiones_ult_90d"] = nv.groupby("cliente_id").size()
    out[["num_devoluciones", "num_cancelaciones"]] = out[["num_devoluciones", "num_cancelaciones"]].fillna(0).astype(int)
    attrs = [c for c in ("segmento", "canal_adquisicion", "ciudad", "miembro_fidelidad", "dispositivo_preferido",
                         "genero") if c in clientes]
    out = out.join(clientes.set_index("cliente_id")[attrs], how="left")
    out["churn"] = (~out.index.isin(cur.cliente_id)).astype(int)
    return out.rename_axis("cliente_id").reset_index()


# ---------- factores asociados al abandono (features al 30-jun → churn jul–sep) ----------
EXCLUIR_FACTORES = {"cliente_id", "churn"}


def factores_numericos(feat: pd.DataFrame) -> pd.DataFrame:
    tab = diferencias_por_grupo(feat.drop(columns=[c for c in ("cliente_id",) if c in feat]), "churn")
    return tab.rename(columns={"promedio churn=0": "promedio_retenidos", "promedio churn=1": "promedio_abandono",
                               "mediana churn=0": "mediana_retenidos", "mediana churn=1": "mediana_abandono"})


def factores_categoricos(feat: pd.DataFrame, cols=("segmento", "canal_adquisicion", "ciudad",
                                                    "dispositivo_preferido", "genero")) -> pd.DataFrame:
    base = feat.churn.mean()
    rows = []
    for c in cols:
        if c not in feat:
            continue
        for v, g in feat.groupby(c):
            rows.append({"variable": c, "valor": v, "clientes": len(g), "churn": g.churn.mean(),
                         "diferencia_pp": (g.churn.mean() - base) * 100, "muestra_pequena": len(g) < MUESTRA_MINIMA})
    return pd.DataFrame(rows).sort_values("diferencia_pp", ascending=False, ignore_index=True)


# Factores accionables: (clave, columna, descripción del grupo expuesto, función de máscara, acción, métrica)
def _reglas(feat: pd.DataFrame):
    q75 = lambda c: feat[c].quantile(.75)  # noqa: E731
    q25 = lambda c: feat[c].quantile(.25)  # noqa: E731
    return [
        ("entregas", "pct_entregas_tardias", "con al menos una entrega tardía",
         lambda d: d.pct_entregas_tardias > 0, lambda d: d.pct_entregas_tardias.notna(),
         "Priorizar el cumplimiento de la promesa de entrega (seguimiento a la transportadora por ciudad y "
         "aviso proactivo cuando un pedido se retrasa).", "% de entregas tardías y churn de clientes afectados"),
        ("reembolsos", "dias_reembolso_promedio", "con reembolsos de más de 10 días",
         lambda d: d.dias_reembolso_promedio > 10, lambda d: d.dias_reembolso_promedio.notna(),
         "Acortar el tiempo de reembolso de devoluciones (SLA ≤ 7 días y estado visible para el cliente).",
         "días promedio de reembolso"),
        ("web", "tiempo_carga_ms_prom_90d", f"con carga de página en el 25 % más lento (> {q75('tiempo_carga_ms_prom_90d'):,.0f} ms)",
         lambda d: d.tiempo_carga_ms_prom_90d > q75("tiempo_carga_ms_prom_90d"), lambda d: d.tiempo_carga_ms_prom_90d.notna(),
         "Optimizar el rendimiento del sitio en los dispositivos más lentos (presupuesto de rendimiento y monitoreo).",
         "tiempo de carga promedio y conversión por sesión"),
        ("cancelaciones", "num_cancelaciones", "con al menos una cancelación",
         lambda d: d.num_cancelaciones > 0, lambda d: d.num_cancelaciones.notna(),
         "Contactar a los clientes tras una cancelación para resolver la causa y ofrecer una alternativa.",
         "churn de clientes con cancelaciones"),
        ("descuentos", "descuento_promedio_pct", f"con descuento promedio en el 25 % más bajo (≤ {q25('descuento_promedio_pct'):.1f} %)",
         lambda d: d.descuento_promedio_pct <= q25("descuento_promedio_pct"), lambda d: d.descuento_promedio_pct.notna(),
         "Probar incentivos focalizados (A/B) en los clientes sensibles al precio en lugar de descuentos generales.",
         "churn y margen del grupo de prueba vs control"),
        ("fidelidad", "miembro_fidelidad", "que no pertenecen al programa de fidelidad",
         lambda d: d.miembro_fidelidad == 0, lambda d: d.miembro_fidelidad.notna(),
         "Invitar al programa de fidelidad a clientes frecuentes que aún no son miembros.",
         "churn de nuevos miembros vs no miembros"),
    ] + _reglas_categoricas(feat)


def _reglas_categoricas(feat: pd.DataFrame):
    """Para segmento y canal: el valor con mayor churn (n ≥ mínimo) frente al resto."""
    reglas = []
    textos = {"segmento": ("del segmento «{v}»", "Diseñar una estrategia de retención específica para el segmento «{v}» "
                           "(ofertas y comunicación según su comportamiento de compra)."),
              "canal_adquisicion": ("adquiridos por «{v}»", "Revisar la calidad de los clientes captados por «{v}» y "
                                    "ajustar la segmentación o la bienvenida de ese canal.")}
    for col, (desc, accion) in textos.items():
        if col not in feat or feat[col].nunique() < 2:
            continue
        g = feat.groupby(col).churn.agg(["mean", "size"])
        g = g[g["size"] >= MUESTRA_MINIMA]
        if g.empty:
            continue
        v = g["mean"].idxmax()
        reglas.append((f"{col}:{v}", col, desc.format(v=v), lambda d, c=col, x=v: d[c] == x,
                       lambda d, c=col: d[c].notna(), accion.format(v=v), f"churn del grupo «{v}»"))
    return reglas


def evidencia_factores(feat: pd.DataFrame, alpha: float = 0.05, min_pp: float = 3.0) -> pd.DataFrame:
    """Compara churn del grupo expuesto vs resto para cada factor accionable (prueba χ² 2×2)."""
    rows = []
    for key, col, desc, expuesto, valido, accion, metrica in _reglas(feat):
        if col not in feat:
            continue
        d = feat[valido(feat)]
        e = expuesto(d)
        n1, n0 = int(e.sum()), int((~e).sum())
        if min(n1, n0) < MUESTRA_MINIMA:
            continue
        r1, r0 = d.churn[e].mean(), d.churn[~e].mean()
        tabla = [[d.churn[e].sum(), n1 - d.churn[e].sum()], [d.churn[~e].sum(), n0 - d.churn[~e].sum()]]
        p = chi2_contingency(tabla)[1] if min(map(min, tabla)) > 0 else np.nan
        rows.append({"factor": key, "columna": col, "grupo": desc, "clientes_expuestos": n1, "clientes_resto": n0,
                     "churn_expuestos": r1, "churn_resto": r0, "diferencia_pp": (r1 - r0) * 100, "p_valor": p,
                     "abandonos_en_exceso": n1 * (r1 - r0), "accion": accion, "metrica": metrica,
                     "significativo": bool(p == p and p < alpha and (r1 - r0) * 100 >= min_pp)})
    return pd.DataFrame(rows).sort_values("abandonos_en_exceso", ascending=False, ignore_index=True)


def recomendaciones(feat: pd.DataFrame, n: int = 3, referencia: pd.DataFrame | None = None) -> list[dict]:
    """Hasta n recomendaciones, solo de factores con asociación significativa y relevante en los datos.

    `referencia` (opcional): misma tabla en un periodo anterior, para informar cómo cambió la exposición.
    """
    if feat.empty or feat.churn.nunique() < 2:
        return []
    ev = evidencia_factores(feat)
    if ev.empty:
        return []
    reglas = {r[0]: r for r in _reglas(feat)}
    out = []
    for r in ev[ev.significativo].head(n).itertuples():
        txt = (f"Los clientes {r.grupo} ({r.clientes_expuestos:,}) tuvieron un churn de {r.churn_expuestos:.1%} "
               f"frente a {r.churn_resto:.1%} del resto (+{r.diferencia_pp:.1f} pp; p = {r.p_valor:.3g}). "
               f"Equivale a unos {r.abandonos_en_exceso:,.0f} abandonos más que si tuvieran la tasa del resto.")
        exp_ref = None
        if referencia is not None and not referencia.empty and r.columna in referencia:
            _, _, _, expuesto, valido, _, _ = reglas[r.factor]
            dr, dc = referencia[valido(referencia)], feat[valido(feat)]
            if len(dr):
                exp_ref, exp_cur = expuesto(dr).mean(), expuesto(dc).mean()
                txt += f" Proporción de clientes en esa situación: {exp_cur:.1%} (vs {exp_ref:.1%} en el periodo de referencia)."
        out.append({"factor": r.factor, "accion": r.accion, "metrica": r.metrica, "evidencia": txt,
                    "diferencia_pp": r.diferencia_pp, "exposicion_referencia": exp_ref})
    return out
