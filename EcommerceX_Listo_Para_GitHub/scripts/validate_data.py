#!/usr/bin/env python3
"""🔎 Valida integridad y coherencia del dataset E-Commerce X.
Escribe reports/validacion.md y sale con código 1 si alguna regla falla.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OBS, CUTOFF_END = pd.Timestamp("2026-09-30 23:59:59"), pd.Timestamp("2026-07-01")

D = lambda s: pd.to_datetime(s)  # noqa: E731
cl = pd.read_csv(RAW / "clientes.csv")
tx = pd.read_csv(RAW / "transacciones.csv")
nv = pd.read_csv(RAW / "navegacion_web.csv")
dv = pd.read_csv(RAW / "devoluciones.csv")
cc = pd.read_csv(RAW / "cancelaciones.csv")
ft = pd.read_csv(ROOT / "data" / "processed" / "features_modelo.csv")
tx["fp"], nv["ts"] = D(tx.fecha_pedido), D(nv.inicio_sesion)
results = []


def check(name, ok_mask_or_bool):
    if isinstance(ok_mask_or_bool, (bool, np.bool_)):
        bad = 0 if ok_mask_or_bool else 1
    else:
        bad = int((~ok_mask_or_bool).sum())
    results.append((name, bad))


# 1. Claves primarias y nulos obligatorios
for df, pk, req in [
    (cl, "cliente_id", ["fecha_registro", "ciudad", "segmento", "canal_adquisicion", "estado_retencion"]),
    (tx, "pedido_id", ["cliente_id", "sesion_id", "fecha_pedido", "monto_neto", "estado_pedido"]),
    (nv, "sesion_id", ["cliente_id", "inicio_sesion", "dispositivo", "tiempo_carga_ms"]),
    (dv, "devolucion_id", ["pedido_id", "fecha_devolucion", "motivo", "estado_devolucion"]),
    (cc, "cancelacion_id", ["pedido_id", "fecha_cancelacion", "motivo", "etapa"])]:
    check(f"PK única: {pk}", ~df[pk].duplicated())
    check(f"Sin nulos obligatorios ({pk[:-3]})", df[req].notna().all(axis=1))

# 2. Integridad referencial
ids = set(cl.cliente_id)
for df, n in [(tx, "transacciones"), (nv, "navegacion_web"), (dv, "devoluciones"), (cc, "cancelaciones")]:
    check(f"FK cliente_id válido en {n}", df.cliente_id.isin(ids))
o_cli = tx.set_index("pedido_id").cliente_id
for df, n in [(dv, "devoluciones"), (cc, "cancelaciones")]:
    check(f"FK pedido_id y cliente coherente en {n}", df.pedido_id.map(o_cli).eq(df.cliente_id))
    check(f"Máx. un registro por pedido en {n}", ~df.pedido_id.duplicated())
s = nv.set_index("sesion_id")
check("Pedido ↔ sesión de compra (compro=1, mismo cliente)",
      tx.sesion_id.map(s.pedido_id).eq(tx.pedido_id) & tx.sesion_id.map(s.cliente_id).eq(tx.cliente_id))
check("Sesiones con compra tienen pedido y viceversa", nv.compro.eq(1) == nv.pedido_id.notna())
check("Embudo: compro ≤ checkout ≤ carrito", (nv.compro <= nv.inicio_checkout) & (nv.inicio_checkout <= nv.agrego_carrito))

# 3. Coherencia temporal
reg = cl.set_index("cliente_id").fecha_registro.pipe(D)
check("Pedido ≥ registro del cliente", tx.fp.dt.normalize() >= tx.cliente_id.map(reg))
check("Sesión ≥ registro del cliente", nv.ts.dt.normalize() >= nv.cliente_id.map(reg))
check("Pedidos dentro de ene-2025…sep-2026", tx.fp.between("2025-01-01", OBS))
dlv = tx.fecha_entrega.notna()
check("fecha_entrega = fecha_pedido + dias_entrega_reales",
      ~dlv | (D(tx.fecha_entrega) == tx.fp.dt.normalize() + pd.to_timedelta(tx.dias_entrega_reales, "D")))
check("Fechas de entrega ≤ fecha de observación", ~dlv | (D(tx.fecha_entrega) <= OBS))
ent = dv.pedido_id.map(tx.set_index("pedido_id").fecha_entrega).pipe(D)
check("Devolución posterior a la entrega", D(dv.fecha_devolucion) > ent)
rf = dv.fecha_reembolso.notna()
check("fecha_reembolso = devolución + dias_reembolso",
      ~rf | (D(dv.fecha_reembolso) == D(dv.fecha_devolucion) + pd.to_timedelta(dv.dias_reembolso, "D")))
check("Reembolsos ≤ fecha de observación", ~rf | (D(dv.fecha_reembolso) <= OBS))
check("Cancelación ≥ fecha del pedido",
      D(cc.fecha_cancelacion) >= cc.pedido_id.map(tx.set_index("pedido_id").fp).dt.normalize())

# 4. Montos
check("monto_neto = bruto − descuento + envío", tx.monto_neto == tx.monto_bruto - tx.monto_descuento + tx.costo_envio)
check("monto_descuento = bruto × pct (redondeo a 100)", (tx.monto_bruto * tx.descuento_pct / 100).round(-2) == tx.monto_descuento)
check("Montos no negativos", (tx[["monto_bruto", "monto_descuento", "costo_envio", "monto_neto"]] >= 0).all(axis=1))
check("Envío gratis desde $150.000 (neto de descuento)",
      (tx.monto_bruto - tx.monto_descuento >= 150000) == (tx.costo_envio == 0))
pagado = dv.pedido_id.map(tx.set_index("pedido_id").eval("monto_bruto - monto_descuento"))
check("Reembolso ≤ valor pagado sin envío", dv.monto_reembolso <= pagado)
check("Devolución rechazada ⇒ reembolso 0", (dv.estado_devolucion != "Rechazada") | (dv.monto_reembolso == 0))

# 5. Estados
canc = set(cc.pedido_id); devs = dv.set_index("pedido_id")
check("Cancelado ⇔ registro en cancelaciones", (tx.estado_pedido == "Cancelado") == tx.pedido_id.isin(canc))
check("Devuelto/Devuelto parcial ⇔ registro en devoluciones",
      tx.estado_pedido.isin(["Devuelto", "Devuelto parcial"]) == tx.pedido_id.isin(devs.index))
n_it = dv.pedido_id.map(tx.set_index("pedido_id").num_articulos)
st = dv.pedido_id.map(tx.set_index("pedido_id").estado_pedido)
check("Devuelto total ⇔ todos los artículos devueltos", (st == "Devuelto") == (dv.articulos_devueltos == n_it))
check("Pedidos cancelados/en tránsito sin entrega", ~tx.estado_pedido.isin(["Cancelado", "En tránsito"]) | ~dlv)
check("Dominio estado_pedido", tx.estado_pedido.isin(["Entregado", "Devuelto", "Devuelto parcial", "Cancelado", "En tránsito"]))

# 6. Agregados de clientes y etiqueta
ok = tx[tx.estado_pedido != "Cancelado"]
g = ok.groupby("cliente_id")
c = cl.set_index("cliente_id")
check("total_pedidos coincide con transacciones", c.total_pedidos.eq(g.size().reindex(c.index, fill_value=0)))
check("ingresos_brutos coincide con transacciones", c.ingresos_brutos.eq(g.monto_neto.sum().reindex(c.index, fill_value=0)))
check("ingresos_netos = brutos − reembolsado", c.ingresos_netos.eq(c.ingresos_brutos - c.total_reembolsado))
pre = set(ok[(ok.fp >= "2026-01-01") & (ok.fp < CUTOFF_END)].cliente_id); post = set(ok[ok.fp >= CUTOFF_END].cliente_id)
exp_churn = c.index.to_series().map(lambda i: (0 if i in post else 1) if i in pre else np.nan)
check("Etiqueta churn recalculada coincide", (c.churn.isna() & exp_churn.isna()) | (c.churn == exp_churn))
check("estado_retencion coherente con churn",
      c.estado_retencion.map({"Activo": 0, "Abandonó": 1}).fillna(-1).eq(c.churn.fillna(-1)))
check("Features: una fila por cliente elegible", set(ft.cliente_id) == set(c.index[c.churn.notna()]) and not ft.cliente_id.duplicated().any())
check("Features sin fuga temporal (recencia ≥ 1 día)", ft.recencia_dias >= 1)

# Informe
fails = [r for r in results if r[1] > 0]
q = ok.assign(q=ok.fp.dt.to_period("Q")).groupby("q").cliente_id.apply(set)
ret = [(str(b), len(q[a] & q[b]), len(q[a])) for a, b in zip(q.index, q.index[1:])]
lines = ["# Informe de validación · E-Commerce X", "",
         f"Reglas evaluadas: **{len(results)}** · Fallidas: **{len(fails)}**", "",
         "| Regla | Filas que incumplen | Estado |", "|---|---:|:---:|"]
lines += [f"| {n} | {b} | {'✅' if b == 0 else '❌'} |" for n, b in results]
lines += ["", "## Volumen", "", "| Archivo | Filas |", "|---|---:|"]
lines += [f"| {n} | {len(df):,} |" for n, df in [("clientes.csv", cl), ("transacciones.csv", tx),
          ("navegacion_web.csv", nv), ("devoluciones.csv", dv), ("cancelaciones.csv", cc),
          ("features_modelo.csv", ft)]]
lines += ["", "## Verificación de la premisa (KPI de retención trimestral)", "",
          "Retención T = clientes con compra no cancelada en T−1 que vuelven a comprar en T ÷ compradores de T−1.", "",
          "| Trimestre | Retenidos | Base T−1 | Retención |", "|---|---:|---:|---:|"]
lines += [f"| {t} | {r:,} | {b:,} | {r / b:.1%} |" for t, r, b in ret]
rd = {t: r / b for t, r, b in ret}
lines += ["", f"Variación interanual: 2026Q2 vs 2025Q2 = {rd['2026Q2'] / rd['2025Q2'] - 1:+.1%} · "
          f"2026Q3 vs 2025Q3 = {rd['2026Q3'] / rd['2025Q3'] - 1:+.1%}",
          "", f"Balance de la etiqueta en features_modelo.csv: churn=1 → {ft.churn.mean():.1%}"]
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "validacion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(f"{'✅' if b == 0 else '❌'} {n} ({b})" for n, b in results))
print(f"\n{len(results) - len(fails)}/{len(results)} reglas OK")
sys.exit(1 if fails else 0)
