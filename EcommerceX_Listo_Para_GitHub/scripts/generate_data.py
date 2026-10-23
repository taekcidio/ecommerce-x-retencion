#!/usr/bin/env python3
"""🛍️ Generador reproducible del dataset sintético de E-Commerce X.

Simula mes a mes (ene-2025 → sep-2026) el ciclo de vida de cada cliente:
registro, navegación, compras, devoluciones, cancelaciones y abandono.
Salida: data/raw/*.csv y data/processed/features_modelo.csv
Uso:  python scripts/generate_data.py [--seed 42] [--clientes 8000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"

START, OBS = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-09-30")
CUTOFF = pd.Timestamp("2026-06-30")  # corte de features; etiqueta = jul–sep 2026
ELIG_FROM = pd.Timestamp("2026-01-01")  # base elegible: compró en ene–jun 2026

# Eventos operativos del negocio (ver README · Contexto operativo)
EV_NEQUI = pd.Timestamp("2026-02-01")       # nuevo medio de pago
EV_TIKTOK = pd.Timestamp("2026-03-01")      # nuevo canal de adquisición
EV_CARRIER = pd.Timestamp("2026-04-01")     # cambio de transportadora
EV_FRONTEND = pd.Timestamp("2026-04-15")    # relanzamiento del sitio
EV_RETURNS = pd.Timestamp("2026-05-01")     # nuevo operador de devoluciones
EV_PROMO = pd.Timestamp("2026-06-01")       # recorte de presupuesto promocional

CITIES = ["Bogotá", "Medellín", "Cali", "Barranquilla", "Bucaramanga", "Cartagena",
          "Pereira", "Manizales", "Cúcuta", "Ibagué", "Santa Marta", "Villavicencio"]
CITY_P = [.34, .17, .11, .08, .06, .05, .04, .03, .04, .03, .03, .02]
MAIN = {"Bogotá", "Medellín"}

SEG = {  # prob. compra mensual, riesgo base mensual, precio, perfil descuento, actividad web
    "Joven tendencia":    dict(p=.32, buy=.36, haz=.065, price=.85, disc="mid",  act=1.6, age=(23, 3)),
    "Profesional urbano": dict(p=.28, buy=.32, haz=.050, price=1.0, disc="low",  act=1.0, age=(33, 6)),
    "Cazador de ofertas": dict(p=.25, buy=.30, haz=.075, price=.75, disc="high", act=1.3, age=(31, 8)),
    "Premium":            dict(p=.15, buy=.38, haz=.040, price=1.6, disc="low",  act=1.1, age=(41, 8)),
}
CATS = {"Vestidos": 120000, "Camisas y blusas": 75000, "Pantalones y jeans": 110000,
        "Calzado": 160000, "Accesorios": 45000, "Ropa deportiva": 90000,
        "Abrigos y chaquetas": 190000, "Ropa interior": 40000}
CAT_NAMES = list(CATS)
DISC_LV = [0, 10, 15, 20, 30, 40]
DISC_P = {"low": [.60, .20, .10, .07, .03, .00],
          "mid": [.40, .25, .15, .12, .06, .02],
          "high": [.15, .20, .20, .20, .17, .08]}
SEASON = {1: .85, 2: .90, 3: .95, 4: 1, 5: 1.12, 6: 1.08, 7: 1, 8: .95, 9: .95, 10: 1, 11: 1.3, 12: 1.4}
LOAD_BASE = {"Móvil": 1700, "Escritorio": 1150, "Tablet": 1450}
TRAFFIC = ["Directo", "Búsqueda orgánica", "Redes sociales", "Email", "Pago (SEM)", "Referido"]
RET_REASONS = ["Talla incorrecta", "No coincide con la foto", "Calidad inferior a lo esperado",
               "Cambio de opinión", "Producto defectuoso", "Llegó tarde"]
CANCEL_REASONS = ["Encontró mejor precio", "Cambio de opinión", "Error en el pedido", "Problema con el pago"]


def choice(rng, items, p=None):
    return items[rng.choice(len(items), p=p)]


def make_customers(rng, n):
    n_old = int(n * .22)
    old = pd.Timestamp("2024-07-01") + pd.to_timedelta(rng.integers(0, 184, n_old), "D")
    span = (pd.Timestamp("2026-08-31") - START).days + 1
    w = 1 + .6 * np.arange(span) / span
    new = START + pd.to_timedelta(rng.choice(span, n - n_old, p=w / w.sum()), "D")
    signup = np.sort(np.concatenate([old.values, new.values]))
    rows = []
    for i, sd in enumerate(pd.to_datetime(signup)):
        seg = choice(rng, list(SEG), [v["p"] for v in SEG.values()])
        cfg = SEG[seg]
        ch_w = np.array([.28, .22, .20, .12, .0, .18])
        if sd >= EV_TIKTOK:
            ch_w = np.array([.21, .17, .15, .09, .25, .13])
        channel = choice(rng, ["Instagram Ads", "Google Ads", "Orgánico", "Referido", "TikTok Ads",
                               "Email/Afiliados"], ch_w / ch_w.sum())
        age = int(np.clip(rng.normal(*cfg["age"]), 18, 70)) if rng.random() > .03 else np.nan
        mob = {"Joven tendencia": .82, "Premium": .50}.get(seg, .65)
        dev = choice(rng, ["Móvil", "Escritorio", "Tablet"], [mob, (1 - mob) * .78, (1 - mob) * .22])
        rows.append(dict(
            cliente_id=f"C{i + 1:05d}", fecha_registro=sd.date(), edad=age,
            genero=choice(rng, ["F", "M", "No informa"], [.68, .28, .04]),
            ciudad=choice(rng, CITIES, CITY_P), segmento=seg, canal_adquisicion=channel,
            miembro_fidelidad=int(rng.random() < {"Premium": .6, "Profesional urbano": .35}.get(seg, .25)),
            acepta_newsletter=int(rng.random() < .55), dispositivo_preferido=dev,
            categoria_preferida=choice(rng, CAT_NAMES),
            _never=rng.random() < .12))
    return rows


def simulate(rng, customers):
    months = pd.period_range("2025-01", "2026-09", freq="M")
    orders, sessions, returns, cancels = [], [], [], []
    ids = dict(o=0, s=0, r=0, c=0)

    def nid(k, pre):
        ids[k] += 1
        return f"{pre}{ids[k]:07d}"

    def new_session(c, day, converted, order_id=None):
        dev = c["dispositivo_preferido"] if rng.random() < .8 else choice(rng, list(LOAD_BASE))
        mult = {"Móvil": 1.6, "Tablet": 1.3, "Escritorio": 1.03}[dev] if day >= EV_FRONTEND else 1
        load = int(LOAD_BASE[dev] * mult * rng.lognormal(0, .3))
        start = day + pd.Timedelta(seconds=int(rng.integers(6 * 3600, 24 * 3600)))
        if converted:
            pages, cart, chk = 4 + rng.poisson(8), 1, 1
        else:
            bounce = rng.random() < np.clip(.25 + .00008 * (load - 1500), .1, .7)
            pages = 1 if bounce else 2 + rng.poisson(5)
            cart = int(not bounce and rng.random() < .28 * (1 - min(.5, max(0, load - 1500) / 6000)))
            chk = int(cart and rng.random() < .35)
        dur = int(pages * 40 * rng.lognormal(0, .35)) + 5
        sid = nid("s", "S")
        sessions.append(dict(
            sesion_id=sid, cliente_id=c["cliente_id"], inicio_sesion=start, dispositivo=dev,
            fuente_trafico=choice(rng, TRAFFIC, [.25, .22, .2, .12, .13, .08]),
            paginas_vistas=pages, duracion_seg=dur, tiempo_carga_ms=load,
            uso_buscador=int(rng.random() < .45), agrego_carrito=cart, inicio_checkout=chk,
            compro=int(converted), pedido_id=order_id))
        return sid, min(start + pd.Timedelta(seconds=dur), day + pd.Timedelta(hours=23, minutes=59)), load

    def make_order(c, day):
        cfg = SEG[c["segmento"]]
        oid = nid("o", "P")
        sid, ts, load = new_session(c, day, True, oid)
        cat = c["categoria_preferida"] if rng.random() < .55 else choice(rng, CAT_NAMES)
        n_items = int(min(6, 1 + rng.poisson(.8)))
        gross = sum(round(CATS[cat if j == 0 else choice(rng, CAT_NAMES)] * cfg["price"]
                          * rng.lognormal(0, .25), -2) for j in range(n_items))
        dp = np.array(DISC_P[cfg["disc"]])
        if day.month == 11:
            dp = dp * np.array([.6, .8, 1, 1.4, 1.8, 2])
        disc = int(choice(rng, DISC_LV, dp / dp.sum()))
        if day >= EV_PROMO and disc > 15:
            disc = int(choice(rng, [10, 15]))
        d_amt = round(gross * disc / 100, -2)
        main = c["ciudad"] in MAIN
        ship = 0 if gross - d_amt >= 150000 else (9900 if main else 14900)
        pays = ["Tarjeta de crédito", "PSE", "Tarjeta débito", "Contraentrega"]
        pw = [.38, .27, .17, .18]
        if day >= EV_NEQUI:
            pays, pw = pays + ["Nequi/Daviplata"], [.33, .23, .14, .15, .15]
        pay = choice(rng, pays, pw)
        promised = 2 if main else 4
        actual = promised + int(rng.integers(-1, 2)) + int(rng.random() < .1) * int(rng.integers(1, 4))
        if day >= EV_CARRIER:
            actual += int(round(rng.gamma(1, .5) if main else rng.gamma(2, 1.5)))
        actual = max(1, actual)
        late = actual > promised + 1
        net = gross - d_amt + ship
        o = dict(pedido_id=oid, cliente_id=c["cliente_id"], sesion_id=sid, fecha_pedido=ts,
                 categoria_principal=cat, num_articulos=n_items, monto_bruto=gross, descuento_pct=disc,
                 monto_descuento=d_amt, costo_envio=ship, monto_neto=net, metodo_pago=pay,
                 cupon_usado=int(disc > 0 and rng.random() < .6), dias_entrega_prometidos=promised,
                 dias_entrega_reales=np.nan, fecha_entrega=pd.NaT, estado_pedido="Entregado")
        bad = 0.0
        p_cancel = .025 + (.07 if late else 0) + (.03 if pay == "Contraentrega" else 0)
        if rng.random() < p_cancel:
            if late:
                cd = ts.normalize() + pd.Timedelta(days=int(rng.integers(promised + 1, actual)))
                stage, reason = "En tránsito", ("Demora en el envío" if rng.random() < .65
                                                else choice(rng, CANCEL_REASONS))
            else:
                cd = ts.normalize() + pd.Timedelta(days=int(rng.integers(0, 2)))
                stage, reason = "Antes del envío", choice(rng, CANCEL_REASONS, [.3, .3, .2, .2])
            cd = min(max(cd, ts.normalize()), OBS)
            o["estado_pedido"] = "Cancelado"
            cancels.append(dict(cancelacion_id=nid("c", "X"), pedido_id=oid, cliente_id=c["cliente_id"],
                                fecha_cancelacion=cd.date(), etapa=stage, motivo=reason,
                                reembolsado="No aplica" if pay == "Contraentrega" else "Sí"))
            orders.append(o)
            return o, 1.0
        delivered = ts.normalize() + pd.Timedelta(days=actual)
        if delivered > OBS:
            o["estado_pedido"] = "En tránsito"
            orders.append(o)
            return o, 0.0
        o["dias_entrega_reales"], o["fecha_entrega"] = actual, delivered.date()
        bad += 1.0 if late else 0
        p_ret = .17 + (.03 if c["segmento"] == "Joven tendencia" else 0) \
            + (.06 if cat in ("Vestidos", "Calzado") else 0) + (.05 if late else 0)
        rd = delivered + pd.Timedelta(days=1 + min(29, int(rng.gamma(2, 3))))
        if rng.random() < p_ret and rd <= OBS:
            k = n_items if rng.random() < .6 else int(rng.integers(1, n_items + 1))
            rw = np.array([.38, .18, .14, .14, .08, .30 if late else .08])
            refund = round((gross - d_amt) * k / n_items, -2)
            rdays = int(round(3 + rng.gamma(3, 2.8) if rd >= EV_RETURNS else 1 + rng.gamma(2, 2.2)))
            status, fref = "Reembolsada", rd + pd.Timedelta(days=rdays)
            if rng.random() < .05:
                status, refund, fref, rdays = "Rechazada", 0, pd.NaT, np.nan
                bad += 1.5
            elif fref > OBS:
                status, fref = "En proceso", pd.NaT
            if status != "Rechazada" and rdays > 10:
                bad += 1.2
            returns.append(dict(devolucion_id=nid("r", "D"), pedido_id=oid, cliente_id=c["cliente_id"],
                                fecha_devolucion=rd.date(), articulos_devueltos=k,
                                motivo=choice(rng, RET_REASONS, rw / rw.sum()), monto_reembolso=refund,
                                estado_devolucion=status, dias_reembolso=rdays,
                                fecha_reembolso=None if pd.isna(fref) else fref.date()))
            o["estado_pedido"] = "Devuelto" if k == n_items else "Devuelto parcial"
        orders.append(o)
        return o, bad

    for c in customers:
        cfg, sd = SEG[c["segmento"]], pd.Timestamp(c["fecha_registro"])
        alive, bad, bought, n_ok = True, 0.0, sd < START, (int(rng.integers(1, 6)) if sd < START else 0)
        for m in months:
            ms, me = m.start_time, min(m.end_time.normalize(), OBS)
            if me < sd:
                continue
            lo = max(ms, sd)
            frac = ((me - lo).days + 1) / m.days_in_month

            def rand_day():
                return lo + pd.Timedelta(days=int(rng.integers(0, (me - lo).days + 1)))

            month_bad, loads = 0.0, []
            if alive and not c["_never"]:
                if not bought:
                    p = (.75 if lo == sd else .25) * frac
                else:
                    p = cfg["buy"] * SEASON[m.month] * (1.15 if c["miembro_fidelidad"] else 1) * frac
                    if ms >= EV_FRONTEND and c["dispositivo_preferido"] == "Móvil":
                        p *= .90
                if rng.random() < p:
                    for _ in range(1 + int(rng.random() < .12)):
                        o, b = make_order(c, rand_day())
                        month_bad += b
                        if o["estado_pedido"] != "Cancelado":
                            n_ok += 1
                    bought = True
            lam = (cfg["act"] * 1.3 if (alive and bought) else (.6 if alive else .2)) * frac
            for _ in range(rng.poisson(lam)):
                loads.append(new_session(c, rand_day(), False)[2])
            # dinámica de abandono
            bad = bad * .5 + month_bad
            if alive and bought:
                h = cfg["haz"] * np.exp(.6 * bad)
                h *= .65 if c["miembro_fidelidad"] else 1
                h *= 1.5 if n_ok <= 1 else 1
                h *= 1.35 if c["canal_adquisicion"] == "TikTok Ads" else 1
                h *= 1.25 if (ms >= EV_PROMO and c["segmento"] == "Cazador de ofertas") else 1
                h *= 1.2 if (loads and np.mean(loads) > 2600) else 1
                alive = rng.random() >= min(h, .6)
            elif alive and not bought and not (lo == sd):
                alive = rng.random() >= .2
            elif not alive and bought and rng.random() < .015:
                alive, bad = True, 0.0
    return [pd.DataFrame(x) for x in (orders, sessions, returns, cancels)]


def customer_summary(cust, orders, returns):
    ok = orders[orders.estado_pedido != "Cancelado"]
    g = ok.groupby("cliente_id")
    s = pd.DataFrame({
        "fecha_primera_compra": g.fecha_pedido.min().dt.date,
        "fecha_ultima_compra": g.fecha_pedido.max().dt.date,
        "total_pedidos": g.size(),
        "ingresos_brutos": g.monto_neto.sum()})
    s["total_reembolsado"] = returns.groupby("cliente_id").monto_reembolso.sum()
    s["total_cancelaciones"] = orders[orders.estado_pedido == "Cancelado"].groupby("cliente_id").size()
    s["total_devoluciones"] = returns.groupby("cliente_id").size()
    cust = cust.merge(s, left_on="cliente_id", right_index=True, how="left")
    for col in ["total_pedidos", "ingresos_brutos", "total_reembolsado", "total_cancelaciones",
                "total_devoluciones"]:
        cust[col] = cust[col].fillna(0).astype(int)
    cust.loc[cust.total_cancelaciones.isna(), "total_cancelaciones"] = 0
    cust["ingresos_netos"] = cust.ingresos_brutos - cust.total_reembolsado
    end = CUTOFF + pd.Timedelta(days=1)
    base = ok[(ok.fecha_pedido >= ELIG_FROM) & (ok.fecha_pedido < end)].cliente_id.unique()
    after = ok[ok.fecha_pedido >= end].cliente_id.unique()
    elig, act = cust.cliente_id.isin(base), cust.cliente_id.isin(after)
    cust["churn"] = np.where(elig, (~act).astype(int), np.nan)
    cust["estado_retencion"] = np.select(
        [elig & act, elig & ~act, ~elig & act, cust.total_pedidos > 0],
        ["Activo", "Abandonó", "Nuevo o reactivado", "Inactivo previo"], "Sin compras")
    cust["churn"] = cust["churn"].astype("Int64")
    return cust


def build_features(cust, orders, sessions, returns):
    end = CUTOFF + pd.Timedelta(days=1)
    w90 = end - pd.Timedelta(days=90)
    o = orders[orders.fecha_pedido < end]
    ok = o[o.estado_pedido != "Cancelado"]
    r = returns[pd.to_datetime(returns.fecha_devolucion) < end]
    s = sessions[(sessions.inicio_sesion < end) & (sessions.inicio_sesion >= w90)]
    f = cust.loc[cust.churn.notna(), ["cliente_id", "fecha_registro", "edad", "genero", "ciudad",
                                      "segmento", "canal_adquisicion", "miembro_fidelidad",
                                      "acepta_newsletter", "dispositivo_preferido", "churn"]].copy()
    f["antiguedad_dias"] = (end - pd.to_datetime(f.fecha_registro)).dt.days
    g = ok.groupby("cliente_id")
    delivered = ok[ok.dias_entrega_reales.notna()]
    late = (delivered.dias_entrega_reales > delivered.dias_entrega_prometidos + 1)
    feats = pd.DataFrame({
        "recencia_dias": (end - g.fecha_pedido.max().dt.normalize()).dt.days,
        "frecuencia_pedidos": g.size(),
        "valor_monetario": g.monto_neto.sum(),
        "ticket_promedio": g.monto_neto.mean().round(0),
        "descuento_promedio_pct": g.descuento_pct.mean().round(2),
        "pedidos_ult_90d": ok[ok.fecha_pedido >= w90].groupby("cliente_id").size(),
        "dias_entrega_promedio": delivered.groupby("cliente_id").dias_entrega_reales.mean().round(2),
        "pct_entregas_tardias": late.groupby(delivered.cliente_id).mean().round(3),
        "num_cancelaciones": o[o.estado_pedido == "Cancelado"].groupby("cliente_id").size(),
        "num_devoluciones": r.groupby("cliente_id").size(),
        "dias_reembolso_promedio": r.groupby("cliente_id").dias_reembolso.mean().round(2),
        "sesiones_ult_90d": s.groupby("cliente_id").size(),
        "tiempo_carga_ms_prom_90d": s.groupby("cliente_id").tiempo_carga_ms.mean().round(0),
        "carritos_abandonados_90d": s[(s.agrego_carrito == 1) & (s.compro == 0)].groupby("cliente_id").size(),
    })
    f = f.merge(feats, left_on="cliente_id", right_index=True, how="left")
    zero = ["pedidos_ult_90d", "num_cancelaciones", "num_devoluciones", "sesiones_ult_90d",
            "carritos_abandonados_90d"]
    f[zero] = f[zero].fillna(0).astype(int)
    f["tasa_devolucion"] = (f.num_devoluciones / f.frecuencia_pedidos).round(3)
    cols = [c for c in f.columns if c != "churn"] + ["churn"]
    return f[cols].drop(columns="fecha_registro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clientes", type=int, default=8000)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cust_rows = make_customers(rng, a.clientes)
    orders, sessions, returns, cancels = simulate(rng, cust_rows)
    orders = orders.sort_values("fecha_pedido").reset_index(drop=True)
    sessions = sessions.sort_values("inicio_sesion").reset_index(drop=True)
    cust = pd.DataFrame(cust_rows).drop(columns="_never")
    cust = customer_summary(cust, orders, returns)
    feats = build_features(cust, orders, sessions, returns)
    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    for df, name in [(cust, "clientes"), (orders, "transacciones"), (sessions, "navegacion_web"),
                     (returns, "devoluciones"), (cancels, "cancelaciones")]:
        for col in df.select_dtypes("datetime").columns:
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(RAW / f"{name}.csv", index=False, encoding="utf-8")
        print(f"✅ {name}.csv: {len(df):,} filas")
    feats.to_csv(PROC / "features_modelo.csv", index=False, encoding="utf-8")
    print(f"✅ features_modelo.csv: {len(feats):,} filas")


if __name__ == "__main__":
    main()
