"""🛍️ Espacio de trabajo inicial del caso E-Commerce X (único módulo específico del caso).

Carga los CSV de la fase 1 con el módulo de ingesta y verifica la cifra de retención
según la definición documentada en README.md · Definiciones.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import ingesta as ig

from .perfil import retencion_periodos

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
CIFRA_REPORTADA = -0.15
DEFINICION = ("Retención trimestral = clientes con un pedido no cancelado en T−1 que vuelven a comprar en T "
              "÷ compradores de T−1 (README · Definiciones). La dirección reporta una caída cercana al 15 % "
              "en 2026-Q2 y Q3 frente al mismo periodo del año anterior.")


def cargar_tabla(nombre: str, raw: Path = RAW) -> pd.DataFrame:
    """Lee y tipa un CSV de la fase 1 reutilizando ingesta (el archivo no se modifica)."""
    r = ig.load_file((raw / f"{nombre}.csv").read_bytes(), f"{nombre}.csv")
    plan, _ = ig.sugerir_plan(r.df)
    plan.acciones = {k: "conservar" for k in ig.ACCIONES}  # el espacio de análisis no excluye registros
    return ig.aplicar(r.df, plan, nombre).limpio


def cargar_caso(raw: Path = RAW, incluir_navegacion: bool = False) -> dict[str, pd.DataFrame]:
    nombres = ["clientes", "transacciones", "devoluciones", "cancelaciones"] + (
        ["navegacion_web"] if incluir_navegacion else [])
    t = {n: cargar_tabla(n, raw) for n in nombres}
    cli = t["clientes"][["cliente_id", "segmento", "ciudad", "canal_adquisicion", "miembro_fidelidad"]]
    t["transacciones_con_cliente"] = ig.unir(t["transacciones"], "cliente_id", cli, "cliente_id")
    return t


def verificar_cifra(transacciones: pd.DataFrame, col_id: str = "cliente_id", col_fecha: str = "fecha_pedido",
                    col_estado: str = "estado_pedido", excluir: str = "Cancelado") -> dict:
    """Recalcula la retención trimestral y compara con la cifra reportada. No asume que se confirme."""
    mask = transacciones[col_estado].ne(excluir) if col_estado in transacciones else None
    ret = retencion_periodos(transacciones, col_id, col_fecha, "Q", mask)
    tasas = dict(zip(ret.periodo, ret.tasa))
    comps = []
    for cur in ret.periodo[-2:]:
        p = pd.Period(cur, "Q")
        for ref, tipo in [(p - 4, "interanual"), (p - 1, "trimestre anterior")]:
            a, b = tasas.get(str(ref)), tasas.get(cur)
            cambio = (b / a - 1) if a else np.nan
            comps.append({"trimestre": cur, "comparado_con": str(ref), "tipo": tipo,
                          "retencion_ref": a, "retencion": b, "cambio_relativo": cambio,
                          "cambio_pp": (b - a) * 100 if a is not None and b is not None else np.nan,
                          "coincide_15pct": bool(abs(cambio - CIFRA_REPORTADA) <= 0.02) if not np.isnan(cambio) else False})
    comp = pd.DataFrame(comps)
    yoy = comp[comp.tipo == "interanual"]
    ok = yoy[yoy.coincide_15pct].trimestre.tolist()
    no = yoy[~yoy.coincide_15pct]
    partes = [f"{r.trimestre} vs {r.comparado_con}: {r.cambio_relativo:+.1%}" for r in yoy.itertuples()]
    if len(ok) == len(yoy):
        veredicto = "Los datos confirman una caída cercana al 15 % en ambos trimestres. " + "; ".join(partes)
    elif ok:
        veredicto = (f"La caída cercana al 15 % solo se confirma en {', '.join(ok)}; en "
                     f"{', '.join(no.trimestre)} es menor. " + "; ".join(partes))
    else:
        veredicto = "Los datos NO confirman una caída del 15 %. " + "; ".join(partes)
    return {"definicion": DEFINICION, "serie": ret, "comparaciones": comp, "veredicto": veredicto}
