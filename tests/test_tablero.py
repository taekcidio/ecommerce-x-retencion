"""🧪 Consistencia de cifras del dashboard, modelo y recomendaciones."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analisis import modelo as md  # noqa: E402
from analisis import tablero as tb  # noqa: E402
from analisis.ecommerce_x import cargar_caso, verificar_cifra  # noqa: E402


@pytest.fixture(scope="module")
def t():
    t = cargar_caso(incluir_navegacion=True)
    t["features"] = md.cargar_features()
    return t


@pytest.fixture(scope="module")
def d(t):
    return tb.filtrar(t, tb.Filtros())


def test_kpis_coinciden_con_fuentes(t, d):
    k = tb.kpis(d, tb.Filtros())
    cli = t["clientes"]
    assert k["ingresos_netos"].valor == cli.ingresos_netos.sum()
    assert k["ingresos_brutos"].valor == cli.ingresos_brutos.sum()
    assert k["pedidos"].valor == cli.total_pedidos.sum()
    assert k["churn"].valor == pytest.approx(cli.churn.dropna().mean())
    assert k["churn"].valor == pytest.approx(t["features"].churn.mean())
    v = verificar_cifra(t["transacciones"])
    q3 = v["serie"].set_index("periodo").loc["2026Q3", "tasa"]
    assert k["retencion"].valor == pytest.approx(q3)
    assert k["retencion_info"]["cambio_relativo"] == pytest.approx(
        v["comparaciones"].set_index(["trimestre", "tipo"]).loc[("2026Q3", "interanual"), "cambio_relativo"])


def test_tendencias_suman_kpis(d):
    tr, k = tb.tendencias(d), tb.kpis(d, tb.Filtros())
    assert tr.ingresos_netos.sum() == k["ingresos_netos"].valor
    assert tr.pedidos.sum() == k["pedidos"].valor
    assert len(tr) == 21  # ene-2025 … sep-2026


def test_segmentacion_suma_total(d):
    k = tb.kpis(d, tb.Filtros())
    for dim in tb.DIMENSIONES:
        s = tb.segmentacion(d, dim)
        assert s.ingresos_netos.sum() == k["ingresos_netos"].valor
        assert s.pct_ingresos.sum() == pytest.approx(1)


def test_filtros_y_conjunto_vacio(t):
    f = tb.Filtros(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31"), {"segmento": ["Premium"]})
    d = tb.filtrar(t, f)
    assert d["tx"].segmento.eq("Premium").all() and d["tx"].fecha_pedido.dt.quarter.eq(1).all()
    k = tb.kpis(d, f)
    assert k["retencion_info"]["periodo"] == "2026Q1"
    vacio = tb.Filtros(atributos={"segmento": []})
    dv = tb.filtrar(t, vacio)
    kv = tb.kpis(dv, vacio)
    assert not kv["ingresos_netos"].calculable and not kv["churn"].calculable and not kv["retencion"].calculable
    assert tb.tendencias(dv).empty and tb.recomendaciones(dv["features"]) == []


def test_cohorte_es_base_de_retencion(d):
    ret = tb.retencion(d["tx_atributos"]).set_index("periodo")
    for T in ("2025Q3", "2026Q3"):
        c = tb.cohorte(d["tx_atributos"], d["clientes"], d["devoluciones_todas"], d["navegacion"], T)
        assert len(c) == ret.loc[T, "base_anterior"]
        assert 1 - c.churn.mean() == pytest.approx(ret.loc[T, "tasa"])


def test_recomendaciones_respaldadas(d):
    c = tb.cohorte(d["tx_atributos"], d["clientes"], d["devoluciones_todas"], d["navegacion"], "2026Q3")
    ev = tb.evidencia_factores(c).set_index("factor")
    recos = tb.recomendaciones(c)
    assert 1 <= len(recos) <= 3
    for r in recos:
        e = ev.loc[r["factor"]]
        assert e.significativo and e.p_valor < 0.05 and e.diferencia_pp >= 3
        assert f"{e.churn_expuestos:.1%}" in r["evidencia"]
    e = ev.loc["entregas"]  # recalcular a mano un factor
    exp = c.pct_entregas_tardias > 0
    base = c[c.pct_entregas_tardias.notna()]
    assert e.churn_expuestos == pytest.approx(base.churn[base.pct_entregas_tardias > 0].mean())
    assert exp.sum() == e.clientes_expuestos


def test_modelo_reproducible_y_mejor_que_base(t):
    a, b = md.entrenar(t["features"]), md.entrenar(t["features"])
    ma = a["metricas"]
    assert ma == b["metricas"]
    assert ma["auc"] > 0.55 and ma["accuracy"] >= ma["accuracy_linea_base"]
    assert ma["n_prueba"] + ma["n_entrenamiento"] == len(t["features"])
    assert a["matriz"].sum() == ma["n_prueba"]
    assert a["calibracion"].churn_real.iloc[-1] > a["calibracion"].churn_real.iloc[0]
    with pytest.raises(ValueError):
        md.entrenar(t["features"].head(50))


def test_interfaz_dashboard():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "dashboard.py"), default_timeout=300).run()
    assert not at.exception, at.exception
    assert any("Retención" in m.value for m in at.markdown)
    at.sidebar.multiselect[0].set_value(["Premium"]).run()
    assert not at.exception
    at.sidebar.multiselect[0].set_value([]).run()  # filtro sin datos: no debe romperse
    assert not at.exception
