"""🧪 Pruebas del módulo de análisis: cálculos con resultados conocidos, filtros, KPIs y consultas."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analisis as an  # noqa: E402
from analisis import ia  # noqa: E402
from analisis.filtros import Filtro  # noqa: E402
from analisis.kpis import KPI, Condicion, Metrica  # noqa: E402


@pytest.fixture
def df():
    # Datos de resultado conocido (no E-Commerce X): 6 registros
    return pd.DataFrame({
        "pedido_id": ["a", "b", "c", "d", "e", "f"],
        "fecha": pd.to_datetime(["2026-01-05", "2026-01-20", "2026-02-03", "2026-02-25", "2026-03-10", None]),
        "zona": ["Norte", "Norte", "Sur", "Sur", "Sur", None],
        "importe": [10.0, 20.0, 30.0, 40.0, np.nan, 60.0],
        "devuelto": pd.array([1, 0, 1, 1, 0, pd.NA], dtype="Int64"),
        "nota": ["x1", "x2", "x3", "x4", "x5", "x6"],
    })


# ---------- perfil y descriptivos ----------
def test_tipos_columnas(df):
    assert an.perfil(df) == {"pedido_id": "id", "fecha": "fecha", "zona": "categorica", "importe": "numerica",
                             "devuelto": "booleano", "nota": "texto"}


def test_resumen_numerico_conocido(df):
    r = an.resumen_numerico(df, ["importe"]).iloc[0]
    assert (r.suma, r.promedio, r.mediana, r.faltantes, r.con_valor) == (160, 32, 30, 1, 5)


def test_distribuciones(df):
    d = an.distribucion(df, "zona")
    assert dict(zip(d.zona, d.registros)) == {"Sur": 3, "Norte": 2, "(vacío)": 1}
    h = an.distribucion(df, "importe", bins=5)
    assert h.registros.sum() == 5


def test_serie_temporal_y_faltantes(df):
    s = an.serie_temporal(df, "fecha", "M")
    assert list(s.valor) == [2, 2, 1]
    s2 = an.serie_temporal(df, "fecha", "M", df.importe, "suma")
    assert list(s2.valor) == [30, 70, 0] and list(s2.registros) == [2, 2, 0]
    assert an.faltantes(df).set_index("columna").loc["importe", "faltantes"] == 1
    with pytest.raises(ValueError):
        an.serie_temporal(df, "zona", "M")


def test_retencion_periodos_conocida():
    t = pd.DataFrame({"cli": ["A", "B", "C", "A", "C", "D", "A"],
                      "f": pd.to_datetime(["2026-01-10", "2026-02-01", "2026-03-01", "2026-04-02", "2026-05-01",
                                           "2026-06-01", "2026-07-15"])})
    r = an.retencion_periodos(t, "cli", "f", "Q")
    assert list(r.periodo) == ["2026Q2", "2026Q3"]
    assert list(r.retenidos) == [2, 1] and list(r.base_anterior) == [3, 3]


# ---------- filtros ----------
def test_filtros(df):
    f = an.aplicar_filtros(df, [Filtro("fecha", "rango_fecha", ("2026-01-20", "2026-02-25"))])
    assert list(f.pedido_id) == ["b", "c", "d"]  # extremos incluidos
    f = an.aplicar_filtros(df, [Filtro("zona", "valores", ["Sur", "(vacío)"])])
    assert list(f.pedido_id) == ["c", "d", "e", "f"]
    f = an.aplicar_filtros(df, [Filtro("importe", "rango_num", (15, 40))])
    assert list(f.pedido_id) == ["b", "c", "d"]
    assert an.aplicar_filtros(df, [Filtro("zona", "valores", [])]).empty
    with pytest.raises(ValueError, match="no existe"):
        an.aplicar_filtros(df, [Filtro("color", "valores", ["rojo"])])


def test_filtros_actualizan_kpis(df):
    k = KPI("Total", Metrica("suma", "importe"))
    assert an.calcular_kpi(df, k).valor == 160
    sur = an.aplicar_filtros(df, [Filtro("zona", "valores", ["Sur"])])
    assert an.calcular_kpi(sur, k).valor == 70


# ---------- KPIs ----------
def test_kpis_conocidos_y_no_calculables(df):
    tasa = KPI("Tasa dev", Metrica("tasa", "devuelto", evento=Condicion("devuelto", "==", "1")), formato="porcentaje")
    v = an.calcular_kpi(df, tasa)
    assert v.valor == pytest.approx(3 / 5) and an.formatear(v, "porcentaje") == "60.0%"
    assert "pequeña" in v.aviso
    ticket = KPI("Ticket", Metrica("suma", "importe"), Metrica("conteo_distinto", "pedido_id"))
    assert an.calcular_kpi(df, ticket).valor == pytest.approx(160 / 6)
    cond = KPI("Norte", Metrica("suma", "importe", [Condicion("zona", "==", "Norte")]))
    assert an.calcular_kpi(df, cond).valor == 30
    cero = KPI("Div0", Metrica("suma", "importe"), Metrica("suma", "importe", [Condicion("zona", "==", "Este")]))
    r = an.calcular_kpi(df, cero)
    assert not r.calculable and an.formatear(r, "numero") == "No calculable"
    assert "numérica" in an.calcular_kpi(df, KPI("x", Metrica("suma", "zona"))).motivo
    assert not an.calcular_kpi(df.iloc[0:0], KPI("x", Metrica("conteo"))).calculable
    assert "no está disponible" in an.calcular_kpi(df, KPI("x", Metrica("promedio", "precio"))).motivo
    assert not an.calcular_kpi(df, KPI("x", Metrica("tasa", "devuelto"))).calculable  # sin evento


def test_kpis_sugeridos_genericos():
    inv = pd.DataFrame({"SKU": ["A", "B"], "Stock actual": [5, 0], "Precio unitario": [10.0, 20.0],
                        "activo": [True, False]})
    nombres = [k.nombre for k in an.sugerir_kpis(inv)]
    assert "Registros" in nombres and "Total Precio unitario" in nombres and "Tasa activo" in nombres
    tasa = next(k for k in an.sugerir_kpis(inv) if k.nombre == "Tasa activo")
    assert an.calcular_kpi(inv, tasa).valor == 0.5


# ---------- consultas en lenguaje natural ----------
def test_consulta_ranking_tasa(df):
    r = an.responder("¿Qué zona presenta la mayor tasa de devueltos?", df)
    assert r.estado == "ok" and r.interpretacion["grupo"] == "zona"
    t = r.tabla.set_index("zona")
    assert t.loc["Sur", "tasa"] == pytest.approx(2 / 3) and t.loc["Norte", "tasa"] == 0.5
    assert "**Sur**" in r.texto and any("menos de 30" in a for a in r.avisos)
    assert set(r.registros.pedido_id) == {"c", "d"}  # registros que respaldan la respuesta


def test_consulta_evolucion_y_agregado(df):
    r = an.responder("¿Cómo evolucionaron los ingresos por mes?", df)
    assert r.estado == "ok" and r.interpretacion["medida"] == "importe" and list(r.tabla.valor) == [30, 70, 0]
    r = an.responder("¿Cuál es el promedio de importe por zona?", df)
    assert r.estado == "ok" and r.tabla.set_index("zona").loc["Norte", "valor"] == 15


def test_consulta_comparar_grupos(df):
    r = an.responder("¿Qué diferencias existen entre devueltos y no devueltos?", df)
    assert r.estado == "ok" and r.interpretacion["grupo"] == "devuelto"
    fila = r.tabla.set_index("columna").loc["importe"]
    assert fila["promedio devuelto=0"] == 20 and fila["promedio devuelto=1"] == pytest.approx(80 / 3)


def test_consulta_ambigua_y_aclaracion():
    d = pd.DataFrame({"f1": pd.to_datetime(["2026-01-01", "2026-02-01"]),
                      "f2": pd.to_datetime(["2026-01-05", "2026-02-05"]), "monto": [1, 2]})
    r = an.responder("¿Cómo evolucionó el monto?", d)
    assert r.estado == "aclaracion" and r.opciones == {"fecha": ["f1", "f2"]}
    r = an.responder("¿Cómo evolucionó el monto?", d, {"fecha": "f2"})
    assert r.estado == "ok" and r.interpretacion["fecha"] == "f2"


def test_consultas_no_compatibles(df):
    assert an.responder("¿Cuál es el color favorito del gerente?", df).estado == "no_soportada"
    sin_fecha = df.drop(columns="fecha")
    r = an.responder("¿Cómo evolucionaron los ingresos?", sin_fecha)
    assert r.estado == "no_soportada" and "fecha" in r.texto
    assert an.responder("¿Qué zona tiene más devueltos?", df.iloc[0:0]).estado == "no_soportada"
    assert an.responder("", df).estado == "no_soportada"
    r = an.responder("¿Qué zona tiene la mayor tasa de satisfacción?", df)
    assert r.estado == "no_soportada" and "Columnas disponibles" in r.texto


def test_ia_opcional_desactivada(monkeypatch, df):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert ia.disponible()[0] is False
    assert ia.interprete_ia("¿algo?", df).estado == "no_soportada"
    assert ia.validar({"intencion": "agregado", "medida": "inventada"}, df).estado == "no_soportada"
    assert isinstance(ia.validar({"intencion": "agregado", "medida": "importe"}, df), an.Interpretacion)


# ---------- caso E-Commerce X ----------
@pytest.fixture(scope="module")
def caso():
    from analisis.ecommerce_x import cargar_caso
    return cargar_caso()


def test_caso_verifica_cifra_sin_asumirla(caso):
    from analisis.ecommerce_x import verificar_cifra
    v = verificar_cifra(caso["transacciones"])
    c = v["comparaciones"].set_index(["trimestre", "tipo"])
    assert c.loc[("2026Q3", "interanual"), "coincide_15pct"]
    assert not c.loc[("2026Q2", "interanual"), "coincide_15pct"]  # Q2 no confirma el 15 %
    assert "solo se confirma en 2026Q3" in v["veredicto"]


def test_caso_preguntas_ejemplo(caso):
    cli, tx = caso["clientes"], caso["transacciones_con_cliente"]
    r = an.responder(an.EJEMPLOS[0], cli)
    elig = cli[cli.churn.notna()]
    esperado = elig.groupby("segmento").churn.mean().idxmax()
    assert r.estado == "ok" and f"**{esperado}**" in r.texto
    r = an.responder(an.EJEMPLOS[2], tx)
    assert r.estado == "ok" and r.interpretacion["evento"]["columna"] == "estado_pedido"
    t = r.tabla.set_index("categoria_principal")
    manual = tx[tx.estado_pedido.isin(["Devuelto", "Devuelto parcial"])].groupby("categoria_principal").size()
    assert (t.eventos == manual.reindex(t.index)).all()
    r = an.responder(an.EJEMPLOS[1], tx)
    assert r.estado == "aclaracion"  # varias fechas/medidas posibles: pide aclarar
    r = an.responder(an.EJEMPLOS[3], cli)
    assert r.estado == "ok" and r.interpretacion["grupo"] == "churn"


def test_interfaz_analisis():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "analisis.py"), default_timeout=120).run()
    assert not at.exception and any("No hay datos" in i.value for i in at.info)
    at.button[0].click().run()
    assert not at.exception
    assert any(m.label == "Registros" for m in at.metric)
    at.sidebar.selectbox[0].set_value("clientes").run()
    at.text_input(key="q_input").set_value(an.EJEMPLOS[0]).run()
    assert not at.exception and any("segmento" in s.value for s in at.success)
