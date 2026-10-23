"""🧪 Pruebas del módulo de ingesta (lógica + interfaz)."""
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ingesta as ig  # noqa: E402

RAW, SAMPLES = ROOT / "data" / "raw", ROOT / "data" / "samples"


def rb(p):
    return Path(p).read_bytes()


def run(path, **plan_kw):
    r = ig.load_file(rb(path), Path(path).name)
    plan, _ = ig.sugerir_plan(r.df)
    for k, v in plan_kw.items():
        setattr(plan, k, v)
    return r, ig.aplicar(r.df, plan, r.source)


# ---------- importación válida ----------
def test_importacion_valida_ecommerce_x():
    r, res = run(RAW / "devoluciones.csv")
    assert ig.match_schema(list(r.df.columns))[0].nombre == "devoluciones"
    inf = res.informe
    assert inf["errores"] == 0 and inf["registros_validos"] == inf["registros_leidos"] == len(r.df)
    assert str(res.limpio.fecha_devolucion.dtype).startswith("datetime")


def test_formatos_tsv_excel_json_sqlite():
    tsv = ig.load_file(rb(SAMPLES / "inventario.tsv"), "inventario.tsv")
    assert tsv.options["delimitador"] == "\t" and tsv.df.shape == (4, 6)
    assert ig.excel_sheets(rb(SAMPLES / "empresa.xlsx"), "empresa.xlsx") == ["Empleados", "Departamentos"]
    xl = ig.load_file(rb(SAMPLES / "empresa.xlsx"), "empresa.xlsx", sheet="Empleados")
    assert xl.df.ingreso.iloc[0] == "2023-02-01"
    js = ig.load_file(rb(SAMPLES / "catalogo.json"), "catalogo.json", path="productos")
    assert {"dimensiones.talla", "etiquetas"} <= set(js.df.columns) and js.df.precio.isna().sum() == 1
    db = rb(SAMPLES / "tienda.sqlite")
    assert ig.sqlite_tables(db, "tienda.sqlite") == ["pedidos", "productos"]
    assert len(ig.load_file(db, "tienda.sqlite", table="productos").df) == 3


# ---------- archivo vacío y errores de formato ----------
@pytest.mark.parametrize("name,msg", [("vacio.csv", "vacío"), ("solo_encabezados.csv", "no hay registros")])
def test_archivo_vacio(name, msg):
    with pytest.raises(ig.IngestaError, match=msg):
        ig.load_file(rb(SAMPLES / name), name)


def test_formato_no_soportado_y_disfrazado():
    with pytest.raises(ig.IngestaError, match="no soportado"):
        ig.load_file(b"a,b\n1,2", "datos.parquet")
    with pytest.raises(ig.IngestaError, match="no es un Excel"):
        ig.load_file(b"a,b\n1,2", "falso.xlsx")
    with pytest.raises(ig.IngestaError, match="JSON inválido"):
        ig.load_file(b"{rota", "x.json")
    assert "no están habilitadas" in ig.db_server_support()


def test_sqlite_rechaza_tabla_no_listada():
    with pytest.raises(ig.IngestaError, match="no existe"):
        ig.load_sqlite(rb(SAMPLES / "tienda.sqlite"), 'productos"; DROP TABLE productos; --', "tienda.sqlite")


# ---------- datos con errores ----------
def test_datos_con_errores_detectados_y_tratados():
    r, res = run(SAMPLES / "ventas_con_errores.csv", clave=["factura"], requeridas=["factura", "fecha"],
                 acciones={"variantes_categoria": "corregir", "tipo_invalido": "conservar"})
    codes = {p["codigo"] for p in res.informe["problemas"]}
    assert {"nulo_requerido", "fecha_invalida", "tipo_invalido", "fila_duplicada", "clave_duplicada",
            "espacios", "variantes_categoria", "formula", "negativo", "fecha_fuera_rango"} <= codes
    assert res.informe["registros_leidos"] == 10
    assert res.informe["registros_validos"] + res.informe["registros_excluidos"] == 10
    assert set(res.excluidos._registro) == {5, 6, 9, 10}
    assert res.limpio.ciudad.isin(["Bogotá", "Medellín", "Cali"]).all()
    assert res.limpio.loc[3, "_observaciones"] == "monto inválido: 'abc'"  # conservado y anotado
    assert pd.isna(res.limpio.loc[2, "fecha"])  # fecha imposible no se inventa
    assert res.limpio.loc[1, "monto"] == 85500.5


def test_original_no_se_modifica():
    path = SAMPLES / "ventas_con_errores.csv"
    before = hashlib.sha256(rb(path)).hexdigest()
    r, res = run(path)
    snapshot = r.df.copy()
    ig.aplicar(r.df, ig.sugerir_plan(r.df)[0])
    assert hashlib.sha256(rb(path)).hexdigest() == before
    pd.testing.assert_frame_equal(r.df, snapshot)


# ---------- columnas desconocidas ----------
def test_columnas_desconocidas_y_faltantes():
    r, res = run(SAMPLES / "clientes_columnas_extra.csv")
    probs = res.informe["problemas"]
    unknown = {p["columna"] for p in probs if p["codigo"] == "columna_desconocida"}
    missing = {p["columna"] for p in probs if p["codigo"] == "columna_faltante"}
    assert {"puntos_app", "nivel_vip"} <= unknown
    assert "edad" in missing and "edad" not in res.limpio.columns  # no se inventa
    assert any(p["codigo"] == "fecha_invalida" for p in probs)  # 2025-13-01


def test_estructura_generica_sin_esquema():
    r, res = run(SAMPLES / "inventario.tsv")
    assert res.informe["esquema_detectado"] == "genérico"
    assert res.informe["columnas"]["Stock actual"] == "entero"
    assert res.informe["columnas"]["Última revisión"] == "fecha"


# ---------- datos manuales ----------
def test_datos_manuales():
    df = pd.DataFrame({"producto": ["Camisa", "Jean", None, "  "], "cantidad": ["2", "x", None, None],
                       "fecha": ["2026-10-01", "2026-10-02", None, None]})
    r = ig.load_manual(df)
    assert len(r.df) == 2  # filas totalmente vacías se descartan, no se rellenan
    plan, _ = ig.sugerir_plan(r.df)
    plan.tipos["cantidad"] = "entero"
    res = ig.aplicar(r.df, plan, "manual")
    assert any(p["codigo"] == "tipo_invalido" for p in res.informe["problemas"])
    with pytest.raises(ig.IngestaError, match="no tiene registros"):
        ig.load_manual(pd.DataFrame({"a": [None, ""]}))


# ---------- exportación ----------
def test_exportacion_csv_excel_informe():
    r, res = run(SAMPLES / "ventas_con_errores.csv", clave=["factura"])
    csv = ig.to_csv_bytes(res.limpio).decode("utf-8-sig")
    assert "'=HYPERLINK" in csv  # fórmula neutralizada
    back = pd.read_csv(io.StringIO(csv))
    assert len(back) == len(res.limpio)
    xls = pd.read_excel(io.BytesIO(ig.to_excel_bytes({"limpio": res.limpio, "excluidos": res.excluidos})),
                        sheet_name=None)
    assert set(xls) == {"limpio", "excluidos"} and len(xls["limpio"]) == len(res.limpio)
    assert not str(xls["limpio"].cliente.dropna().iloc[-2]).startswith("=")
    inf = json.loads(ig.informe_json(res.informe))
    assert inf["registros_leidos"] == 10
    assert "# Informe de calidad" in ig.informe_markdown(res.informe)


# ---------- relaciones ----------
def test_relaciones_y_union():
    emp = ig.load_file(rb(SAMPLES / "empresa.xlsx"), "empresa.xlsx", sheet="Empleados").df
    dep = ig.load_file(rb(SAMPLES / "empresa.xlsx"), "empresa.xlsx", sheet="Departamentos").df
    sug = ig.sugerir_relaciones({"emp": emp, "dep": dep})
    assert sug[0]["hija"] == "emp" and sug[0]["columna_padre"] == "depto_id"
    v = ig.verificar(emp, "depto_id", dep, "depto_id")
    assert v["huerfanos"] == 1 and v["ejemplos_huerfanos"] == ["D9"]
    assert len(ig.unir(emp, "depto_id", dep, "depto_id")) == 4
    assert len(ig.unir(emp, "depto_id", dep, "depto_id", "inner")) == 3
    with pytest.raises(ValueError, match="repetidos"):
        ig.unir(dep, "depto_id", emp, "depto_id")


# ---------- interfaz ----------
def test_interfaz_flujo_con_ejemplo():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "importador.py"), default_timeout=60).run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Ejemplos incluidos").run()
    sb = at.selectbox[0]
    sb.set_value(next(o for o in sb.options if "ventas_con_errores" in str(o))).run()
    at.button(key=next(b.key for b in at.button if b.key and b.key.endswith("_go"))).click().run()
    assert not at.exception and not at.error
    at.button[-1].click().run()  # aplicar y generar informe
    assert not at.exception
    assert any(m.label == "Procesados" and m.value == "10" for m in at.metric)


def test_interfaz_muestra_error_archivo_vacio():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "importador.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Ejemplos incluidos").run()
    sb = at.selectbox[0]
    sb.set_value(next(o for o in sb.options if str(o).endswith("vacio.csv"))).run()
    at.button(key=next(b.key for b in at.button if b.key and b.key.endswith("_go"))).click().run()
    assert not at.exception
    assert any("vacío" in e.value for e in at.error)
