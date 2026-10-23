"""🧭 Importador de datos · E-Commerce X (interfaz Streamlit).

Ejecutar:  streamlit run app/importador.py
Toda la lógica vive en el paquete `ingesta`; este archivo solo orquesta la interfaz.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ingesta as ig  # noqa: E402

st.set_page_config(page_title="Importador · E-Commerce X", page_icon="📥", layout="wide")
ss = st.session_state
ss.setdefault("carga", None)        # LoadResult actual
ss.setdefault("carga_id", 0)        # cambia en cada carga para reiniciar widgets
ss.setdefault("resultado", None)
ss.setdefault("tablas", {})         # espacio de trabajo: nombre -> DataFrame limpio
EJEMPLOS = sorted((ROOT / "data" / "samples").glob("*")) + sorted((ROOT / "data" / "raw").glob("*.csv"))


def set_carga(fn, *args, **kw):
    try:
        with st.spinner("Leyendo datos…"):
            ss.carga = fn(*args, **kw)
        ss.carga_id += 1
        ss.resultado = None
    except ig.IngestaError as e:
        ss.carga = None
        st.error(f"⚠️ {e}")
    except Exception as e:  # error inesperado: mensaje comprensible, sin traza
        ss.carga = None
        st.error(f"⚠️ No se pudo leer la fuente ({type(e).__name__}). Revisa el formato del archivo.")


def file_options(data: bytes, name: str, key: str):
    """Opciones según formato y botón de lectura."""
    try:
        fmt = ig.detect_format(name, data)
    except ig.IngestaError as e:
        st.error(f"⚠️ {e}")
        return
    opts = {}
    try:
        if fmt == "delimitado":
            c1, c2, c3 = st.columns(3)
            default = 2 if name.lower().endswith((".tsv", ".tab")) else 0
            d = c1.selectbox("Delimitador", ["Detectar automáticamente", *ig.DELIMITERS], index=default + (default > 0),
                             key=f"{key}_sep")
            opts["sep"] = ig.DELIMITERS.get(d)
            enc = c2.selectbox("Codificación", ["Automática", "utf-8", "cp1252", "latin-1"], key=f"{key}_enc")
            opts["encoding"] = None if enc == "Automática" else enc
            opts["header"] = c3.checkbox("La primera fila es encabezado", True, key=f"{key}_hdr")
        elif fmt.startswith("excel"):
            opts["sheet"] = st.selectbox("Hoja", ig.excel_sheets(data, name), key=f"{key}_sheet")
            opts["header"] = st.checkbox("La primera fila es encabezado", True, key=f"{key}_hdr")
        elif fmt == "json":
            paths = ig.json_tables(data)
            if paths != [""]:
                opts["path"] = st.selectbox("Lista de registros", paths, key=f"{key}_path")
        elif fmt == "sqlite":
            tables = ig.sqlite_tables(data, name)
            if not tables:
                st.warning("La base de datos no tiene tablas.")
                return
            opts["table"] = st.selectbox("Tabla", tables, key=f"{key}_tbl")
    except ig.IngestaError as e:
        st.error(f"⚠️ {e}")
        return
    if st.button("📥 Leer datos", type="primary", key=f"{key}_go"):
        set_carga(ig.load_file, data, name, **opts)


# ---------------- 1. Fuente ----------------
st.title("📥 Importador y preparación de datos")
st.caption("Carga, revisa y limpia datos de E-Commerce X o de cualquier otra empresa. "
           "Los archivos originales nunca se modifican.")

fuente = st.sidebar.radio("Fuente de datos", ["Archivo", "Ejemplos incluidos", "Ingreso manual", "Base de datos"])
st.sidebar.caption("Formatos: CSV, TSV/TXT, Excel (.xlsx, .xls), JSON / JSON Lines, SQLite.")
if ss.tablas:
    st.sidebar.markdown("**Espacio de trabajo**")
    for n, d in ss.tablas.items():
        st.sidebar.write(f"🗂️ {n} · {len(d):,} filas")

st.header("1 · Fuente")
if fuente == "Archivo":
    up = st.file_uploader("Selecciona un archivo", type=[e.strip(".") for e in ig.loaders.SUPPORTED])
    if up:
        file_options(up.getvalue(), up.name, f"up_{up.name}")
elif fuente == "Ejemplos incluidos":
    ej = st.selectbox("Archivo de ejemplo", EJEMPLOS, format_func=lambda p: f"{p.parent.name}/{p.name}")
    if ej:
        file_options(ej.read_bytes(), ej.name, f"ej_{ej.name}")  # solo lectura
elif fuente == "Ingreso manual":
    cols = st.text_input("Nombres de columnas (separados por coma)", "producto, cantidad, precio, fecha")
    names = [c.strip() for c in cols.split(",") if c.strip()]
    if len(set(names)) != len(names):
        st.error("⚠️ Hay nombres de columna repetidos.")
    elif names:
        base = pd.DataFrame({c: pd.Series(dtype="string") for c in names})
        edit = st.data_editor(base, num_rows="dynamic", width="stretch", key=f"manual_{cols}")
        if st.button("📥 Usar estos datos", type="primary"):
            set_carga(ig.load_manual, edit)
else:
    st.info("🔌 " + ig.db_server_support())
    db = st.file_uploader("Archivo SQLite (se abre en modo solo lectura)", type=["sqlite", "sqlite3", "db"])
    if db:
        file_options(db.getvalue(), db.name, f"db_{db.name}")

carga = ss.carga
if carga is None:
    st.stop()

# ---------------- 2. Vista previa ----------------
k = ss.carga_id
st.header("2 · Vista previa")
st.write(f"**{carga.source}** · {len(carga.df):,} registros · {carga.df.shape[1]} columnas · "
         f"opciones: {carga.options or '—'}")
for w in carga.warnings:
    st.warning(w)
st.dataframe(carga.df.head(50), width="stretch")

# ---------------- 3. Columnas y tipos ----------------
st.header("3 · Columnas, nombres y tipos")
plan, esquema = ig.sugerir_plan(carga.df)
st.caption(f"Esquema reconocido: **{esquema}**" if esquema else "Estructura genérica: tipos inferidos del contenido.")
cfg = pd.DataFrame({
    "importar": True, "columna": carga.df.columns, "nuevo_nombre": carga.df.columns,
    "tipo": [plan.tipos[c] for c in carga.df.columns],
    "clave": [c in plan.clave for c in carga.df.columns],
    "obligatoria": [c in plan.requeridas for c in carga.df.columns]})
cfg = st.data_editor(cfg, hide_index=True, width="stretch", key=f"cfg_{k}", disabled=["columna"],
                     column_config={"tipo": st.column_config.SelectboxColumn(options=ig.TIPOS, required=True)})
sel = cfg[cfg.importar]
final = dict(zip(sel.columna, sel.nuevo_nombre.fillna(sel.columna).astype(str).str.strip()))
final = {c: (n or c) for c, n in final.items()}
user_plan = ig.Plan(columnas=list(sel.columna), renombrar={c: n for c, n in final.items() if n != c},
                    tipos={final[c]: t for c, t in zip(sel.columna, sel.tipo)},
                    clave=[final[c] for c in sel.columna[sel.clave]],
                    requeridas=[final[c] for c in sel.columna[sel.obligatoria]])

# ---------------- 4. Diagnóstico ----------------
st.header("4 · Diagnóstico de calidad")
try:
    prep = ig.preparar(carga.df, user_plan)
except ValueError as e:
    st.error(f"⚠️ {e}")
    st.stop()
with st.spinner("Analizando calidad…"):
    issues = ig.diagnose(prep, user_plan.tipos, user_plan.clave, user_plan.requeridas,
                         ig.match_schema(list(prep.columns))[0])
if issues:
    color = {"error": "🔴", "advertencia": "🟠", "info": "🔵"}
    tab = pd.DataFrame([i.as_dict() for i in issues])
    tab.insert(0, "", tab.severidad.map(color))
    st.dataframe(tab, hide_index=True, width="stretch")
else:
    st.success("✅ No se detectaron problemas.")

st.subheader("Qué hacer con los registros problemáticos")
acciones = {}
codes = sorted({i.codigo for i in issues if i.codigo in ig.ACCIONES})
for c in codes:
    default, opciones = ig.ACCIONES[c]
    a, b = st.columns([1, 2])
    acciones[c] = a.selectbox(c, opciones, index=opciones.index(default), key=f"acc_{k}_{c}")
    b.caption(ig.DESCRIPCION_ACCION.get(c, "excluir = sacar el registro al archivo de excluidos · "
                                            "conservar = mantenerlo y anotarlo en _observaciones"))
user_plan.acciones = acciones

if st.button("🧹 Aplicar y generar informe", type="primary"):
    with st.spinner("Limpiando y validando…"):
        ss.resultado = ig.aplicar(carga.df, user_plan, carga.source)

res = ss.resultado
if res is None:
    st.stop()

# ---------------- 5. Resultado ----------------
st.header("5 · Resultado e informe")
inf = res.informe
m = st.columns(5)
m[0].metric("Procesados", f"{inf['registros_leidos']:,}")
m[1].metric("Válidos", f"{inf['registros_validos']:,}")
m[2].metric("Excluidos", f"{inf['registros_excluidos']:,}")
m[3].metric("Errores", inf["errores"])
m[4].metric("Advertencias", inf["advertencias"])
for c in res.cambios:
    st.write("• " + c)
t1, t2 = st.tabs(["Datos limpios", f"Excluidos ({len(res.excluidos)})"])
t1.dataframe(res.limpio.head(200), width="stretch")
t2.dataframe(res.excluidos, width="stretch")

base = Path(carga.source.split(":")[-1]).stem
d = st.columns(5)
d[0].download_button("⬇️ CSV limpio", ig.to_csv_bytes(res.limpio), f"{base}_limpio.csv", "text/csv")
d[1].download_button("⬇️ Excel", ig.to_excel_bytes({"limpio": res.limpio, "excluidos": res.excluidos}),
                     f"{base}_limpio.xlsx")
d[2].download_button("⬇️ Excluidos CSV", ig.to_csv_bytes(res.excluidos), f"{base}_excluidos.csv", "text/csv",
                     disabled=res.excluidos.empty)
d[3].download_button("⬇️ Informe .md", ig.informe_markdown(inf).encode(), f"{base}_informe.md", "text/markdown")
d[4].download_button("⬇️ Informe .json", ig.informe_json(inf), f"{base}_informe.json", "application/json")

c1, c2 = st.columns([2, 1])
nombre = c1.text_input("Nombre de la tabla en el espacio de trabajo", base, key=f"ws_{k}")
if c2.button("➕ Agregar al espacio de trabajo"):
    ss.tablas[nombre] = res.limpio
    st.success(f"Tabla «{nombre}» agregada.")

# ---------------- 6. Relaciones ----------------
st.header("6 · Relacionar tablas")
if len(ss.tablas) < 2:
    st.info("Agrega al menos dos tablas al espacio de trabajo para relacionarlas por claves comunes.")
    st.stop()
sug = ig.sugerir_relaciones(ss.tablas)
if sug:
    st.caption("Relaciones sugeridas (mismo nombre de columna y clave única en la tabla padre):")
    st.dataframe(pd.DataFrame(sug), hide_index=True, width="stretch")
names = list(ss.tablas)
a, b = st.columns(2)
hija = a.selectbox("Tabla hija", names, key="r_h")
fk = a.selectbox("Columna en la hija", list(ss.tablas[hija].columns), key="r_fk")
padre = b.selectbox("Tabla padre", [n for n in names if n != hija], key="r_p")
pk = b.selectbox("Columna clave en el padre", list(ss.tablas[padre].columns), key="r_pk")
v = ig.verificar(ss.tablas[hija], fk, ss.tablas[padre], pk)
st.json(v, expanded=False)
if v["huerfanos"]:
    st.warning(f"{v['huerfanos']} registro(s) de «{hija}» no encuentran su clave en «{padre}».")
tipo = st.radio("Tipo de unión", ["left", "inner"], horizontal=True,
                format_func=lambda x: {"left": "Conservar todos los de la hija", "inner": "Solo coincidencias"}[x])
try:
    unida = ig.unir(ss.tablas[hija], fk, ss.tablas[padre], pk, tipo)
    st.dataframe(unida.head(100), width="stretch")
    st.download_button("⬇️ Descargar tabla unida", ig.to_csv_bytes(unida), f"{hija}_{padre}.csv", "text/csv")
    if st.button("➕ Agregar la unión al espacio de trabajo (para analizarla)"):
        ss.tablas[f"{hija}_con_{padre}"] = unida
        st.success(f"Tabla «{hija}_con_{padre}» agregada.")
except ValueError as e:
    st.error(f"⚠️ {e}")
