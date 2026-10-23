"""📊 Espacio de análisis: KPIs, estadísticas, gráficos, filtros y preguntas en lenguaje natural.

Funciona con cualquier tabla del espacio de trabajo (importada en la página principal)
y ofrece un espacio inicial para el caso E-Commerce X.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import analisis as an  # noqa: E402
import ingesta as ig  # noqa: E402
from analisis import graficos as gr  # noqa: E402
from analisis import ia  # noqa: E402
from analisis.ecommerce_x import cargar_caso, verificar_cifra  # noqa: E402

st.set_page_config(page_title="Análisis · E-Commerce X", page_icon="📊", layout="wide")
ss = st.session_state
ss.setdefault("tablas", {})
ss.setdefault("kpis_propios", {})   # dataset -> list[KPI]
ss.setdefault("pregunta", "")
ss.setdefault("aclaraciones", {})


@st.cache_data(show_spinner=False)
def _caso():
    return cargar_caso()


def csv(df):
    return ig.to_csv_bytes(df)


st.title("📊 Análisis exploratorio")
c1, c2 = st.columns([3, 1])
if c2.button("🛍️ Cargar caso E-Commerce X", width="stretch"):
    with st.spinner("Cargando y tipando los CSV de la fase 1…"):
        ss.tablas.update(_caso())
if not ss.tablas:
    st.info("No hay datos en el espacio de trabajo. Importa una tabla en la página principal o carga el caso E-Commerce X.")
    st.stop()

# ---------------- selección y filtros ----------------
sb = st.sidebar
nombre = sb.selectbox("Conjunto de datos", list(ss.tablas))
base = ss.tablas[nombre]
tipos = an.perfil(base)
cols = sb.multiselect("Columnas a analizar", list(tipos), default=list(tipos), key=f"cols_{nombre}")
if not cols:
    st.warning("Selecciona al menos una columna.")
    st.stop()
df0 = base[cols]

sb.markdown("**Filtros**")
candidatos = [c for c in cols if tipos[c] in ("fecha", "categorica", "booleano", "numerica")]
default_f = [c for c in candidatos if tipos[c] == "fecha"][:1] + [c for c in candidatos if tipos[c] == "categorica"][:2]
fcols = sb.multiselect("Filtrar por", candidatos, default=default_f, key=f"fcols_{nombre}")
filtros = []
for c in fcols:
    s, k = df0[c], f"f_{nombre}_{c}"
    if tipos[c] == "fecha" and s.notna().any():
        lo, hi = s.min().date(), s.max().date()
        r = sb.date_input(c, (lo, hi), min_value=lo, max_value=hi, key=k)
        if isinstance(r, (list, tuple)) and len(r) == 2:
            filtros.append(an.Filtro(c, "rango_fecha", r))
    elif tipos[c] == "numerica" and s.notna().any():
        lo, hi = float(s.min()), float(s.max())
        if lo < hi:
            filtros.append(an.Filtro(c, "rango_num", sb.slider(c, lo, hi, (lo, hi), key=k)))
    elif tipos[c] in ("categorica", "booleano"):
        opts = sorted(s.astype(object).where(s.notna(), an.filtros.VACIO).astype(str).unique())
        sel = sb.multiselect(c, opts, default=opts, key=k)
        filtros.append(an.Filtro(c, "valores", sel))
df = an.aplicar_filtros(df0, filtros)
st.caption(f"**{nombre}** · {len(df):,} de {len(df0):,} registros tras filtros · {len(cols)} columnas")
if df.empty:
    st.warning("Los filtros no dejan registros: los indicadores no se pueden calcular.")

tab_k, tab_d, tab_g, tab_q, tab_c = st.tabs(["🎯 KPIs", "📐 Descriptivo", "📈 Gráficos", "💬 Preguntas",
                                              "🛍️ Caso E-Commerce X"])

# ---------------- KPIs ----------------
with tab_k:
    sugeridos = an.sugerir_kpis(df0)
    nombres = [k.nombre for k in sugeridos]
    elegidos = st.multiselect("KPIs sugeridos según las columnas", nombres, default=nombres, key=f"ks_{nombre}")
    kpis = [k for k in sugeridos if k.nombre in elegidos] + ss.kpis_propios.get(nombre, [])
    filas = []
    grid = st.columns(4)
    for i, k in enumerate(kpis):
        v = an.calcular_kpi(df, k)
        txt = an.formatear(v, k.formato)
        grid[i % 4].metric(k.nombre, txt, help=k.descripcion or None)
        if not v.calculable:
            grid[i % 4].caption(f"⚠️ {v.motivo}")
        elif v.aviso:
            grid[i % 4].caption(f"ℹ️ {v.aviso}")
        filas.append({"kpi": k.nombre, "valor": v.valor, "formato": k.formato, "calculable": v.calculable,
                      "motivo": v.motivo, "n": v.n})
    st.download_button("⬇️ Exportar KPIs (CSV)", csv(pd.DataFrame(filas)), f"kpis_{nombre}.csv", "text/csv")

    with st.expander("➕ Definir un KPI propio"):
        with st.form(f"kform_{nombre}"):
            a, b, c = st.columns(3)
            kn = a.text_input("Nombre")
            agg = b.selectbox("Agregación", an.AGREGACIONES)
            col = c.selectbox("Columna", ["(ninguna)"] + cols)
            st.caption("Tasa: % de registros (con valor en la columna) que cumplen el evento.")
            e1, e2, e3 = st.columns(3)
            ecol = e1.selectbox("Evento / condición: columna", ["(ninguna)"] + cols)
            eop = e2.selectbox("Operador", an.OPERADORES)
            eval_ = e3.text_input("Valor")
            uso = st.radio("La condición se usa como", ["evento de la tasa", "filtro previo de la métrica"],
                           horizontal=True)
            d1, d2, d3 = st.columns(3)
            dagg = d1.selectbox("Dividir por (opcional)", ["(nada)"] + [x for x in an.AGREGACIONES if x != "tasa"])
            dcol = d2.selectbox("Columna del denominador", ["(ninguna)"] + cols)
            fmt = d3.selectbox("Formato", an.FORMATOS)
            if st.form_submit_button("Guardar KPI"):
                cond = an.Condicion(ecol, eop, eval_) if ecol != "(ninguna)" else None
                if not kn.strip():
                    st.error("El KPI necesita un nombre.")
                elif agg == "tasa" and (cond is None or uso != "evento de la tasa"):
                    st.error("Una tasa requiere una condición usada como evento.")
                else:
                    num = an.Metrica(agg, None if col == "(ninguna)" else col,
                                     [cond] if cond and uso != "evento de la tasa" else [],
                                     cond if agg == "tasa" else None)
                    den = None if dagg == "(nada)" else an.Metrica(dagg, None if dcol == "(ninguna)" else dcol)
                    ss.kpis_propios.setdefault(nombre, []).append(an.KPI(kn.strip(), num, den, fmt))
                    st.rerun()
        if ss.kpis_propios.get(nombre) and st.button("🗑️ Borrar KPIs propios", key=f"kdel_{nombre}"):
            ss.kpis_propios[nombre] = []
            st.rerun()

# ---------------- Descriptivo ----------------
with tab_d:
    num, cat = an.cols_de(df, "numerica"), an.cols_de(df, "categorica", "booleano")
    st.subheader("Columnas numéricas")
    rn = an.resumen_numerico(df, num)
    st.dataframe(rn, hide_index=True, width="stretch") if not rn.empty else st.caption("Sin columnas numéricas.")
    st.subheader("Columnas categóricas")
    rc = an.resumen_categorico(df, cat)
    st.dataframe(rc, hide_index=True, width="stretch") if not rc.empty else st.caption("Sin columnas categóricas.")
    st.subheader("Valores faltantes")
    fa = an.faltantes(df)
    st.dataframe(fa, hide_index=True, width="stretch",
                 column_config={"pct": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    st.subheader("Distribución")
    dc = st.selectbox("Columna", num + cat, key=f"dist_{nombre}") if num + cat else None
    if dc and not df.empty:
        dd = an.distribucion(df, dc)
        st.dataframe(dd, hide_index=True, width="stretch")
    st.subheader("Evolución temporal")
    fechas = an.cols_de(df, "fecha")
    if not fechas:
        st.caption("El conjunto no tiene columnas de fecha: la evolución temporal no está disponible.")
    elif not df.empty:
        a, b, c, d = st.columns(4)
        fc = a.selectbox("Fecha", fechas, key=f"ef_{nombre}")
        mc = b.selectbox("Medida", ["(conteo de registros)"] + num, key=f"em_{nombre}")
        ag = c.selectbox("Agregación", ["suma", "promedio", "mediana"], key=f"ea_{nombre}",
                         disabled=mc.startswith("("))
        fq = d.selectbox("Frecuencia", list(an.FRECUENCIAS), format_func=an.FRECUENCIAS.get,
                         index=list(an.FRECUENCIAS).index(an.frecuencia_auto(df[fc])), key=f"eq_{nombre}")
        ser = an.serie_temporal(df, fc, fq, None if mc.startswith("(") else df[mc],
                                "conteo" if mc.startswith("(") else ag)
        if ser.empty:
            st.caption("No hay fechas con valor tras los filtros.")
        else:
            st.altair_chart(gr.linea(ser, "inicio", "valor"), width="stretch")
            st.download_button("⬇️ Serie (CSV)", csv(ser), f"serie_{nombre}.csv", "text/csv")

# ---------------- Gráficos ----------------
with tab_g:
    gc = st.selectbox("Columna", cols, key=f"gc_{nombre}")
    kind = tipos[gc]
    opciones = gr.TIPOS_GRAFICO[kind]
    gt = st.radio("Tipo de gráfico", opciones, horizontal=True, key=f"gt_{nombre}_{gc}")
    if df.empty:
        st.caption("Sin registros tras los filtros.")
    elif gt == "tabla":
        tb = an.distribucion(df, gc) if kind != "texto" and kind != "id" else df[[gc]].head(500)
        st.dataframe(tb, hide_index=True, width="stretch")
        st.download_button("⬇️ Tabla (CSV)", csv(tb), f"{gc}.csv", "text/csv")
    elif kind == "fecha":
        fq = an.frecuencia_auto(df[gc])
        ser = an.serie_temporal(df, gc, fq)
        ch = gr.linea(ser, "inicio", "valor") if gt == "linea" else gr.barras(ser, "periodo", "valor", horizontal=False)
        st.altair_chart(ch, width="stretch")
        st.download_button("⬇️ Gráfico (HTML)", gr.a_html(ch), f"{gc}.html", "text/html")
    else:
        ch = gr.grafico_columna(df, gc, gt)
        if ch is None:
            st.caption(f"No se puede dibujar «{gt}» para «{gc}» (sin valores o tipo incompatible).")
        else:
            st.altair_chart(ch, width="stretch")
            st.download_button("⬇️ Gráfico (HTML)", gr.a_html(ch), f"{gc}_{gt}.html", "text/html")

# ---------------- Preguntas ----------------
with tab_q:
    st.caption("Las respuestas se calculan con los datos filtrados del conjunto seleccionado.")
    ex = st.columns(len(an.EJEMPLOS))
    for i, e in enumerate(an.EJEMPLOS):
        if ex[i].button(e, key=f"ex_{i}", width="stretch"):
            ss.pregunta, ss.aclaraciones = e, {}
    q = st.text_input("Pregunta", ss.pregunta, key="q_input")
    if q != ss.pregunta:
        ss.pregunta, ss.aclaraciones = q, {}
    ia_ok, ia_msg = ia.disponible()
    usar_ia = st.toggle("Interpretar con IA (opcional)", value=False, disabled=not ia_ok, help=ia_msg)
    st.caption(("🤖 " if ia_ok else "⚙️ ") + ia_msg + ("" if ia_ok else " Se usa el motor determinista."))
    if ss.pregunta:
        r = an.responder(ss.pregunta, df, ss.aclaraciones, ia.interprete_ia if usar_ia else None)
        if r.estado == "aclaracion":
            st.warning("❓ " + r.texto)
            rol, cands = next(iter(r.opciones.items()))
            elec = st.radio(f"Elige la columna ({rol})", cands, key=f"acl_{rol}_{ss.pregunta}")
            if st.button("Continuar"):
                ss.aclaraciones[rol] = elec
                st.rerun()
        elif r.estado == "no_soportada":
            st.error("🚫 " + r.texto)
        else:
            st.success(r.texto)
            for a in r.avisos:
                st.info(a)
            with st.expander("🔍 Consulta interpretada y cálculo", expanded=True):
                st.write("**Columnas utilizadas:** " + ", ".join(f"`{c}`" for c in r.columnas))
                st.write("**Cálculo:** " + r.calculo)
                st.json(r.interpretacion, expanded=False)
            if r.tabla is not None:
                ch = gr.desde_respuesta(r.tabla, r.grafico)
                if ch is not None:
                    st.altair_chart(ch, width="stretch")
                st.dataframe(r.tabla, hide_index=True, width="stretch")
            if r.registros is not None and not r.registros.empty:
                with st.expander(f"📄 Registros de respaldo (muestra de {len(r.registros)})"):
                    st.dataframe(r.registros, hide_index=True, width="stretch")
            d1, d2, d3 = st.columns(3)
            if r.tabla is not None:
                d1.download_button("⬇️ Resultado (CSV)", csv(r.tabla), "resultado.csv", "text/csv")
                if ch is not None:
                    d2.download_button("⬇️ Gráfico (HTML)", gr.a_html(ch), "resultado.html", "text/html")
            d3.download_button("⬇️ Respuesta (JSON)", json.dumps(
                {"pregunta": ss.pregunta, "respuesta": r.texto, "calculo": r.calculo, "columnas": r.columnas,
                 "interpretacion": r.interpretacion, "avisos": r.avisos}, ensure_ascii=False, indent=2, default=str),
                "respuesta.json", "application/json")

# ---------------- Caso E-Commerce X ----------------
with tab_c:
    if "transacciones" not in ss.tablas:
        st.info("Pulsa «Cargar caso E-Commerce X» para abrir este espacio de trabajo.")
    else:
        st.caption("Se calcula sobre la tabla completa de transacciones (los filtros laterales no aplican aquí).")
        v = verificar_cifra(ss.tablas["transacciones"])
        st.markdown("**Cómo está definida la cifra.** " + v["definicion"])
        (st.success if "solo" not in v["veredicto"] and "NO" not in v["veredicto"] else st.warning)(
            "**Verificación con los datos:** " + v["veredicto"])
        comp = v["comparaciones"]
        st.dataframe(comp, hide_index=True, width="stretch", column_config={
            "retencion_ref": st.column_config.NumberColumn(format="percent"),
            "retencion": st.column_config.NumberColumn(format="percent"),
            "cambio_relativo": st.column_config.NumberColumn(format="percent")})
        st.altair_chart(gr.linea(v["serie"], "periodo", "tasa", "Retención trimestral (calculada)", True),
                        width="stretch")
        st.download_button("⬇️ Serie de retención (CSV)", csv(v["serie"]), "retencion_trimestral.csv", "text/csv")
        st.caption("Pistas para investigar (no conclusiones): selecciona «clientes» o «transacciones_con_cliente» "
                   "y usa las preguntas de ejemplo, los filtros por fecha y segmento y los KPIs.")
