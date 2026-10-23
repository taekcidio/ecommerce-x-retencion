"""🛍️ Dashboard de E-Commerce X: retención, churn, ingresos, segmentos, factores, modelo y recomendaciones.

Todas las cifras salen de analisis.tablero (fórmulas únicas) y analisis.modelo; nada está escrito a mano.
"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ingesta as ig  # noqa: E402
from analisis import modelo as md  # noqa: E402
from analisis import tablero as tb  # noqa: E402
from analisis.ecommerce_x import cargar_caso  # noqa: E402
from analisis.kpis import formatear  # noqa: E402

st.set_page_config(page_title="Dashboard · E-Commerce X", page_icon="🛍️", layout="wide")

C = {"primario": "#4F46E5", "ok": "#0D9488", "alerta": "#E11D48", "neutro": "#94A3B8", "suave": "#C7D2FE"}
EVENTOS = pd.DataFrame({"fecha": pd.to_datetime(["2026-02-01", "2026-03-01", "2026-04-01", "2026-04-15",
                                                 "2026-05-01", "2026-06-01"]),
                        "evento": ["Pago Nequi", "TikTok Ads", "Nueva transportadora", "Nuevo sitio web",
                                   "Nuevo operador devoluciones", "Recorte promociones"]})

st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 1280px;}}
.kpi {{border: 1px solid rgba(148,163,184,.35); border-radius: 14px; padding: 14px 16px; height: 100%;
       background: rgba(148,163,184,.06);}}
.kpi .t {{font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; opacity: .7;}}
.kpi .v {{font-size: 1.65rem; font-weight: 650; margin: 2px 0;}}
.kpi .d {{font-size: .8rem; opacity: .8;}}
.kpi .neg {{color: {C['alerta']}; font-weight: 600;}} .kpi .pos {{color: {C['ok']}; font-weight: 600;}}
.reco {{border-left: 4px solid {C['primario']}; border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;
        background: rgba(79,70,229,.06);}}
.reco h4 {{margin: 0 0 6px 0; font-size: 1.02rem;}}
.nota {{font-size: .8rem; opacity: .7;}}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Cargando y validando datos de E-Commerce X…")
def datos():
    t = cargar_caso(incluir_navegacion=True)
    t["features"] = md.cargar_features()
    return t


@st.cache_resource(show_spinner="Entrenando el modelo predictivo…")
def modelo(_feat, n):
    return md.entrenar(_feat)


def kpi_card(col, titulo, valor, detalle="", clase=""):
    col.markdown(f'<div class="kpi"><div class="t">{titulo}</div><div class="v">{valor}</div>'
                 f'<div class="d {clase}">{detalle}</div></div>', unsafe_allow_html=True)


def csv(df):
    return ig.to_csv_bytes(df)


t = datos()
tx_all = t["transacciones_con_cliente"]

# ---------------- filtros ----------------
sb = st.sidebar
sb.header("Filtros")
fmin, fmax = tx_all.fecha_pedido.min().date(), tx_all.fecha_pedido.max().date()
rango = sb.date_input("Fecha del pedido", (fmin, fmax), min_value=fmin, max_value=fmax)
desde, hasta = (rango if isinstance(rango, (list, tuple)) and len(rango) == 2 else (fmin, fmax))
atrib = {}
for col, label in tb.DIMENSIONES.items():
    vals = sorted(t["clientes"][col].dropna().unique(), key=str)
    fmt = (lambda v: "Sí" if v in (1, True) else "No") if col == "miembro_fidelidad" else str
    sel = sb.multiselect(label, vals, default=vals, format_func=fmt)
    if len(sel) < len(vals):
        atrib[col] = sel
sb.caption("El rango de fechas afecta ingresos, pedidos y tendencias. La retención usa trimestres completos "
           "dentro del rango; el churn usa su ventana fija (jul–sep 2026). Los atributos filtran todo.")
f = tb.Filtros(pd.Timestamp(desde), pd.Timestamp(hasta), atrib)
d = tb.filtrar(t, f)

# ---------------- encabezado y KPIs ----------------
st.title("🛍️ E-Commerce X · Retención de clientes")
st.caption(f"Datos sintéticos validados · pedidos del {desde:%d-%m-%Y} al {hasta:%d-%m-%Y} · "
           f"{len(d['clientes']):,} clientes en el filtro")
k = tb.kpis(d, f)
ri = k["retencion_info"]
cols = st.columns(6)
if k["retencion"].calculable:
    cambio = ri.get("cambio_relativo")
    det = (f"{ri['periodo']} · {cambio:+.1%} vs {ri['ref_periodo']}" if cambio == cambio and cambio is not None
           else f"{ri['periodo']} · sin trimestre de referencia")
    kpi_card(cols[0], "Retención trimestral", formatear(k["retencion"], "porcentaje"), det,
             "neg" if cambio == cambio and cambio is not None and cambio < 0 else "pos")
else:
    kpi_card(cols[0], "Retención trimestral", "No calculable", k["retencion"].motivo)
kpi_card(cols[1], "Churn jul–sep 2026", formatear(k["churn"], "porcentaje"),
         f"{k['churn'].n:,} clientes elegibles" if k["churn"].calculable else k["churn"].motivo)
kpi_card(cols[2], "Ingresos netos", formatear(k["ingresos_netos"], "moneda"),
         f"brutos {formatear(k['ingresos_brutos'], 'moneda')}")
kpi_card(cols[3], "Compradores", formatear(k["compradores"], "numero"), f"{formatear(k['pedidos'], 'numero')} pedidos")
kpi_card(cols[4], "Ticket promedio", formatear(k["ticket_promedio"], "moneda"), "por pedido válido")
kpi_card(cols[5], "Tasa de devolución", formatear(k["tasa_devolucion"], "porcentaje"),
         f"cancelación {formatear(k['tasa_cancelacion'], 'porcentaje')}")

tabs = st.tabs(["📈 Tendencias", "👥 Segmentación", "🔍 Factores de abandono", "🤖 Modelo predictivo",
                "💡 Recomendaciones", "📘 Metodología"])

# ---------------- tendencias ----------------
with tabs[0]:
    tr = tb.tendencias(d)
    ret = tb.retencion(d["tx_atributos"])
    if tr.empty:
        st.info("No hay pedidos con los filtros actuales.")
    else:
        metricas = {"ingresos_netos": ("Ingresos netos", ",.0f"), "pedidos": ("Pedidos", ",.0f"),
                    "compradores": ("Compradores", ",.0f"), "ticket_promedio": ("Ticket promedio", ",.0f"),
                    "tasa_devolucion": ("Tasa de devolución", ".1%"),
                    "pct_entregas_tardias": ("% entregas tardías", ".1%"),
                    "tasa_cancelacion": ("Tasa de cancelación", ".1%")}
        a, b = st.columns([2, 1])
        m = a.selectbox("Indicador mensual", list(metricas), format_func=lambda x: metricas[x][0])
        ver_ev = b.toggle("Mostrar eventos operativos", True)
        base = alt.Chart(tr).mark_line(point=True, color=C["primario"]).encode(
            x=alt.X("inicio:T", title=None), y=alt.Y(f"{m}:Q", title=metricas[m][0], axis=alt.Axis(format=metricas[m][1])),
            tooltip=["periodo", alt.Tooltip(f"{m}:Q", format=metricas[m][1])])
        ev = EVENTOS[(EVENTOS.fecha >= tr.inicio.min()) & (EVENTOS.fecha <= tr.inicio.max() + pd.offsets.MonthEnd())]
        capas = base
        if ver_ev and not ev.empty:
            capas = base + alt.Chart(ev).mark_rule(strokeDash=[4, 4], color=C["neutro"]).encode(
                x="fecha:T", tooltip=["evento", alt.Tooltip("fecha:T", format="%d-%m-%Y")])
        st.altair_chart(capas.properties(height=320), width="stretch")
        st.caption("Líneas punteadas: eventos operativos del negocio (README · contexto). Indican cuándo ocurrieron, "
                   "no que hayan causado cambios.")
        st.subheader("Retención trimestral")
        if ret.empty:
            st.caption("No hay trimestres consecutivos con compras para estos filtros.")
        else:
            rc = alt.Chart(ret).mark_line(point=True, color=C["ok"]).encode(
                x=alt.X("periodo:N", title=None), y=alt.Y("tasa:Q", axis=alt.Axis(format="%"), title="Retención",
                                                          scale=alt.Scale(zero=False)),
                tooltip=["periodo", alt.Tooltip("tasa:Q", format=".1%"), "base_anterior", "retenidos"])
            st.altair_chart(rc.properties(height=260), width="stretch")
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Tendencias mensuales (CSV)", csv(tr), "tendencias.csv", "text/csv")
        c2.download_button("⬇️ Retención trimestral (CSV)", csv(ret), "retencion.csv", "text/csv")

# ---------------- segmentación ----------------
with tabs[1]:
    dim = st.selectbox("Dimensión", list(tb.DIMENSIONES), format_func=tb.DIMENSIONES.get)
    seg = tb.segmentacion(d, dim)
    if seg.empty:
        st.info("Sin datos para segmentar con los filtros actuales.")
    else:
        churn_global = k["churn"].valor
        a, b = st.columns(2)
        bars = alt.Chart(seg.dropna(subset=["churn"])).mark_bar(color=C["primario"]).encode(
            y=alt.Y(f"{dim}:N", sort="-x", title=None), x=alt.X("churn:Q", axis=alt.Axis(format="%"), title="Churn"),
            tooltip=[dim, alt.Tooltip("churn:Q", format=".1%"), "clientes_elegibles"])
        if churn_global is not None:
            bars = bars + alt.Chart(pd.DataFrame({"x": [churn_global]})).mark_rule(color=C["alerta"]).encode(x="x:Q")
        a.markdown("**Churn por grupo** · línea roja: churn del filtro")
        a.altair_chart(bars.properties(height=max(200, 34 * len(seg))), width="stretch")
        b.markdown("**Participación en ingresos netos**")
        b.altair_chart(alt.Chart(seg).mark_arc(innerRadius=60).encode(
            theta="ingresos_netos:Q", color=alt.Color(f"{dim}:N", legend=alt.Legend(orient="bottom", columns=2)),
            tooltip=[dim, alt.Tooltip("pct_ingresos:Q", format=".1%"), alt.Tooltip("ingresos_netos:Q", format=",.0f")]
        ).properties(height=300), width="stretch")
        st.dataframe(seg, hide_index=True, width="stretch", column_config={
            "churn": st.column_config.NumberColumn(format="percent"),
            "pct_ingresos": st.column_config.NumberColumn(format="percent"),
            "ingresos_netos": st.column_config.NumberColumn(format="localized"),
            "ingresos_brutos": st.column_config.NumberColumn(format="localized"),
            "ticket_promedio": st.column_config.NumberColumn(format="localized")})
        if seg.muestra_pequena.any():
            st.caption("⚠️ Grupos marcados con muestra pequeña (< 30 clientes elegibles): churn poco robusto.")
        st.download_button("⬇️ Segmentación (CSV)", csv(seg), f"segmentacion_{dim}.csv", "text/csv")

# ---------------- factores ----------------
ret_rango = tb.retencion(d["tx_atributos"])
q = pd.PeriodIndex(ret_rango.periodo, freq="Q") if len(ret_rango) else pd.PeriodIndex([], freq="Q")
trimestres = [str(p) for p in q if pd.Period(desde, "Q") <= p <= pd.Period(hasta, "Q")] or [str(p) for p in q]


def cohorte_de(T):
    return tb.cohorte(d["tx_atributos"], d["clientes"], d["devoluciones_todas"], d["navegacion"], T)


with tabs[2]:
    opciones = [f"Cohorte {T}" for T in reversed(trimestres)] + ["Variables del modelo (al 30-jun-2026)"]
    base_sel = st.selectbox("Base de análisis", opciones)
    if base_sel.startswith("Cohorte"):
        T = base_sel.split()[1]
        base_f = cohorte_de(T)
        st.caption(f"Compradores de {pd.Period(T, 'Q') - 1} y su experiencia en ese trimestre; churn = no volvieron a "
                   f"comprar en {T}. Es la misma base que la retención de {T}.")
    else:
        base_f = d["features"]
        st.caption("Clientes elegibles: variables hasta el 30-jun-2026 y churn en jul–sep 2026 (tabla del modelo).")
    if base_f.empty or base_f.churn.nunique() < 2:
        st.info("No hay suficientes clientes en ambos grupos para comparar con los filtros actuales.")
    else:
        ev = tb.evidencia_factores(base_f)
        st.markdown(f"**{len(base_f):,} clientes · churn {base_f.churn.mean():.1%}** — diferencia de churn entre el "
                    "grupo expuesto y el resto (prueba χ²; significativo si p < 0,05 y diferencia ≥ 3 pp).")
        if not ev.empty:
            ev["estado"] = ev.significativo.map({True: "Significativo", False: "No significativo"})
            st.altair_chart(alt.Chart(ev).mark_bar().encode(
                y=alt.Y("grupo:N", sort="-x", title=None), x=alt.X("diferencia_pp:Q", title="Diferencia de churn (pp)"),
                color=alt.Color("estado:N", scale=alt.Scale(domain=["Significativo", "No significativo"],
                                                             range=[C["alerta"], C["neutro"]]), legend=alt.Legend(orient="bottom")),
                tooltip=["grupo", alt.Tooltip("churn_expuestos:Q", format=".1%"), alt.Tooltip("churn_resto:Q", format=".1%"),
                         alt.Tooltip("p_valor:Q", format=".3g"), "clientes_expuestos"]
            ).properties(height=max(220, 40 * len(ev))), width="stretch")
            st.dataframe(ev.drop(columns=["accion", "metrica", "estado"]), hide_index=True, width="stretch",
                         column_config={"churn_expuestos": st.column_config.NumberColumn(format="percent"),
                                        "churn_resto": st.column_config.NumberColumn(format="percent")})
        with st.expander("Diferencias en todas las variables numéricas (retenidos vs abandono)"):
            num = tb.factores_numericos(base_f)
            st.dataframe(num, hide_index=True, width="stretch")
        with st.expander("Churn por categoría"):
            st.dataframe(tb.factores_categoricos(base_f), hide_index=True, width="stretch",
                         column_config={"churn": st.column_config.NumberColumn(format="percent")})
        st.caption("Son asociaciones observadas, no relaciones causales probadas.")
        st.download_button("⬇️ Evidencia de factores (CSV)", csv(ev), "factores.csv", "text/csv")

# ---------------- modelo ----------------
with tabs[3]:
    res = modelo(t["features"], len(t["features"]))
    mt = res["metricas"]
    st.caption(f"Regresión logística · {mt['n_entrenamiento']:,} clientes de entrenamiento y {mt['n_prueba']:,} de "
               "prueba (25 %, estratificado, semilla 42) · variables al 30-jun-2026 → churn jul–sep 2026.")
    c = st.columns(5)
    kpi_card(c[0], "AUC", f"{mt['auc']:.3f}", "línea base 0,500")
    kpi_card(c[1], "Exactitud", f"{mt['accuracy']:.1%}", f"línea base {mt['accuracy_linea_base']:.1%}")
    kpi_card(c[2], "Precisión", f"{mt['precision']:.1%}", f"umbral {mt['umbral']}")
    kpi_card(c[3], "Recall", f"{mt['recall']:.1%}", "abandonos detectados")
    kpi_card(c[4], "F1", f"{mt['f1']:.3f}", f"Brier {mt['brier']:.3f}")
    calidad = "moderada" if mt["auc"] < 0.7 else "buena" if mt["auc"] < 0.8 else "alta"
    st.info(f"Capacidad de discriminación {calidad} (AUC {mt['auc']:.2f}). Útil para priorizar contactos por riesgo, "
            "no para decisiones individuales automáticas.")
    a, b = st.columns(2)
    a.markdown("**Curva ROC (conjunto de prueba)**")
    roc = alt.Chart(res["roc"]).mark_line(color=C["primario"]).encode(x=alt.X("fpr:Q", title="Falsos positivos"),
                                                                       y=alt.Y("tpr:Q", title="Verdaderos positivos"))
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color=C["neutro"]).encode(x="x", y="y")
    a.altair_chart((roc + diag).properties(height=280), width="stretch")
    b.markdown("**Churn real por decil de riesgo**")
    b.altair_chart(alt.Chart(res["calibracion"]).mark_bar(color=C["ok"]).encode(
        x=alt.X("decil_riesgo:O", title="Decil (10 = mayor riesgo)"), y=alt.Y("churn_real:Q", axis=alt.Axis(format="%")),
        tooltip=["clientes", alt.Tooltip("prob_media:Q", format=".1%"), alt.Tooltip("churn_real:Q", format=".1%")]
    ).properties(height=280), width="stretch")
    a, b = st.columns(2)
    a.markdown("**Variables con mayor peso** (coeficientes estandarizados)")
    imp = res["importancia"].head(12)
    a.altair_chart(alt.Chart(imp).mark_bar().encode(
        y=alt.Y("variable:N", sort=alt.EncodingSortField("coeficiente", op="max", order="descending"), title=None),
        x=alt.X("coeficiente:Q", title="← menos riesgo · más riesgo →"),
        color=alt.condition("datum.coeficiente > 0", alt.value(C["alerta"]), alt.value(C["ok"])),
        tooltip=["variable", alt.Tooltip("coeficiente:Q", format=".3f")]).properties(height=360), width="stretch")
    mx = res["matriz"]
    b.markdown("**Matriz de confusión (prueba)**")
    b.dataframe(pd.DataFrame(mx, index=["Real: retenido", "Real: abandono"],
                             columns=["Pred.: retenido", "Pred.: abandono"]), width="stretch")
    pr = res["pruebas"]
    for col, vals in f.atributos.items():
        if col in pr:
            pr = pr[pr[col].isin(vals)]
    b.markdown(f"**Riesgo medio por segmento** ({len(pr):,} clientes de prueba en el filtro)")
    if not pr.empty:
        rs = pr.groupby("segmento").agg(clientes=("real", "size"), riesgo_medio=("probabilidad", "mean"),
                                        churn_real=("real", "mean")).reset_index()
        b.dataframe(rs, hide_index=True, width="stretch", column_config={
            "riesgo_medio": st.column_config.NumberColumn(format="percent"),
            "churn_real": st.column_config.NumberColumn(format="percent")})
    st.caption("Para puntuar clientes actuales hay que recalcular las variables a la fecha más reciente; aquí se "
               "muestran solo resultados de evaluación.")
    st.download_button("⬇️ Predicciones de prueba (CSV)", csv(pr), "predicciones_prueba.csv", "text/csv")

# ---------------- recomendaciones ----------------
with tabs[4]:
    if not trimestres:
        st.info("No hay trimestres con retención calculable para generar recomendaciones.")
    else:
        T = trimestres[-1]
        ref_T = str(pd.Period(T, "Q") - 4)
        coh = cohorte_de(T)
        ref = cohorte_de(ref_T) if ref_T in set(ret_rango.periodo) else None
        recos = tb.recomendaciones(coh, 3, ref)
        st.caption(f"Basadas en la cohorte de {T} (compradores de {pd.Period(T, 'Q') - 1}). Solo se incluyen factores "
                   "con asociación significativa (p < 0,05) y diferencia ≥ 3 pp, ordenados por abandonos en exceso."
                   + (f" Comparación de exposición con {ref_T}." if ref is not None else ""))
        if not recos:
            st.warning("Con los filtros actuales ningún factor cumple el criterio de evidencia: no se generan recomendaciones.")
        elif len(recos) < 3:
            st.warning(f"Solo {len(recos)} factor(es) cumplen el criterio de evidencia con estos filtros.")
        for i, r in enumerate(recos, 1):
            st.markdown(f'<div class="reco"><h4>{i}. {r["accion"]}</h4><div>{r["evidencia"]}</div>'
                        f'<div class="nota">Métrica de seguimiento: {r["metrica"]}. Validar con una prueba controlada '
                        f'antes de escalar.</div></div>', unsafe_allow_html=True)
        if recos:
            md_txt = "# Recomendaciones · E-Commerce X\n\n" + "\n".join(
                f"{i}. **{r['accion']}**\n   - Evidencia: {r['evidencia']}\n   - Seguimiento: {r['metrica']}"
                for i, r in enumerate(recos, 1))
            st.download_button("⬇️ Recomendaciones (.md)", md_txt.encode(), "recomendaciones.md", "text/markdown")

# ---------------- metodología ----------------
with tabs[5]:
    st.markdown(tb.__doc__.split("\n", 1)[1])
    st.markdown("""
- **Cohorte de T:** compradores válidos de T−1 con su experiencia en ese trimestre (entregas, devoluciones,
  cancelaciones, descuentos y tiempo de carga web). churn = sin pedidos válidos en T, así que 1 − churn = retención de T.
- **Evidencia de factores:** churn del grupo expuesto frente al resto, prueba χ² 2×2, mínimo 30 clientes por grupo.
- **Modelo:** regresión logística con imputación por mediana, estandarización y one-hot; evaluado en un 25 % de
  clientes no usados para entrenar.
- **Fuentes:** `data/raw/*.csv` (validados con `scripts/validate_data.py`) y `data/processed/features_modelo.csv`.
""")
