# 🛍️ E-Commerce X · Caso de retención de clientes

Caso empresarial **ficticio** con datos sintéticos para investigar el abandono de clientes y entrenar un modelo predictivo de churn.

## 📌 Situación

E-Commerce X es una tienda de moda online que vende en Colombia (montos en **COP**). La dirección reporta que la **retención trimestral de clientes cayó cerca de un 15 %** en los dos últimos trimestres (2026-Q2 y Q3) frente al mismo periodo del año anterior. Se desconocen las causas: este repositorio entrega los datos para investigarlas.

> ⚠️ Este README describe hechos y estructura, **no conclusiones**. Las causas del abandono deben salir del análisis.

### Contexto operativo (hechos del negocio, no causas confirmadas)

| Fecha | Evento |
|---|---|
| 2026-02-01 | Se habilita el pago con Nequi/Daviplata |
| 2026-03-01 | Inicia la adquisición de clientes por TikTok Ads |
| 2026-04-01 | Cambio de empresa transportadora |
| 2026-04-15 | Relanzamiento del sitio web (nuevo frontend) |
| 2026-05-01 | Nuevo operador logístico de devoluciones |
| 2026-06-01 | Recorte del presupuesto promocional (descuentos máx. 15 %) |

## 📁 Estructura

```
data/raw/            clientes, transacciones, navegacion_web, devoluciones, cancelaciones
data/processed/      features_modelo.csv (tabla lista para ML)
scripts/generate_data.py   generador reproducible (semilla 42)
scripts/validate_data.py   48 reglas de calidad → reports/validacion.md
ingesta/             módulo de importación, diagnóstico, limpieza y relaciones (sin interfaz)
app/importador.py    interfaz Streamlit del importador
analisis/            BI genérico: perfil, filtros, KPIs, gráficos y consultas en lenguaje natural
app/pages/analisis.py  página de análisis (misma app, segunda página)
app/pages/dashboard.py dashboard ejecutivo de E-Commerce X
analisis/tablero.py  fórmulas únicas de KPIs, tendencias, segmentos, cohortes y recomendaciones
analisis/modelo.py   modelo predictivo de churn (regresión logística)
data/samples/        archivos de prueba genéricos (scripts/make_samples.py)
tests/               pruebas automáticas (pytest)
```

```bash
pip install -r requirements.txt
python scripts/generate_data.py      # --seed 42 --clientes 8000
python scripts/validate_data.py      # sale con código 1 si alguna regla falla
```

## 📐 Definiciones

- **Periodo transaccional:** 2025-01-01 → 2026-09-30 (fecha de observación). Clientes registrados en 2024 existen, pero su historial previo a 2025 no está en los datos.
- **Retención trimestral (KPI):** clientes con un pedido no cancelado en T−1 que vuelven a comprar en T ÷ compradores de T−1.
- **Entrega tardía:** `dias_entrega_reales > dias_entrega_prometidos + 1`.
- **Base elegible para churn:** clientes con ≥1 pedido no cancelado entre 2026-01-01 y 2026-06-30.
- **churn = 1:** cliente elegible sin pedidos no cancelados entre 2026-07-01 y 2026-09-30.

## 🧾 Diccionario de datos

Formato: CSV UTF-8, separador `,`, fechas `YYYY-MM-DD`, marcas de tiempo `YYYY-MM-DD HH:MM:SS`. Celda vacía = nulo.

### `clientes.csv` · 1 fila por cliente

| Columna | Tipo | Descripción |
|---|---|---|
| cliente_id | str (PK) | `C00001`… |
| fecha_registro | fecha | Alta en la tienda |
| edad | int, nulo ≈3 % | 18–70 |
| genero | cat | F, M, No informa |
| ciudad | cat | 12 ciudades de Colombia |
| segmento | cat | Joven tendencia, Profesional urbano, Cazador de ofertas, Premium (asignado al registro) |
| canal_adquisicion | cat | Instagram Ads, Google Ads, Orgánico, Referido, TikTok Ads, Email/Afiliados |
| miembro_fidelidad | 0/1 | Pertenece al programa de lealtad |
| acepta_newsletter | 0/1 | Consentimiento de email marketing |
| dispositivo_preferido | cat | Móvil, Escritorio, Tablet |
| categoria_preferida | cat | Categoría declarada en el perfil |
| fecha_primera_compra / fecha_ultima_compra | fecha, nulo | Sobre pedidos no cancelados |
| total_pedidos | int | Pedidos no cancelados |
| ingresos_brutos | int COP | Suma de `monto_neto` de pedidos no cancelados |
| total_reembolsado | int COP | Suma de reembolsos aprobados |
| total_cancelaciones / total_devoluciones | int | Conteos |
| ingresos_netos | int COP | `ingresos_brutos − total_reembolsado` |
| churn | 0/1, nulo | Etiqueta (nulo si no es elegible) |
| estado_retencion | cat | Activo, Abandonó, Nuevo o reactivado, Inactivo previo, Sin compras |

> ⚠️ Los agregados (fechas de compra, totales) cubren todo el periodo e incluyen el trimestre de la etiqueta: **no usarlos como features**. Para modelar usa `features_modelo.csv`.

### `transacciones.csv` · 1 fila por pedido

| Columna | Tipo | Descripción |
|---|---|---|
| pedido_id | str (PK) | `P0000001`… |
| cliente_id | FK | → clientes |
| sesion_id | FK | Sesión en que se compró → navegacion_web |
| fecha_pedido | timestamp | Momento de la compra |
| categoria_principal | cat | Categoría del primer artículo (8 categorías) |
| num_articulos | int | 1–6 |
| monto_bruto | int COP | Precio de lista de los artículos |
| descuento_pct | int | 0, 10, 15, 20, 30, 40 |
| monto_descuento | int COP | `monto_bruto × descuento_pct / 100`, redondeado a 100 |
| costo_envio | int COP | 0 si `bruto − descuento ≥ 150.000`; 9.900 Bogotá/Medellín; 14.900 resto |
| monto_neto | int COP | `bruto − descuento + envío` |
| metodo_pago | cat | Tarjeta de crédito, PSE, Tarjeta débito, Contraentrega, Nequi/Daviplata |
| cupon_usado | 0/1 | El descuento vino de un cupón |
| dias_entrega_prometidos | int | 2 Bogotá/Medellín, 4 resto |
| dias_entrega_reales | int, nulo | Nulo si cancelado o en tránsito |
| fecha_entrega | fecha, nulo | `fecha_pedido + dias_entrega_reales` |
| estado_pedido | cat | Entregado, Devuelto, Devuelto parcial, Cancelado, En tránsito |

### `navegacion_web.csv` · 1 fila por sesión

| Columna | Tipo | Descripción |
|---|---|---|
| sesion_id | str (PK) | `S0000001`… |
| cliente_id | FK | Solo sesiones de usuarios identificados |
| inicio_sesion | timestamp | |
| dispositivo | cat | Móvil, Escritorio, Tablet |
| fuente_trafico | cat | Directo, Búsqueda orgánica, Redes sociales, Email, Pago (SEM), Referido |
| paginas_vistas | int | 1 = rebote |
| duracion_seg | int | |
| tiempo_carga_ms | int | Tiempo medio de carga de página en la sesión |
| uso_buscador | 0/1 | |
| agrego_carrito / inicio_checkout / compro | 0/1 | Embudo: `compro ≤ checkout ≤ carrito` |
| pedido_id | FK, nulo | Presente solo si `compro = 1` |

### `devoluciones.csv` · máx. 1 por pedido

| Columna | Tipo | Descripción |
|---|---|---|
| devolucion_id | str (PK) | `D0000001`… |
| pedido_id / cliente_id | FK | |
| fecha_devolucion | fecha | Posterior a `fecha_entrega` |
| articulos_devueltos | int | = `num_articulos` ⇒ pedido "Devuelto"; menor ⇒ "Devuelto parcial" |
| motivo | cat | Talla incorrecta, No coincide con la foto, Calidad inferior a lo esperado, Cambio de opinión, Producto defectuoso, Llegó tarde |
| monto_reembolso | int COP | Proporcional a artículos devueltos, sin envío; 0 si rechazada |
| estado_devolucion | cat | Reembolsada, En proceso, Rechazada |
| dias_reembolso | int, nulo | Días entre devolución y reembolso (nulo si rechazada) |
| fecha_reembolso | fecha, nulo | Nulo si en proceso o rechazada |

### `cancelaciones.csv` · máx. 1 por pedido

| Columna | Tipo | Descripción |
|---|---|---|
| cancelacion_id | str (PK) | `X0000001`… |
| pedido_id / cliente_id | FK | |
| fecha_cancelacion | fecha | ≥ fecha del pedido |
| etapa | cat | Antes del envío, En tránsito |
| motivo | cat | Demora en el envío, Encontró mejor precio, Cambio de opinión, Error en el pedido, Problema con el pago |
| reembolsado | cat | Sí, No aplica (contraentrega) |

### `features_modelo.csv` · 1 fila por cliente elegible

Todas las variables se calculan **solo con datos hasta 2026-06-30**; la etiqueta usa jul–sep 2026 (sin fuga temporal). Incluye los atributos estáticos del cliente más:

| Columna | Descripción |
|---|---|
| antiguedad_dias | Días desde el registro hasta el corte |
| recencia_dias | Días desde la última compra no cancelada |
| frecuencia_pedidos / valor_monetario / ticket_promedio | RFM sobre pedidos no cancelados |
| descuento_promedio_pct | Descuento medio recibido |
| pedidos_ult_90d | Pedidos en abr–jun 2026 |
| dias_entrega_promedio / pct_entregas_tardias | Sobre pedidos entregados |
| num_cancelaciones / num_devoluciones / tasa_devolucion | Historial post-venta |
| dias_reembolso_promedio | Nulo si no tiene devoluciones con reembolso |
| sesiones_ult_90d / tiempo_carga_ms_prom_90d / carritos_abandonados_90d | Navegación abr–jun 2026 |
| churn | Etiqueta objetivo |

## 📥 Importador de datos (fase 2)

```bash
streamlit run app/importador.py   # interfaz
python -m pytest -q tests         # pruebas
```

- **Fuentes:** CSV/TXT (delimitador automático o manual), TSV, Excel `.xlsx`/`.xls` (selección de hoja), JSON y JSON Lines (selección de lista, objetos anidados aplanados), SQLite en solo lectura (selección de tabla) e ingreso manual en tabla editable.
- **No soportado:** conexiones a servidores (PostgreSQL, MySQL, SQL Server); la interfaz lo indica en lugar de simularlo. Límite de 50 MB por archivo.
- **Flujo:** fuente → vista previa → columnas, nombres, tipos, clave y obligatorias → diagnóstico → acción por problema (corregir / excluir / conservar) → informe y descargas (CSV, Excel, excluidos, informe .md/.json) → relación de tablas por claves comunes.
- **Diagnóstico:** columnas vacías, nulos, filas y claves duplicadas, fechas inválidas, tipos incorrectos, espacios, variantes de una misma categoría, magnitudes negativas, fechas fuera de rango, fórmulas, columnas desconocidas o faltantes frente a los esquemas de E-Commerce X.
- **Seguridad:** el original nunca se modifica, todo se lee como texto (sin `eval`, macros ni fórmulas), la extensión se valida contra la firma del archivo, SQLite usa lista blanca de tablas y las fórmulas se neutralizan al exportar. Nunca se rellenan valores ni se crean registros.

## 📊 Análisis exploratorio y consultas (fase 3)

```bash
streamlit run app/importador.py           # página «analisis» en el menú lateral
python -m pytest -q tests/test_analisis.py
```

1. Importa tablas (página principal → «Agregar al espacio de trabajo») o pulsa **Cargar caso E-Commerce X**.
2. Elige conjunto, columnas y filtros (fechas, categorías, rangos); KPIs, tablas y gráficos se recalculan con los filtros.
3. **KPIs:** sugeridos según los tipos de columna y propios (conteo, distinto, suma, promedio, mediana, mín., máx., tasa con evento, cocientes, condiciones). Si no se pueden calcular (división por cero, columna no numérica, sin registros) se muestra «No calculable» y el motivo.
4. **Preguntas:** motor determinista que resuelve columnas por nombre, sinónimos y valores. Muestra la consulta interpretada, las columnas usadas, el cálculo y los registros de respaldo. Si hay varias columnas posibles, pide aclaración; si no hay datos para responder, explica la limitación. Tipos soportados: ranking de un evento por grupo, evolución temporal, comparación entre dos grupos y agregados (por grupo o global).
5. **Exportación:** CSV de KPIs, series y resultados; gráficos en HTML (Altair); respuesta en JSON.

**IA opcional** (`analisis/ia.py`): con `pip install anthropic` y `ANTHROPIC_API_KEY` en el entorno (opcional `ECOMX_IA_MODELO`), la IA solo traduce la pregunta a una consulta estructurada, que se valida contra las columnas y se ejecuta con el motor determinista. Envía nombres y tipos de columnas, nunca filas. Sin configuración el interruptor queda deshabilitado.

**Caso E-Commerce X:** la pestaña del caso recalcula la retención trimestral con la definición de este README y compara cada uno de los dos últimos trimestres con el mismo trimestre del año anterior y con el trimestre previo.

## 🛍️ Dashboard ejecutivo (fase 4)

### Instalación y ejecución

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/importador.py                     # abre http://localhost:8501
```

En el menú lateral elige **dashboard**. La primera carga tarda unos segundos: lee y valida los CSV con el módulo de ingesta y entrena el modelo; después todo queda en caché.

```bash
python -m pytest -q tests                           # 41 pruebas (ingesta, análisis y dashboard)
python scripts/validate_data.py                     # 48 reglas de calidad de los datos
```

### Contenido

| Sección | Qué muestra |
|---|---|
| KPIs | Retención trimestral (con variación interanual), churn jul–sep 2026, ingresos netos y brutos, compradores, ticket promedio, devolución y cancelación |
| Tendencias | Indicadores mensuales con eventos operativos marcados y retención trimestral |
| Segmentación | Churn e ingresos por segmento, canal, ciudad o fidelidad |
| Factores de abandono | Churn del grupo expuesto vs el resto (prueba χ²) en la cohorte trimestral o en las variables del modelo |
| Modelo predictivo | AUC, exactitud frente a la línea base, precisión, recall, ROC, churn por decil de riesgo, coeficientes y matriz de confusión (conjunto de prueba) |
| Recomendaciones | Hasta 3, solo de factores significativos (p < 0,05 y ≥ 3 pp), con la evidencia calculada |

Filtros: rango de fechas, segmento, canal, ciudad y programa de fidelidad. Todas las fórmulas están en `analisis/tablero.py` y las pruebas verifican que los KPIs coincidan con `clientes.csv`, con la suma de las tendencias y segmentos y con la verificación de retención de la fase 3.

## ✅ Calidad

`reports/validacion.md` contiene el resultado de las 48 reglas (PK, FK, fechas, montos, estados, agregados, etiqueta, fuga temporal) y la verificación del KPI de retención.
