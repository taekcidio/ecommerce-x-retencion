# Informe de validación · E-Commerce X

Reglas evaluadas: **48** · Fallidas: **0**

| Regla | Filas que incumplen | Estado |
|---|---:|:---:|
| PK única: cliente_id | 0 | ✅ |
| Sin nulos obligatorios (cliente) | 0 | ✅ |
| PK única: pedido_id | 0 | ✅ |
| Sin nulos obligatorios (pedido) | 0 | ✅ |
| PK única: sesion_id | 0 | ✅ |
| Sin nulos obligatorios (sesion) | 0 | ✅ |
| PK única: devolucion_id | 0 | ✅ |
| Sin nulos obligatorios (devolucion) | 0 | ✅ |
| PK única: cancelacion_id | 0 | ✅ |
| Sin nulos obligatorios (cancelacion) | 0 | ✅ |
| FK cliente_id válido en transacciones | 0 | ✅ |
| FK cliente_id válido en navegacion_web | 0 | ✅ |
| FK cliente_id válido en devoluciones | 0 | ✅ |
| FK cliente_id válido en cancelaciones | 0 | ✅ |
| FK pedido_id y cliente coherente en devoluciones | 0 | ✅ |
| Máx. un registro por pedido en devoluciones | 0 | ✅ |
| FK pedido_id y cliente coherente en cancelaciones | 0 | ✅ |
| Máx. un registro por pedido en cancelaciones | 0 | ✅ |
| Pedido ↔ sesión de compra (compro=1, mismo cliente) | 0 | ✅ |
| Sesiones con compra tienen pedido y viceversa | 0 | ✅ |
| Embudo: compro ≤ checkout ≤ carrito | 0 | ✅ |
| Pedido ≥ registro del cliente | 0 | ✅ |
| Sesión ≥ registro del cliente | 0 | ✅ |
| Pedidos dentro de ene-2025…sep-2026 | 0 | ✅ |
| fecha_entrega = fecha_pedido + dias_entrega_reales | 0 | ✅ |
| Fechas de entrega ≤ fecha de observación | 0 | ✅ |
| Devolución posterior a la entrega | 0 | ✅ |
| fecha_reembolso = devolución + dias_reembolso | 0 | ✅ |
| Reembolsos ≤ fecha de observación | 0 | ✅ |
| Cancelación ≥ fecha del pedido | 0 | ✅ |
| monto_neto = bruto − descuento + envío | 0 | ✅ |
| monto_descuento = bruto × pct (redondeo a 100) | 0 | ✅ |
| Montos no negativos | 0 | ✅ |
| Envío gratis desde $150.000 (neto de descuento) | 0 | ✅ |
| Reembolso ≤ valor pagado sin envío | 0 | ✅ |
| Devolución rechazada ⇒ reembolso 0 | 0 | ✅ |
| Cancelado ⇔ registro en cancelaciones | 0 | ✅ |
| Devuelto/Devuelto parcial ⇔ registro en devoluciones | 0 | ✅ |
| Devuelto total ⇔ todos los artículos devueltos | 0 | ✅ |
| Pedidos cancelados/en tránsito sin entrega | 0 | ✅ |
| Dominio estado_pedido | 0 | ✅ |
| total_pedidos coincide con transacciones | 0 | ✅ |
| ingresos_brutos coincide con transacciones | 0 | ✅ |
| ingresos_netos = brutos − reembolsado | 0 | ✅ |
| Etiqueta churn recalculada coincide | 0 | ✅ |
| estado_retencion coherente con churn | 0 | ✅ |
| Features: una fila por cliente elegible | 0 | ✅ |
| Features sin fuga temporal (recencia ≥ 1 día) | 0 | ✅ |

## Volumen

| Archivo | Filas |
|---|---:|
| clientes.csv | 8,000 |
| transacciones.csv | 22,296 |
| navegacion_web.csv | 120,373 |
| devoluciones.csv | 4,299 |
| cancelaciones.csv | 1,020 |
| features_modelo.csv | 3,324 |

## Verificación de la premisa (KPI de retención trimestral)

Retención T = clientes con compra no cancelada en T−1 que vuelven a comprar en T ÷ compradores de T−1.

| Trimestre | Retenidos | Base T−1 | Retención |
|---|---:|---:|---:|
| 2025Q2 | 809 | 1,256 | 64.4% |
| 2025Q3 | 953 | 1,618 | 58.9% |
| 2025Q4 | 1,214 | 1,812 | 67.0% |
| 2026Q1 | 1,300 | 2,240 | 58.0% |
| 2026Q2 | 1,327 | 2,185 | 60.7% |
| 2026Q3 | 1,230 | 2,466 | 49.9% |

Variación interanual: 2026Q2 vs 2025Q2 = -5.7% · 2026Q3 vs 2025Q3 = -15.3%

Balance de la etiqueta en features_modelo.csv: churn=1 → 55.1%
