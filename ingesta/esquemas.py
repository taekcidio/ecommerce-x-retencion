"""🗂️ Esquemas conocidos (opcionales). Si un archivo no coincide con ninguno, se trata como genérico.

Basados en el diccionario de datos de la fase 1 (README.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .tipos import normalize_name


@dataclass
class Esquema:
    nombre: str
    columnas: dict[str, str]  # columna -> tipo
    clave: list[str]
    requeridas: list[str] = field(default_factory=list)
    foraneas: dict[str, str] = field(default_factory=dict)  # columna -> "tabla.columna"


_E = Esquema
ECOMMERCE_X = {
    "clientes": _E("clientes", {
        "cliente_id": "texto", "fecha_registro": "fecha", "edad": "entero", "genero": "texto",
        "ciudad": "texto", "segmento": "texto", "canal_adquisicion": "texto", "miembro_fidelidad": "booleano",
        "acepta_newsletter": "booleano", "dispositivo_preferido": "texto", "categoria_preferida": "texto",
        "fecha_primera_compra": "fecha", "fecha_ultima_compra": "fecha", "total_pedidos": "entero",
        "ingresos_brutos": "entero", "total_reembolsado": "entero", "total_cancelaciones": "entero",
        "total_devoluciones": "entero", "ingresos_netos": "entero", "churn": "entero",
        "estado_retencion": "texto"}, ["cliente_id"], ["cliente_id", "fecha_registro"]),
    "transacciones": _E("transacciones", {
        "pedido_id": "texto", "cliente_id": "texto", "sesion_id": "texto", "fecha_pedido": "fecha_hora",
        "categoria_principal": "texto", "num_articulos": "entero", "monto_bruto": "entero",
        "descuento_pct": "entero", "monto_descuento": "entero", "costo_envio": "entero", "monto_neto": "entero",
        "metodo_pago": "texto", "cupon_usado": "booleano", "dias_entrega_prometidos": "entero",
        "dias_entrega_reales": "entero", "fecha_entrega": "fecha", "estado_pedido": "texto"},
        ["pedido_id"], ["pedido_id", "cliente_id", "fecha_pedido", "monto_neto"],
        {"cliente_id": "clientes.cliente_id", "sesion_id": "navegacion_web.sesion_id"}),
    "navegacion_web": _E("navegacion_web", {
        "sesion_id": "texto", "cliente_id": "texto", "inicio_sesion": "fecha_hora", "dispositivo": "texto",
        "fuente_trafico": "texto", "paginas_vistas": "entero", "duracion_seg": "entero",
        "tiempo_carga_ms": "entero", "uso_buscador": "booleano", "agrego_carrito": "booleano",
        "inicio_checkout": "booleano", "compro": "booleano", "pedido_id": "texto"},
        ["sesion_id"], ["sesion_id", "cliente_id", "inicio_sesion"], {"cliente_id": "clientes.cliente_id"}),
    "devoluciones": _E("devoluciones", {
        "devolucion_id": "texto", "pedido_id": "texto", "cliente_id": "texto", "fecha_devolucion": "fecha",
        "articulos_devueltos": "entero", "motivo": "texto", "monto_reembolso": "entero",
        "estado_devolucion": "texto", "dias_reembolso": "entero", "fecha_reembolso": "fecha"},
        ["devolucion_id"], ["devolucion_id", "pedido_id", "fecha_devolucion"],
        {"pedido_id": "transacciones.pedido_id", "cliente_id": "clientes.cliente_id"}),
    "cancelaciones": _E("cancelaciones", {
        "cancelacion_id": "texto", "pedido_id": "texto", "cliente_id": "texto", "fecha_cancelacion": "fecha",
        "etapa": "texto", "motivo": "texto", "reembolsado": "texto"},
        ["cancelacion_id"], ["cancelacion_id", "pedido_id", "fecha_cancelacion"],
        {"pedido_id": "transacciones.pedido_id", "cliente_id": "clientes.cliente_id"}),
}


def match_schema(columns: list[str], min_overlap: float = 0.5) -> tuple[Esquema | None, float]:
    """Esquema con mayor coincidencia (o None si es genérico).

    Exige la clave del esquema y que la mayoría de columnas del archivo pertenezcan a él,
    así una exportación parcial también se reconoce.
    """
    norm = {normalize_name(c) for c in columns}
    best, score = None, 0.0
    for esq in ECOMMERCE_X.values():
        if not set(esq.clave) <= norm:
            continue
        s = len(norm & set(esq.columnas)) / max(len(norm), 1)
        if s > score:
            best, score = esq, s
    return (best, score) if score >= min_overlap else (None, score)
