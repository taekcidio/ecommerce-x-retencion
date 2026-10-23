"""📊 Análisis exploratorio genérico: perfil, filtros, KPIs, gráficos y consultas en lenguaje natural."""
from .consultas import EJEMPLOS, Interpretacion, Respuesta, ejecutar, interpretar, responder
from .filtros import Filtro, aplicar_filtros
from .kpis import AGREGACIONES, FORMATOS, KPI, OPERADORES, Condicion, Metrica, Valor, calcular_kpi, \
    calcular_metrica, formatear, sugerir_kpis
from .perfil import FRECUENCIAS, cols_de, distribucion, faltantes, frecuencia_auto, perfil, resumen_categorico, \
    resumen_numerico, retencion_periodos, serie_temporal, tipo_columna

__all__ = [n for n in dir() if not n.startswith("_")]
