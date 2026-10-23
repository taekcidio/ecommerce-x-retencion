"""📦 Módulo de ingesta de E-Commerce X: carga, diagnóstico, limpieza y relación de datos.

La lógica es independiente de la interfaz (app/importador.py usa solo esta API pública).
"""
from .calidad import ACCIONES, DESCRIPCION_ACCION, Issue, diagnose
from .esquemas import ECOMMERCE_X, match_schema
from .limpieza import (Plan, Resultado, aplicar, informe_json, informe_markdown, preparar, sugerir_plan,
                       to_csv_bytes, to_excel_bytes)
from .loaders import (DELIMITERS, IngestaError, LoadResult, db_server_support, detect_format, excel_sheets,
                      json_tables, load_delimited, load_excel, load_file, load_json, load_manual, load_sqlite,
                      sqlite_tables)
from .relaciones import sugerir_relaciones, unir, verificar
from .tipos import TIPOS, convert, infer_type

__all__ = [n for n in dir() if not n.startswith("_")]
