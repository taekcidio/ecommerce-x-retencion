#!/usr/bin/env python3
"""🧪 Crea archivos de ejemplo (no E-Commerce X) para probar el importador: data/samples/."""
import json
import sqlite3
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "samples"
OUT.mkdir(parents=True, exist_ok=True)

# 1) CSV con punto y coma y problemas de calidad deliberados
(OUT / "ventas_con_errores.csv").write_text(
    "factura;fecha;ciudad;cliente;monto;unidades;pagado\n"
    "F001;2026-03-01;Bogotá;Ana Ruiz;120000;2;si\n"
    "F002;2026-03-02;bogota ;Luis Pérez;85.500,50;1;no\n"
    "F003;31/02/2026;Medellín;Marta Gil;99000;3;si\n"
    "F004;2026-03-04;Medellin;  Juan Díaz;abc;1;si\n"
    "F005;;Cali;Rosa León;45000;-2;no\n"
    "F001;2026-03-05;Cali;Pedro Mora;30000;1;si\n"
    "F006;2026-03-06;Cali;=HYPERLINK(\"http://x.co\",\"clic\");15000;1;si\n"
    "F007;2031-01-01;Bogotá;Sara Peña;70000;1;si\n"
    "F002;2026-03-02;bogota ;Luis Pérez;85.500,50;1;no\n"
    ";2026-03-08;Pasto;Iván Ríos;52000;1;no\n", encoding="utf-8")

# 2) TSV genérico con estructura totalmente distinta
(OUT / "inventario.tsv").write_text(
    "SKU\tProducto\tStock actual\tPrecio unitario\tÚltima revisión\tObservación\n"
    "A-01\tCamiseta básica\t120\t29900\t2026-09-01\t\n"
    "A-02\tJean slim\t45\t119900\t2026-09-03\tTallas 28-34\n"
    "A-03\tChaqueta\t0\t249900\t2026-08-30\tAgotado\n"
    "A-04\tGorra\t310\t35000\t2026-09-05\t\n", encoding="utf-8")

# 3) JSON anidado
(OUT / "catalogo.json").write_text(json.dumps({"generado": "2026-10-01", "productos": [
    {"id": 1, "nombre": "Vestido lino", "precio": 159900, "dimensiones": {"talla": "M", "color": "beige"},
     "etiquetas": ["verano", "nuevo"]},
    {"id": 2, "nombre": "Tenis urbanos", "precio": 210000, "dimensiones": {"talla": "40", "color": "blanco"},
     "etiquetas": []},
    {"id": 3, "nombre": "Bolso cuero", "precio": None, "dimensiones": {"talla": "única", "color": "café"}}]},
    ensure_ascii=False, indent=2), encoding="utf-8")

# 4) Excel con dos hojas relacionables
emp = pd.DataFrame({"empleado_id": [1, 2, 3, 4], "nombre": ["Laura", "Diego", "Camila", "Andrés"],
                    "depto_id": ["D1", "D2", "D1", "D9"],
                    "ingreso": pd.to_datetime(["2023-02-01", "2024-07-15", "2025-01-10", "2026-05-02"]),
                    "salario": [4200000, 3800000, 5100000, 3500000]})
dep = pd.DataFrame({"depto_id": ["D1", "D2", "D3"], "departamento": ["Ventas", "Logística", "Marketing"]})
with pd.ExcelWriter(OUT / "empresa.xlsx") as xw:
    emp.to_excel(xw, sheet_name="Empleados", index=False)
    dep.to_excel(xw, sheet_name="Departamentos", index=False)

# 5) SQLite pequeña
db = OUT / "tienda.sqlite"
db.unlink(missing_ok=True)
con = sqlite3.connect(db)
con.executescript("""
CREATE TABLE productos(producto_id TEXT PRIMARY KEY, nombre TEXT, precio INTEGER);
INSERT INTO productos VALUES ('P1','Camisa',89900),('P2','Falda',99900),('P3','Botas',259900);
CREATE TABLE pedidos(pedido TEXT, producto_id TEXT, cantidad INTEGER, fecha TEXT);
INSERT INTO pedidos VALUES ('O1','P1',2,'2026-09-10'),('O2','P3',1,'2026-09-11'),('O3','P7',1,'2026-09-12');
""")
con.commit()
con.close()

# 6) Casos borde
(OUT / "vacio.csv").write_text("", encoding="utf-8")
(OUT / "solo_encabezados.csv").write_text("a,b,c\n", encoding="utf-8")

# 7) Esquema E-Commerce X con nombres distintos y columnas desconocidas
(OUT / "clientes_columnas_extra.csv").write_text(
    "Cliente ID,fecha_registro,Ciudad,segmento,puntos_app,nivel_vip\n"
    "C00001,2025-01-03,Bogotá,Premium,1200,oro\n"
    "C00002,2025-02-11,Cali,Joven tendencia,300,\n"
    "C00003,2025-13-01,Medellín,Premium,850,plata\n", encoding="utf-8")

print("\n".join(f"✅ {p.name}" for p in sorted(OUT.iterdir())))
