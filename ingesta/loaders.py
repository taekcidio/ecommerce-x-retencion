"""📥 Carga segura de datos tabulares desde bytes (sin ejecutar nada del contenido).

Todos los loaders reciben bytes en memoria: nunca escriben ni modifican el archivo original.
Los valores se leen como texto (dtype=str) para que la inferencia de tipos sea explícita
y nada se convierta en silencio.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass, field

import pandas as pd

MAX_BYTES = 50 * 1024 * 1024  # 50 MB
NA_VALUES = ["", "NA", "N/A", "n/a", "null", "NULL", "None", "nan", "NaN", "-"]
SUPPORTED = {
    ".csv": "delimitado", ".txt": "delimitado", ".tsv": "delimitado", ".tab": "delimitado",
    ".xlsx": "excel", ".xlsm": "excel", ".xls": "excel_legacy",
    ".json": "json", ".sqlite": "sqlite", ".sqlite3": "sqlite", ".db": "sqlite",
}
DELIMITERS = {"Coma (,)": ",", "Punto y coma (;)": ";", "Tabulación": "\t", "Barra (|)": "|"}


class IngestaError(Exception):
    """Error con mensaje comprensible para el usuario final."""


@dataclass
class LoadResult:
    df: pd.DataFrame
    source: str
    fmt: str
    options: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------- utilidades ----------
def _check_size(data: bytes, name: str) -> None:
    if not data or not data.strip():
        raise IngestaError(f"El archivo «{name}» está vacío.")
    if len(data) > MAX_BYTES:
        raise IngestaError(f"«{name}» supera el límite de {MAX_BYTES // 2**20} MB.")


def detect_format(filename: str, data: bytes) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext not in SUPPORTED:
        raise IngestaError(
            f"Formato «{ext or 'sin extensión'}» no soportado. Formatos válidos: "
            + ", ".join(sorted(SUPPORTED)) + ".")
    fmt = SUPPORTED[ext]
    head = data[:16]
    # La extensión debe coincidir con la firma del contenido (evita archivos disfrazados)
    if fmt == "excel" and not head.startswith(b"PK"):
        raise IngestaError(f"«{filename}» no es un Excel .xlsx válido (contenido no coincide con la extensión).")
    if fmt == "excel_legacy" and not head.startswith(b"\xd0\xcf\x11\xe0"):
        raise IngestaError(f"«{filename}» no es un Excel .xls válido.")
    if fmt == "sqlite" and not head.startswith(b"SQLite format 3\x00"):
        raise IngestaError(f"«{filename}» no es una base de datos SQLite válida.")
    if fmt in ("delimitado", "json") and (head.startswith(b"PK") or b"\x00" in data[:1024]):
        raise IngestaError(f"«{filename}» parece un archivo binario, no texto.")
    return fmt


def decode(data: bytes, encoding: str | None = None) -> tuple[str, str]:
    for enc in ([encoding] if encoding else ["utf-8-sig", "cp1252", "latin-1"]):
        try:
            return data.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    raise IngestaError("No se pudo leer el texto con la codificación indicada.")


def sniff_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:50])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        counts = {d: sample.count(d) for d in ",;\t|"}
        return max(counts, key=counts.get) if any(counts.values()) else ","


def _finish(df: pd.DataFrame, name: str, fmt: str, opts: dict, warns: list[str]) -> LoadResult:
    if df.shape[1] == 0 or (df.shape[0] == 0 and df.columns.size == 0):
        raise IngestaError(f"«{name}» no contiene columnas ni datos.")
    if df.shape[0] == 0:
        raise IngestaError(f"«{name}» solo tiene encabezados: no hay registros para importar.")
    cols = [str(c).strip() for c in df.columns]
    unnamed = [i for i, c in enumerate(cols) if not c or c.lower().startswith("unnamed:")]
    for i in unnamed:
        cols[i] = f"columna_{i + 1}"
    if unnamed:
        warns.append(f"{len(unnamed)} columna(s) sin nombre se renombraron como columna_N.")
    seen: dict[str, int] = {}
    for i, c in enumerate(cols):
        if c in seen:
            seen[c] += 1
            cols[i] = f"{c}_{seen[c]}"
            warns.append(f"Encabezado duplicado «{c}» renombrado a «{cols[i]}».")
        else:
            seen[c] = 0
    df.columns = cols
    return LoadResult(df.reset_index(drop=True), name, fmt, opts, warns)


# ---------- loaders ----------
def load_delimited(data: bytes, name: str = "archivo", sep: str | None = None,
                   encoding: str | None = None, header: bool = True) -> LoadResult:
    _check_size(data, name)
    text, enc = decode(data, encoding)
    if not text.strip():
        raise IngestaError(f"El archivo «{name}» está vacío.")
    sep = sep or sniff_delimiter(text)
    warns: list[str] = []
    bad_lines: list[int] = []

    def on_bad(line):
        bad_lines.append(len(line))
        return None

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, header=0 if header else None,
                         na_values=NA_VALUES, keep_default_na=False, skipinitialspace=False,
                         engine="python", on_bad_lines=on_bad)
    except pd.errors.EmptyDataError:
        raise IngestaError(f"El archivo «{name}» está vacío.")
    except (pd.errors.ParserError, csv.Error) as e:
        raise IngestaError(f"No se pudo interpretar «{name}» con el delimitador {sep!r}: {e}")
    if bad_lines:
        warns.append(f"{len(bad_lines)} línea(s) con número de campos distinto al encabezado fueron omitidas.")
    if not header:
        df.columns = [f"columna_{i + 1}" for i in range(df.shape[1])]
    return _finish(df, name, "delimitado", {"delimitador": sep, "codificacion": enc, "encabezado": header}, warns)


def excel_sheets(data: bytes, name: str = "archivo.xlsx") -> list[str]:
    fmt = detect_format(name, data)
    try:
        return pd.ExcelFile(io.BytesIO(data), engine="xlrd" if fmt == "excel_legacy" else "openpyxl").sheet_names
    except ImportError:
        raise IngestaError("Leer .xls requiere la biblioteca xlrd, que no está instalada.")
    except Exception as e:  # archivo corrupto
        raise IngestaError(f"No se pudo abrir el Excel «{name}»: {e}")


def load_excel(data: bytes, name: str = "archivo.xlsx", sheet: str | int = 0, header: bool = True) -> LoadResult:
    _check_size(data, name)
    fmt = detect_format(name, data)
    try:
        # openpyxl lee valores en caché: las fórmulas y macros NO se ejecutan
        df = pd.read_excel(io.BytesIO(data), sheet_name=sheet, dtype=object,
                           header=0 if header else None, na_values=NA_VALUES, keep_default_na=False,
                           engine="xlrd" if fmt == "excel_legacy" else "openpyxl")
    except ImportError:
        raise IngestaError("Leer .xls requiere la biblioteca xlrd, que no está instalada.")
    except ValueError as e:
        raise IngestaError(f"No se encontró la hoja {sheet!r} en «{name}»: {e}")
    except Exception as e:
        raise IngestaError(f"No se pudo leer el Excel «{name}»: {e}")
    df = df.dropna(how="all").dropna(axis=1, how="all") if df.size else df
    if not header:
        df.columns = [f"columna_{i + 1}" for i in range(df.shape[1])]
    # normaliza a texto para inferencia uniforme (fechas Excel → ISO)
    for c in df.columns:
        df[c] = df[c].map(lambda v: None if pd.isna(v) else
                          (v.isoformat(sep=" ") if hasattr(v, "isoformat") else str(v)))
        df[c] = df[c].str.replace(r" 00:00:00$", "", regex=True)
    return _finish(df, name, "excel", {"hoja": sheet, "encabezado": header}, [])


def json_tables(data: bytes) -> list[str]:
    """Rutas candidatas a tabla dentro del JSON ('' = raíz)."""
    obj = _parse_json(data)
    if isinstance(obj, list):
        return [""]
    if isinstance(obj, dict):
        keys = [k for k, v in obj.items() if isinstance(v, list) and v and isinstance(v[0], dict)]
        return keys or [""]
    return [""]


def _parse_json(data: bytes):
    text, _ = decode(data)
    if not text.strip():
        raise IngestaError("El archivo JSON está vacío.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:  # JSON Lines
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as e:
            raise IngestaError(f"JSON inválido (línea {e.lineno}, columna {e.colno}): {e.msg}.")


def load_json(data: bytes, name: str = "archivo.json", path: str = "") -> LoadResult:
    _check_size(data, name)
    obj = _parse_json(data)
    if path:
        if not isinstance(obj, dict) or path not in obj:
            raise IngestaError(f"La clave «{path}» no existe en el JSON.")
        obj = obj[path]
    if isinstance(obj, dict):
        if all(isinstance(v, list) for v in obj.values()):  # formato columnar
            lens = {len(v) for v in obj.values()}
            if len(lens) > 1:
                raise IngestaError("JSON columnar con listas de distinta longitud.")
            obj = pd.DataFrame(obj).to_dict("records")
        else:
            obj = [obj]
    if not isinstance(obj, list) or not obj:
        raise IngestaError(f"«{name}» no contiene una lista de registros.")
    if not all(isinstance(r, dict) for r in obj):
        raise IngestaError("El JSON debe ser una lista de objetos (registros clave-valor).")
    df = pd.json_normalize(obj, sep=".")
    warns = []
    for c in df.columns:
        if df[c].map(lambda v: isinstance(v, list)).any():
            warns.append(f"La columna «{c}» contiene listas anidadas; se conservan como texto JSON.")
            df[c] = df[c].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)
    df = df.map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
    return _finish(df, name, "json", {"ruta": path}, warns)


# ---------- SQLite (solo lectura) ----------
def _sqlite_connect(data: bytes):
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.write(data)
    tmp.close()
    con = sqlite3.connect(f"file:{tmp.name}?mode=ro", uri=True)
    return con, tmp.name


def sqlite_tables(data: bytes, name: str = "base.sqlite") -> list[str]:
    _check_size(data, name)
    detect_format(name, data)
    con, path = _sqlite_connect(data)
    try:
        return [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    except sqlite3.DatabaseError as e:
        raise IngestaError(f"No se pudo leer la base SQLite: {e}")
    finally:
        con.close()
        os.unlink(path)


def load_sqlite(data: bytes, table: str, name: str = "base.sqlite", limit: int | None = None) -> LoadResult:
    tables = sqlite_tables(data, name)
    if table not in tables:  # lista blanca: nunca se interpola SQL arbitrario
        raise IngestaError(f"La tabla «{table}» no existe. Disponibles: {', '.join(tables) or 'ninguna'}.")
    con, path = _sqlite_connect(data)
    try:
        q = 'SELECT * FROM "' + table.replace('"', '""') + '"'
        if limit:
            q += f" LIMIT {int(limit)}"
        df = pd.read_sql_query(q, con)
    finally:
        con.close()
        os.unlink(path)
    df = df.map(lambda v: None if v is None else str(v))
    return _finish(df, f"{name}:{table}", "sqlite", {"tabla": table}, [])


def db_server_support() -> str:
    return ("Las conexiones a servidores (PostgreSQL, MySQL, SQL Server) no están habilitadas: "
            "el proyecto no incluye drivers ni un gestor seguro de credenciales. "
            "Exporta la tabla a CSV/Excel o usa un archivo SQLite.")


# ---------- datos manuales ----------
def load_manual(df: pd.DataFrame, name: str = "ingreso_manual") -> LoadResult:
    df = df.copy().dropna(how="all")
    df = df.loc[:, [c for c in df.columns if str(c).strip()]]
    df = df.map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == ""
                else str(v))
    df = df.dropna(how="all")
    if df.empty:
        raise IngestaError("La tabla manual no tiene registros: escribe al menos una fila.")
    return _finish(df, name, "manual", {}, [])


def load_file(data: bytes, filename: str, **opts) -> LoadResult:
    """Punto de entrada único por archivo."""
    _check_size(data, filename)
    fmt = detect_format(filename, data)
    if fmt == "delimitado":
        if filename.lower().endswith((".tsv", ".tab")) and not opts.get("sep"):
            opts["sep"] = "\t"
        return load_delimited(data, filename, **opts)
    if fmt in ("excel", "excel_legacy"):
        return load_excel(data, filename, **opts)
    if fmt == "json":
        return load_json(data, filename, **opts)
    if fmt == "sqlite":
        return load_sqlite(data, name=filename, **opts)
    raise IngestaError(f"Formato {fmt} no soportado.")
