"""🤖 Intérprete opcional con la API de Anthropic.

Solo traduce la pregunta a una `Interpretacion` estructurada; los números siempre los calcula
el motor determinista (consultas.ejecutar). Se envían nombres y tipos de columnas, nunca filas.

Configuración (opcional):
  pip install anthropic
  export ANTHROPIC_API_KEY=...        # nunca se guarda ni se muestra
  export ECOMX_IA_MODELO=claude-sonnet-5-5   # opcional
Sin estas condiciones `disponible()` es False y la app usa solo el motor determinista.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from .consultas import Interpretacion, Respuesta
from .perfil import perfil

INTENCIONES = {"ranking_evento", "evolucion", "comparar_grupos", "agregado"}


def disponible() -> tuple[bool, str]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, "IA desactivada: no hay ANTHROPIC_API_KEY configurada."
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, "IA desactivada: la biblioteca «anthropic» no está instalada."
    return True, "IA activa (solo interpreta la pregunta; los cálculos son deterministas)."


def validar(d: dict, df: pd.DataFrame) -> Interpretacion | Respuesta:
    """Valida la salida del modelo contra las columnas reales antes de ejecutar nada."""
    if d.get("intencion") not in INTENCIONES:
        return Respuesta("no_soportada", "La IA no pudo clasificar la pregunta en un tipo compatible.")
    for k in ("grupo", "medida", "fecha"):
        if d.get(k) and d[k] not in df.columns:
            return Respuesta("no_soportada", f"La IA propuso la columna inexistente «{d[k]}»; se descarta.")
    ev = d.get("evento")
    if ev and (ev.get("columna") not in df.columns or ev.get("tipo") not in ("booleano", "valor", "suma")):
        return Respuesta("no_soportada", "La IA propuso un evento no válido; se descarta.")
    campos = Interpretacion.__dataclass_fields__
    it = Interpretacion(**{k: v for k, v in d.items() if k in campos})
    it.notas = list(it.notas or []) + ["Interpretación propuesta por IA y validada contra las columnas."]
    return it


def interprete_ia(pregunta: str, df: pd.DataFrame):
    ok, msg = disponible()
    if not ok:
        return Respuesta("no_soportada", msg)
    import anthropic

    cols = perfil(df)
    prompt = (
        "Traduce la pregunta a JSON con claves: intencion (ranking_evento|evolucion|comparar_grupos|agregado), "
        "grupo, evento {tipo: booleano|valor|suma, columna, valores}, medida, agregacion "
        "(suma|promedio|mediana|conteo), fecha, frecuencia (D|W|M|Q|Y), direccion (mayor|menor), "
        "criterio (tasa|conteo). Usa SOLO estas columnas: " + json.dumps(cols, ensure_ascii=False) +
        ". Si no se puede responder, devuelve {\"intencion\": null}. Responde solo con JSON.\nPregunta: " + pregunta)
    try:
        msg = anthropic.Anthropic().messages.create(
            model=os.environ.get("ECOMX_IA_MODELO", "claude-sonnet-5-5"), max_tokens=400,
            messages=[{"role": "user", "content": prompt}])
        d = json.loads(msg.content[0].text.strip().strip("`").removeprefix("json"))
    except Exception as e:  # red, cuota o JSON inválido: no se inventa una respuesta
        return Respuesta("no_soportada", f"La IA no respondió de forma utilizable ({type(e).__name__}).")
    return validar(d, df)
