import json
import re

# Same datatype -> (Format, Number of registers) mapping used previously.
# Kept identical so output is consistent with any historical LLM-generated
# JSON files already produced by this pipeline.
_DTYPE_MAP = {
    # 32-bit floats
    "float":         ("Float",  2),
    "float32":       ("Float",  2),
    "ieee754":       ("Float",  2),
    "real":          ("Float",  2),
    # 32-bit integers
    "int32":         ("Int32",  2),
    "integer32":     ("Int32",  2),
    "long":          ("Int32",  2),
    "int":           ("Int32",  2),
    "uint32":        ("UInt32", 2),
    "unsignedint32": ("UInt32", 2),
    "dword":         ("UInt32", 2),
    # 16-bit integers
    "int16":         ("Int16",  1),
    "integer16":     ("Int16",  1),
    "word":          ("Int16",  1),
    "uint16":        ("UInt16", 1),
    "unsignedint16": ("UInt16", 1),
    "unsigned":      ("UInt16", 1),
    "short":         ("Int16",  1),
    # 64-bit
    "int64":         ("Int64",  4),
    "double":        ("Double", 4),
    # string / other
    "string":        ("String", 1),
    "ascii":         ("String", 1),
    "boolean":       ("Bool",   1),
    "bool":          ("Bool",   1),
    "bit":           ("Bool",   1),
    "enum":          ("UInt16", 1),
    "enumeration":   ("UInt16", 1),
}

_DEFAULT_FORMAT    = "Float"
_DEFAULT_NUM_REGS  = 2
_DEFAULT_MULTIPLIER = 1
_DEFAULT_SCALING    = 1


def _resolve_format_and_regs(raw_datatype: str) -> tuple:
    if not raw_datatype or raw_datatype.strip().lower() in ("", "nan", "none", "unknown"):
        return _DEFAULT_FORMAT, _DEFAULT_NUM_REGS
    norm = re.sub(r"[^a-z0-9]", "", raw_datatype.lower())

    if norm in _DTYPE_MAP:
        return _DTYPE_MAP[norm]

    for key, value in _DTYPE_MAP.items():
        if key in norm:
            return value

    return _DEFAULT_FORMAT, _DEFAULT_NUM_REGS


def _make_label(description: str) -> str:
    if not description:
        return ""
    return (
        description.lower().strip()
        .replace(" ", "_").replace("-", "_").replace("/", "_")
        .replace("(", "").replace(")", "").replace(".", "")
    )


def _resolve_scaling(raw_scaling) -> float | int:
    if raw_scaling is None:
        return _DEFAULT_SCALING
    s = str(raw_scaling).strip()
    if s.lower() in ("", "nan", "none"):
        return _DEFAULT_SCALING
    # Allow values like "10", "0.1", "x10"
    s_clean = s.lstrip("xX").replace(",", "")
    try:
        as_float = float(s_clean)
        # Use an int in the output when the value is a whole number, to match
        # the SCHEMA_EXAMPLE style ("Scaling": 1) rather than always emitting
        # a float like 1.0.
        return int(as_float) if as_float.is_integer() else as_float
    except ValueError:
        return _DEFAULT_SCALING

def _resolve_num_registers(raw_num_registers, fallback: int) -> int:
    if raw_num_registers is None:
        return fallback
    s = str(raw_num_registers).strip()
    if s.lower() in ("", "nan", "none"):
        return fallback
    s_clean = s.replace(",", "")
    if s_clean.endswith(".0"):
        s_clean = s_clean[:-2]
    try:
        return int(s_clean)
    except ValueError:
        return fallback

def generate_json(register_rows: list, company: str = "", device_family: str = "") -> str:
    """
    Deterministic replacement for llm_formatter.generate_json().
    Same signature, same return type (a JSON string), same output schema:
        [
          {
            "Label": "...",
            "Address": "...",
            "Format": "...",
            "Number of registers": N,
            "Scaling": N,
            "Multiplier": 1,
            "Description": "...",
            "Display Name": null
          },
          ...
        ]
    """
    results = []

    for row in register_rows:
        description = row.get("description", "") or ""
        label        = row.get("label") or _make_label(description)
        address      = row.get("address", "")
        fmt, num_regs_fallback = _resolve_format_and_regs(row.get("datatype", ""))
        scaling      = _resolve_scaling(row.get("scaling"))
        num_regs     = _resolve_num_registers(row.get("num_registers"), num_regs_fallback)

        results.append({
            "Label":                label,
            "Address":              address,
            "Format":                fmt,
            "Number of registers":   num_regs,
            "Scaling":               scaling,
            "Multiplier":            _DEFAULT_MULTIPLIER,
            "Description":           description,
            "Display Name":          None,
        })

    return json.dumps(results, indent=2)