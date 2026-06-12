import os
import json
import re
from langchain_openai import ChatOpenAI

# $env:OPENAI_API_KEY="key"  -> need to set in terminal

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)

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
    "enumeration":   ("UInt16", 1)
}

_DEFAULT_FORMAT= "Float"
_DEFAULT_NUM_REGS = 2

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

def _preprocess_rows(rows: list) -> list:
    enriched = []
    for row in rows:
        fmt, num_regs = _resolve_format_and_regs(row.get("datatype", ""))
        enriched.append({
            **row,
            "_format":       fmt,
            "_num_regs":     num_regs
        })
    return enriched

SCHEMA_EXAMPLE = """[
  {
    "Label": "van",
    "Address": "1",
    "Format": "Float",
    "Number of registers": 2,
    "Scaling": 1,
    "Multiplier": 1,
    "Description": "Voltage Va-n",
    "Display Name": null
  }
]"""
 
FORMAT_RULES = """Rules (read carefully — some fields are PRE-FILLED, do not change them):
 
PRE-FILLED by the system (copy exactly as given, do not modify):
  - Format              : already resolved from the datatype string
  - Number of registers : already resolved from the datatype string
 
YOU must fill in:
- Label        : snake_case version of the description (e.g. "watts_total")
- Address      : keep the numeric address EXACTLY as extracted (e.g. "40101")
- Scaling      : use 1 unless the register map explicitly states a scale factor.
- Multiplier   : use 1 unless the register map explicitly states a multiplier.
- Description  : the human-readable parameter name exactly as in the source.
- Display Name : null (leave as null unless a separate display name is given).
"""

def generate_json(register_rows: list, company: str = "", device_family: str = "") -> str:
    if not register_rows:
        return "[]"

    enriched = _preprocess_rows(register_rows)
    batches = [enriched[i:i+80] for i in range(0, len(enriched), 80)]
    all_results = []

    for batch_idx, batch in enumerate(batches):
        print(f"  [llm_formatter] Processing batch {batch_idx+1}/{len(batches)} "
              f"({len(batch)} rows)...")

        prompt = f"""You are formatting Modbus register data for the device:
  Company      : {company}
  Device family: {device_family}

Convert the following register rows into the JSON format shown below.

Target format example:
{SCHEMA_EXAMPLE}

{FORMAT_RULES}

Input data (fields prefixed with _ are pre-filled system values):
{json.dumps(batch, indent=2)}
 
Return ONLY a valid JSON array. No markdown, no explanation, no extra text.
"""

        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            parsed = json.loads(raw)
            for out_row, in_row in zip(parsed, batch):
                out_row["Format"]              = in_row["_format"]
                out_row["Number of registers"] = in_row["_num_regs"]
            all_results.extend(parsed)
        except json.JSONDecodeError as e:
            print(f"  [llm_formatter] JSON parse error in batch {batch_idx+1}: {e}")
            print(f"  Raw output (first 500 chars): {raw[:500]}")
 
    return json.dumps(all_results, indent=2)