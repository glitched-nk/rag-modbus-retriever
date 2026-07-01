import os
import re
import json

MIN_ROWS_PER_500_CHARS = 1
MIN_TEXT_LEN_FOR_ZERO_ROW_FALLBACK = 200
 
 
def should_use_llm_fallback(ocr_text: str, cell_parser_row_count: int) -> bool:
    text_len = len(ocr_text.strip()) if ocr_text else 0
 
    if text_len < 50:
        return False
 
    if cell_parser_row_count == 0 and text_len >= MIN_TEXT_LEN_FOR_ZERO_ROW_FALLBACK:
        return True
 
    expected_min_rows = (text_len / 500) * MIN_ROWS_PER_500_CHARS
    return cell_parser_row_count < expected_min_rows
 
 
def _get_llm():
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise RuntimeError(
            "langchain_openai is not installed, but the OCR LLM fallback "
            "was triggered. Run: pip install langchain_openai"
        ) from e
 
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Run:  $env:OPENAI_API_KEY='sk-...'  (PowerShell) "
            "or    set OPENAI_API_KEY=sk-...      (CMD) "
            "in your terminal before starting the program."
        )
    return ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
 
 
_SCHEMA_EXAMPLE = """[
  {
    "address": "40150",
    "description": "Voltage Va-n",
    "datatype": "UINT16",
    "scaling": "10",
    "num_registers": "1"
  }
]"""
 
_INSTRUCTIONS = """You are extracting a Modbus register table from raw OCR text.
The OCR output below may have misaligned columns, merged cells, or noisy
characters — use context and domain knowledge of Modbus register maps to
recover the correct structure anyway.
 
For every register row you can identify, extract:
  - address       : the numeric Modbus register address (REQUIRED — skip any
                     row where you cannot confidently determine this)
  - description    : the human-readable parameter name
  - datatype       : the data type if shown (e.g. UINT16, UINT32, Float,
                     Int32) — use "Unknown" if not present in the text
  - scaling        : the scale factor if shown as its own column or value —
                     use null if not present (do NOT guess a default; a
                     downstream step applies the correct default for missing
                     values)
  - num_registers  : the "number of registers" value if shown as its own
                     column — use null if not present (same rule: do not
                     guess, a downstream step applies the datatype-derived
                     default when this is null)
 
Rules:
  - Only include rows where you can identify a real register address.
  - Do not invent values that aren't supported by the text.
  - Return ONLY a valid JSON array in the exact shape shown below, no
    markdown, no explanation, no extra text.
 
Target shape:
{schema}
"""
 
 
def extract_rows_via_llm(ocr_text: str, company: str = "", device_family: str = "", device_rack: str = "") -> list[dict]:
    if not ocr_text or not ocr_text.strip():
        return []
 
    context_line = ""
    if company or device_family or device_rack:
        context_line = (f"This text is from a Modbus document for "
                        f"company={company!r}, device_family={device_family!r}, "
                        f"device_rack={device_rack!r}.\n\n")
 
    prompt = (
        context_line
        + _INSTRUCTIONS.format(schema=_SCHEMA_EXAMPLE)
        + f"\nRaw OCR text:\n---\n{ocr_text[:6000]}\n---\n"
    )
 
    try:
        response = _get_llm().invoke(prompt)
        raw = response.content.strip()
    except Exception as e:
        print(f"    [ocr-llm-fallback] LLM call failed: {e}")
        return []
 
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
 
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"    [ocr-llm-fallback] JSON parse error: {e}")
        print(f"    Raw output (first 300 chars): {raw[:300]}")
        return []
 
    if not isinstance(parsed, list):
        print("    [ocr-llm-fallback] LLM did not return a JSON array — discarding.")
        return []
 
    addr_pat = re.compile(r"^\d{3,7}$")
    rows = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        address = str(item.get("address", "")).strip()
        if not addr_pat.match(address):
            continue
        rows.append({
            "address":       address,
            "description":   str(item.get("description", "")).strip(),
            "datatype":      str(item.get("datatype") or "Unknown").strip(),
            "scaling":       (str(item.get("scaling")).strip()
                              if item.get("scaling") not in (None, "", "null") else None),
            "num_registers": (str(item.get("num_registers")).strip()
                              if item.get("num_registers") not in (None, "", "null") else None),
        })
 
    print(f"    [ocr-llm-fallback] LLM recovered {len(rows)} row(s) from raw OCR text")
    return rows
 