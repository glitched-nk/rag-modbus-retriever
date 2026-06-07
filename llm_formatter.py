import os
import json
from langchain_openai import ChatOpenAI

# $env:OPENAI_API_KEY="your_key"  -> need to set in terminal

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)

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
 
FORMAT_RULES = """Rules for filling each field:
- Label        : snake_case version of the description (e.g. "watts_total")
- Address      : keep the numeric address EXACTLY as extracted (e.g. "40101")
- Format       : the data type from the register map ("Float", "Int32", "Long", etc.)
- Scaling      : use 1 unless the register map explicitly states a scale factor.
- Multiplier   : use 1 unless the register map explicitly states a multiplier.
- Description  : the human-readable parameter name exactly as in the source.
- Display Name : null (leave as null unless a separate display name is given).
"""
def infer_register_count(datatype: str) -> int:

    dt = str(datatype).lower()

    if any(x in dt for x in [
        "float",
        "int32",
        "uint32",
        "dword",
        "long",
        "double"
    ]):
        return 2

    return 1
 
def generate_json(register_rows: list, company: str = "", device_family: str = "") -> str:
    if not register_rows:
        return "[]"
    
    for row in register_rows:
        row["register_count"] = infer_register_count(
            row.get("datatype", "")
        )

    # Chunk into batches of 80 rows to stay within token limits
    batches = [register_rows[i:i+80] for i in range(0, len(register_rows), 80)]
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
 
Input data:
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
            all_results.extend(parsed)
        except json.JSONDecodeError as e:
            print(f"  [llm_formatter] JSON parse error in batch {batch_idx+1}: {e}")
            print(f"  Raw output (first 500 chars): {raw[:500]}")
 
    return json.dumps(all_results, indent=2)