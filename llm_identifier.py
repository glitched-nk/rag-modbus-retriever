import os
import json
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)

def identify_device_from_text(raw_text: str, user_company: str = "", user_family: str = "") -> dict:
    prompt = f"""You are analyzing a Modbus register map document.
The user told you:
  - Company     : "{user_company}"
  - Device family: "{user_family}"
 
Below is the raw text extracted from the document (first ~3000 characters):

---
{raw_text[:3000]}
---
 
Based on this text, identify:
1. The manufacturer/company that produced the device.
2. The device family or product line (e.g. "EN8400N", "MFM376", "i-LINK").
 
Return ONLY a JSON object with these exact keys (no markdown, no explanation):
{{
  "company": "<company name>",
  "device_family": "<device family or model series>",
  "confidence": "<high|medium|low>",
  "notes": "<any relevant caveats or observations>"
}}
"""
    
    response = llm.invoke(prompt)
    raw = response.content.strip()
 
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
 
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "company": user_company,
            "device_family": user_family,
            "confidence": "low",
            "notes": f"LLM response could not be parsed: {raw[:200]}"
        }
 
 
def identify_device_rack_from_text(raw_text: str, company: str, device_family: str, user_rack: str = "") -> dict:
    prompt = f"""You are analyzing a Modbus register map for:
  Company      : {company}
  Device family: {device_family}
 
The user believes the device rack/variant is: "{user_rack}"
 
Here is a sample of the document text:
---
{raw_text[:2000]}
---
 
Identify the specific device rack, variant, or model number mentioned in this document
(e.g. "EN8400N", "MFM376-C-CE", "i-Link 310"). This is usually a sub-model within the family.
 
Return ONLY a JSON object:
{{
  "device_rack": "<specific model/variant>",
  "confidence": "<high|medium|low>",
  "notes": "<any caveats>"
}}
"""
    
    response = llm.invoke(prompt)
    raw = response.content.strip()
    
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
 
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "device_rack": user_rack,
            "confidence": "low",
            "notes": f"LLM parse error: {raw[:200]}"
        }
    
def identify_rack_from_context(context: str, company: str, device_family: str, known_rack: str = "") -> dict:
    if not context or len(context.strip()) < 20:
        return {"device_rack": known_rack, "confidence": "low"}
 
    prompt = f"""You are analyzing one section of a Modbus register map document.
The document is for:
  Company      : {company}
  Device family: {device_family}
 
The text below is from the page(s) immediately surrounding a specific register table.
It typically contains a section heading that names the device rack or model variant
this table belongs to.
 
Page text:
---
{context[:800]}
---
 
Which device rack or model variant does this table belong to?
Examples of expected values: "EM6400N", "EM6430", "EM6433", "PM5110", "MFM376-C"
 
Rules:
- Return the most specific model identifier you can find in the text.
- If the text clearly names a model, confidence = "high".
- If you're inferring from partial info, confidence = "medium".
- If the text gives no clues, return known_rack="{known_rack}" with confidence = "low".
 
Return ONLY a JSON object, no markdown:
{{"device_rack": "", "confidence": "high|medium|low"}}
"""
    response = llm.invoke(prompt)
    raw = response.content.strip()
    
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
 
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "device_rack": known_rack,
            "confidence": "low",
            "notes": f"LLM parse error: {raw[:200]}"
        }
 