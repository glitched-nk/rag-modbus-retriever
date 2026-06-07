import os
import json

def save_json(data: str, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
 
    parsed = json.loads(data)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
 
    print(f"[json_export] Saved {len(parsed)} registers → {filepath}")