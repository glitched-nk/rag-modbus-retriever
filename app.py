# Version-aware retrieval: different doc versions coexist; newest wins per address when doc_hash=None, or pinned to a specific hash.
# FTS5 full-text search index maintained automatically via SQL triggers; search_registers() in database.py is the public search interface.

import os
import re
import pandas as pd
import hashlib
import gc
from datetime import datetime, timezone

from extractor import *
from ocr import *
from table_detect import *
from database import *
#from retriever import retrieve_registers
from llm_identifier import identify_device_from_text, identify_device_rack_from_text, identify_rack_from_context
from json_formatter import generate_json
from json_export import save_json
from ocr_llm_fallback import *

UPLOAD_FOLDER  = "uploads"
IMAGE_FOLDER   = "temp_images"
OUTPUT_FOLDER  = "output"
TABLE_CSV_DIR  = "tables"

os.makedirs(IMAGE_FOLDER, exist_ok=True)

for d in [IMAGE_FOLDER, OUTPUT_FOLDER, TABLE_CSV_DIR]:
    os.makedirs(d, exist_ok=True)

TEXT_GAP_THRESHOLD = 300

def _file_hash(path: str) -> str:   #returns the SHA-256 hex digest of a file's raw bytes
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _make_label(description: str) -> str:
    return (
        description.lower().strip()
        .replace(" ", "_").replace("-", "_").replace("/", "_")
        .replace("(", "").replace(")", "").replace(".", "")
    )

def normalize_rack(rack: str) -> str: #lowercase and strip spaces/hyphens for fuzzy rack matching
    return re.sub(r"[\s\-_]", "", rack.lower())

def _racks_match(rack_a: str, rack_b: str) -> bool: #Return True if two rack strings refer to the same model
    return normalize_rack(rack_a) == normalize_rack(rack_b)

#multi-page table
def _tables_are_compatible(a: dict, b: dict) -> bool:
    df_a = a["table"]
    df_b = b["table"]

    if df_a.empty or df_b.empty:
        return False
    if df_a.shape[1] != df_b.shape[1]:
        return False
    
    first_row_values = [str(v) for v in df_b.iloc[1].tolist()]      #checks 2nd row
    first_row_text   = " ".join(first_row_values).lower()
    header_keywords = [
        "address", "register", "parameter", "datatype",
        "data type", "description", "sl.no", "sl no", "name", "type"]
    
    """ first row of b shouldnt look like a header:
    if any(kw in first_row_text for kw in header_keywords):
        return False
    """
 
    addr_pat = re.compile(r"^\d{3,7}$")
    if not any(addr_pat.match(cell.strip()) for cell in first_row_values):
        return False
 
    gap_text = b.get("gap_text_before", "")
    if len(gap_text.strip()) > TEXT_GAP_THRESHOLD:
        print(f"    [merge-guard] Gap = {len(gap_text)} chars → NOT merging")
        return False
 
    page_a = a.get("page")
    page_b = b.get("page")
    if page_a is not None and page_b is not None and page_b != page_a:
        if len(gap_text.strip()) > 60:
            print(f"    [merge-guard] Page {page_a}→{page_b}, "
                  f"gap={len(gap_text)} chars → NOT merging")
            return False
 
    return True
 
def merge_continued_tables(tables: list) -> list:
    if not tables:
        return []
    merged = []
    current = {
        "table": tables[0]["table"].copy(),
        "context": tables[0].get("context", ""),
        "page": tables[0].get("page", 0)}

    for nxt in tables[1:]:

        if _tables_are_compatible(current, nxt):
            current["table"] = pd.concat([current["table"], nxt["table"]], ignore_index=True)
            #append nearby text context too
            current["context"] += "\n" + nxt.get("context", "")
            print(f"    [merge] Appended continuation -> now {len(current['table'])} rows")

        else:
            merged.append(current)
            current = {
                "table": nxt["table"].copy(),
                "context": nxt.get("context", ""),
                "page": nxt.get("page", 0)
            }
    merged.append(current)

    print(f"    [merge] {len(tables)} raw tables -> {len(merged)} after merging")
    return merged

#if useful tables exist
def _validate_roles(roles: dict, label: str) -> bool:
    if roles.get("address", -1) == -1:
        print(f"  [SKIP] {label}: address column not detected — skipping.")
        print(f"         Roles: {roles}")
        return False
    if roles.get("description", -1) == -1:
        print(f"  [WARN] {label}: description column not detected.")
    return True

def _looks_like_address(value: str) -> bool:
    if not value:
        return False
    cleaned = value.strip().replace(",", "")
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]
    if not cleaned.isdigit():
        return False
    return 3 <= len(cleaned) <= 7

def _clean_address(value: str) -> str:
    cleaned = value.strip().replace(",", "")
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]
    return cleaned

def _find_best_address_column(df, start_row: int) -> int:

    n_rows = len(df) - start_row
    if n_rows <= 0:
        return -1

    best_col, best_score = -1, 0.0
    for col_idx in range(df.shape[1]):
        hits = 0
        for i in range(start_row, len(df)):
            cell = str(df.iloc[i, col_idx]).strip()
            if _looks_like_address(cell):
                hits += 1
        score = hits / n_rows
        if score > best_score:
            best_score, best_col = score, col_idx

    # Require at least half the rows to look like addresses before trusting this column as a genuine address column
    return best_col if best_score >= 0.5 else -1


def _try_insert_with_column(df, start_row, addr_col, desc_col, type_col, scaling_col, num_regs_col, company, device_family, device_rack, doc_hash, dry_run: bool) -> tuple:

    inserted = skipped = 0

    for i in range(start_row, len(df)):
        row = df.iloc[i]
        raw_address = str(row.iloc[addr_col]).strip() if addr_col != -1 else ""
        description = str(row.iloc[desc_col]).strip() if desc_col != -1 else ""
        datatype    = str(row.iloc[type_col]).strip() if type_col != -1 else "Unknown"
        scaling     = str(row.iloc[scaling_col]).strip() if scaling_col != -1 else None
        if scaling is not None and scaling.lower() in ("", "nan", "none"):
            scaling = None
        
        num_registers = str(row.iloc[num_regs_col]).strip() if num_regs_col != -1 else None
        if num_registers is not None and num_registers.lower() in ("", "nan", "none"):
            num_registers = None
        
        if not _looks_like_address(raw_address):
            skipped += 1
            continue

        address = _clean_address(raw_address)

        if not dry_run:
            insert_register_v2(company, device_family, device_rack, doc_hash, _make_label(description), address, datatype, description, scaling, num_registers)
        inserted += 1

    return inserted, skipped


def _insert_rows_from_table(df, roles, start_row, company, device_family, device_rack, doc_hash) -> int :
    addr_col = roles.get("address", -1)
    desc_col = roles.get("description", -1)
    type_col = roles.get("datatype", -1)
    scaling_col = roles.get("scaling", -1)
    num_regs_col = roles.get("num_registers", -1)

    # First pass with the column detect_modbus_columns() chose
    inserted, skipped = _try_insert_with_column(df, start_row, addr_col, desc_col, type_col, scaling_col, num_regs_col,
        company, device_family, device_rack, doc_hash, dry_run=True)

    # Auto-correction: if every row failed, the detected column is wrong — re-scan all columns for the one that actually looks like addresses.
    if inserted == 0 and skipped > 0:
        sample = [str(df.iloc[start_row + i, addr_col]) if addr_col != -1 else ""
                  for i in range(min(3, len(df) - start_row))]
        print(f"    [address-fix] Detected address column {addr_col} produced "
              f"0 valid rows out of {skipped}. Sample values: {sample}")

        corrected_col = _find_best_address_column(df, start_row)
        if corrected_col != -1 and corrected_col != addr_col:
            print(f"    [address-fix] Re-routing address column "
                  f"{addr_col} → {corrected_col} and retrying.")
            addr_col = corrected_col
        else:
            print("    [address-fix] No alternative column qualifies "
                  "(need ≥50% address-shaped values) — leaving as-is.")

    # Real insert pass (with corrected column if applicable)
    inserted, skipped = _try_insert_with_column(df, start_row, addr_col, desc_col, type_col, scaling_col, num_regs_col,
        company, device_family, device_rack, doc_hash, dry_run=False)

    print(f"    Inserted {inserted}, skipped {skipped}")
    return inserted

def _insert_rows_from_ocr(ocr_rows, company, device_family, device_rack, doc_hash) -> int:
    import re
    addr_pat = re.compile(r"^\d{3,7}$")
    dtype_words = {"float", "int32", "int16", "uint16", "long", "unsigned","uint32", "string", "word", "dword", "double"}
    inserted = 0
    for row in ocr_rows:
        if len(row) < 2:
            continue
        address = ""; addr_idx = -1
        for idx, cell in enumerate(row):
            if addr_pat.match(cell.strip()):
                address = cell.strip(); addr_idx = idx; break
        if not address:
            continue
        datatype = "Unknown"; type_idx = -1
        for idx, cell in enumerate(row):
            cell_lower = cell.strip().lower()
            if any(dt in cell_lower for dt in dtype_words):
                datatype = cell.strip(); type_idx = idx; break
        remaining = [(i, c) for i, c in enumerate(row)
                     if i != addr_idx and i != type_idx]
        if not remaining:       #check if needed
            continue
        description = max(remaining, key=lambda t: len(t[1]))[1].strip()
        if not description:
            continue
        insert_register(company, device_family, device_rack, _make_label(description), address, datatype, description, doc_hash, scaling= None, num_registers=None)
        inserted += 1
    print(f"    OCR inserted {inserted} rows")
    return inserted

def _insert_rows_from_llm_fallback(llm_rows: list, company, device_family, device_rack, doc_hash) -> int:
    inserted = 0
    for row in llm_rows:
        address = row.get("address", "")
        description = row.get("description", "")
        if not address or not description:
            continue
        insert_register_v2(
            company, device_family, device_rack, doc_hash,
            _make_label(description), address,
            row.get("datatype") or "Unknown", description,
            row.get("scaling"), row.get("num_registers"),
        )
        inserted += 1
    print(f"    LLM fallback inserted {inserted} rows")
    return inserted

def _collect_all_text(files: list) -> str:
    combined = []
    for file_path in files:
        ext = file_path.lower().rsplit(".", 1)[-1]
        print(f"  [TEXT] Collecting from: {os.path.basename(file_path)}")
        text = ""

        # PDFs
        if ext == "pdf":
            text = extract_pdf_text_with_ocr_fallback(file_path, IMAGE_FOLDER, extract_text_from_image)
            if text.strip():
                combined.append(text[:4000])

        # Images
        elif ext in ("png", "jpg", "jpeg", "bmp", "tiff"):
            text = extract_text_from_image(file_path)
            if text.strip():
                combined.append(text[:4000])

        # DOCX
        elif ext == "docx":
            text = extract_docx_text(file_path)
            if text.strip():
                combined.append(text[:4000])

        # Excel
        elif ext in ("xlsx", "xls"):
            text = extract_excel_text(file_path)
            if text.strip():
                combined.append(text[:4000])

    return "\n\n------\n\n".join(combined)

def _run_llm_identification(all_text, user_company, user_family, user_rack):
    company       = user_company
    device_family = user_family
    device_rack   = user_rack

    if not all_text.strip():
        print("  [LLM] No text found — using user-supplied values.")
        return company, device_family, device_rack
 
    if not company or not device_family:
        print("  [LLM] Identifying company + device family (1 API call)...")
        ident = identify_device_from_text(all_text, company, device_family)
        print(f"         {ident}")
        if not company and ident.get("company"):
            company = ident["company"]
        if not device_family and ident.get("device_family"):
            device_family = ident["device_family"]
 
    if not device_rack:
        print("  [LLM] Identifying device rack (1 API call)...")
        rack_info = identify_device_rack_from_text(all_text, company, device_family, device_rack)
        print(f"         {rack_info}")
        if rack_info.get("device_rack"):
            device_rack = rack_info["device_rack"]
 
    return company, device_family, device_rack

def _new_rack_state() -> dict:
    return {"last_known_rack": ""}

def _keyword_search_rack(context: str, target_rack: str) -> tuple:
    if not context or not target_rack:
        return ("miss", "")
    
    norm_target  = normalize_rack(target_rack)
    norm_context = normalize_rack(context)
 
    # 1st check: normalised substring match
    if norm_target in norm_context:
        print(f"    [rack/keyword] Pass 1 exact hit for '{target_rack}'")
        return ("found", target_rack)
 
    # 2nd check: target appears on a line that also contains a heading keyword
    heading_keywords = ["register", "modbus", "map", "section", "table", "chapter", "device", "meter", "model", "type"]
    for line in context.splitlines():
        norm_line = normalize_rack(line)
        if norm_target in norm_line:
            line_lower = line.lower()
            if any(kw in line_lower for kw in heading_keywords):
                print(f"    [rack/keyword] Pass 2 heading hit: {line.strip()!r}")
                return ("found", target_rack)
 
    # 3rd check: all characters of target appear in order within one line
    target_chars = norm_target
    for line in context.splitlines():
        norm_line = normalize_rack(line)
        idx = 0
        for ch in target_chars:
            idx = norm_line.find(ch, idx)
            if idx == -1:
                break
            idx += 1
        else:
            print(f"    [rack/keyword] Pass 3 partial hit on line: {line.strip()!r}")
            return ("partial", target_rack)
 
    return ("miss", "")

def _resolve_table_rack(context, company, device_family, target_rack, fallback_rack, rack_state) -> tuple:
    if not context or not context.strip():
        print(f"    [rack] No context for this table — cannot confirm "
              f"'{target_rack or fallback_rack}' → SKIP")
        return (target_rack or fallback_rack), False

    if target_rack:
        hit_type, found_rack = _keyword_search_rack(context, target_rack)

        if hit_type in ("found", "partial"):
            rack_state["last_known_rack"] = found_rack
            print(f"    [rack] Keyword {hit_type} → '{found_rack}'")
            return found_rack, True

        print("    [rack] Keyword miss in this table's context — calling LLM "
              "for an independent check (no inheritance)...")
        try:
            result     = identify_rack_from_context(context, company, device_family, fallback_rack)
            table_rack = result.get("device_rack") or ""
            confidence = result.get("confidence", "low")
        except Exception as e:
            print(f"    [rack] LLM error: {e} — treating as unconfirmed")
            table_rack = ""
            confidence = "low"

        print(f"    [rack] LLM → '{table_rack or '(none)'}' (confidence: {confidence})")

        if table_rack and _racks_match(table_rack, target_rack) and confidence != "low":
            rack_state["last_known_rack"] = table_rack
            return table_rack, True

        print(f"    [rack] No fresh confirmation for '{target_rack}' in this "
              f"table's own context → SKIP (likely a merge gap or different "
              f"rack's section)")
        return (table_rack or target_rack), False

    try:
        result     = identify_rack_from_context(context, company, device_family, fallback_rack)
        table_rack = result.get("device_rack") or fallback_rack
        confidence = result.get("confidence", "low")
    except Exception as e:
        print(f"    [rack] LLM error: {e} — using fallback")
        table_rack = fallback_rack
        confidence = "low"

    print(f"    [rack] LLM → '{table_rack}' (confidence: {confidence})")
    if table_rack:
        rack_state["last_known_rack"] = table_rack
    return table_rack, True

def _process_pdf(file_path, file_name, company, device_family, target_rack, fallback_rack, doc_hash) -> int:
    total    = 0
    rack_state = _new_rack_state()
    raw_text = extract_pdf_text(file_path)
    scanned  = is_scanned_pdf(file_path) or len(raw_text.strip()) < 50
 
    if not scanned:
        print("  Text PDF — reading page text…")
        page_texts = extract_pdf_pages_with_text(file_path)
        print("  Extracting tables…")
        raw_tables = extract_tables_from_pdf(file_path, page_texts)
        gc.collect()
 
        if not raw_tables:
            print("  Camelot found nothing — trying pdfplumber…")
            raw_tables = extract_tables_from_pdfplumber(file_path, page_texts)
 
        print(f"  {len(raw_tables)} raw table(s) found")

        for t in raw_tables:
            t["table"] = decode_cid_dataframe(t["table"])

        # Populate gap_text_before on each table dict before merging
        raw_tables    = enrich_tables_with_gap_text(raw_tables, page_texts)
        tables        = merge_continued_tables(raw_tables)
        modbus_tables = [t for t in tables if is_modbus_table(t["table"])]
        print(f"  {len(modbus_tables)} Modbus table(s) after merge+filter")
 
        for t_idx, tbl in enumerate(modbus_tables):
            df      = tbl["table"]
            context = tbl.get("context", "")
            label   = f"{file_name} table {t_idx + 1}"
            roles   = detect_modbus_columns(df)
            start   = find_data_start_row(df, roles)
 
            if not _validate_roles(roles, label):
                continue
 
            try:
                table_rack, should_insert = _resolve_table_rack(context, company, device_family, target_rack, fallback_rack, rack_state)
            except Exception as e:
                print(f"  [rack] error: {e} — fallback")
                table_rack, should_insert = fallback_rack, True
 
            if not should_insert:
                continue
 
            print(f"  Table {t_idx+1}: rack='{table_rack}', "
                  f"data_rows={len(df) - start}")
            df.to_csv(
                os.path.join(TABLE_CSV_DIR, f"{file_name}_t{t_idx+1}.csv"),
                index=False)
            total += _insert_rows_from_table(df, roles, start, company, device_family, table_rack, doc_hash)
 
    else:
        print("  Scanned PDF — running OCR…")
        image_paths = convert_pdf_to_images(file_path, IMAGE_FOLDER)
 
        for img_path in image_paths:
            print(f"  OCR page: {os.path.basename(img_path)}")
            page_context = extract_text_from_image(img_path)

            if not page_context.strip():
                from ocr import PADDLE_AVAILABLE
                if not PADDLE_AVAILABLE:
                    print("  [WARNING] OCR text is empty AND PaddleOCR is not installed — this page cannot be processed at all. "
                          "Install PaddleOCR (pip install paddleocr) to enable scanned-document support.")
                else:
                    print("  [OCR] No text recognised on this page — skipping.")
                        
            try:
                page_rack, should_insert = _resolve_table_rack(page_context, company, device_family, target_rack, fallback_rack, rack_state)
            except Exception as e:
                print(f"  [rack] error: {e} — fallback")
                page_rack, should_insert = fallback_rack, True
 
            if not should_insert:
                print("  Skipping page — rack mismatch")
                continue
 
            rows     = extract_table_rows_from_image(img_path)
            reg_rows = filter_register_rows(rows)
            print(f"  {len(reg_rows)} candidate rows, rack='{page_rack}'")

            if should_use_llm_fallback(page_context, len(reg_rows)):
                print(f"  [hybrid] Cell parser yielded {len(reg_rows)} row(s) "
                      f"for {len(page_context.strip())} chars of OCR text — "
                      f"too sparse, falling back to LLM on raw text.")
                llm_rows = extract_rows_via_llm(page_context, company, device_family, page_rack)
                total += _insert_rows_from_llm_fallback(llm_rows, company, device_family, page_rack, doc_hash)
            else:
                total += _insert_rows_from_ocr(reg_rows, company, device_family, page_rack, doc_hash)

        for img_path in image_paths:
            try:
                os.remove(img_path)
            except Exception:
                pass
 
    return total
 
 
def _process_image(file_path, file_name, company, device_family, target_rack, fallback_rack, doc_hash) -> int:
    rack_state = _new_rack_state()
    print("  Running OCR on image…")
    image_context = extract_text_from_image(file_path)

    if not image_context.strip():
        from ocr import PADDLE_AVAILABLE
        if not PADDLE_AVAILABLE:
            print("  [WARNING] OCR text is empty AND PaddleOCR is not installed — this image cannot be processed at all. "
                  "Install PaddleOCR (pip install paddleocr) to enable image OCR support.")
        else:
            print("  [OCR] No text recognised in this image.")
    
    try:
        img_rack, should_insert = _resolve_table_rack(image_context, company, device_family, target_rack, fallback_rack, rack_state)
    except Exception as e:
        print(f"  [rack] error: {e} — fallback")
        img_rack, should_insert = fallback_rack, True
 
    if not should_insert:
        print("  Skipping image — rack mismatch")
        return 0
 
    rows     = extract_table_rows_from_image(file_path)
    reg_rows = filter_register_rows(rows)
    print(f"  {len(reg_rows)} candidate rows, rack='{img_rack}'")
    if should_use_llm_fallback(image_context, len(reg_rows)):
        print(f"  [hybrid] Cell parser yielded {len(reg_rows)} row(s) "
              f"for {len(image_context.strip())} chars of OCR text — "
              f"too sparse, falling back to LLM on raw text.")
        llm_rows = extract_rows_via_llm(image_context, company, device_family, img_rack)
        return _insert_rows_from_llm_fallback(llm_rows, company, device_family, img_rack, doc_hash)
    
    return _insert_rows_from_ocr(reg_rows, company, device_family, img_rack, doc_hash)
 
 
def _process_excel(file_path, file_name, company, device_family, target_rack, fallback_rack, doc_hash) -> int:
    total  = 0
    rack_state = _new_rack_state()
    sheets = extract_excel(file_path)
 
    for sheet_name, df in sheets.items():
        label = f"{file_name} sheet '{sheet_name}'"
        print(f"\n  Sheet: {sheet_name}")
 
        if not is_modbus_table(df):
            print("  Not a Modbus table — skipping.")
            continue
 
        roles = detect_modbus_columns(df)
        start = find_data_start_row(df, roles)
 
        if not _validate_roles(roles, label):
            continue
 
        sheet_context = "\n".join(
            " ".join(str(v) for v in df.iloc[r].tolist()
                     if str(v) not in ("nan", ""))
            for r in range(min(10, len(df)))
        )
 
        try:
            sheet_rack, should_insert = _resolve_table_rack(sheet_context, company, device_family, target_rack, fallback_rack, rack_state)
        except Exception as e:
            print(f"  [rack] error: {e} — fallback")
            sheet_rack, should_insert = fallback_rack, True
 
        if not should_insert:
            print("  Skipping sheet — rack mismatch")
            continue
 
        print(f"  rack='{sheet_rack}', data_rows={len(df) - start}")
        total += _insert_rows_from_table(df, roles, start, company, device_family, sheet_rack, doc_hash)
 
    return total
 
 
def _process_docx(file_path, file_name, company, device_family, target_rack, fallback_rack, doc_hash) -> int:
    total = 0
    rack_state = _new_rack_state()
    print("  Extracting tables from DOCX…")
    raw_tables    = extract_tables_from_docx(file_path)
    # No page_texts for DOCX — gap_text_before falls back to empty string
    raw_tables    = enrich_tables_with_gap_text(raw_tables, page_texts=None)
    modbus_tables = [t for t in raw_tables if is_modbus_table(t["table"])]
    print(f"  {len(modbus_tables)} Modbus table(s)")
 
    for t_idx, tbl in enumerate(modbus_tables):
        df      = tbl["table"]
        context = tbl.get("context", "")
        label   = f"{file_name} table {t_idx + 1}"
        roles   = detect_modbus_columns(df)
        start   = find_data_start_row(df, roles)
 
        if not _validate_roles(roles, label):
            continue
 
        try:
            table_rack, should_insert = _resolve_table_rack(context, company, device_family, target_rack, fallback_rack, rack_state)
        except Exception as e:
            print(f"  [rack] error: {e} — fallback")
            table_rack, should_insert = fallback_rack, True
 
        if not should_insert:
            print(f"  Skipping DOCX table {t_idx+1} — rack mismatch")
            continue
 
        print(f"  DOCX Table {t_idx+1}: rack='{table_rack}', "
              f"data_rows={len(df) - start}")
        total += _insert_rows_from_table(df, roles, start, company, device_family, table_rack, doc_hash)
 
    return total


## main program
def main():

    init_db() 

    user_company = input("Company Name : ").strip()
    user_family = input("Device Family : ").strip()
    user_rack = input("Device Rack : ").strip()

    files = []

    for f in sorted(os.listdir(UPLOAD_FOLDER)):
        if not f.startswith("."):
            full_path = os.path.join(UPLOAD_FOLDER, f)
            files.append(full_path)
    if not files:
        print(f"No files found in '{UPLOAD_FOLDER}/'. Exiting.")
        return
    
    print(f"\nFiles: {[os.path.basename(f) for f in files]}\n")

    files_needing_full_extraction:   list[tuple[str, str]] = []  # (path, hash)
    files_needing_rack_extraction:   list[tuple[str, str, dict]] = []  # (path, hash, cached_meta)
    cached_meta: dict | None = None  # metadata from any already-seen file
 
    for file_path in files:
        doc_hash = _file_hash(file_path)
        existing = get_document(doc_hash)
 
        if existing is None:
            # Never seen this file before
            files_needing_full_extraction.append((file_path, doc_hash))
        elif rack_already_extracted(doc_hash, user_rack):
            # File seen AND this exact rack already extracted — pure cache hit
            print(f"  [cache] '{os.path.basename(file_path)}' already processed "
                  f"for rack '{user_rack}' — skipping extraction.")
            cached_meta = existing
        else:
            # File seen but this rack is new — re-extract for this rack only
            print(f"  [cache] '{os.path.basename(file_path)}' known file, "
                  f"but rack '{user_rack}' not yet extracted — re-extracting.")
            files_needing_rack_extraction.append((file_path, doc_hash, existing))
            cached_meta = existing
 
    #Determine company / device_family
    # We always need these two values before extraction can run.
    # They come from LLM only for brand-new files; for cached files we reuse
    # the stored values and skip the LLM identification step entirely.
 
    if files_needing_full_extraction:
        new_paths = [fp for fp, _ in files_needing_full_extraction]
        print("Collecting text for device identification…")
        all_text = _collect_all_text(new_paths)
        company, device_family, device_rack = _run_llm_identification(
            all_text, user_company, user_family, user_rack)
        if not company:       company       = "Unknown"
        if not device_family: device_family = "Unknown"
        target_rack   = device_rack
        fallback_rack = target_rack or device_family
        print(f"\nIdentified: {company} / {device_family} / "
              f"rack='{target_rack or '(all)'}' ")
 
    elif files_needing_rack_extraction or cached_meta:
        # All files are known — reuse stored company/family
        meta          = (files_needing_rack_extraction[0][2]
                         if files_needing_rack_extraction else cached_meta)
        company       = meta["company"]       or user_company  or "Unknown"
        device_family = meta["device_family"] or user_family   or "Unknown"
        target_rack   = user_rack
        fallback_rack = target_rack or device_family
        print(f"\nUsing cached identity: {company} / {device_family} / "
              f"rack='{target_rack}'")
 
    else:
        print("No files found to process.")
        return
 
    # Extract: brand-new files
    total_inserted = 0
 
    for file_path, doc_hash in files_needing_full_extraction:
        file_name = os.path.basename(file_path)
        ext       = file_name.lower().rsplit(".", 1)[-1]
 
        print(f"\n{'='*10}\nProcessing (new): {file_name}\n{'='*10}")
 
        if ext == "pdf":
            n = _process_pdf(file_path, file_name, company, device_family, target_rack, fallback_rack, doc_hash)
        elif ext in ("png", "jpg", "jpeg", "tiff", "bmp"):
            n = _process_image(file_path, file_name, company, device_family,  target_rack, fallback_rack, doc_hash)
        elif ext in ("xlsx", "xls"):
            n = _process_excel(file_path, file_name, company, device_family, target_rack, fallback_rack, doc_hash)
        elif ext == "docx":
            n = _process_docx(file_path, file_name, company, device_family, target_rack, fallback_rack, doc_hash)
        else:
            print(f"  Unsupported extension: .{ext}")
            n = 0
 
        if n > 0:
            register_document(doc_hash, file_name, company, device_family,
                              target_rack or device_family)
        total_inserted += n
 
    # Extract: known file, new rack
    for file_path, doc_hash, meta in files_needing_rack_extraction:
        file_name = os.path.basename(file_path)
        ext       = file_name.lower().rsplit(".", 1)[-1]
 
        print(f"\n{'='*10}\nRe-extracting (new rack '{target_rack}'): "
              f"{file_name}\n{'='*10}")
 
        if ext == "pdf":
            n = _process_pdf(file_path, file_name, company, device_family, target_rack, fallback_rack, doc_hash)
        elif ext in ("png", "jpg", "jpeg", "tiff", "bmp"):
            n = _process_image(file_path, file_name, company, device_family, target_rack, fallback_rack, doc_hash)
        elif ext in ("xlsx", "xls"):
            n = _process_excel(file_path, file_name, company, device_family, target_rack, fallback_rack, doc_hash)
        elif ext == "docx":
            n = _process_docx(file_path, file_name, company, device_family, target_rack, fallback_rack, doc_hash)
        else:
            print(f"  Unsupported extension: .{ext}")
            n = 0
 
        if n > 0:
            # Append the new rack to this document's extracted_racks list
            register_document(doc_hash, file_name, company, device_family, target_rack)
        total_inserted += n
 
    # Report extraction results (only when extraction actually ran)
    if files_needing_full_extraction or files_needing_rack_extraction:
        print(f"\n{'='*20}\nTotal rows inserted: {total_inserted}")
        if total_inserted == 0:
            print("WARNING: Nothing extracted — check the extraction pipeline.")
            return
    
    retrieve_rack = target_rack if target_rack else fallback_rack
    print(f"\nRetrieving for {company} / {device_family} / {retrieve_rack}...")
    
    raw_registers = retrieve_registers_v2(company, device_family, retrieve_rack)
    print(f"Retrieved {len(raw_registers)} register(s)")
 
    if not raw_registers:
        print("No registers found. Check rack name matching.")
        return
    
    #register_rows = [{"label": r[0], "address": r[1], "datatype": r[2], "description": r[3]}
    #                for r in registers]

    seen_addresses  = set()
    header_patterns = {"address", "register", "parameter", "description",
                       "data type", "datatype", "type", "label"}
    register_rows   = []
 
    for r in raw_registers:
        label       = r["label"]
        address     = r["address"]
        datatype    = r["datatype"]
        description = r["description"]
        scaling     = r.get("scaling")
        num_registers = r.get("num_registers")
 
        if description.strip().lower() in header_patterns:
            continue
        if address.strip().lower() in header_patterns:
            continue
        if address in seen_addresses:
            continue
        seen_addresses.add(address)
 
        register_rows.append({
            "label":       label,
            "address":     address,
            "datatype":    datatype,
            "description": description,
            "scaling":     scaling,
            "num_registers": num_registers,
        })
 
    print(f"  {len(register_rows)} unique register(s) after dedup + header strip")

    print("Formatting deterministically")
    json_output = generate_json(register_rows, company, device_family)
 
    safe_rack = device_family.replace(" ", "_").replace("/", "_")
    output_file = os.path.join(OUTPUT_FOLDER, f"{safe_rack}.json")
    save_json(json_output, output_file)
    print(f"\nSaved: {output_file}")

if __name__ == "__main__":
    main()