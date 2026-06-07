import os
import re
import pandas as pd
import gc

from extractor import *
from ocr import *
from table_detect import *
from database import insert_register, clear_registers
from retriever import retrieve_registers
from llm_identifier import identify_device_from_text, identify_device_rack_from_text, identify_rack_from_context
from llm_formatter import generate_json
from json_export import save_json

UPLOAD_FOLDER  = "uploads"
IMAGE_FOLDER   = "temp_images"
OUTPUT_FOLDER  = "output"
TABLE_CSV_DIR  = "tables"

os.makedirs(IMAGE_FOLDER, exist_ok=True)

for d in [IMAGE_FOLDER, OUTPUT_FOLDER, TABLE_CSV_DIR]:
    os.makedirs(d, exist_ok=True)
 
 
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
    first_row_values = [str(v) for v in df_b.iloc[0].tolist()]
    first_row_text   = " ".join(first_row_values).lower()
    header_keywords = [
        "address", "register", "parameter", "datatype",
        "data type", "description", "sl.no", "sl no", "name", "type"]
    if any(kw in first_row_text for kw in header_keywords):
        return False
 
    addr_pat = re.compile(r"^\d{2,7}$")
    return any(addr_pat.match(cell.strip()) for cell in first_row_values) 
 
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

    print(f"    [merge] {len(tables)} raw tables "
        f"-> {len(merged)} after merging")
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
 
def _insert_rows_from_table(df, roles, start_row, company, device_family, device_rack):
    addr_col = roles.get("address", -1)
    desc_col = roles.get("description", -1)
    type_col = roles.get("datatype", -1)
    inserted = skipped = 0
    for i in range(start_row, len(df)):
        row = df.iloc[i]
        address = str(row.iloc[addr_col]).strip() if addr_col != -1 else ""
        description = str(row.iloc[desc_col]).strip() if desc_col != -1 else ""
        datatype = str(row.iloc[type_col]).strip() if type_col != -1 else "Unknown"
        if not address or not address.isdigit():
            skipped += 1
            continue
        #if description.lower() in ("", "nan", "none"):
            #skipped += 1
            #continue
        insert_register(company, device_family, device_rack, _make_label(description), address, datatype, description)
        inserted += 1
    print(f"    Inserted {inserted}, skipped {skipped}")
    return inserted
 
 
def _insert_rows_from_ocr(ocr_rows, company, device_family, device_rack):
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
        insert_register(
            company, device_family, device_rack,
            _make_label(description), address, datatype, description
        )
        inserted += 1
    print(f"    OCR inserted {inserted} rows")
    return inserted

def _collect_all_text(files: list) -> str:
    combined = []
    for file_path in files:
        ext = file_path.lower().rsplit(".", 1)[-1]
        print(f"  [TEXT] Collecting from: {os.path.basename(file_path)}")

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
        rack_info = identify_device_rack_from_text(
            all_text, company, device_family, device_rack
        )
        print(f"         {rack_info}")
        if rack_info.get("device_rack"):
            device_rack = rack_info["device_rack"]
 
    return company, device_family, device_rack

_last_known_rack: str = ""

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

def _resolve_table_rack(context: str, company: str, device_family: str, target_rack: str, fallback_rack: str) -> tuple:
    if not context or not context.strip():
        print(f"    [rack] No context — using fallback '{fallback_rack}'")
        return fallback_rack, True
    
    if target_rack:
        hit_type, found_rack = _keyword_search_rack(context, target_rack)
 
        if hit_type == "found":     #no LLM
            return found_rack, True
 
        if hit_type == "partial":       #uncertain, nut no LLM
            print(f"    [rack] Partial keyword match — accepting '{found_rack}'")
            return found_rack, True
        
        print(f"    [rack] Keyword miss — falling back to LLM...")
        try:
            result = identify_rack_from_context(context, company, device_family, fallback_rack)
            table_rack = result.get("device_rack") or fallback_rack
            confidence = result.get("confidence", "low")
        except Exception as e:
            print(f"    [rack] LLM error: {e} — using fallback")
            table_rack = fallback_rack
            confidence = "low"
        
        print(f" [rack] LLM identified: '{table_rack}' (confidence: {confidence})")

        if _racks_match(table_rack, target_rack):
            return table_rack, True
        if confidence == "low":
            print(f"    [rack] Low confidence mismatch — including anyway")
            return fallback_rack, True
        print(f"    [rack] SKIP — '{table_rack}' ≠ target '{target_rack}'")
        return table_rack, False
    
    #no target rack given by user
    try:
        result = identify_rack_from_context(context, company, device_family, fallback_rack)
        table_rack = result.get("device_rack") or fallback_rack
        confidence = result.get("confidence", "low")
    except Exception as e:
        print(f"    [rack] LLM error: {e} — using fallback")
        table_rack = fallback_rack
        confidence = "low"
    
    print(f"    [rack] LLM identified: '{table_rack}' (confidence: {confidence})")
    return table_rack, True

## main program 
def main():
    user_company = input("Company Name : ").strip()
    user_device_family = input("Device Family : ").strip()
    user_device_rack = input("Device Rack : ").strip()

    files = []

    for f in sorted(os.listdir(UPLOAD_FOLDER)):
        if not f.startswith("."):
            full_path = os.path.join(UPLOAD_FOLDER, f)
            files.append(full_path)
    if not files:
        print(f"No files found in '{UPLOAD_FOLDER}/'. Exiting.")
        return
    
    print(f"\nFiles: {[os.path.basename(f) for f in files]}\n")

    print("Collecting text for device identification")
    all_text = _collect_all_text(files)
    company, device_family, device_rack = _run_llm_identification(all_text, user_company, user_device_family, user_device_rack)
    if not company: company = "Unknown"
    if not device_family: device_family = "Unknown"
    target_rack = device_rack   # what the user (or initial LLM) said
    fallback_rack = target_rack if target_rack else device_family

    print(f"\nUsing: {company} / {device_family} / rack='{target_rack or '(all)'}'")
    if target_rack:
        clear_registers(company, device_family, target_rack)
    
    total_inserted = 0

    for file_path in files:
        file_name = os.path.basename(file_path)
        ext = file_name.lower().rsplit(".", 1)[-1]
        print(f"\n{'='*10}\nProcessing: {file_name}\n{'='*10}")

        if ext == "pdf":
            raw_text = extract_pdf_text(file_path)
            scanned = is_scanned_pdf(file_path) or len(raw_text.strip()) < 50
 
            if not scanned:
                print("  Reading page text for context…")
                page_texts = extract_pdf_pages_with_text(file_path)
                print("  Extracting tables...")
                raw_tables = extract_tables_from_pdf(file_path, page_texts)
                gc.collect()

                if not raw_tables:
                    print("  Camelot found nothing — trying pdfplumber…")
                    raw_tables = extract_tables_from_pdfplumber(file_path, page_texts)

                print(f"  {len(raw_tables)} raw table(s) found")

                tables = merge_continued_tables(raw_tables) 
                modbus_tables = [t for t in tables if is_modbus_table(t["table"])]
                print(f"  {len(modbus_tables)} Modbus table(s) after merge+filter")

                for t_idx, table_info in enumerate(modbus_tables):
                    df = table_info["table"]
                    context = table_info.get("context", "")
                    label = f"{file_name} table {t_idx+1}"
 
                    roles = detect_modbus_columns(df)
                    start_row = find_data_start_row(df, roles)
 
                    if not _validate_roles(roles, label):
                        continue

                    try:        #CHECK
                        table_rack, should_insert = _resolve_table_rack(context, company, device_family, target_rack, fallback_rack)

                    except Exception as e:
                        print(f"  [context-identification] failed: {e}")
                        table_rack   = fallback_rack
                        should_insert = True
                    
                    if not should_insert:
                            continue
                    
                    print(f"  Table {t_idx+1}: rack='{table_rack}',"
                          f"start_row={start_row}, "
                          f"data_rows={len(df)-start_row}")
                    df.to_csv(os.path.join(TABLE_CSV_DIR, f"{file_name}_t{t_idx+1}.csv"),index=False)
                    total_inserted += _insert_rows_from_table(df, roles, start_row, company, device_family, table_rack)
 
            else:
                print("  Scanned PDF — running OCR...")
                image_paths = convert_pdf_to_images(file_path, IMAGE_FOLDER)

                for img_path in image_paths:
                    page_num = os.path.basename(img_path)  # for logging
                    print(f"  OCR page: {page_num}")
                    page_context = extract_text_from_image(img_path)

                    try:
                        page_rack, should_insert = _resolve_table_rack(page_context, company, device_family, target_rack, fallback_rack)
                    
                    except Exception as e:
                        print(f"  [rack resolution] error: {e} — using fallback")
                        page_rack = fallback_rack
                        should_insert = True
 
                    if not should_insert:
                        print(f"  Skipping page — rack mismatch")
                        continue

                    # TABLE ROW OCR
                    rows     = extract_table_rows_from_image(img_path)
                    reg_rows = filter_register_rows(rows)
                    print(f"    {len(reg_rows)} candidate rows, rack='{page_rack}'")
 
                    total_inserted += _insert_rows_from_ocr(reg_rows, company, device_family, page_rack)
                
                # cleanup temp OCR images
                for img_path in image_paths:
                    try:
                        os.remove(img_path)
                    except Exception:
                        pass
        
        elif ext in ("png", "jpg", "jpeg", "tiff", "bmp"):
            print("Running OCR on image...")

            # Use the full OCR text of this image as context for rack detection
            image_context = extract_text_from_image(file_path)

            try:
                img_rack, should_insert = _resolve_table_rack(image_context, company, device_family, target_rack, fallback_rack)
            except Exception as e:
                print(f" [rack resolution] error: {e} — using fallback")
                img_rack = fallback_rack
                should_insert = True
 
            if not should_insert:
                print(f"  Skipping image — rack mismatch")
            else:
                rows = extract_table_rows_from_image(file_path)
                reg_rows = filter_register_rows(rows)
                print(f"  {len(reg_rows)} candidate rows, rack='{img_rack}'")
                total_inserted += _insert_rows_from_ocr(reg_rows, company, device_family, img_rack)
            
        elif ext in ("xlsx", "xls"):
            sheets = extract_excel(file_path)

            for sheet_name, df in sheets.items():
                label = f"{file_name} sheet '{sheet_name}'"
                print(f"\n  Sheet: {sheet_name}")

                if not is_modbus_table(df):
                    print(f"  Not a Modbus table — skipping.")
                    continue

                roles= detect_modbus_columns(df)
                start_row = find_data_start_row(df, roles)

                if not _validate_roles(roles, label):
                    continue
                
                sheet_context = "\n".join(" ".join(str(v) for v in df.iloc[r].tolist() if str(v) not in ("nan", ""))
                    for r in range(min(10, len(df))))

                try:
                    sheet_rack, should_insert = _resolve_table_rack(sheet_context, company, device_family, target_rack, fallback_rack)
                except Exception as e:
                    print(f"  [rack resolution] error: {e} — using fallback")
                    sheet_rack= fallback_rack
                    should_insert = True
                
                if not should_insert:
                    print(f"  Skipping sheet — rack mismatch")
                    continue

                print(f"  rack='{sheet_rack}', start_row={start_row}, "
                      f"data_rows={len(df) - start_row}")
                total_inserted += _insert_rows_from_table(df, roles, start_row, company, device_family, sheet_rack)
        
        elif ext in ("docx",):
            print("  Extracting tables from DOCX...")
            raw_tables = extract_tables_from_docx(file_path)
            print(f"  {len(raw_tables)} table(s) found in DOCX")
            modbus_tables = [t for t in raw_tables if is_modbus_table(t["table"])]
            print(f"  {len(modbus_tables)} Modbus table(s) detected")

            for t_idx, table_info in enumerate(modbus_tables):
                df= table_info["table"]
                context = table_info.get("context", "")
                label= f"{file_name} table {t_idx+1}"

                roles= detect_modbus_columns(df)
                start_row = find_data_start_row(df, roles)

                if not _validate_roles(roles, label):
                    continue

                try:
                    table_rack, should_insert = _resolve_table_rack(context, company, device_family, target_rack, fallback_rack)
                except Exception as e:
                    print(f"  [rack resolution] error: {e} — using fallback")
                    table_rack = fallback_rack
                    should_insert = True
                
                if not should_insert:
                    print(f"  Skipping DOCX table {t_idx+1} — rack mismatch")
                    continue
 
                print(f"  DOCX Table {t_idx+1}: rack='{table_rack}', "
                      f"start_row={start_row}, "
                      f"data_rows={len(df) - start_row}")
                total_inserted += _insert_rows_from_table(df, roles, start_row, company, device_family, table_rack)
        
        else:
            print(f"  Unsupported: .{ext}")
    
    print(f"\n{'='*20}\nTotal rows inserted: {total_inserted}")
    if total_inserted == 0:
        print("WARNING: Nothing extracted")             #run test_extractigon.py to debu
        return
    
    retrieve_rack = target_rack if target_rack else fallback_rack
    print(f"\nRetrieving for {company} / {device_family} / {retrieve_rack}...")
    registers = retrieve_registers(company, device_family, retrieve_rack)
    print(f"Retrieved {len(registers)} register(s)")

    if not registers:
        print("No registers found in DB for those keys. Check rack name matching.")
        return
    
    register_rows = [{"label": r[0], "address": r[1], "datatype": r[2], "description": r[3]}
                    for r in registers]
 
    print("Formatting with LLM...")
    json_output = generate_json(register_rows, company, device_family)
 
    safe_rack = device_family.replace(" ", "_").replace("/", "_")
    output_file = os.path.join(OUTPUT_FOLDER, f"{safe_rack}.json")
    save_json(json_output, output_file)
    print(f"\nSaved: {output_file}")

if __name__ == "__main__":
    main()