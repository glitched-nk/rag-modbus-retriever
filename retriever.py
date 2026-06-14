from database import retrieve_registers_v2

def retrieve_registers(company: str, family: str, rack: str, doc_hash: str | None = None) -> list:
    rows = retrieve_registers_v2(company, family, rack, doc_hash)
 
    seen_addresses: set = set()
    deduped: list = []
 
    for r in rows:
        address = r["address"]
        if address not in seen_addresses:
            seen_addresses.add(address)
            # Return as tuple to preserve backward compatibility
            deduped.append((
                r["label"],
                r["address"],
                r["datatype"],
                r["description"],
            ))
 
    if len(deduped) < len(rows):
        print(f"  [retriever] Removed {len(rows) - len(deduped)} "
              f"duplicate address(es)")
 
    return deduped