from database import cursor

def retrieve_registers(company: str, family: str, rack: str) -> list:
    cursor.execute("""
        SELECT DISTINCT label, address, data_type, description
        FROM   registers
        WHERE  company=? AND device_family=? AND device_rack=?
        ORDER  BY CAST(address AS INTEGER)
    """, (company, family, rack))
    rows = cursor.fetchall()
    
    seen_addresses = set()
    deduped = []
    for row in rows:
        address = row[1]
        if address not in seen_addresses:
            seen_addresses.add(address)
            deduped.append(row)
    
    if len(deduped) < len(rows):
        print(f"  [retriever] Deduplicated {len(rows) - len(deduped)} "
              f"duplicate address(es) from query results")
    
    return deduped