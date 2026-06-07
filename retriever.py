from database import cursor

def retrieve_registers(company: str, family: str, rack: str) -> list:
    cursor.execute("""
        SELECT label, address, data_type, description
        FROM   registers
        WHERE  company=? AND device_family=? AND device_rack=?
        ORDER  BY CAST(address AS INTEGER)
    """, (company, family, rack))
    return cursor.fetchall()