import sqlite3
 
DB_PATH = "modbus.db"
 
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
 
cursor.execute("""
CREATE TABLE IF NOT EXISTS registers (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    company      TEXT,
    device_family TEXT,
    device_rack  TEXT,
    label        TEXT,
    address      TEXT,
    data_type    TEXT,
    description  TEXT
)
""")
conn.commit()

def insert_register(company, family, rack, label, address, datatype, description):
    cursor.execute("""
        INSERT OR IGNORE INTO registers
            (company, device_family, device_rack, label, address, data_type, description)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (company, family, rack, label, address, datatype, description))
    conn.commit()

def clear_registers(company, family, rack):     #to avoid duplicate rows
    cursor.execute("""
        DELETE FROM registers
        WHERE company=? AND device_family=? AND device_rack=?
    """, (company, family, rack))
    conn.commit()
    cursor.execute("VACUUM")
    conn.commit()

def delete_db():
    """Drop and recreate the registers table. Use for clean test runs."""
    cursor.executescript("""
        DROP TABLE IF EXISTS registers;
        CREATE TABLE registers (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            company       TEXT,
            device_family TEXT,
            device_rack   TEXT,
            label         TEXT,
            address       TEXT,
            data_type     TEXT,
            description   TEXT,
            UNIQUE (company, device_family, device_rack, address)
        );
    """)
    conn.commit()