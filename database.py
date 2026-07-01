import sqlite3
import json
from datetime import datetime, timezone
 
DB_PATH = "modbus_store.db"
 
# One connection per call;we do NOT keep a module-level connection: that pattern is not thread-safe 
 
def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
 
def init_db():
    with _get_conn() as conn:
 
        # Document registry
        conn.execute("""
            CREATE TABLE IF NOT EXISTS doc_registry (
                doc_hash      TEXT PRIMARY KEY,
                filename      TEXT NOT NULL,
                company       TEXT,
                device_family TEXT,
                extracted_racks TEXT NOT NULL DEFAULT '[]',
                first_seen_at   TEXT NOT NULL,
                last_updated_at TEXT NOT NULL
            )
        """)
 
        # Versioned registers — keyed by (company, device_family, device_rack, doc_hash, address) so different document versions never collide.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS registers_v2 (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                company       TEXT    NOT NULL,
                device_family TEXT    NOT NULL,
                device_rack   TEXT    NOT NULL,
                doc_hash      TEXT    NOT NULL,
                label         TEXT,
                address       TEXT    NOT NULL,
                datatype      TEXT,
                description   TEXT,
                scaling       TEXT,
                num_registers TEXT,
                inserted_at   TEXT    NOT NULL,
                UNIQUE(company, device_family, device_rack, doc_hash, address)
            )
        """)

        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(registers_v2)").fetchall()} ##for tables stored wo scaling
        if "scaling" not in existing_cols:
            conn.execute("ALTER TABLE registers_v2 ADD COLUMN scaling TEXT")
        if "num_registers" not in existing_cols:
            conn.execute("ALTER TABLE registers_v2 ADD COLUMN num_registers TEXT")

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reg_v2_lookup
            ON registers_v2(company, device_family, device_rack, doc_hash)
        """)
 
        # FTS5 content table — mirrors registers_v2, kept in sync by triggers.
        # tokenize='porter ascii' gives stemmed English search.
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS registers_fts
            USING fts5(
                company,
                device_family,
                device_rack,
                doc_hash,
                label,
                address,
                datatype,
                description,
                scaling,
                num_registers,
                content='registers_v2',
                content_rowid='id',
                tokenize='porter ascii'
            )
        """)
 
        # Triggers to keep FTS in sync automatically
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS reg_v2_ai
            AFTER INSERT ON registers_v2 BEGIN
                INSERT INTO registers_fts(
                    rowid, company, device_family, device_rack,
                    doc_hash, label, address, datatype, description, scaling, num_registers)
                VALUES (
                    new.id, new.company, new.device_family, new.device_rack,
                    new.doc_hash, new.label, new.address,
                    new.datatype, new.description, new.scaling, new.num_registers);
            END
        """)
 
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS reg_v2_ad
            AFTER DELETE ON registers_v2 BEGIN
                INSERT INTO registers_fts(
                    registers_fts, rowid, company, device_family, device_rack,
                    doc_hash, label, address, datatype, description, scaling, num_registers)
                VALUES (
                    'delete', old.id, old.company, old.device_family,
                    old.device_rack, old.doc_hash, old.label,
                    old.address, old.datatype, old.description, old.scaling, old.num_registers);
            END
        """)
 
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS reg_v2_au
            AFTER UPDATE ON registers_v2 BEGIN
                INSERT INTO registers_fts(
                    registers_fts, rowid, company, device_family, device_rack,
                    doc_hash, label, address, datatype, description, scaling, num_registers)
                VALUES (
                    'delete', old.id, old.company, old.device_family,
                    old.device_rack, old.doc_hash, old.label,
                    old.address, old.datatype, old.description, old.scaling, old.num_registers);
                INSERT INTO registers_fts(
                    rowid, company, device_family, device_rack,
                    doc_hash, label, address, datatype, description, scaling, num_registers)
                VALUES (
                    new.id, new.company, new.device_family, new.device_rack,
                    new.doc_hash, new.label, new.address,
                    new.datatype, new.description, new.scaling, new.num_registers);
            END
        """)
 
        conn.commit()
 
 
# Document registry
 
def register_document(doc_hash: str, filename: str, company: str, device_family: str, rack: str):
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT extracted_racks FROM doc_registry WHERE doc_hash=?",
            (doc_hash,)
        ).fetchone()
 
        if existing is None:
            racks = [rack] if rack else []
            conn.execute("""
                INSERT INTO doc_registry
                    (doc_hash, filename, company, device_family,
                     extracted_racks, first_seen_at, last_updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (doc_hash, filename, company, device_family,
                  json.dumps(racks), now, now))
        else:
            racks = json.loads(existing[0])
            if rack and rack not in racks:
                racks.append(rack)
            conn.execute("""
                UPDATE doc_registry
                SET extracted_racks=?, last_updated_at=?
                WHERE doc_hash=?
            """, (json.dumps(racks), now, doc_hash))
 
        conn.commit()
 
def get_document(doc_hash: str) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT filename, company, device_family, "
            "extracted_racks, first_seen_at "
            "FROM doc_registry WHERE doc_hash=?",
            (doc_hash,)
        ).fetchone()
    if row:
        return {
            "filename":        row[0],
            "company":         row[1],
            "device_family":   row[2],
            "extracted_racks": json.loads(row[3]),
            "first_seen_at":   row[4],
        }
    return None

def rack_already_extracted(doc_hash: str, rack: str) -> bool:
    doc = get_document(doc_hash)
    if not doc:
        return False
    norm = _norm_rack(rack)
    return any(_norm_rack(r) == norm for r in doc["extracted_racks"])
 
 
def _norm_rack(rack: str) -> str:
    import re
    return re.sub(r"[\s\-_]", "", rack.lower())

# Register insert
 
def insert_register_v2(company: str, device_family: str, device_rack: str, doc_hash: str, label: str, address: str, 
                       datatype: str, description: str, scaling: str = None, num_registers: str = None):
    #Insert one register row.  Duplicate (same key + address) is silently ignore
    with _get_conn() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO registers_v2
                (company, device_family, device_rack, doc_hash,
                 label, address, datatype, description, scaling, num_registers, inserted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (company, device_family, device_rack, doc_hash,
              label, address, datatype, description, scaling, num_registers,
              datetime.now(timezone.utc).isoformat()))
        conn.commit()

# Backward-compat shims
# These keep any code that still imports the old names working.
# insert_register() requires a doc_hash; callers in app.py already supply one.
# For truly legacy callers that don't pass doc_hash we use a sentinel value.
 
_LEGACY_HASH = "legacy"
 
def insert_register(company, family, rack, label, address, datatype, description, doc_hash=_LEGACY_HASH, scaling = None, num_registers = None):
    """Shim: delegates to insert_register_v2."""
    insert_register_v2(company, family, rack, doc_hash, label, address, datatype, description, scaling, num_registers)
 
 
def clear_registers(company, family, rack):
    """No-op shim: v2 storage is non-destructive (versioned by doc_hash).
    Kept so existing call-sites don't break with an ImportError.
    """
    pass
 
 
def delete_db():
    """Drop and recreate all tables.  Use only for clean test runs."""
    with _get_conn() as conn:
        conn.executescript("""
            DROP TABLE IF EXISTS registers_fts;
            DROP TABLE IF EXISTS registers_v2;
            DROP TABLE IF EXISTS doc_registry;
        """)
        conn.commit()
    init_db()
 
 
# Register retrieval
 
def retrieve_registers_v2(company: str, device_family: str, device_rack: str, doc_hash: str | None = None) -> list[dict]:
    """
    Retrieve registers for a company / device_family / device_rack triple.
 
    doc_hash supplied  →  return only rows from that specific document version.
    doc_hash=None      →  return the union of all versions; when the same address appears in multiple versions the newest inserted_at wins.
    """
    with _get_conn() as conn:
        if doc_hash:
            rows = conn.execute("""
                SELECT label, address, datatype, description, scaling, num_registers, inserted_at
                FROM   registers_v2
                WHERE  company=? AND device_family=? AND device_rack=?
                       AND doc_hash=?
                ORDER  BY CAST(address AS INTEGER)
            """, (company, device_family, device_rack, doc_hash)).fetchall()
        else:
            # Window function: newest inserted_at wins per address
            rows = conn.execute("""
                SELECT label, address, datatype, description, scaling, num_registers, inserted_at
                FROM (
                    SELECT label, address, datatype, description, scaling, num_registers, inserted_at,
                           ROW_NUMBER() OVER (
                               PARTITION BY address
                               ORDER BY inserted_at DESC
                           ) AS rn
                    FROM   registers_v2
                    WHERE  company=? AND device_family=? AND device_rack=?
                ) WHERE rn = 1
                ORDER  BY CAST(address AS INTEGER)
            """, (company, device_family, device_rack)).fetchall()
 
    return [
        {
            "label":        r[0],
            "address":      r[1],
            "datatype":     r[2],
            "description":  r[3],
            "scaling":      r[4],
            "num_registers": r[5],
            "inserted_at":   r[6],
        }
        for r in rows
    ]
 
 
# full-text search
 
def search_registers(query: str,
                     company: str | None = None,
                     device_family: str | None = None) -> list[dict]:
    """
    BM25-ranked full-text search across all stored registers. Optionally narrow by company and/or device_family.
    Returns up to 200 results, best match first.
    """
    conditions = ["registers_fts MATCH ?"]
    params: list = [query]
 
    if company:
        conditions.append("company = ?")
        params.append(company)
    if device_family:
        conditions.append("device_family = ?")
        params.append(device_family)
 
    sql = f"""
        SELECT company, device_family, device_rack, doc_hash,
               label, address, datatype, description, scaling, num_registers,
               bm25(registers_fts) AS score
        FROM   registers_fts
        WHERE  {' AND '.join(conditions)}
        ORDER  BY score
        LIMIT  200
    """
    with _get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
 
    return [
        {
            "company":       r[0],
            "device_family": r[1],
            "device_rack":   r[2],
            "doc_hash":      r[3],
            "label":         r[4],
            "address":       r[5],
            "datatype":      r[6],
            "description":   r[7],
            "scaling":       r[8],
            "num_registers": r[9],
            "score":         r[10],
        }
        for r in rows
    ]