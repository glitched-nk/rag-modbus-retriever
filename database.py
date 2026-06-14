import sqlite3
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
                device_rack   TEXT,
                processed_at  TEXT NOT NULL
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
                inserted_at   TEXT    NOT NULL,
                UNIQUE(company, device_family, device_rack, doc_hash, address)
            )
        """)
 
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
                    doc_hash, label, address, datatype, description)
                VALUES (
                    new.id, new.company, new.device_family, new.device_rack,
                    new.doc_hash, new.label, new.address,
                    new.datatype, new.description);
            END
        """)
 
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS reg_v2_ad
            AFTER DELETE ON registers_v2 BEGIN
                INSERT INTO registers_fts(
                    registers_fts, rowid, company, device_family, device_rack,
                    doc_hash, label, address, datatype, description)
                VALUES (
                    'delete', old.id, old.company, old.device_family,
                    old.device_rack, old.doc_hash, old.label,
                    old.address, old.datatype, old.description);
            END
        """)
 
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS reg_v2_au
            AFTER UPDATE ON registers_v2 BEGIN
                INSERT INTO registers_fts(
                    registers_fts, rowid, company, device_family, device_rack,
                    doc_hash, label, address, datatype, description)
                VALUES (
                    'delete', old.id, old.company, old.device_family,
                    old.device_rack, old.doc_hash, old.label,
                    old.address, old.datatype, old.description);
                INSERT INTO registers_fts(
                    rowid, company, device_family, device_rack,
                    doc_hash, label, address, datatype, description)
                VALUES (
                    new.id, new.company, new.device_family, new.device_rack,
                    new.doc_hash, new.label, new.address,
                    new.datatype, new.description);
            END
        """)
 
        conn.commit()
 
 
# Document registry
 
def register_document(doc_hash: str, filename: str, company: str, device_family: str, device_rack: str):
    with _get_conn() as conn:       # Record that a document has been fully processed
        conn.execute("""
            INSERT OR REPLACE INTO doc_registry
                (doc_hash, filename, company, device_family, device_rack, processed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_hash, filename, company, device_family, device_rack,
              datetime.now(timezone.utc).isoformat()))
        conn.commit()
 
def get_document(doc_hash: str) :  #"Return the registry row for this hash, or None if unseen."""
    with _get_conn() as conn:
            row = conn.execute(
                "SELECT filename, company, device_family, device_rack, processed_at "
                "FROM doc_registry WHERE doc_hash = ?",
                (doc_hash,)
            ).fetchone()
    if row:
            return {
                "filename":      row[0],
                "company":       row[1],
                "device_family": row[2],
                "device_rack":   row[3],
                "processed_at":  row[4],
            }
    return None
 
# Register insert
 
def insert_register_v2(company: str, device_family: str, device_rack: str,
                        doc_hash: str, label: str, address: str, datatype: str, description: str):
    #Insert one register row.  Duplicate (same key + address) is silently ignore
    with _get_conn() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO registers_v2
                (company, device_family, device_rack, doc_hash,
                 label, address, datatype, description, inserted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (company, device_family, device_rack, doc_hash,
              label, address, datatype, description,
              datetime.now(timezone.utc).isoformat()))
        conn.commit()
 
 
# ── Backward-compat shims ──────────────────────────────────────────────────────
# These keep any code that still imports the old names working.
# insert_register() requires a doc_hash; callers in app.py already supply one.
# For truly legacy callers that don't pass doc_hash we use a sentinel value.
 
_LEGACY_HASH = "legacy"
 
def insert_register(company, family, rack, label, address, datatype,
                    description, doc_hash=_LEGACY_HASH):
    """Shim: delegates to insert_register_v2."""
    insert_register_v2(company, family, rack, doc_hash,
                       label, address, datatype, description)
 
 
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
                SELECT label, address, datatype, description, inserted_at
                FROM   registers_v2
                WHERE  company=? AND device_family=? AND device_rack=?
                       AND doc_hash=?
                ORDER  BY CAST(address AS INTEGER)
            """, (company, device_family, device_rack, doc_hash)).fetchall()
        else:
            # Window function: newest inserted_at wins per address
            rows = conn.execute("""
                SELECT label, address, datatype, description, inserted_at
                FROM (
                    SELECT label, address, datatype, description, inserted_at,
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
            "inserted_at":  r[4],
        }
        for r in rows
    ]
 
 
# full-text search
 
def search_registers(query: str,
                     company: str | None = None,
                     device_family: str | None = None) -> list[dict]:
    """
    BM25-ranked full-text search across all stored registers.
    Optionally narrow by company and/or device_family.
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
               label, address, datatype, description,
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
            "score":         r[8],
        }
        for r in rows
    ]