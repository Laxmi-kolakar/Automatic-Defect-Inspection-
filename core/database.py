import sqlite3
import pandas as pd
from config.settings import DATABASE_PATH

COLUMNS = ["inspection_id", "timestamp", "component_id", "batch_id", "production_line", "operator", "filename", "predicted_class", "confidence", "decision", "model_version", "notes"]

def connect():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS inspections (
            inspection_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, component_id TEXT, batch_id TEXT,
            production_line TEXT, operator TEXT, filename TEXT, predicted_class TEXT, confidence REAL,
            decision TEXT NOT NULL, model_version TEXT, notes TEXT)""")
        conn.commit()

def save_inspection(record):
    init_db()
    fields = [c for c in COLUMNS if c in record]
    placeholders = ",".join(["?"] * len(fields))
    sql = f"INSERT OR REPLACE INTO inspections ({','.join(fields)}) VALUES ({placeholders})"
    with connect() as conn:
        conn.execute(sql, [record.get(c) for c in fields])
        conn.commit()

def get_inspections(limit=None):
    init_db()
    sql = "SELECT * FROM inspections ORDER BY timestamp DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params = (int(limit),)
    else:
        params = ()
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)

def clear_history():
    init_db()
    with connect() as conn:
        conn.execute("DELETE FROM inspections")
        conn.commit()
