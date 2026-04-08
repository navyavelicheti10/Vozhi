import sqlite3
import json
import time
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "chat_history.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            messages TEXT,
            updated_at REAL
        )
    ''')
    conn.commit()
    conn.close()

def get_all_sessions():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title, updated_at FROM chat_sessions ORDER BY updated_at DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "updatedAt": r[2]} for r in rows]

def get_session(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT messages FROM chat_sessions WHERE id = ?", (session_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

def save_session(session_id: str, title: str, messages: list):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    msgs_json = json.dumps(messages)
    c.execute('''
        INSERT INTO chat_sessions (id, title, messages, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title=excluded.title,
            messages=excluded.messages,
            updated_at=excluded.updated_at
    ''', (session_id, title, msgs_json, time.time() * 1000))
    conn.commit()
    conn.close()

def delete_session(session_id: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    return deleted
