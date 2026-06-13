# backend/db/database.py

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent.parent.parent / "sessions.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL DEFAULT 'Untitled Research',
            created_at TEXT NOT NULL,
            messages TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'idle',
            eval_score REAL
        )
    """)
    conn.commit()
    conn.close()


def create_session(session_id: str, name: str = "Untitled Research"):
    conn = get_connection()
    conn.execute(
        "INSERT INTO sessions (id, name, created_at, messages, status) VALUES (?, ?, ?, ?, ?)",
        (session_id, name, datetime.now(timezone.utc).isoformat(), "[]", "idle")
    )
    conn.commit()
    conn.close()


def update_session(session_id: str, messages: list = None, status: str = None,
                    eval_score: float = None, name: str = None):
    conn = get_connection()
    updates = []
    params = []

    if messages is not None:
        updates.append("messages = ?")
        params.append(json.dumps(messages))
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if eval_score is not None:
        updates.append("eval_score = ?")
        params.append(eval_score)
    if name is not None:
        updates.append("name = ?")
        params.append(name)

    if not updates:
        return

    params.append(session_id)
    conn.execute(f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()
    conn.close()


def get_session(session_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "created_at": row["created_at"],
        "messages": json.loads(row["messages"]),
        "status": row["status"],
        "eval_score": row["eval_score"]
    }


def list_sessions() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT id, name, created_at, status FROM sessions ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_session(session_id: str):
    conn = get_connection()
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()