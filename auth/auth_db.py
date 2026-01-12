import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "users.db"

def get_user(email):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, email, password_hash, role, is_active FROM users WHERE email=?",
            (email,)
        )
        return cur.fetchone()

def create_user(email, password_hash, role="viewer"):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (email, password_hash, role) VALUES (?,?,?)",
            (email, password_hash, role)
        )
        conn.commit()

def list_users():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            "SELECT id, email, role, is_active FROM users"
        ).fetchall()

def update_user(user_id, role, is_active):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE users SET role=?, is_active=? WHERE id=?",
            (role, is_active, user_id)
        )
        conn.commit()
