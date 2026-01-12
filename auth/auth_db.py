import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).parent / "users.db"

def get_conn():
return sqlite3.connect(DB_PATH, check_same_thread=False)

def get_user_by_email(email):
with get_conn() as conn:
return conn.execute(
"SELECT id, email, password_hash, role, org_id, is_active, email_verified FROM users WHERE email=?",
(email,),
).fetchone()

def create_user(email, password_hash, role="viewer", org_id=None):
with get_conn() as conn:
conn.execute(
"INSERT INTO users (email, password_hash, role, org_id) VALUES (?,?,?,?)",
(email, password_hash, role, org_id),
)
conn.commit()

def list_users():
with get_conn() as conn:
return conn.execute(
"SELECT id, email, role, org_id, is_active FROM users"
).fetchall()

def update_user(uid, role, org_id, is_active):
with get_conn() as conn:
conn.execute(
"UPDATE users SET role=?, org_id=?, is_active=? WHERE id=?",
(role, org_id, is_active, uid),
)
conn.commit()

def log_event(user_id, action, details=""):
with get_conn() as conn:
conn.execute(
"INSERT INTO audit_logs (user_id, action, details) VALUES (?,?,?)",
(user_id, action, details),
)
conn.commit()
