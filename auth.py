# auth.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
import json
from streamlit_cookies_manager import EncryptedCookieManager

# ---------------- Firebase Admin ----------------
try:
    if "firebase_admin" in st.secrets and not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
        firebase_admin.initialize_app(cred)
except Exception as e:
    st.error(f"⚠️ Firebase Admin initialization failed: {e}")
    st.stop()

# ---------------- Pyrebase ----------------
firebase_auth = None
firebase_cfg = dict(st.secrets.get("firebase", {}))
if firebase_cfg:
    try:
        firebase = pyrebase.initialize_app(firebase_cfg)
        firebase_auth = firebase.auth()
    except Exception as e:
        st.warning(f"⚠️ Firebase authentication service unavailable: {e}")

# ---------------- Privileged Domains ----------------
PRIVILEGED_DOMAINS = set(d.lower() for d in st.secrets.get("access", {}).get("privileged_domains", []))

# ---------------- Cookie Manager ----------------
def get_cookies_manager():
    if "cookies_manager" not in st.session_state:
        cookie_password = st.secrets["cookie"]["cookie_password"]
        st.session_state["cookies_manager"] = EncryptedCookieManager(
            prefix="myapp",
            password=cookie_password
        )

    cookies = st.session_state["cookies_manager"]

    # Initialize cookies if not ready
    if not cookies.ready():
        cookies.sync()          # trigger initialization
        st.info("🔄 Loading session…")
        st.experimental_rerun() # rerun until ready

    return cookies
# ---------------- Helpers ----------------
def init_state():
    defaults = {
        "user": None,
        "email": "",
        "name": "",
        "user_role": None,
        "email_verified": False,
        "idToken": None,
        "auth_tab": "Login"
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def parse_firebase_error(e):
    try:
        payload = e.args[1] if len(e.args) > 1 else e.args[0]
        error_json = json.loads(payload)
        return error_json.get("error", {}).get("message", str(e))
    except Exception:
        return str(e)

def logout_user():
    cookies = get_cookies_manager()
    for key in ["user", "email", "name", "user_role", "email_verified", "idToken"]:
        st.session_state.pop(key, None)
        if key in cookies:
            del cookies[key]
    cookies.save()
    st.rerun()

def is_privileged():
    return st.session_state.get("user_role") == "privileged" and st.session_state.get("email_verified")

def refresh_id_token():
    if firebase_auth and st.session_state.get("idToken"):
        try:
            refreshed = firebase_auth.refresh(st.session_state.idToken)
            st.session_state.idToken = refreshed["idToken"]
        except Exception:
            pass

# ---------------- Authentication UI ----------------
ERROR_MAP = {
    "EMAIL_EXISTS": "This email is already registered.",
    "INVALID_PASSWORD": "Incorrect email or password.",
    "EMAIL_NOT_FOUND": "Email not registered.",
    "WEAK_PASSWORD": "Password must be at least 6 characters.",
    "INVALID_LOGIN_CREDENTIALS": "Incorrect email or password."
}

def get_email_domain(email: str) -> str:
    return email.strip().split("@")[-1].lower()

def auth_ui():
    cookies = get_cookies_manager()
    st.write("✅ Cookies are ready")