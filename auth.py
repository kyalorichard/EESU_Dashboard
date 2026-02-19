# auth_full_safe.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
import json
from datetime import datetime, timedelta
from streamlit_cookies_manager import EncryptedCookieManager

# ---------------- Firebase Admin Initialization ----------------
def init_firebase_admin():
    if not firebase_admin._apps:
        try:
            cred_dict = st.secrets.get("firebase_admin")
            if not cred_dict:
                st.error("Firebase Admin credentials missing in secrets.toml")
                st.stop()
            cred = credentials.Certificate(dict(cred_dict))
            firebase_admin.initialize_app(cred)
            st.success("✅ Firebase Admin initialized")
        except Exception as e:
            st.error(f"❌ Firebase Admin initialization failed: {e}")
            st.stop()

# Lazy Firestore client to ensure Admin is initialized
def get_firestore_client():
    init_firebase_admin()
    return firestore.client()

db = get_firestore_client()

# ---------------- Pyrebase ----------------
firebase_cfg = dict(st.secrets.get("firebase", {}))
firebase_auth = None

if not firebase_cfg:
    st.error("Firebase config missing in secrets.toml")
else:
    try:
        firebase = pyrebase.initialize_app(firebase_cfg)
        firebase_auth = firebase.auth()
    except Exception as e:
        st.error(f"Firebase Auth initialization failed: {e}")

# ---------------- Privileged Domains ----------------
PRIVILEGED_DOMAINS = set(d.lower() for d in st.secrets.get("access", {}).get("privileged_domains", []))

# ---------------- Cookie Manager ----------------
def get_cookie_manager():
    if "cookies_manager" not in st.session_state:
        cookie_password = st.secrets.get("cookie", {}).get("cookie_password")
        if not cookie_password:
            st.error("Cookie password missing in secrets.toml")
            st.stop()
        st.session_state["cookies_manager"] = EncryptedCookieManager(
            prefix="myapp",
            password=cookie_password
        )
    cookies = st.session_state["cookies_manager"]
    if not cookies.ready():
        try:
            cookies.sync()
        except Exception:
            st.info("🔄 Waiting for browser session…")
        return None
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
        "session_id": None,
        "token_last_refresh": None
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

def get_email_domain(email: str) -> str:
    return email.strip().split("@")[-1].lower()

# ---------------- Session / Firestore ----------------
SESSION_DURATION = timedelta(hours=2)
TOKEN_REFRESH_INTERVAL = timedelta(minutes=50)

def save_session(email, id_token, name, role, verified):
    session_id = str(uuid.uuid4())
    db.collection("sessions").document(session_id).set({
        "email": email,
        "idToken": id_token,
        "name": name,
        "user_role": role,
        "email_verified": verified,
        "created_at": firestore.SERVER_TIMESTAMP,
        "expires_at": datetime.utcnow() + SESSION_DURATION
    })
    st.session_state["session_id"] = session_id
    st.session_state["token_last_refresh"] = datetime.utcnow()

    cookies = get_cookie_manager()
    if cookies and cookies.ready():
        cookies["session_id"] = session_id
        cookies.save()
    return session_id

def restore_session():
    if st.session_state.get("session_id"):
        session_id = st.session_state["session_id"]
    else:
        cookies = get_cookie_manager()
        if not cookies or not cookies.ready():
            return False
        session_id = cookies.get("session_id")
        st.session_state["session_id"] = session_id

    if not session_id:
        return False

    doc = db.collection("sessions").document(session_id).get()
    if not doc.exists:
        return False

    data = doc.to_dict()
    expires_at = data.get("expires_at")
    if expires_at and datetime.utcnow() > expires_at:
        logout_user()
        return False

    st.session_state.update({
        "user": True,
        "email": data.get("email"),
        "name": data.get("name"),
        "user_role": data.get("user_role"),
        "email_verified": data.get("email_verified", False),
        "idToken": data.get("idToken"),
        "token_last_refresh": datetime.utcnow()
    })

    # Auto refresh Firebase token
    if firebase_auth:
        last_refresh = st.session_state.get("token_last_refresh") or datetime.utcnow() - timedelta(hours=1)
        if datetime.utcnow() - last_refresh > TOKEN_REFRESH_INTERVAL:
            try:
                refreshed = firebase_auth.refresh(st.session_state.idToken)
                st.session_state.idToken = refreshed["idToken"]
                st.session_state.token_last_refresh = datetime.utcnow()
                # Update Firestore
                db.collection("sessions").document(session_id).update({
                    "idToken": st.session_state.idToken
                })
            except Exception:
                logout_user()
                return False

    return True

def logout_user():
    session_id = st.session_state.get("session_id")
    if session_id:
        db.collection("sessions").document(session_id).delete()
    cookies = get_cookie_manager()
    if cookies and cookies.ready() and "session_id" in cookies:
        del cookies["session_id"]
        cookies.save()
    for key in ["user", "email", "name", "user_role", "email_verified", "idToken", "session_id", "token_last_refresh"]:
        st.session_state.pop(key, None)
    st.experimental_rerun()

def cleanup_expired_sessions():
    now = datetime.utcnow()
    expired = db.collection("sessions").where("expires_at", "<", now).stream()
    for doc in expired:
        db.collection("sessions").document(doc.id).delete()

# ---------------- Authentication UI ----------------
ERROR_MAP = {
    "EMAIL_EXISTS": "This email is already registered.",
    "INVALID_PASSWORD": "Incorrect email or password.",
    "EMAIL_NOT_FOUND": "Email not registered.",
    "WEAK_PASSWORD": "Password must be at least 6 characters.",
    "INVALID_LOGIN_CREDENTIALS": "Incorrect email or password."
}

def auth_ui():
    init_state()

    # Clean expired sessions
    cleanup_expired_sessions()

    # Restore existing session
    if restore_session():
        sidebar = st.sidebar
        if st.session_state.email_verified:
            sidebar.success(f"👋 {st.session_state.name} ✅ Verified")
        else:
            sidebar.warning(f"👋 {st.session_state.name} ⚠️ Email not verified")
            if sidebar.button("Resend verification email") and firebase_auth:
                try:
                    firebase_auth.send_email_verification(st.session_state.idToken)
                    st.success("Verification email resent.")
                except Exception as e:
                    st.error(f"Failed to resend email: {parse_firebase_error(e)}")
        if sidebar.button("Logout"):
            logout_user()
        return

    sidebar = st.sidebar
    tab_choice = sidebar.radio("Select Action", ["Login", "Register"], index=0)

    # ---------------- LOGIN ----------------
    if tab_choice == "Login":
        with sidebar.form("login_form", clear_on_submit=True):
            email = st.text_input("Email").strip()
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")
            forgot_pass = st.form_submit_button("Forgot Password?")

            if submitted:
                if not firebase_auth:
                    st.error("Authentication service unavailable.")
                    return
                domain = get_email_domain(email)
                if domain not in PRIVILEGED_DOMAINS:
                    st.error(f"Access restricted: {domain} not authorized.")
                    return
                try:
                    user = firebase_auth.sign_in_with_email_and_password(email, password)
                    id_token = user["idToken"]
                    info = firebase_auth.get_account_info(id_token)
                    verified = info["users"][0].get("emailVerified", False)
                    name = email.split("@")[0].title()
                    role = "privileged" if verified else "unverified"

                    save_session(email, id_token, name, role, verified)
                    st.experimental_rerun()
                except Exception as e:
                    code = parse_firebase_error(e)
                    st.error(ERROR_MAP.get(code, f"Login failed: {code}"))

            if forgot_pass and firebase_auth:
                if not email:
                    st.warning("Enter your email above to reset password.")
                else:
                    try:
                        firebase_auth.send_password_reset_email(email)
                        st.success(f"Password reset email sent to {email}.")
                    except Exception as e:
                        st.error(f"Failed to send reset email: {parse_firebase_error(e)}")

    # ---------------- REGISTER ----------------
    if tab_choice == "Register":
        with sidebar.form("register_form", clear_on_submit=True):
            email = st.text_input("Email").strip()
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Register")

            if submitted:
                if not firebase_auth:
                    st.error("Authentication service unavailable.")
                    return
                domain = get_email_domain(email)
                if domain not in PRIVILEGED_DOMAINS:
                    st.error(f"Registration restricted: {domain} not approved.")
                    return
                try:
                    user = firebase_auth.create_user_with_email_and_password(email, password)
                    firebase_auth.send_email_verification(user["idToken"])
                    st.success("Registration successful. Check your email to verify account.")

                    save_session(
                        email,
                        user["idToken"],
                        email.split("@")[0].title(),
                        "unverified",
                        False
                    )
                    st.experimental_rerun()
                except Exception as e:
                    code = parse_firebase_error(e)
                    st.error(ERROR_MAP.get(code, f"Registration failed: {code}"))
