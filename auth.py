# auth.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
from streamlit_cookies_manager import EncryptedCookieManager
import json

# -------------------------------------------------
# Firebase Admin Initialization
# -------------------------------------------------
if not firebase_admin._apps:
    if "firebase_admin" not in st.secrets:
        st.error("Missing firebase_admin in secrets.toml")
        st.stop()
    cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
    firebase_admin.initialize_app(cred)

# -------------------------------------------------
# Firebase Auth (Pyrebase)
# -------------------------------------------------
firebase_auth = None
firebase_cfg = dict(st.secrets.get("firebase", {}))

if not firebase_cfg:
    st.error("Firebase config missing in secrets.toml")
else:
    firebase = pyrebase.initialize_app(firebase_cfg)
    firebase_auth = firebase.auth()

# -------------------------------------------------
# Privileged Domains
# -------------------------------------------------
PRIVILEGED_DOMAINS = set(
    d.lower() for d in st.secrets.get("access", {}).get("privileged_domains", [])
)

# -------------------------------------------------
# Cookie Manager (SAFE VERSION)
# -------------------------------------------------
def get_cookies():
    if "cookies" not in st.session_state:
        password = st.secrets.get("cookie", {}).get("cookie_password")
        if not password:
            st.error("Cookie password missing in secrets.toml")
            st.stop()

        st.session_state.cookies = EncryptedCookieManager(
            prefix="myapp",
            password=password,
        )

    cookies = st.session_state.cookies

    if not cookies.ready():
        cookies.sync()
        st.stop()

    return cookies

# -------------------------------------------------
# Session Initialization
# -------------------------------------------------
def init_session():
    defaults = {
        "user": False,
        "email": None,
        "name": None,
        "role": None,
        "email_verified": False,
        "restored": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

# -------------------------------------------------
# Restore From Cookies
# -------------------------------------------------
def restore_session():
    cookies = get_cookies()

    if not st.session_state.restored:
        if "email" in cookies:
            st.session_state.user = True
            st.session_state.email = cookies["email"]
            st.session_state.name = cookies["name"]
            st.session_state.role = cookies["role"]
            st.session_state.email_verified = cookies["email_verified"]
        st.session_state.restored = True

# -------------------------------------------------
# Logout
# -------------------------------------------------
def logout():
    cookies = get_cookies()
    for key in ["email", "name", "role", "email_verified"]:
        if key in cookies:
            del cookies[key]
    cookies.save()

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.rerun()

# -------------------------------------------------
# Helper
# -------------------------------------------------
def parse_error(e):
    try:
        payload = e.args[1] if len(e.args) > 1 else e.args[0]
        data = json.loads(payload)
        return data.get("error", {}).get("message", str(e))
    except:
        return str(e)

def get_domain(email):
    return email.split("@")[-1].lower()

def is_privileged():
    return (
        st.session_state.user
        and st.session_state.email_verified
        and st.session_state.role == "privileged"
    )

# -------------------------------------------------
# Authentication UI
# -------------------------------------------------
def auth_ui():
    init_session()
    restore_session()

    sidebar = st.sidebar

    # ---------------- Logged In ----------------
    if st.session_state.user:
        sidebar.success(f"👋 {st.session_state.name}")

        if not st.session_state.email_verified:
            sidebar.warning("Email not verified.")
            return

        if sidebar.button("Logout"):
            logout()
        return

    # ---------------- Tabs ----------------
    action = sidebar.radio("Select Action", ["Login", "Register"])

    # ================= LOGIN =================
    if action == "Login":
        with sidebar.form("login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign in")

            if submit:
                if get_domain(email) not in PRIVILEGED_DOMAINS:
                    st.error("Access restricted to approved domains.")
                    return

                try:
                    user = firebase_auth.sign_in_with_email_and_password(email, password)
                    info = firebase_auth.get_account_info(user["idToken"])
                    verified = info["users"][0]["emailVerified"]

                    role = "privileged" if verified else "restricted"

                    # Save session
                    st.session_state.user = True
                    st.session_state.email = email
                    st.session_state.name = email.split("@")[0].title()
                    st.session_state.email_verified = verified
                    st.session_state.role = role

                    # Save safe cookie data
                    cookies = get_cookies()
                    cookies["email"] = email
                    cookies["name"] = st.session_state.name
                    cookies["email_verified"] = verified
                    cookies["role"] = role
                    cookies.save()

                    st.rerun()

                except Exception as e:
                    st.error(parse_error(e))

    # ================= REGISTER =================
    if action == "Register":
        with sidebar.form("register"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Register")

            if submit:
                if get_domain(email) not in PRIVILEGED_DOMAINS:
                    st.error("Registration restricted to approved domains.")
                    return

                try:
                    user = firebase_auth.create_user_with_email_and_password(email, password)
                    firebase_auth.send_email_verification(user["idToken"])
                    st.success("Registration successful. Check your email to verify.")

                except Exception as e:
                    st.error(parse_error(e))