# auth.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
import urllib.parse
import requests
from google.oauth2 import id_token
from google.auth.transport import requests as grequests

# -------------------------------------------------
# Firebase Admin Init
# -------------------------------------------------
if "firebase_admin" in st.secrets and not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
    firebase_admin.initialize_app(cred)

# -------------------------------------------------
# Pyrebase Init (Email login)
# -------------------------------------------------
firebase_auth = None
firebase_cfg = dict(st.secrets.get("firebase", {}))
if firebase_cfg:
    firebase = pyrebase.initialize_app(firebase_cfg)
    firebase_auth = firebase.auth()

# -------------------------------------------------
# Config
# -------------------------------------------------
PRIVILEGED_DOMAINS = set(st.secrets.get("access", {}).get("privileged_domains", []))
GOOGLE_CLIENT_ID = st.secrets.get("oauth", {}).get("client_id")
GOOGLE_CLIENT_SECRET = st.secrets.get("oauth", {}).get("client_secret")
REDIRECT_URI = st.secrets.get("oauth", {}).get("redirect_uri")

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def get_email_domain(email: str) -> str:
    return email.split("@")[-1].lower()

def is_privileged() -> bool:
    return st.session_state.get("user_role") == "privileged"

# -------------------------------------------------
# Google OAuth
# -------------------------------------------------
def get_google_auth_url():
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "prompt": "select_account",
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)

def handle_google_redirect():
    params = st.query_params
    if "code" not in params:
        return

    code = params["code"]

    try:
        token_resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            timeout=10,
        )
        token_resp.raise_for_status()
        tokens = token_resp.json()

        idinfo = id_token.verify_oauth2_token(
            tokens["id_token"], grequests.Request(), GOOGLE_CLIENT_ID
        )

        email = idinfo["email"]
        name = idinfo.get("name", email.split("@")[0].title())

    except Exception:
        st.error("Google login failed.")
        st.query_params.clear()
        return

    if get_email_domain(email) not in PRIVILEGED_DOMAINS:
        st.error("Access denied for this email domain.")
        st.query_params.clear()
        return

    # ✅ Login success
    st.session_state["user"] = "google"
    st.session_state["email"] = email
    st.session_state["name"] = name
    st.session_state["user_role"] = "privileged"

    st.query_params.clear()
    st.rerun()

# -------------------------------------------------
# Logout
# -------------------------------------------------
def logout_user():
    for k in ["user", "email", "name", "photo", "user_role"]:
        st.session_state.pop(k, None)
    st.rerun()

# -------------------------------------------------
# Sidebar Authentication
# -------------------------------------------------
def sidebar_auth():
    handle_google_redirect()

    # ───────── NOT LOGGED IN ─────────
    if "user" not in st.session_state:
        st.sidebar.markdown("## 🔐 Sign in")

        # Google Login
        st.sidebar.markdown(
            f"""
            <a href="{get_google_auth_url()}" style="text-decoration:none;">
                <div style="
                    display:flex; justify-content:center; align-items:center;
                    background:#1a73e8; color:white; font-weight:600;
                    border-radius:8px; padding:0.6rem; font-size:16px;
                    width:100%; cursor:pointer;
                ">
                    🔵 Sign in with Google
                </div>
            </a>
            """,
            unsafe_allow_html=True,
        )

        st.sidebar.markdown("---")

        # Email / Password Login
        with st.sidebar.form("email_login_form", clear_on_submit=True):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign in with Email")

            if submit:
                if not firebase_auth:
                    st.error("Email login is unavailable.")
                    return

                if get_email_domain(email) not in PRIVILEGED_DOMAINS:
                    st.error("Access denied for this email domain.")
                    return

                try:
                    firebase_auth.sign_in_with_email_and_password(email, password)

                    # ✅ Login success
                    st.session_state["user"] = "email"
                    st.session_state["email"] = email
                    st.session_state["name"] = email.split("@")[0].title()
                    st.session_state["user_role"] = "privileged"

                    st.rerun()

                except Exception:
                    st.error("Invalid email or password.")

    # ───────── LOGGED IN ─────────
    else:
        st.sidebar.success(f"👋 Welcome, {st.session_state.get('name','User')}")
        st.sidebar.button("Logout", on_click=logout_user)
