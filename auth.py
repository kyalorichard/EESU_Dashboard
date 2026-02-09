# auth.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
import urllib.parse
import requests
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
import json

# ----------------------------
# Firebase Admin Init
# ----------------------------
if "firebase_admin" in st.secrets and not firebase_admin._apps:
    try:
        cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Firebase Admin initialization failed: {e}")

# ----------------------------
# Pyrebase Init (Email login)
# ----------------------------
firebase_auth = None
firebase_cfg = dict(st.secrets.get("firebase", {}))
if firebase_cfg:
    try:
        firebase = pyrebase.initialize_app(firebase_cfg)
        firebase_auth = firebase.auth()
    except Exception as e:
        st.error(f"Firebase initialization failed: {e}")

# ----------------------------
# Config
# ----------------------------
PRIVILEGED_DOMAINS = set(st.secrets.get("access", {}).get("privileged_domains", []))
GOOGLE_CLIENT_ID = st.secrets.get("oauth", {}).get("client_id")
GOOGLE_CLIENT_SECRET = st.secrets.get("oauth", {}).get("client_secret")
REDIRECT_URI = st.secrets.get("oauth", {}).get("redirect_uri")

# ----------------------------
# Helpers
# ----------------------------
def get_email_domain(email: str) -> str:
    return email.split("@")[-1].lower()

def avatar_initials(email: str) -> str:
    parts = email.split("@")[0].replace(".", " ").split()
    return "".join(p[0].upper() for p in parts[:2])

def is_privileged() -> bool:
    return st.session_state.get("user_role") == "privileged"

# ----------------------------
# Google OAuth
# ----------------------------
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
    try:
        params = st.experimental_get_query_params()
    except Exception:
        params = {}

    if "code" not in params:
        return

    code = params["code"][0]

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

        email = idinfo.get("email")
        name = idinfo.get("name") or email.split("@")[0].title()
        picture = idinfo.get("picture")
    except Exception:
        st.error("Google login failed or token invalid.")
        st.experimental_set_query_params()
        return

    # Domain restriction
    if get_email_domain(email) not in PRIVILEGED_DOMAINS:
        st.error(f"Access denied. Only emails from {', '.join(PRIVILEGED_DOMAINS)} allowed.")
        st.experimental_set_query_params()
        return

    # Set session state
    st.session_state.user = "google"
    st.session_state.email = email
    st.session_state.name = name
    st.session_state.photo = picture
    st.session_state.user_role = "privileged"

    st.experimental_set_query_params()
    st.experimental_rerun()

# ----------------------------
# Logout
# ----------------------------
def logout_user():
    keys_to_clear = ["user", "email", "name", "photo", "user_role"]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

# ----------------------------
# Sidebar Login
# ----------------------------
def sidebar_auth():
    handle_google_redirect()

    if "user" not in st.session_state:
        st.sidebar.markdown("## Sign in")
        login_url = get_google_auth_url()

        # Material-style Google button
        st.sidebar.markdown(f"""
        <a href="{login_url}" style="text-decoration:none;">
            <div style="
                display:flex; align-items:center; justify-content:center;
                background-color:#1a73e8; color:white; font-weight:600;
                border-radius:8px; padding:0.5rem; font-size:16px;
                width:100%; margin-top:0.5rem; cursor:pointer;
                transition: background-color 0.2s ease-in-out;
            " onmouseover="this.style.backgroundColor='#1669c1';" onmouseout="this.style.backgroundColor='#1a73e8';">
                🔵 Sign in with Google
            </div>
        </a>
        """, unsafe_allow_html=True)

    else:
        st.sidebar.markdown(f"👋 Welcome, **{st.session_state.get('name','User')}**!")
        st.sidebar.button("Logout", on_click=logout_user)
