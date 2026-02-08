# auth.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
import urllib.parse
import requests
from google.oauth2 import id_token
from google.auth.transport import requests as grequests

# =================================================
# Firebase Admin (safe init)
# =================================================
if "firebase_admin" in st.secrets and not firebase_admin._apps:
    try:
        cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
        firebase_admin.initialize_app(cred)
    except Exception:
        pass

# =================================================
# Pyrebase (safe init)
# =================================================
firebase_auth = None
firebase_cfg = dict(st.secrets.get("firebase", {}))
if firebase_cfg:
    firebase_cfg.setdefault("databaseURL", "https://dummy.firebaseio.com/")
    try:
        firebase = pyrebase.initialize_app(firebase_cfg)
        firebase_auth = firebase.auth()
    except Exception:
        firebase_auth = None

# =================================================
# Config
# =================================================
PRIVILEGED_DOMAINS = set(
    st.secrets.get("access", {}).get("privileged_domains", [])
)

GOOGLE_CLIENT_ID = st.secrets.get("oauth", {}).get("client_id")
GOOGLE_CLIENT_SECRET = st.secrets.get("oauth", {}).get("client_secret")
REDIRECT_URI = st.secrets.get("oauth", {}).get("redirect_uri")

# =================================================
# Helpers
# =================================================
def get_email_domain(email: str) -> str:
    return email.split("@")[-1].lower()

def avatar_initials(email: str) -> str:
    parts = email.split("@")[0].replace(".", " ").split()
    return "".join(p[0].upper() for p in parts[:2])

def is_privileged() -> bool:
    return st.session_state.get("user_role") == "privileged"

# =================================================
# Google OAuth URL
# =================================================
def get_google_auth_url() -> str:
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "prompt": "select_account",
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)

# =================================================
# REAL Google OAuth handler
# =================================================
def handle_google_redirect():
    params = st.query_params

    if "code" not in params:
        return

    code = params["code"]

    # ---- Exchange code for tokens ----
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

    if token_resp.status_code != 200:
        st.error("Google authentication failed")
        st.query_params.clear()
        return

    tokens = token_resp.json()
    idinfo = id_token.verify_oauth2_token(
        tokens["id_token"],
        grequests.Request(),
        GOOGLE_CLIENT_ID,
    )

    email = idinfo.get("email")
    name = idinfo.get("name")
    picture = idinfo.get("picture")

    if not email:
        st.error("Google login failed: no email")
        return

    # ---- Store session ----
    st.session_state.user = "google"
    st.session_state.email = email
    st.session_state.name = name
    st.session_state.photo = picture
    st.session_state.user_role = (
        "privileged"
        if get_email_domain(email) in PRIVILEGED_DOMAINS
        else "public"
    )

    # ---- Clean URL and rerun ----
    st.query_params.clear()
    st.rerun()

# =================================================
# CSS – top-left avatar
# =================================================
def inject_auth_css():
    st.markdown("""
    <style>
    .block-container { padding-top: 4.2rem; }
    .auth-container {
        position: fixed;
        top: 0.8rem;
        left: 1.2rem;
        z-index: 9999;
    }
    .auth-panel {
        background: white;
        border-radius: 10px;
        padding: 0.8rem;
        width: 260px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# =================================================
# Top-left avatar auth UI
# =================================================
def top_right_auth():
    handle_google_redirect()

    if "auth_open" not in st.session_state:
        st.session_state.auth_open = False

    st.markdown('<div class="auth-container">', unsafe_allow_html=True)

    avatar = (
        avatar_initials(st.session_state["email"])
        if "email" in st.session_state
        else "?"
    )

    if st.button(avatar, key="avatar"):
        st.session_state.auth_open = not st.session_state.auth_open

    if st.session_state.auth_open:
        st.markdown('<div class="auth-panel">', unsafe_allow_html=True)

        if "user" not in st.session_state:
            st.link_button(
                "🔵 Sign in with Google",
                get_google_auth_url(),
                use_container_width=True,
            )

            if firebase_auth:
                st.divider()
                with st.form("email_login"):
                    email = st.text_input("Email")
                    password = st.text_input("Password", type="password")
                    if st.form_submit_button("Sign in with Email"):
                        user = firebase_auth.sign_in_with_email_and_password(email, password)
                        st.session_state.user = user
                        st.session_state.email = email
                        st.session_state.user_role = (
                            "privileged"
                            if get_email_domain(email) in PRIVILEGED_DOMAINS
                            else "public"
                        )
                        st.rerun()

        else:
            st.markdown(f"**{st.session_state.get('name')}**")
            st.caption(st.session_state.get("email"))
            st.caption(st.session_state.get("user_role").capitalize())

            st.divider()
            if st.button("Logout", use_container_width=True):
                st.session_state.clear()
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
