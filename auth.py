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
# Pyrebase Init
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

    if get_email_domain(email) not in PRIVILEGED_DOMAINS:
        st.error(f"Access denied. Only emails from {', '.join(PRIVILEGED_DOMAINS)} are allowed.")
        st.experimental_set_query_params()
        return

    st.session_state.user = "google"
    st.session_state.email = email
    st.session_state.name = name
    st.session_state.photo = picture
    st.session_state.user_role = "privileged"
    st.session_state.auth_open = False
    st.experimental_set_query_params()
    st.experimental_rerun()

# ----------------------------
# CSS
# ----------------------------
def inject_auth_css():
    st.markdown("""
    <style>
    .auth-container {
        position: fixed;
        top: 1rem;
        left: 1rem;
        z-index: 9999;
    }
    .avatar {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        background: #1a73e8;
        color: white;
        font-weight: 600;
        font-size: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
    }
    .avatar-img {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        object-fit: cover;
        cursor: pointer;
    }
    .floating-auth-box {
        position: fixed;
        top: 60px;
        left: 10px;
        background: white;
        border-radius: 10px;
        padding: 1rem;
        width: 280px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        z-index: 10000;
    }
    .floating-auth-box button {
        width: 100%;
        padding: 0.5rem 0;
        margin-top: 0.5rem;
        border-radius: 6px;
        border: none;
        background-color: #1a73e8;
        color: white;
        cursor: pointer;
        font-weight: 600;
    }
    .floating-auth-box button:hover {
        background-color: #1558b0;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# Clickable floating avatar login
# ----------------------------
def top_right_auth():
    handle_google_redirect()
    if "auth_open" not in st.session_state:
        st.session_state.auth_open = False

    st.markdown('<div class="auth-container">', unsafe_allow_html=True)

    email = st.session_state.get("email", "?")
    photo = st.session_state.get("photo")
    name = st.session_state.get("name", "User")

    # Streamlit checkbox hack to toggle login box
    if st.checkbox("", key="avatar_toggle", value=st.session_state.auth_open):
        st.session_state.auth_open = True
    else:
        st.session_state.auth_open = False

    # Show avatar using st.markdown
    if photo:
        st.markdown(f'<img src="{photo}" class="avatar-img">', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="avatar">{avatar_initials(email)}</div>', unsafe_allow_html=True)

    # Floating login box
    if st.session_state.auth_open:
        st.markdown('<div class="floating-auth-box">', unsafe_allow_html=True)

        if "user" not in st.session_state:
            st.markdown(f'<a href="{get_google_auth_url()}"><button>🔵 Sign in with Google</button></a>', unsafe_allow_html=True)
            if firebase_auth:
                st.divider()
                with st.form("email_login"):
                    email_input = st.text_input("Email")
                    password = st.text_input("Password", type="password")
                    submit = st.form_submit_button("Sign in with Email")
                    if submit:
                        try:
                            user = firebase_auth.sign_in_with_email_and_password(email_input, password)
                            domain = get_email_domain(email_input)
                            if domain not in PRIVILEGED_DOMAINS:
                                st.error(f"Access denied. Only emails from {', '.join(PRIVILEGED_DOMAINS)} are allowed.")
                            else:
                                st.session_state.user = user
                                st.session_state.email = email_input
                                st.session_state.name = email_input.split("@")[0].title()
                                st.session_state.user_role = "privileged"
                                st.session_state.auth_open = False
                                st.success(f"✅ Welcome, {st.session_state.name}!")
                                st.experimental_rerun()
                        except Exception:
                            st.error("Firebase login failed")
        else:
            st.markdown(f"**👋 Welcome, {name}!**")
            st.caption(email)
            st.caption(st.session_state.get("user_role", "public").capitalize())
            st.divider()
            if st.button("Logout"):
                st.session_state.clear()
                st.experimental_rerun()

        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
