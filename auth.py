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

    st.session_state.user = "google"
    st.session_state.email = email
    st.session_state.name = name
    st.session_state.photo = picture
    st.session_state.user_role = "privileged"

    st.experimental_set_query_params()
    st.experimental_rerun()

# ----------------------------
# CSS
# ----------------------------
def inject_auth_css():
    st.markdown("""
    <style>
    .auth-container { position: fixed; top: 1rem; left: 1rem; z-index: 9999; }
    .avatar-button { width:50px; height:50px; border-radius:50%; background:#1a73e8; color:white; font-weight:600; display:flex; align-items:center; justify-content:center; cursor:pointer; font-size:18px; }
    .avatar-img { width:50px; height:50px; border-radius:50%; object-fit:cover; cursor:pointer; }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# Top-left avatar + centered modal login
# ----------------------------
def top_right_auth():
    handle_google_redirect()
    if "auth_open" not in st.session_state:
        st.session_state.auth_open = False

    import streamlit.components.v1 as components

    photo = st.session_state.get("photo")
    email = st.session_state.get("email", "?")
    avatar_html = f'<img src="{photo}" class="avatar-img">' if photo else f'<div class="avatar-button">{avatar_initials(email)}</div>'
    
    if st.button("Toggle Login"):
        st.session_state.auth_open = not st.session_state.auth_open
    st.markdown(avatar_html, unsafe_allow_html=True)

    if st.session_state.get("auth_open", False):
        # Modal HTML
        modal_html = f"""
        <div style="
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(0,0,0,0.5); z-index: 99999;
            display: flex; justify-content: center; align-items: center;
        " id="authModal">
            <div style="
                background: white; border-radius: 12px; padding: 2rem;
                width: 400px; max-width: 90%; box-shadow:0 12px 32px rgba(0,0,0,0.4);
                text-align: center; font-family: 'Google Sans', sans-serif;
            ">
                {"<h3>Sign in</h3>" if "user" not in st.session_state else f"👋 Welcome, <strong>{st.session_state.get('name','User')}</strong>!"}
                {f'<a href="{get_google_auth_url()}"><button style="width:100%; margin-top:1rem;">🔵 Sign in with Google</button></a>' if "user" not in st.session_state else ""}
                {"" if "user" in st.session_state else '''
                <hr style="margin:1rem 0;">
                <form>
                    <input type="text" placeholder="Email" style="width:100%; padding:0.5rem; margin-bottom:0.5rem;">
                    <input type="password" placeholder="Password" style="width:100%; padding:0.5rem; margin-bottom:0.5rem;">
                    <button type="button" style="width:100%;">Sign in with Email</button>
                </form>
                '''}
                {f'<br><button style="width:100%; margin-top:1rem;" onclick="alert(\'Logout handled in backend\')">Logout</button>' if "user" in st.session_state else ""}
            </div>
        </div>

        <script>
        const modal = document.getElementById('authModal');
        modal.addEventListener('click', function(e) {{
            if(e.target === modal) {{
                window.parent.postMessage({{func:"closeAuthModal"}}, "*");
            }}
        }});
        </script>
        """
        components.html(modal_html, height=700)

    # Welcome note on dashboard
    if "user" in st.session_state:
        st.markdown(f"👋 Welcome, **{st.session_state.get('name','User')}**!", unsafe_allow_html=True)
