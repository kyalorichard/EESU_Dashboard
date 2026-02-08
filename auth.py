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
        st.error("Google login failed or token invalid. Check your client ID / redirect URI.")
        st.experimental_set_query_params()
        return

    # Domain restriction
    if get_email_domain(email) not in PRIVILEGED_DOMAINS:
        st.error(f"Access denied. Only emails from {', '.join(PRIVILEGED_DOMAINS)} are allowed.")
        st.experimental_set_query_params()
        return

    # Save session
    st.session_state.user = "google"
    st.session_state.email = email
    st.session_state.name = name
    st.session_state.photo = picture
    st.session_state.user_role = "privileged"

    st.experimental_set_query_params()
    st.experimental_rerun()

# ----------------------------
# CSS for floating avatar + Gmail-like popup
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
    .avatar-button {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        background: #1a73e8;
        color: white;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 16px;
        user-select: none;
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
        top: 70px;
        left: 20px;
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        width: 300px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        z-index: 10000;
        animation: fadeIn 0.2s ease-out;
        font-family: "Google Sans", sans-serif;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .floating-auth-box h3 {
        margin-top: 0;
        margin-bottom: 0.8rem;
        font-size: 18px;
        color: #202124;
    }
    .floating-auth-box button {
        padding: 0.55rem 0;
        margin-top: 0.6rem;
        border-radius: 4px;
        border: none;
        background-color: #1a73e8;
        color: white;
        cursor: pointer;
        font-weight: 600;
        width: 100%;
    }
    .floating-auth-box button:hover {
        background-color: #1558b0;
    }
    .floating-auth-box input {
        width: 100%;
        padding: 0.5rem;
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
        border-radius: 4px;
        border: 1px solid #dadce0;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# Floating Avatar Login UI
# ----------------------------
def top_right_auth():
    handle_google_redirect()

    if "auth_open" not in st.session_state:
        st.session_state.auth_open = False

    st.markdown('<div class="auth-container">', unsafe_allow_html=True)

    # Avatar
    photo = st.session_state.get("photo")
    email = st.session_state.get("email", "?")
    avatar_html = (
        f'<img src="{photo}" class="avatar-img">' if photo else f'<div class="avatar-button">{avatar_initials(email)}</div>'
    )

    if st.button("", key="avatar_toggle"):
        st.session_state.auth_open = not st.session_state.auth_open

    # Floating login box
    if st.session_state.auth_open:
        st.markdown('<div class="floating-auth-box">', unsafe_allow_html=True)

        if "user" not in st.session_state:
            st.markdown("<h3>Sign in</h3>", unsafe_allow_html=True)
            # Google login
            st.markdown(f'<a href="{get_google_auth_url()}"><button>🔵 Sign in with Google</button></a>', unsafe_allow_html=True)

            # Email login
            if firebase_auth:
                st.divider()
                with st.form("email_login_form"):
                    email_input = st.text_input("Email")
                    password_input = st.text_input("Password", type="password")
                    submit = st.form_submit_button("Sign in with Email")
                    if submit:
                        try:
                            user = firebase_auth.sign_in_with_email_and_password(email_input, password_input)
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
                        except Exception as e:
                            try:
                                err = json.loads(e.args[1])
                                st.error(err['error']['message'])
                            except Exception:
                                st.error("Firebase login failed")
        else:
            # Logged in
            name = st.session_state.get("name", "User")
            email = st.session_state.get("email")
            role = st.session_state.get("user_role", "public")
            st.markdown(f"**👋 Welcome, {name}!**")
            st.caption(email)
            st.caption(role.capitalize())
            st.divider()
            if st.button("Logout", use_container_width=True):
                st.session_state.clear()
                st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Show small welcome message on dashboard even if popup closed
    if "user" in st.session_state:
        st.markdown(f"👋 Welcome, **{st.session_state.get('name', 'User')}**!", unsafe_allow_html=True)
