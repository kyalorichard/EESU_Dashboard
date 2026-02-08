# auth.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, auth
import urllib.parse
import requests

# ----------------------------
# Firebase Setup
# ----------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
    firebase_admin.initialize_app(cred)

firebase = pyrebase.initialize_app(dict(st.secrets["firebase"]))
firebase_auth = firebase.auth()

PRIVILEGED_DOMAINS = set(st.secrets["access"]["privileged_domains"])
GOOGLE_CLIENT_ID = st.secrets["oauth"]["client_id"]
GOOGLE_CLIENT_SECRET = st.secrets["oauth"]["client_secret"]
REDIRECT_URI = st.secrets["oauth"]["redirect_uri"]

# ----------------------------
# Helpers
# ----------------------------
def get_email_domain(email: str) -> str:
    return email.split("@")[-1].lower()

def avatar_initials(email: str) -> str:
    name = email.split("@")[0]
    parts = name.replace(".", " ").split()
    return (parts[0][0] + (parts[1][0] if len(parts) > 1 else "")).upper()

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
        "access_type": "offline",
        "prompt": "select_account",
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"

def exchange_code_for_token(code: str):
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    resp = requests.post(token_url, data=data)
    resp.raise_for_status()
    return resp.json()  # contains id_token

def handle_google_redirect():
    params = st.experimental_get_query_params()
    if "code" in params:
        code = params["code"][0]
        token_data = exchange_code_for_token(code)
        id_token = token_data["id_token"]
        try:
            decoded = auth.verify_id_token(id_token)
        except Exception:
            st.error("Login failed. Invalid or expired token.")
            st.session_state.clear()
            st.experimental_set_query_params()
            st.rerun()

        st.session_state.user = {"idToken": id_token}
        st.session_state.email = decoded.get("email")
        st.session_state.photo = decoded.get("picture")
        st.session_state.user_role = (
            "privileged" if get_email_domain(st.session_state.email) in PRIVILEGED_DOMAINS else "public"
        )

        st.experimental_set_query_params()
        st.rerun()

# ----------------------------
# CSS for top-right auth
# ----------------------------
def inject_auth_css():
    st.markdown("""
    <style>
    .auth-container {
        position: fixed;
        top: 0.75rem;
        right: 1.5rem;
        z-index: 9999;
    }
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #1a73e8;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        cursor: pointer;
    }
    details summary { list-style: none; }
    details summary::-webkit-details-marker { display: none; }
    .dropdown-panel {
        margin-top: 0.4rem;
        background: white;
        border-radius: 10px;
        padding: 0.75rem;
        width: 240px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.18);
        animation: slideDown 0.25s ease-out;
    }
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# Top-right auth UI
# ----------------------------
def top_right_auth():
    handle_google_redirect()
    inject_auth_css()
    
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)

    # Public user
    if "user" not in st.session_state:
        google_url = get_google_auth_url()
        st.markdown(
            f"""
            <details>
              <summary><div class="avatar">?</div></summary>
              <div class="dropdown-panel">
                <a href="{google_url}">
                    <button style="width:100%">🔵 Sign in with Google</button>
                </a>
                <hr>
            """,
            unsafe_allow_html=True
        )

        # Email/password login
        with st.form("email_login"):
            email = st.text_input("Email", label_visibility="collapsed")
            password = st.text_input("Password", type="password", label_visibility="collapsed")
            if st.form_submit_button("Sign in"):
                try:
                    user = firebase_auth.sign_in_with_email_and_password(email, password)
                    st.session_state.user = user
                    st.session_state.email = email
                    st.session_state.photo = None
                    st.session_state.user_role = (
                        "privileged" if get_email_domain(email) in PRIVILEGED_DOMAINS else "public"
                    )
                    st.experimental_rerun()
                except Exception:
                    st.error("Invalid email or password.")

        st.markdown("</div></details>", unsafe_allow_html=True)

    # Logged-in user
    else:
        email = st.session_state.get("email")
        role = st.session_state.get("user_role")
        photo = st.session_state.get("photo")

        avatar_html = (
            f"<img src='{photo}' class='avatar'>" if photo else f"<div class='avatar'>{avatar_initials(email)}</div>"
        )

        st.markdown(
            f"""
            <details>
              <summary>{avatar_html}</summary>
              <div class="dropdown-panel">
                <strong>{email}</strong><br>
                <small>{role.capitalize()} access</small>
            """,
            unsafe_allow_html=True
        )

        if st.button("Logout"):
            for key in ["user", "email", "photo", "user_role"]:
                st.session_state.pop(key, None)
            st.experimental_rerun()

        st.markdown("</div></details>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
