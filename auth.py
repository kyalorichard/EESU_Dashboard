# auth.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, auth
import urllib.parse
import requests

# ----------------------------
# Safe Firebase Admin Init
# ----------------------------
if "firebase_admin" in st.secrets:
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Firebase Admin initialization failed: {e}")
else:
    st.warning("⚠️ 'firebase_admin' secrets not found. Admin features disabled.")

# ----------------------------
# Safe Pyrebase Init
# ----------------------------
firebase_config = dict(st.secrets.get("firebase", {}))
if "databaseURL" not in firebase_config:
    # Pyrebase requires databaseURL, use dummy if missing
    firebase_config["databaseURL"] = "https://dummy.firebaseio.com/"

try:
    firebase = pyrebase.initialize_app(firebase_config)
    firebase_auth = firebase.auth()
except Exception as e:
    st.error(f"Pyrebase initialization failed: {e}")
    firebase_auth = None

# ----------------------------
# Config
# ----------------------------
PRIVILEGED_DOMAINS = set(st.secrets.get("access", {}).get("privileged_domains", []))
GOOGLE_CLIENT_ID = st.secrets.get("oauth", {}).get("client_id", "")
GOOGLE_CLIENT_SECRET = st.secrets.get("oauth", {}).get("client_secret", "")
REDIRECT_URI = st.secrets.get("oauth", {}).get("redirect_uri", "")

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
    if not GOOGLE_CLIENT_ID or not REDIRECT_URI:
        return "#"
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
    try:
        params = st.experimental_get_query_params()
    except Exception:
        params = {}

    if "code" in params:
        code = params["code"][0]
        try:
            token_data = exchange_code_for_token(code)
            id_token = token_data["id_token"]
            decoded = auth.verify_id_token(id_token)
        except Exception:
            st.error("Google login failed. Invalid or expired token.")
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
    if "user" not in st.session_state or firebase_auth is None:
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
        if firebase_auth is not None:
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
