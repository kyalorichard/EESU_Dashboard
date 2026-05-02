# auth.py
import json
import time
import streamlit as st

DEBUG = False

# ============================================================
# Optional imports
# ============================================================
try:
    import pyrebase
    HAS_PYREBASE = True
except ImportError:
    pyrebase = None
    HAS_PYREBASE = False

try:
    import firebase_admin
    from firebase_admin import credentials
    HAS_FIREBASE_ADMIN = True
except ImportError:
    firebase_admin = None
    credentials = None
    HAS_FIREBASE_ADMIN = False

try:
    from streamlit_cookies_manager import EncryptedCookieManager
    HAS_COOKIES = True
except ImportError:
    EncryptedCookieManager = None
    HAS_COOKIES = False


# ============================================================
# Firebase init
# ============================================================
def init_firebase_admin():
    if not HAS_FIREBASE_ADMIN:
        return None

    secrets_admin = st.secrets.get("firebase_admin", {})
    if not secrets_admin:
        return None

    try:
        private_key = secrets_admin["private_key"].replace("\\n", "\n")

        cred = credentials.Certificate({
            "type": "service_account",
            "project_id": secrets_admin["project_id"],
            "private_key_id": secrets_admin["private_key_id"],
            "private_key": private_key,
            "client_email": secrets_admin["client_email"],
            "client_id": secrets_admin["client_id"],
            "auth_uri": secrets_admin["auth_uri"],
            "token_uri": secrets_admin["token_uri"],
            "auth_provider_x509_cert_url": secrets_admin["auth_provider_x509_cert_url"],
            "client_x509_cert_url": secrets_admin["client_x509_cert_url"],
        })

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        return firebase_admin

    except Exception as e:
        if DEBUG:
            st.warning(f"Firebase Admin init failed: {e}")
        return None


def init_firebase_client():
    cfg = st.secrets.get("firebase", {})
    if not cfg or not HAS_PYREBASE:
        return None, None

    try:
        firebase = pyrebase.initialize_app(cfg)
        auth = firebase.auth()
        return firebase, auth

    except Exception as e:
        if DEBUG:
            st.warning(f"Firebase client init failed: {e}")
        return None, None


firebase_admin_app = init_firebase_admin()
firebase_client, firebase_auth = init_firebase_client()

PRIVILEGED_DOMAINS = set(
    d.lower() for d in st.secrets.get("access", {}).get("privileged_domains", [])
)


# ============================================================
# Session helpers
# ============================================================
def init_session():
    defaults = {
        "user": False,
        "email": None,
        "name": None,
        "role": None,
        "email_verified": False,
        "auth_mode": "Login",
        "auth_view": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def is_authenticated():
    return bool(
        st.session_state.get("user")
        and st.session_state.get("email_verified")
    )


def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ============================================================
# UX (PROFESSIONAL)
# ============================================================
def _auth_page_css():
    st.markdown("""
    <style>
    html, body, .stApp {
        background: linear-gradient(135deg, #020617, #0f172a);
        font-family: Inter, sans-serif;
        color: white;
    }

    .wrapper {
        display: flex;
        height: 100vh;
    }

    .left {
        flex: 1;
        padding: 80px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .title {
        font-size: 56px;
        font-weight: 900;
        line-height: 1;
    }

    .accent {
        background: linear-gradient(90deg,#a855f7,#38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        margin-top: 20px;
        font-size: 16px;
        color: #cbd5e1;
        max-width: 420px;
    }

    .right {
        width: 420px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .card {
        width: 100%;
        padding: 40px;
        border-radius: 20px;
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .stTextInput input {
        border-radius: 10px;
        height: 45px;
    }

    .stButton button {
        width: 100%;
        height: 45px;
        border-radius: 10px;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# Forms
# ============================================================
def _login_form():
    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign in")

    if submit:
        try:
            user = firebase_auth.sign_in_with_email_and_password(email, password)
            info = firebase_auth.get_account_info(user["idToken"])

            verified = info["users"][0]["emailVerified"]

            if not verified:
                st.error("Verify your email first.")
                return

            st.session_state.user = True
            st.session_state.email = email
            st.session_state.email_verified = True

            st.success("Login successful")
            st.rerun()

        except Exception as e:
            st.error("Invalid credentials")


def _register_form():
    with st.form("register"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Create account")

    if submit:
        try:
            user = firebase_auth.create_user_with_email_and_password(email, password)
            firebase_auth.send_email_verification(user["idToken"])
            st.success("Check your email to verify account")
        except Exception:
            st.error("Registration failed")


def _reset_form():
    with st.form("reset"):
        email = st.text_input("Email")
        submit = st.form_submit_button("Reset password")

    if submit:
        try:
            firebase_auth.send_password_reset_email(email)
            st.success("Reset email sent")
        except Exception:
            st.error("Error sending reset email")


# ============================================================
# Main UI
# ============================================================
def _render_auth_page():
    _auth_page_css()

    mode = st.session_state.get("auth_mode", "Login")

    left, right = st.columns([2, 1])

    with left:
        st.markdown("""
        <div class="title">
        Intelligence for a<br>
        <span class="accent">Safer Europe</span>
        </div>

        <div class="subtitle">
        Advanced intelligence platform delivering real-time insights and
        analytics across regions.
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if mode == "Login":
            _login_form()
        elif mode == "Register":
            _register_form()
        else:
            _reset_form()

        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# Entry
# ============================================================
def auth_ui():
    init_session()

    if is_authenticated():
        return

    _render_auth_page()
