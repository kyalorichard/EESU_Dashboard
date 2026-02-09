import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
import urllib.parse
import requests
from google.oauth2 import id_token
from google.auth.transport import requests as grequests

# ---------------------------
# Firebase Admin Init
# ---------------------------
firebase_admin_initialized = False
if "firebase_admin" in st.secrets and not firebase_admin._apps:
    try:
        cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
        firebase_admin.initialize_app(cred)
        firebase_admin_initialized = True
    except Exception as e:
        st.warning(f"Firebase Admin initialization failed: {e}")

# ---------------------------
# Pyrebase Init
# ---------------------------
firebase_auth = None
firebase_cfg = dict(st.secrets.get("firebase", {}))
if firebase_cfg:
    try:
        firebase = pyrebase.initialize_app(firebase_cfg)
        firebase_auth = firebase.auth()
    except Exception as e:
        st.warning(f"Firebase email login not initialized: {e}")

# ---------------------------
# Config
# ---------------------------
PRIVILEGED_DOMAINS = set(st.secrets.get("access", {}).get("privileged_domains", []))

try:
    GOOGLE_CLIENT_ID = st.secrets["oauth"]["client_id"]
    GOOGLE_CLIENT_SECRET = st.secrets["oauth"]["client_secret"]
    REDIRECT_URI = st.secrets["oauth"]["redirect_uri"]
    oauth_enabled = True
except KeyError:
    oauth_enabled = False
    st.warning("⚠️ OAuth credentials missing. Google login will be disabled.")

# ---------------------------
# Helpers
# ---------------------------
def get_email_domain(email: str) -> str:
    return email.split("@")[-1].lower()

def init_state():
    st.session_state.setdefault("show_login", False)
    st.session_state.setdefault("email_input", "")
    st.session_state.setdefault("password_input", "")
    st.session_state.setdefault("login_error", "")

# ---------------------------
# Google OAuth
# ---------------------------
def get_google_auth_url():
    if not oauth_enabled:
        return "#"
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "prompt": "select_account",
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)

def handle_google_redirect():
    if not oauth_enabled or "code" not in st.query_params:
        return

    code = st.query_params["code"][0] if isinstance(st.query_params["code"], list) else st.query_params["code"]

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

        if get_email_domain(email) not in PRIVILEGED_DOMAINS:
            st.error("Access denied")
            st.query_params.clear()
            return

        st.session_state.user = "google"
        st.session_state.email = email
        st.session_state.name = name
        st.session_state.user_role = "privileged"
        st.session_state.show_login = False
        st.query_params.clear()

    except Exception:
        st.error("Google login failed")

# ---------------------------
# Logout
# ---------------------------
def logout_user():
    for k in ["user", "email", "name", "user_role"]:
        st.session_state.pop(k, None)
    st.session_state.show_login = False

# ---------------------------
# CSS (Sliding Drawer)
# ---------------------------
def inject_css():
    st.markdown("""
    <style>
    .drawer-overlay {
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,.4);
        z-index: 9999;
        display:flex;
        justify-content:flex-end;
        transition: opacity 0.3s ease;
    }
    .drawer-card {
        background:#fff;
        width:350px;
        height:100%;
        padding:1.5rem;
        box-shadow:-10px 0 30px rgba(0,0,0,.3);
        transform: translateX(100%);
        transition: transform 0.3s ease;
        display:flex;
        flex-direction:column;
    }
    .drawer-card.show {
        transform: translateX(0);
    }
    .drawer-card input { padding:.5rem; margin-bottom:.5rem; width:100%; border-radius:6px; border:1px solid #ccc; }
    .drawer-card button { padding:.6rem; width:100%; border-radius:6px; border:none; font-weight:600; cursor:pointer; margin-bottom:.3rem; }
    .google-btn { background:#1a73e8; color:white; }
    .cancel-btn { background:#ccc; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# AUTH UI (Sliding Drawer)
# ---------------------------
def auth_ui():
    init_state()
    inject_css()
    handle_google_redirect()

    # --- LOGGED IN ---
    if "user" in st.session_state:
        st.sidebar.success(f"👋 {st.session_state['name']}")
        st.sidebar.button("Logout", on_click=logout_user)
        return

    # --- LOGGED OUT ---
    st.sidebar.markdown("## Account")
    if st.sidebar.button("🔐 Sign in"):
        st.session_state.show_login = True

    if not st.session_state.show_login:
        return

    login_url = get_google_auth_url()

    email = st.session_state.email_input
    password = st.session_state.password_input
    login_error = st.session_state.login_error

    with st.form("email_login_form"):
        st.markdown('<div class="drawer-overlay">', unsafe_allow_html=True)
        st.markdown(f'<div class="drawer-card show">', unsafe_allow_html=True)

        st.markdown("<h3>Sign in</h3>", unsafe_allow_html=True)

        # Google login button (if available)
        if oauth_enabled:
            st.markdown(f"""
            <a href="{login_url}" style="text-decoration:none">
                <button class="google-btn">🔵 Sign in with Google</button>
            </a>
            """, unsafe_allow_html=True)
        else:
            st.info("Google login disabled. Missing OAuth credentials.")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Email login
        email = st.text_input("Email", value=email)
        password = st.text_input("Password", type="password", value=password)
        submitted = st.form_submit_button("Sign in with Email")

        if submitted:
            st.session_state.email_input = email
            st.session_state.password_input = password
            if not firebase_auth:
                st.session_state.login_error = "Firebase not initialized."
            elif get_email_domain(email) not in PRIVILEGED_DOMAINS:
                st.session_state.login_error = "Access denied for domain."
            else:
                try:
                    firebase_auth.sign_in_with_email_and_password(email, password)
                    st.session_state.user = "email"
                    st.session_state.email = email
                    st.session_state.name = email.split("@")[0].title()
                    st.session_state.user_role = "privileged"
                    st.session_state.show_login = False
                    st.session_state.login_error = ""
                except Exception as e:
                    st.session_state.login_error = f"Login failed: {e}"

        if login_error:
            st.error(login_error)

        if st.button("Cancel"):
            st.session_state.show_login = False

        st.markdown('</div></div>', unsafe_allow_html=True)
