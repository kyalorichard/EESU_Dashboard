# ---------------- auth.py ----------------
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
import urllib.parse
import requests
from google.oauth2 import id_token
from google.auth.transport import requests as grequests

# --------------------------- Firebase Admin ---------------------------
if "firebase_admin" in st.secrets and not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
    firebase_admin.initialize_app(cred)

# --------------------------- Pyrebase Auth ---------------------------
firebase_auth = None
firebase_cfg = dict(st.secrets.get("firebase", {}))
if firebase_cfg:
    firebase = pyrebase.initialize_app(firebase_cfg)
    firebase_auth = firebase.auth()

# --------------------------- Config ---------------------------
PRIVILEGED_DOMAINS = set(st.secrets.get("access", {}).get("privileged_domains", []))
GOOGLE_CLIENT_ID = st.secrets["oauth"]["client_id"]
GOOGLE_CLIENT_SECRET = st.secrets["oauth"]["client_secret"]
REDIRECT_URI = st.secrets["oauth"]["redirect_uri"]

# --------------------------- Helpers ---------------------------
def get_email_domain(email: str) -> str:
    return email.split("@")[-1].lower()

def init_state():
    st.session_state.setdefault("show_login", False)
    st.session_state.setdefault("email_input", "")
    st.session_state.setdefault("password_input", "")
    st.session_state.setdefault("login_error", "")
    st.session_state.setdefault("_close_login_state", False)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("user_role", None)
    st.session_state.setdefault("name", None)

# --------------------------- Google OAuth ---------------------------
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
    if "code" not in st.query_params:
        return
    code_param = st.query_params["code"]
    code = code_param[0] if isinstance(code_param, list) else code_param
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
        idinfo = id_token.verify_oauth2_token(tokens["id_token"], grequests.Request(), GOOGLE_CLIENT_ID)

        email = idinfo["email"]
        name = idinfo.get("name", email.split("@")[0].title())

        if get_email_domain(email) not in PRIVILEGED_DOMAINS:
            st.error("Access denied")
            st.query_params = {}
            return

        st.session_state.user = "google"
        st.session_state.email = email
        st.session_state.name = name
        st.session_state.user_role = "privileged"
        st.session_state.show_login = False
        st.query_params = {}

    except requests.exceptions.RequestException:
        st.error("Google login failed: network issue")
        st.query_params = {}
    except ValueError:
        st.error("Google login failed: invalid token")
        st.query_params = {}
    except Exception as e:
        st.error(f"Google login failed: {e}")
        st.query_params = {}

# --------------------------- Logout ---------------------------
def logout_user():
    for k in ["user", "email", "name", "user_role"]:
        st.session_state.pop(k, None)
    st.session_state.show_login = False

# --------------------------- CSS ---------------------------
def inject_css():
    st.markdown("""
    <style>
    /* Overlay */
    .login-overlay {
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,.45);
        z-index: 9998;
        opacity: 0;
        visibility: hidden;
        pointer-events: none;
        transition: opacity 0.3s ease, visibility 0.3s ease;
    }
    .login-overlay.show {
        opacity: 1;
        visibility: visible;
        pointer-events: all;
    }

    /* Drawer */
    .login-card {
        position: fixed;
        top:0; right:-400px;
        width:350px;
        height:100%;
        background:#fff;
        padding:1.5rem;
        box-shadow:-10px 0 30px rgba(0,0,0,.3);
        z-index: 9999;
        display:flex;
        flex-direction:column;
        transition: right 0.3s ease;
    }
    .login-card.show { right:0; }

    /* Inputs & Buttons */
    .login-card input {
        padding:.5rem;
        margin-bottom:.5rem;
        width:100%;
        border-radius:6px;
        border:1px solid #ccc;
    }
    .login-card button {
        padding:.6rem;
        width:100%;
        border-radius:6px;
        border:none;
        font-weight:600;
        cursor:pointer;
        margin-bottom:.5rem;
        transition: all 0.2s ease;
    }
    .google-btn { background:#1a73e8; color:white; }
    .google-btn:hover { background:#1669c1; box-shadow: 0 0 8px rgba(26,115,232,0.6); }
    .email-btn { background:#4caf50; color:white; }
    .email-btn:hover { background:#3b8c40; box-shadow: 0 0 8px rgba(76,175,80,0.6); }
    .cancel-btn { background:#ccc; }
    .cancel-btn:hover { background:#999; }
    </style>
    """, unsafe_allow_html=True)

# --------------------------- AUTH UI ---------------------------
def auth_ui():
    init_state()
    inject_css()
    handle_google_redirect()

    # Already logged in
    if st.session_state.user:
        st.sidebar.success(f"👋 {st.session_state['name']}")
        st.sidebar.button("Logout", on_click=logout_user)
        return

    # Sidebar sign-in button
    st.sidebar.markdown("## Account")
    if st.sidebar.button("🔐 Sign in"):
        st.session_state.show_login = True

    # Login overlay & drawer
    login_url = get_google_auth_url()
    overlay_class = "login-overlay show" if st.session_state.show_login else "login-overlay"
    card_class = "login-card show" if st.session_state.show_login else "login-card"

    st.markdown(f"""
    <div class="{overlay_class}" id="loginOverlay"></div>
    <div class="{card_class}" id="loginCard">
        <h3>Sign in</h3>

        <!-- Google Login -->
        <a href="{login_url}" style="text-decoration:none">
            <button class="google-btn">🔵 Sign in with Google</button>
        </a>

        <hr>

        <!-- Email Login -->
        <input type="text" id="email_input" placeholder="Email" value="{st.session_state.email_input}" title="Use a privileged domain email">
        <input type="password" id="password_input" placeholder="Password" value="{st.session_state.password_input}">
        <button class="email-btn" id="email_login_btn">Sign in with Email</button>

        <p style="color:red; text-align:center;">{st.session_state.login_error}</p>
        <button class="cancel-btn" onclick="window.parent.postMessage({{func:'closeLoginState'}}, '*')">Cancel</button>
    </div>

    <script>
    const overlay = document.getElementById('loginOverlay');
    overlay.addEventListener('click', () => {{
        window.parent.postMessage({{func:'closeLoginState'}}, '*');
    }});
    const card = document.getElementById('loginCard');
    card.addEventListener('click', e => e.stopPropagation());
    </script>
    """, unsafe_allow_html=True)

    # Streamlit Email Login Handling
    email_input = st.text_input("Email", value=st.session_state.email_input, key="email_field")
    password_input = st.text_input("Password", value=st.session_state.password_input, type="password", key="password_field")
    if st.button("Sign in with Email", key="email_submit"):
        st.session_state.email_input = email_input
        st.session_state.password_input = password_input

        if not firebase_auth:
            st.session_state.login_error = "Firebase not initialized."
        elif get_email_domain(email_input) not in PRIVILEGED_DOMAINS:
            st.session_state.login_error = "Access denied for domain."
        else:
            try:
                firebase_auth.sign_in_with_email_and_password(email_input, password_input)
                st.session_state.user = "email"
                st.session_state.email = email_input
                st.session_state.name = email_input.split("@")[0].title()
                st.session_state.user_role = "privileged"
                st.session_state.show_login = False
                st.session_state.login_error = ""
            except Exception as e:
                st.session_state.login_error = f"Login failed: {e}"

    # Close overlay if requested
    if st.session_state.get("_close_login_state", False):
        st.session_state.show_login = False
        st.session_state["_close_login_state"] = False
