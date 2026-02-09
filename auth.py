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
if "firebase_admin" in st.secrets and not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
    firebase_admin.initialize_app(cred)

# ---------------------------
# Pyrebase Init (Email login)
# ---------------------------
firebase_auth = None
firebase_cfg = dict(st.secrets.get("firebase", {}))
if firebase_cfg:
    firebase = pyrebase.initialize_app(firebase_cfg)
    firebase_auth = firebase.auth()

# ---------------------------
# Config
# ---------------------------
PRIVILEGED_DOMAINS = set(st.secrets.get("access", {}).get("privileged_domains", []))
GOOGLE_CLIENT_ID = st.secrets["oauth"]["client_id"]
GOOGLE_CLIENT_SECRET = st.secrets["oauth"]["client_secret"]
REDIRECT_URI = st.secrets["oauth"]["redirect_uri"]

# ---------------------------
# Helpers
# ---------------------------
def get_email_domain(email: str) -> str:
    return email.split("@")[-1].lower()

def init_state():
    st.session_state.setdefault("show_login", False)

# ---------------------------
# Google OAuth
# ---------------------------
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

    code = st.query_params["code"]

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
# CSS (Floating Card)
# ---------------------------
def inject_css():
    st.markdown("""
    <style>
    .login-overlay {
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,.45);
        z-index: 9999;
        display:flex;
        align-items:center;
        justify-content:center;
    }
    .login-card {
        background:#fff;
        width:380px;
        padding:1.5rem;
        border-radius:14px;
        box-shadow:0 25px 60px rgba(0,0,0,.35);
        animation: pop .25s ease-out;
    }
    @keyframes pop {
        from {opacity:0; transform:scale(.9)}
        to   {opacity:1; transform:scale(1)}
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# AUTH UI (Floating Card with Google + Email)
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

    # Floating login card (Google + Email)
    st.markdown(f"""
    <div class="login-overlay" id="loginOverlay">
      <div class="login-card" onclick="event.stopPropagation()">
        <h3>Sign in</h3>

        <!-- Google Login -->
        <a href="{login_url}" style="text-decoration:none">
          <div style="
            background:#1a73e8;color:white;
            padding:.6rem;border-radius:8px;
            text-align:center;font-weight:600;">
            🔵 Sign in with Google
          </div>
        </a>

        <hr>

        <!-- Email/Password Login -->
        <form method="post" id="emailForm">
          <input name="email" placeholder="Email" style="width:100%;margin-bottom:.5rem">
          <input type="password" name="password" placeholder="Password" style="width:100%;margin-bottom:.5rem">
          <button type="submit" style="width:100%; margin-top:.3rem;">Sign in</button>
        </form>

        <button style="width:100%; margin-top:.5rem;" onclick="window.parent.postMessage({{func:'closeLogin'}}, '*')">
          Cancel
        </button>
      </div>
    </div>

    <script>
    // Close overlay when clicking outside
    const overlay = document.getElementById('loginOverlay');
    overlay.addEventListener('click', () => {{
        window.parent.postMessage({{func:'closeLoginState'}}, '*');
    }});

    const card = document.querySelector('.login-card');
    card.addEventListener('click', e => e.stopPropagation());
    </script>
    """, unsafe_allow_html=True)

    # Streamlit side: handle closing overlay
    if "_close_login_state" not in st.session_state:
        st.session_state["_close_login_state"] = False

    if st.session_state["_close_login_state"]:
        st.session_state.show_login = False
        st.session_state["_close_login_state"] = False

    # --- Email login backend ---
    if "email" in st.experimental_get_query_params() and firebase_auth:
        form_data = st.experimental_get_query_params()
        email = form_data.get("email", [""])[0]
        password = form_data.get("password", [""])[0]
        if email and password:
            try:
                user = firebase_auth.sign_in_with_email_and_password(email, password)
                st.session_state.user = "email"
                st.session_state.email = email
                st.session_state.name = email.split("@")[0].title()
                st.session_state.user_role = "privileged"
                st.session_state.show_login = False
            except Exception as e:
                st.error(f"Email login failed: {e}")
