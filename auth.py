import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
import urllib.parse
import requests
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
import streamlit.components.v1 as components

# ---------------------------
# Firebase Admin Init
# ---------------------------
if "firebase_admin" in st.secrets and not firebase_admin._apps:
    try:
        cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.warning(f"Firebase Admin init failed: {e}")

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
    st.warning("⚠️ OAuth credentials missing. Google login disabled.")

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

        idinfo = id_token.verify_oauth2_token(tokens["id_token"], grequests.Request(), GOOGLE_CLIENT_ID)
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
        st.experimental_rerun()

    except Exception:
        st.error("Google login failed")

# ---------------------------
# Logout
# ---------------------------
def logout_user():
    for k in ["user", "email", "name", "user_role"]:
        st.session_state.pop(k, None)
    st.session_state.show_login = False
    st.experimental_rerun()

# ---------------------------
# AUTH UI (Sliding Drawer)
# ---------------------------
def auth_ui():
    init_state()
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

    # Use Streamlit components for drawer + overlay with JS handling
    login_url = get_google_auth_url()
    email = st.session_state.email_input
    password = st.session_state.password_input
    login_error = st.session_state.login_error

    drawer_html = f"""
    <div id="overlay" style="position:fixed; inset:0; background: rgba(0,0,0,0.45); z-index:9998;"></div>
    <div id="drawer" style="position:fixed; top:0; right:0; width:350px; height:100%; background:#fff; z-index:9999; padding:1.5rem; box-shadow:-10px 0 30px rgba(0,0,0,.3); display:flex; flex-direction:column;">
        <h3>Sign in</h3>

        {"<a href='" + login_url + "' style='text-decoration:none'><button style='background:#1a73e8;color:white;padding:.6rem;width:100%;border-radius:6px;margin-bottom:.5rem;'>🔵 Sign in with Google</button></a>" if oauth_enabled else "<p>Google login disabled</p>"}

        <hr>

        <input id="email_input" placeholder="Email" value="{email}" style="margin-bottom:.5rem;padding:.5rem;border-radius:6px;border:1px solid #ccc;width:100%;">
        <input id="password_input" type="password" placeholder="Password" value="{password}" style="margin-bottom:.5rem;padding:.5rem;border-radius:6px;border:1px solid #ccc;width:100%;">

        <button onclick="window.parent.postMessage({{func:'closeLogin'}}, '*')" style="margin-top:.3rem;padding:.6rem;width:100%;border-radius:6px;background:#ccc;">Cancel</button>

        <p style='color:red;text-align:center;'>{login_error}</p>
        <button onclick="window.parent.postMessage({{func:'submitEmailLogin', email: document.getElementById('email_input').value, password: document.getElementById('password_input').value}}, '*')" style="margin-top:.3rem;padding:.6rem;width:100%;border-radius:6px;background:#666;color:white;">Sign in with Email</button>
    </div>

    <script>
    document.getElementById('overlay').onclick = function() {{
        window.parent.postMessage({{func:'closeLogin'}}, '*');
    }};
    </script>
    """

    components.html(drawer_html, height=700)

    # --- Handle postMessage from JS ---
    msg = st.experimental_get_query_params().get("msg")
    if msg == ["closeLogin"]:
        st.session_state.show_login = False
        st.experimental_rerun()
    elif msg and msg[0].startswith("submitEmailLogin"):
        import json
        data = json.loads(msg[0].replace("submitEmailLogin", ""))
        email_input = data["email"]
        password_input = data["password"]
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
                st.experimental_rerun()
            except Exception as e:
                st.session_state.login_error = f"Login failed: {e}"
