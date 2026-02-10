# auth.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials

# ---------------------------
# Firebase Admin Init
# ---------------------------
if "firebase_admin" in st.secrets and not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
    firebase_admin.initialize_app(cred)

# ---------------------------
# Pyrebase Init
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
    st.session_state.setdefault("_close_login_state", False)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("email", None)
    st.session_state.setdefault("name", "")
    st.session_state.setdefault("user_role", "")

# ---------------------------
# Logout
# ---------------------------
def logout_user():
    for k in ["user", "email", "name", "user_role", "email_input", "password_input"]:
        st.session_state.pop(k, None)
    st.session_state.show_login = False

# ---------------------------
# CSS (Animated Drawer + Buttons)
# ---------------------------
def inject_css():
    st.markdown("""
    <style>
    .login-overlay {
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,.45);
        z-index: 9998;
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.3s ease, visibility 0.3s ease;
    }
    .login-overlay.show {
        opacity: 1;
        visibility: visible;
    }
    .login-card {
        position: fixed;
        top:0; right:-400px;
        width:350px;
        max-width:95%;
        height:100%;
        background:#fff;
        padding:1.5rem;
        box-shadow:-10px 0 30px rgba(0,0,0,.3);
        z-index: 9999;
        display:flex;
        flex-direction:column;
        transition: right 0.3s ease;
    }
    .login-card.show {
        right:0;
    }
    .login-card input { padding:.5rem; margin-bottom:.5rem; width:100%; border-radius:6px; border:1px solid #ccc; }
    .login-card button { padding:.6rem; width:100%; border-radius:6px; border:none; font-weight:600; cursor:pointer; margin-bottom:.5rem; transition: background 0.2s; }
    .email-btn { background:#4caf50; color:white; }
    .email-btn:hover { background:#3b8c40; }
    .cancel-btn { background:#ccc; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# AUTH UI (Drawer)
# ---------------------------
def auth_ui():
    init_state()
    inject_css()

    # Logged in
    if st.session_state.user:
        st.sidebar.success(f"👋 {st.session_state['name']}")
        st.sidebar.button("Logout", on_click=logout_user)
        return

    # Logged out sidebar
    st.sidebar.markdown("## Account")
    if st.sidebar.button("🔐 Sign in"):
        st.session_state.show_login = True

    email = st.session_state.email_input
    password = st.session_state.password_input
    login_error = st.session_state.login_error

    overlay_class = "login-overlay show" if st.session_state.show_login else "login-overlay"
    card_class = "login-card show" if st.session_state.show_login else "login-card"

    # Login Drawer HTML
    st.markdown(f"""
    <div class="{overlay_class}" id="loginOverlay"></div>
    <div class="{card_class}" id="loginCard">
        <h3>Sign in</h3>

        <!-- Email Login -->
        <input type="text" id="email_input" placeholder="Email" value="{email}">
        <input type="password" id="password_input" placeholder="Password" value="{password}">
        <button class="email-btn" id="email_login_btn">Sign in with Email</button>

        <p style="color:red; text-align:center;">{login_error}</p>
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

    # Streamlit Email login fields
    email_input = st.text_input("Email", value=email, key="email_field")
    password_input = st.text_input("Password", value=password, type="password", key="password_field")
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

    # Close overlay if triggered
    if st.session_state.get("_close_login_state"):
        st.session_state.show_login = False
        st.session_state["_close_login_state"] = False
