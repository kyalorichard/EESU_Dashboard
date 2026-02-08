# auth.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials
import urllib.parse

# =================================================
# Firebase Admin (safe init)
# =================================================
if "firebase_admin" in st.secrets and not firebase_admin._apps:
    try:
        cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
        firebase_admin.initialize_app(cred)
    except Exception:
        pass

# =================================================
# Pyrebase (safe init)
# =================================================
firebase_auth = None
firebase_cfg = dict(st.secrets.get("firebase", {}))
if firebase_cfg:
    firebase_cfg.setdefault("databaseURL", "https://dummy.firebaseio.com/")
    try:
        firebase = pyrebase.initialize_app(firebase_cfg)
        firebase_auth = firebase.auth()
    except Exception:
        firebase_auth = None

# =================================================
# Config
# =================================================
PRIVILEGED_DOMAINS = set(
    st.secrets.get("access", {}).get("privileged_domains", [])
)

GOOGLE_CLIENT_ID = st.secrets.get("oauth", {}).get("client_id", "")
REDIRECT_URI = st.secrets.get("oauth", {}).get("redirect_uri", "")

# =================================================
# Helpers
# =================================================
def get_email_domain(email: str) -> str:
    return email.split("@")[-1].lower()

def avatar_initials(email: str) -> str:
    name = email.split("@")[0].replace(".", " ").split()
    return "".join(part[0].upper() for part in name[:2])

def is_privileged() -> bool:
    return st.session_state.get("user_role") == "privileged"

# =================================================
# Google OAuth URL
# =================================================
def get_google_auth_url() -> str:
    if not GOOGLE_CLIENT_ID or not REDIRECT_URI:
        return "#"

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "prompt": "select_account",
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)

# =================================================
# OAuth Redirect Handler (SAFE NO-OP)
# =================================================
def handle_google_redirect():
    params = st.experimental_get_query_params()

    # If your backend later injects ?email=...
    if "email" not in params:
        return

    email = params.get("email", [None])[0]
    name = params.get("name", [None])[0]

    if not email:
        return

    st.session_state.user = "google"
    st.session_state.email = email
    st.session_state.name = name or email.split("@")[0].title()
    st.session_state.user_role = (
        "privileged"
        if get_email_domain(email) in PRIVILEGED_DOMAINS
        else "public"
    )

    st.experimental_set_query_params()
    st.experimental_rerun()

# =================================================
# CSS – top-left, non-intrusive
# =================================================
def inject_auth_css():
    st.markdown("""
    <style>
    .block-container {
        padding-top: 4.2rem;
    }
    .auth-container {
        position: fixed;
        top: 0.8rem;
        left: 1.2rem;
        z-index: 9999;
    }
    .auth-panel {
        background: white;
        border-radius: 10px;
        padding: 0.8rem;
        width: 260px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    .avatar-btn button {
        width: 36px !important;
        height: 36px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        background: #1a73e8 !important;
        color: white !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =================================================
# Top-Left Avatar Auth UI (STREAMLIT SAFE)
# =================================================
def top_right_auth():
    handle_google_redirect()

    if "auth_open" not in st.session_state:
        st.session_state.auth_open = False

    st.markdown('<div class="auth-container">', unsafe_allow_html=True)

    # -------- Avatar button --------
    if "user" in st.session_state:
        avatar = avatar_initials(st.session_state.get("email", "U"))
    else:
        avatar = "?"

    with st.container():
        with st.container():
            if st.button(avatar, key="avatar", help="Account"):
                st.session_state.auth_open = not st.session_state.auth_open

    # -------- Dropdown panel --------
    if st.session_state.auth_open:
        st.markdown('<div class="auth-panel">', unsafe_allow_html=True)

        # ---------- LOGGED OUT ----------
        if "user" not in st.session_state:

            # ✅ WORKING Google login
            st.link_button(
                "🔵 Sign in with Google",
                get_google_auth_url(),
                use_container_width=True,
            )

            st.divider()

            # ✅ WORKING Email login
            if firebase_auth:
                with st.form("email_login", clear_on_submit=False):
                    email = st.text_input("Email")
                    password = st.text_input("Password", type="password")
                    submit = st.form_submit_button("Sign in with Email")

                    if submit:
                        try:
                            user = firebase_auth.sign_in_with_email_and_password(
                                email, password
                            )
                            st.session_state.user = user
                            st.session_state.email = email
                            st.session_state.user_role = (
                                "privileged"
                                if get_email_domain(email) in PRIVILEGED_DOMAINS
                                else "public"
                            )
                            st.session_state.auth_open = False
                            st.experimental_rerun()
                        except Exception:
                            st.error("Invalid email or password")

        # ---------- LOGGED IN ----------
        else:
            email = st.session_state.get("email")
            role = st.session_state.get("user_role", "public")
            name = st.session_state.get(
                "name",
                email.split("@")[0].replace(".", " ").title()
            )

            st.markdown(f"**{name}**")
            st.caption(email)
            st.caption(f"{role.capitalize()} access")
            st.divider()

            if st.button("Logout", use_container_width=True):
                for k in ["user", "email", "user_role", "auth_open", "name"]:
                    st.session_state.pop(k, None)
                st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
