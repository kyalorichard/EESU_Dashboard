# auth.py
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, auth
import urllib.parse
import requests
import hashlib

# ============================
# Safe Firebase Admin Init
# ============================
if "firebase_admin" in st.secrets:
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Firebase Admin init failed: {e}")
else:
    st.warning("⚠️ firebase_admin secrets missing")

# ============================
# Safe Pyrebase Init
# ============================
firebase_config = dict(st.secrets.get("firebase", {}))
firebase_config.setdefault("databaseURL", "https://dummy.firebaseio.com/")

try:
    firebase = pyrebase.initialize_app(firebase_config)
    firebase_auth = firebase.auth()
except Exception:
    firebase_auth = None

# ============================
# Config
# ============================
PRIVILEGED_DOMAINS = set(st.secrets.get("access", {}).get("privileged_domains", []))
GOOGLE_CLIENT_ID = st.secrets.get("oauth", {}).get("client_id", "")
GOOGLE_CLIENT_SECRET = st.secrets.get("oauth", {}).get("client_secret", "")
REDIRECT_URI = st.secrets.get("oauth", {}).get("redirect_uri", "")

# ============================
# Helpers
# ============================
def get_email_domain(email: str) -> str:
    return email.split("@")[-1].lower()

def avatar_initials(email: str) -> str:
    name = email.split("@")[0].replace(".", " ")
    parts = name.split()
    return (parts[0][0] + (parts[1][0] if len(parts) > 1 else "")).upper()

def get_avatar_color(email: str) -> str:
    colors = [
        "#1a73e8", "#ea4335", "#fbbc05", "#34a853",
        "#ff6d01", "#46bdc6", "#ab47bc", "#f06292"
    ]
    h = int(hashlib.md5(email.encode()).hexdigest(), 16)
    return colors[h % len(colors)]

# ============================
# Google OAuth
# ============================
def get_google_auth_url():
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

def exchange_code_for_token(code: str):
    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        },
    )
    resp.raise_for_status()
    return resp.json()

def handle_google_redirect():
    params = st.experimental_get_query_params()
    if "code" not in params:
        return

    try:
        token_data = exchange_code_for_token(params["code"][0])
        decoded = auth.verify_id_token(token_data["id_token"])
    except Exception:
        st.error("Google login failed")
        st.experimental_set_query_params()
        return

    email = decoded.get("email")
    st.session_state.user = {"provider": "google"}
    st.session_state.email = email
    st.session_state.name = decoded.get("name", email.split("@")[0].title())
    st.session_state.photo = decoded.get("picture")
    st.session_state.user_role = (
        "privileged" if get_email_domain(email) in PRIVILEGED_DOMAINS else "public"
    )

    st.experimental_set_query_params()
    st.experimental_rerun()

# ============================
# Top-right Auth UI
# ============================
def top_right_auth():
    handle_google_redirect()

    if "show_email_form" not in st.session_state:
        st.session_state.show_email_form = False

    # ---------- CSS ----------
    st.markdown("""
    <style>
    .auth-container { position: fixed; top: 0.75rem; right: 1.5rem; z-index: 9999; }
    .avatar {
        width: 36px; height: 36px; border-radius: 50%;
        color: white; font-weight: 600;
        display: flex; align-items: center; justify-content: center;
        cursor: pointer; transition: transform 0.2s ease;
    }
    .avatar:hover { transform: scale(1.1); }

    details summary { list-style: none; }
    details summary::-webkit-details-marker { display: none; }

    .dropdown-panel {
        margin-top: 0.4rem; background: white;
        border-radius: 10px; padding: 0.75rem;
        width: 240px; box-shadow: 0 6px 20px rgba(0,0,0,0.18);
        opacity: 0; transform: translateY(-10px);
        transition: all 0.25s ease-out; pointer-events: none;
    }
    details[open] .dropdown-panel {
        opacity: 1; transform: translateY(0); pointer-events: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="auth-container">', unsafe_allow_html=True)

    # ---------- LOGGED OUT ----------
    if "user" not in st.session_state:
        google_url = get_google_auth_url()

        st.markdown(f"""
        <details>
          <summary>
            <div class="avatar" style="background:#1a73e8">?</div>
          </summary>
          <div class="dropdown-panel">
            <a href="{google_url}">
                <button style="width:100%;margin-bottom:0.5rem;">🔵 Sign in with Google</button>
            </a>
            <hr>
            <button id="email-toggle" style="width:100%;">✉️ Sign in with Email</button>
            <div id="email-form" style="display:{'block' if st.session_state.show_email_form else 'none'};">
        """, unsafe_allow_html=True)

        if firebase_auth and st.session_state.show_email_form:
            with st.form("email_login"):
                email = st.text_input("Email", label_visibility="collapsed")
                password = st.text_input("Password", type="password", label_visibility="collapsed")
                if st.form_submit_button("Sign in"):
                    try:
                        firebase_auth.sign_in_with_email_and_password(email, password)
                        st.session_state.user = {"provider": "email"}
                        st.session_state.email = email
                        st.session_state.name = email.split("@")[0].replace(".", " ").title()
                        st.session_state.photo = None
                        st.session_state.user_role = (
                            "privileged" if get_email_domain(email) in PRIVILEGED_DOMAINS else "public"
                        )
                        st.experimental_rerun()
                    except Exception:
                        st.error("Invalid email or password")

        st.markdown("</div></div></details>", unsafe_allow_html=True)

        st.markdown("""
        <script>
        document.getElementById("email-toggle")?.addEventListener("click", () => {
            const f = document.getElementById("email-form");
            f.style.display = f.style.display === "none" ? "block" : "none";
        });
        </script>
        """, unsafe_allow_html=True)

    # ---------- LOGGED IN ----------
    else:
        email = st.session_state.email
        name = st.session_state.name
        role = st.session_state.user_role
        photo = st.session_state.photo

        if photo:
            avatar = f"<img src='{photo}' class='avatar'>"
        else:
            color = get_avatar_color(email)
            avatar = f"<div class='avatar' style='background:{color}'>{avatar_initials(email)}</div>"

        st.markdown(f"""
        <details>
          <summary>{avatar}</summary>
          <div class="dropdown-panel">
            <strong>{name}</strong><br>
            <small>{email}</small><br>
            <small>{role.capitalize()} access</small>
            <hr>
            <button id="logout-btn" style="width:100%;">Logout</button>
          </div>
        </details>
        """, unsafe_allow_html=True)

        st.markdown("""
        <script>
        document.getElementById("logout-btn")?.addEventListener("click", () => {
            window.location.href = window.location.pathname + "?logout=1";
        });
        </script>
        """, unsafe_allow_html=True)

        if "logout" in st.experimental_get_query_params():
            for k in list(st.session_state.keys()):
                st.session_state.pop(k)
            st.experimental_set_query_params()
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)
