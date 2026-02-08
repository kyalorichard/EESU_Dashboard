# auth.py
import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
import pyrebase
import urllib.parse

# ----------------------------
# Firebase initialization
# ----------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
    firebase_admin.initialize_app(cred)

firebase = pyrebase.initialize_app(dict(st.secrets["firebase"]))
firebase_auth = firebase.auth()

# ----------------------------
# Config
# ----------------------------
PRIVILEGED_DOMAINS = set(st.secrets["access"]["privileged_domains"])

# ----------------------------
# Helpers
# ----------------------------
def get_email_domain(email: str) -> str:
    return email.split("@")[-1].lower()

def get_user_role_from_domain():
    if "user" not in st.session_state:
        return "public"

    decoded = auth.verify_id_token(st.session_state.user["idToken"])
    email = decoded.get("email", "")
    domain = get_email_domain(email)

    return "privileged" if domain in PRIVILEGED_DOMAINS else "public"

def avatar_initials(email: str) -> str:
    name = email.split("@")[0]
    parts = name.replace(".", " ").split()
    return (parts[0][0] + (parts[1][0] if len(parts) > 1 else "")).upper()

def is_privileged():
    return st.session_state.get("user_role") == "privileged"

# ----------------------------
# Google OAuth
# ----------------------------
def google_oauth_url():
    params = {
        "client_id": st.secrets["firebase"]["appId"],
        "redirect_uri": st.secrets["oauth"]["redirect_uri"],
        "response_type": "token",
        "scope": "email profile",
        "provider": "google.com",
    }
    return (
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithIdp"
        f"?{urllib.parse.urlencode(params)}"
    )

def handle_google_redirect():
    params = st.experimental_get_query_params()

    if "id_token" in params:
        id_token = params["id_token"][0]
        decoded = auth.verify_id_token(id_token)

        st.session_state.user = {"idToken": id_token}
        st.session_state.email = decoded.get("email")
        st.session_state.photo = decoded.get("picture")
        st.session_state.user_role = get_user_role_from_domain()

        st.experimental_set_query_params()
        st.rerun()

# ----------------------------
# UI
# ----------------------------
def inject_auth_css():
    st.markdown(
        """
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
        """,
        unsafe_allow_html=True
    )

def top_right_auth():
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)

    # Public user
    if "user" not in st.session_state:
        st.markdown(
            f"""
            <details>
              <summary><div class="avatar">?</div></summary>
              <div class="dropdown-panel">
                <a href="{google_oauth_url()}">
                    <button style="width:100%">🔵 Sign in with Google</button>
                </a>
                <hr>
            """,
            unsafe_allow_html=True
        )

        with st.form("email_login"):
            email = st.text_input("Email", label_visibility="collapsed")
            password = st.text_input("Password", type="password", label_visibility="collapsed")
            if st.form_submit_button("Sign in"):
                user = firebase_auth.sign_in_with_email_and_password(email, password)
                st.session_state.user = user
                st.session_state.email = email
                st.session_state.user_role = get_user_role_from_domain()
                st.rerun()

        st.markdown("</div></details>", unsafe_allow_html=True)

    # Logged-in user
    else:
        email = st.session_state.get("email")
        role = st.session_state.get("user_role")
        photo = st.session_state.get("photo")

        avatar = (
            f"<img src='{photo}' class='avatar'>"
            if photo else f"<div class='avatar'>{avatar_initials(email)}</div>"
        )

        st.markdown(
            f"""
            <details>
              <summary>{avatar}</summary>
              <div class="dropdown-panel">
                <strong>{email}</strong><br>
                <small>{role.capitalize()} access</small>
            """,
            unsafe_allow_html=True
        )

        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

        st.markdown("</div></details>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
