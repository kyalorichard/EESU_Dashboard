# auth.py
import streamlit as st
import hashlib

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
PRIVILEGED_DOMAINS = {"icarda.org", "gmail.com", "government.go.ke"}

# firebase_auth and google helpers must already exist
# Example:
# from firebase import firebase_auth
# from google_oauth import get_google_auth_url, handle_google_redirect

firebase_auth = None  # set this if using Firebase email auth


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def get_email_domain(email: str) -> str:
    return email.split("@")[-1].lower()


def is_privileged() -> bool:
    return st.session_state.get("user_role") == "privileged"


def avatar_initials(email: str) -> str:
    name = email.split("@")[0]
    parts = name.replace(".", " ").split()
    return "".join(p[0].upper() for p in parts[:2])


def avatar_color(email: str) -> str:
    """Deterministic Gmail-style color"""
    colors = [
        "#1a73e8", "#d93025", "#188038",
        "#f9ab00", "#9334e6", "#c5221f"
    ]
    h = int(hashlib.md5(email.encode()).hexdigest(), 16)
    return colors[h % len(colors)]


# -------------------------------------------------
# GOOGLE REDIRECT HANDLER (SAFE NO-OP IF UNUSED)
# -------------------------------------------------
def handle_google_redirect():
    """
    Expected to populate:
      st.session_state.user
      st.session_state.email
      st.session_state.name
      st.session_state.photo
    """
    pass


# -------------------------------------------------
# GLOBAL AUTH CSS
# -------------------------------------------------
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
        width: 38px;
        height: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        color: white;
        cursor: pointer;
        transition: transform 0.2s ease;
    }
    .avatar:hover { transform: scale(1.08); }

    details summary { list-style: none; }
    details summary::-webkit-details-marker { display: none; }

    .dropdown {
        margin-top: 0.4rem;
        background: white;
        border-radius: 12px;
        padding: 0.8rem;
        width: 250px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        opacity: 0;
        transform: translateY(-10px);
        transition: all 0.25s ease;
        pointer-events: none;
    }

    details[open] .dropdown {
        opacity: 1;
        transform: translateY(0);
        pointer-events: auto;
    }

    .dropdown button {
        width: 100%;
        margin-top: 0.4rem;
    }
    </style>
    """, unsafe_allow_html=True)


# -------------------------------------------------
# MAIN AUTH UI (TOP RIGHT)
# -------------------------------------------------
def top_right_auth():
    handle_google_redirect()

    st.markdown('<div class="auth-container">', unsafe_allow_html=True)

    # ----------------------------
    # LOGGED OUT
    # ----------------------------
    if "user" not in st.session_state:
        google_url = "#"

        st.markdown(f"""
        <details>
          <summary>
            <div class="avatar" style="background:#9aa0a6">?</div>
          </summary>
          <div class="dropdown">
            <a href="{google_url}">
              <button>🔵 Sign in with Google</button>
            </a>
            <hr>
            <button onclick="document.getElementById('email-form').style.display='block'">
              ✉️ Sign in with Email
            </button>
            <div id="email-form" style="display:none; margin-top:0.5rem;">
        """, unsafe_allow_html=True)

        if firebase_auth:
            with st.form("email_login"):
                email = st.text_input("Email", label_visibility="collapsed")
                password = st.text_input("Password", type="password", label_visibility="collapsed")
                if st.form_submit_button("Sign in"):
                    user = firebase_auth.sign_in_with_email_and_password(email, password)
                    st.session_state.user = user
                    st.session_state.email = email
                    st.session_state.photo = None
                    st.session_state.name = email.split("@")[0].replace(".", " ").title()
                    st.session_state.user_role = (
                        "privileged"
                        if get_email_domain(email) in PRIVILEGED_DOMAINS
                        else "public"
                    )
                    st.experimental_rerun()

        st.markdown("</div></div></details>", unsafe_allow_html=True)

    # ----------------------------
    # LOGGED IN
    # ----------------------------
    else:
        email = st.session_state["email"]
        name = st.session_state.get("name", "User")
        photo = st.session_state.get("photo")
        role = st.session_state.get("user_role", "public")

        if photo:
            avatar = f"<img src='{photo}' class='avatar'>"
        else:
            color = avatar_color(email)
            initials = avatar_initials(email)
            avatar = f"<div class='avatar' style='background:{color}'>{initials}</div>"

        st.markdown(f"""
        <details>
          <summary>{avatar}</summary>
          <div class="dropdown">
            <strong>{name}</strong><br>
            <small>{email}</small><br>
            <small>{role.capitalize()} access</small>
            <hr>
            <form method="get">
              <button name="logout" value="1">Logout</button>
            </form>
          </div>
        </details>
        """, unsafe_allow_html=True)

        # Logout
        params = st.experimental_get_query_params()
        if "logout" in params:
            st.session_state.clear()
            st.experimental_set_query_params()
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)
