import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, auth as admin_auth
from streamlit_cookies_manager import EncryptedCookieManager
import json

# ---------------- Firebase Initialization ----------------
if not firebase_admin._apps:
    if "firebase_admin" not in st.secrets:
        st.error("Missing firebase_admin in secrets.toml")
        st.stop()
    cred_dict = dict(st.secrets["firebase_admin"])
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

firebase_cfg = dict(st.secrets.get("firebase", {}))
firebase = pyrebase.initialize_app(firebase_cfg)
firebase_auth = firebase.auth()

PRIVILEGED_DOMAINS = set(d.lower() for d in st.secrets.get("access", {}).get("privileged_domains", []))

# ---------------- Cookie Management ----------------
def get_cookies():
    if "cookies" not in st.session_state:
        st.session_state.cookies = EncryptedCookieManager(
            prefix="myapp_auth_v2", 
            password=st.secrets.get("cookie", {}).get("cookie_password")
        )
    cookies = st.session_state.cookies
    if not cookies.ready():
        st.stop() 
    return cookies

# ---------------- Session Lifecycle ----------------
def init_session():
    defaults = {
        "user": False, "email": None, "name": None, 
        "role": None, "email_verified": False, "restored": False,
        "show_profile": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def restore_session():
    cookies = get_cookies()
    if not st.session_state.restored:
        if "email" in cookies:
            st.session_state.user = True
            st.session_state.email = cookies.get("email")
            st.session_state.name = cookies.get("name")
            st.session_state.role = cookies.get("role")
            st.session_state.email_verified = cookies.get("email_verified") == "True"
        st.session_state.restored = True

def logout():
    cookies = get_cookies()
    for key in ["email", "name", "role", "email_verified"]:
        if key in cookies:
            del cookies[key]
    cookies.save()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ---------------- Logic Handlers ----------------
def parse_error(e):
    try:
        return json.loads(e.args[1])['error']['message']
    except:
        return str(e)

def handle_login(email, password):
    domain = email.split("@")[-1].lower() if "@" in email else ""
    if domain not in PRIVILEGED_DOMAINS:
        st.sidebar.error("Access restricted to approved domains.")
        return

    try:
        user = firebase_auth.sign_in_with_email_and_password(email, password)
        info = firebase_auth.get_account_info(user["idToken"])
        user_data = info["users"][0]
        
        verified = user_data.get("emailVerified", False)
        # Use Firebase Display Name if set, otherwise fallback to email prefix
        display_name = user_data.get("displayName") or email.split("@")[0].title()
        
        st.session_state.update({
            "user": True,
            "email": email,
            "name": display_name,
            "email_verified": verified,
            "role": "privileged" if verified else "restricted"
        })
        
        cookies = get_cookies()
        cookies["email"] = email
        cookies["name"] = display_name
        cookies["role"] = st.session_state.role
        cookies["email_verified"] = str(verified)
        cookies.save()
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Login failed: {parse_error(e)}")

# ---------------- Profile Editor ----------------
def profile_editor():
    """Provides UI for updating user details."""
    st.markdown("---")
    st.subheader("Edit Profile")
    
    with st.form("profile_form"):
        new_name = st.text_input("Display Name", value=st.session_state.name)
        save = st.form_submit_button("Update Name", use_container_width=True)
        
        if save:
            try:
                # Use Admin SDK to update the user record
                user_record = admin_auth.get_user_by_email(st.session_state.email)
                admin_auth.update_user(user_record.uid, display_name=new_name)
                
                # Update local state
                st.session_state.name = new_name
                cookies = get_cookies()
                cookies["name"] = new_name
                cookies.save()
                st.success("Profile updated!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("Send Password Reset Email", use_container_width=True):
        try:
            firebase_auth.send_password_reset_email(st.session_state.email)
            st.info("Check your inbox for the reset link.")
        except Exception as e:
            st.error(parse_error(e))

# ---------------- Sidebar UI ----------------
def auth_ui():
    init_session()
    restore_session()
    
    sb = st.sidebar
    sb.title("🔐 Account Access")

    if st.session_state.user:
        sb.success(f"Hello, **{st.session_state.name}**")
        
        # Profile Toggle
        if sb.checkbox("User Settings"):
            profile_editor()
            
        if sb.button("Log Out", use_container_width=True, type="secondary"):
            logout()
        return

    mode = sb.radio("Action", ["Login", "Register"], horizontal=True)
    
    with sb.form("auth_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        if mode == "Login":
            if st.form_submit_button("Sign In", use_container_width=True):
                handle_login(email, password)
        else:
            if st.form_submit_button("Create Account", use_container_width=True):
                try:
                    user = firebase_auth.create_user_with_email_and_password(email, password)
                    firebase_auth.send_email_verification(user['idToken'])
                    st.success("Verification link sent! Check your email.")
                except Exception as e:
                    st.error(parse_error(e))

    if mode == "Login":
        if sb.button("Forgot Password?"):
            if email:
                try:
                    firebase_auth.send_password_reset_email(email)
                    sb.info("Reset link sent.")
                except Exception as e:
                    sb.error(parse_error(e))
            else:
                sb.warning("Enter email above.")