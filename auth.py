import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, auth as admin_auth
import requests
import json

# -----------------------------
# FIREBASE CONFIG
# -----------------------------
firebase_config = {
    "apiKey": st.secrets["firebase"]["apiKey"],
    "authDomain": st.secrets["firebase"]["authDomain"],
    "projectId": st.secrets["firebase"]["projectId"],
    "storageBucket": st.secrets["firebase"]["storageBucket"],
    "messagingSenderId": st.secrets["firebase"]["messagingSenderId"],
    "appId": st.secrets["firebase"]["appId"],
    "databaseURL": ""
}

# Initialize Pyrebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# Initialize Firebase Admin (only once)
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
    firebase_admin.initialize_app(cred)

# -----------------------------
# SETTINGS
# -----------------------------
ALLOWED_DOMAIN = "yourdomain.com"  # change this

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "user_role" not in st.session_state:
    st.session_state.user_role = "public"


# -----------------------------
# REGISTER FUNCTION
# -----------------------------
def register_user(email, password):
    try:
        if not email.endswith(f"@{ALLOWED_DOMAIN}"):
            st.error(f"Registration restricted to @{ALLOWED_DOMAIN} emails.")
            return

        user = auth.create_user_with_email_and_password(email, password)

        # Send verification email
        auth.send_email_verification(user["idToken"])

        st.success("Account created successfully.")
        st.info("Please verify your email before logging in.")

    except Exception as e:
        error_json = json.loads(e.args[1])
        error_code = error_json["error"]["message"]

        if error_code == "EMAIL_EXISTS":
            st.error("This email is already registered.")
        elif error_code == "WEAK_PASSWORD":
            st.error("Password must be at least 6 characters.")
        else:
            st.error("Registration failed. Please try again.")


# -----------------------------
# LOGIN FUNCTION
# -----------------------------
def login_user(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)

        # Refresh user to check email verification
        user_info = auth.get_account_info(user["idToken"])
        email_verified = user_info["users"][0]["emailVerified"]

        if not email_verified:
            st.warning("Please verify your email before logging in.")
            return

        if not email.endswith(f"@{ALLOWED_DOMAIN}"):
            st.error("Access restricted to approved domain users.")
            return

        st.session_state.user = user
        st.session_state.user_role = "privileged"
        st.success("Login successful.")
        st.rerun()

    except Exception as e:
        try:
            error_json = json.loads(e.args[1])
            error_code = error_json["error"]["message"]

            if error_code == "INVALID_LOGIN_CREDENTIALS":
                st.error("Incorrect email or password.")
            elif error_code == "EMAIL_NOT_FOUND":
                st.error("No account found with this email.")
            elif error_code == "INVALID_PASSWORD":
                st.error("Incorrect password.")
            else:
                st.error("Login failed. Please try again.")

        except:
            st.error("Login failed. Please try again.")


# -----------------------------
# LOGOUT FUNCTION
# -----------------------------
def logout_user():
    st.session_state.user = None
    st.session_state.user_role = "public"
    st.success("Logged out successfully.")
    st.rerun()


# -----------------------------
# AUTH UI
# -----------------------------
def show_auth():

    if st.session_state.user is None:

        tab1, tab2 = st.tabs(["Login", "Register"])

        # -------- LOGIN --------
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")

                if submitted:
                    login_user(email, password)

        # -------- REGISTER --------
        with tab2:
            with st.form("register_form"):
                email = st.text_input("Email", key="reg_email")
                password = st.text_input("Password", type="password", key="reg_pass")
                submitted = st.form_submit_button("Register")

                if submitted:
                    register_user(email, password)

    else:
        st.sidebar.success(f"Logged in as {st.session_state.user['email']}")
        if st.sidebar.button("Logout"):
            logout_user()
