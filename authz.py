import streamlit as st

# Firebase/Auth remains responsible for login and identity.
# This file controls dashboard access levels using .streamlit/secrets.toml.

DEFAULT_ROLE_PERMISSIONS = {
    "admin": "ALL",
    "analyst": [
        "view_dashboard",
        "view_overview",
        "view_negative_alerts",
        "view_map",
        "view_manual",
        "view_country_counts",
        "use_ai_copilot",
        "download_data",
        "view_admin_summary",
    ],
    "viewer": [
        "view_dashboard",
        "view_overview",
        "view_map",
        "view_manual",
    ],
    "guest": [
        "view_dashboard",
        "view_overview",
        "view_map",
    ],
}


def _clean_email(value):
    return str(value or "").strip().lower()


def _get_secret_list(section, key):
    try:
        value = st.secrets.get(section, {}).get(key, [])
    except Exception:
        value = []
    if isinstance(value, str):
        value = [value]
    return [_clean_email(v) for v in value if _clean_email(v)]


def get_current_email():
    """Return current Firebase-authenticated email from Streamlit session state."""
    for key in ["email", "user_email", "firebase_email", "auth_email"]:
        email = _clean_email(st.session_state.get(key, ""))
        if email:
            return email

    user = st.session_state.get("user") or st.session_state.get("firebase_user") or {}
    if isinstance(user, dict):
        for key in ["email", "user_email"]:
            email = _clean_email(user.get(key, ""))
            if email:
                return email
    return ""


def get_admin_emails():
    return _get_secret_list("auth", "admin_emails")


def get_power_user_emails():
    return _get_secret_list("auth", "power_users")


def get_power_users():
    return get_power_user_emails()


def is_logged_in():
    return bool(get_current_email())


def is_authenticated_user():
    return is_logged_in()


def is_admin():
    return get_current_email() in get_admin_emails()


def is_power_user():
    return get_current_email() in get_power_user_emails()


def get_current_role():
    if is_admin():
        return "admin"
    if is_power_user():
        return "analyst"
    if is_logged_in():
        return "viewer"
    return "guest"


def get_role():
    return get_current_role()


def is_privileged():
    return get_current_role() in ["admin", "analyst"]


def has_permission(permission_name):
    role = get_current_role()
    permissions = DEFAULT_ROLE_PERMISSIONS.get(role, [])
    if permissions == "ALL":
        return True
    return permission_name in permissions


def get_allowed_scope_for_current_user():
    email = get_current_email()
    if not email:
        return {}
    try:
        return dict(st.secrets.get("access_scope", {}).get(email, {}))
    except Exception:
        return {}


def apply_data_scope(df):
    """Filter dataframe by optional user-level scope. Admins are unrestricted."""
    if df is None or getattr(df, "empty", True) or is_admin():
        return df

    scope = get_allowed_scope_for_current_user()
    if not scope:
        return df

    scoped = df.copy()
    regions = scope.get("regions", []) or []
    countries = scope.get("countries", []) or []
    years = scope.get("years", []) or []

    if regions and "region" in scoped.columns:
        scoped = scoped[scoped["region"].isin(regions)]
    if countries and "alert-country" in scoped.columns:
        scoped = scoped[scoped["alert-country"].isin(countries)]
    if years and "year" in scoped.columns:
        scoped = scoped[scoped["year"].isin(years)]

    return scoped


def render_access_badge():
    role = get_current_role().title()
    email = get_current_email() or "Not signed in"
    st.sidebar.caption(f"Access: **{role}**")
    st.sidebar.caption(email)
