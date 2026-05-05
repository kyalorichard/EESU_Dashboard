import streamlit as st

# Firebase Auth remains responsible for login and identity.
# This file reads admin/power-user email lists from .streamlit/secrets.toml.

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


def is_logged_in():
    return bool(get_current_email())


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


def is_privileged():
    """Compatibility wrapper used by the existing dashboard."""
    return get_current_role() in ["admin", "analyst"]


def has_permission(permission_name):
    role = get_current_role()
    permissions = DEFAULT_ROLE_PERMISSIONS.get(role, [])
    if permissions == "ALL":
        return True
    return permission_name in permissions


def apply_basic_data_scope(df):
    """
    Optional static data scoping from secrets.toml.

    Example:
    [access_scope."viewer@example.org"]
    regions = ["Africa"]
    countries = ["Kenya", "Ethiopia"]
    years = [2024, 2025]
    """
    email = get_current_email()
    if not email or df is None or df.empty:
        return df

    try:
        scope = st.secrets.get("access_scope", {}).get(email, {})
    except Exception:
        scope = {}

    scoped = df.copy()
    regions = list(scope.get("regions", [])) if scope else []
    countries = list(scope.get("countries", [])) if scope else []
    years = list(scope.get("years", [])) if scope else []

    if regions and "region" in scoped.columns:
        scoped = scoped[scoped["region"].isin(regions)]
    if countries and "alert-country" in scoped.columns:
        scoped = scoped[scoped["alert-country"].isin(countries)]
    if years and "year" in scoped.columns:
        scoped = scoped[scoped["year"].isin(years)]
    return scoped


def render_access_badge():
    email = get_current_email() or "unknown user"
    role = get_current_role().title()
    st.sidebar.markdown(
        f"""
        <div style="background:#FFFFFF;border:1px solid #E6E8EF;border-radius:14px;padding:10px 11px;box-shadow:0 6px 16px rgba(16,24,40,.05);font-family:Arial,sans-serif;">
            <div style="font-size:9px;font-weight:900;color:#660094;letter-spacing:.12em;text-transform:uppercase;">Access role</div>
            <div style="font-size:12px;font-weight:900;color:#23152F;margin-top:3px;">{role}</div>
            <div style="font-size:10px;color:#667085;margin-top:3px;word-break:break-word;">{email}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
