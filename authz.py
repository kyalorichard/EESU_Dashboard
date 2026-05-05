# authz.py
"""
EU SEE Dashboard authorization helpers.

Firebase Auth handles identity in auth.py. This file reads the logged-in email from
st.session_state and compares it with [auth].admin_emails in .streamlit/secrets.toml.
"""

import streamlit as st


def get_current_email() -> str:
    email = (
        st.session_state.get("email")
        or st.session_state.get("user_email")
        or st.session_state.get("firebase_email")
        or ""
    )
    return str(email).lower().strip()


def get_admin_emails() -> list[str]:
    return [
        str(e).lower().strip()
        for e in st.secrets.get("auth", {}).get("admin_emails", [])
        if str(e).strip()
    ]


def get_power_user_emails() -> list[str]:
    return [
        str(e).lower().strip()
        for e in st.secrets.get("auth", {}).get("power_users", [])
        if str(e).strip()
    ]


def is_admin() -> bool:
    email = get_current_email()
    return bool(email and email in get_admin_emails())


def is_power_user() -> bool:
    email = get_current_email()
    return bool(email and email in get_power_user_emails())


def get_current_role() -> str:
    if is_admin():
        return "admin"
    if is_power_user():
        return "analyst"
    if get_current_email():
        return "viewer"
    return "guest"


def get_role() -> str:
    return get_current_role()


def has_permission(permission: str) -> bool:
    if is_admin():
        return True

    role_permissions = {
        "analyst": [
            "view_dashboard",
            "view_overview",
            "view_negative_alerts",
            "view_maps",
            "view_data_table",
            "view_country_counts",
            "download_data",
            "use_ai_copilot",
        ],
        "viewer": [
            "view_dashboard",
            "view_overview",
            "view_maps",
        ],
        "guest": ["view_dashboard", "view_overview"],
    }
    return permission in role_permissions.get(get_current_role(), [])


def apply_data_scope(df):
    """Optional role-based data scoping from secrets. Admins see all data."""
    if df is None or is_admin():
        return df

    try:
        scoped_df = df.copy()
        role = get_current_role()
        scope = st.secrets.get("data_scope", {}).get(role, {})

        regions = scope.get("regions", [])
        countries = scope.get("countries", [])
        years = scope.get("years", [])

        if regions and "region" in scoped_df.columns:
            scoped_df = scoped_df[scoped_df["region"].isin(regions)]
        if countries and "alert-country" in scoped_df.columns:
            scoped_df = scoped_df[scoped_df["alert-country"].isin(countries)]
        if years and "year" in scoped_df.columns:
            scoped_df = scoped_df[scoped_df["year"].isin(years)]
        return scoped_df
    except Exception:
        return df
