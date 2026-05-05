import streamlit as st
import pandas as pd
from datetime import datetime

try:
    from authz import (
        get_admin_emails,
        get_power_user_emails,
        get_power_users,
        get_current_email,
        get_current_role,
        get_role,
        has_permission,
        is_admin,
    )
except Exception:
    def get_current_email():
        return st.session_state.get("email", "").lower().strip()
    def get_admin_emails():
        return [e.lower().strip() for e in st.secrets.get("auth", {}).get("admin_emails", [])]
    def get_power_user_emails():
        return [e.lower().strip() for e in st.secrets.get("auth", {}).get("power_users", [])]
    def get_power_users():
        return get_power_user_emails()
    def is_admin():
        return get_current_email() in get_admin_emails()
    def get_current_role():
        if is_admin():
            return "admin"
        if get_current_email() in get_power_user_emails():
            return "analyst"
        if get_current_email():
            return "viewer"
        return "guest"
    def get_role():
        return get_current_role()
    def has_permission(permission):
        return is_admin()

PERMISSION_LABELS = {
    "view_dashboard": "View dashboard",
    "view_overview": "View Overview tab",
    "view_negative_alerts": "View Negative Alerts tab",
    "view_map": "View Visualization Map tab",
    "view_manual": "View User Manual tab",
    "view_country_counts": "View country counts",
    "use_ai_copilot": "Use AI Copilot",
    "download_data": "Download/export data",
    "manage_users": "Manage users",
}


def _admin_css():
    st.markdown(
        """
        <style>
        .admin-hero {
            background: radial-gradient(circle at 100% 0%, rgba(102,0,148,.10), transparent 30%), linear-gradient(135deg,#FFFFFF 0%,#F7ECFB 100%);
            border:1px solid rgba(102,0,148,.16);
            border-radius:20px;
            padding:18px 20px;
            box-shadow:0 14px 34px rgba(16,24,40,.08);
            margin-bottom:14px;
            font-family:Arial,sans-serif;
        }
        .admin-eyebrow {font-size:10px;font-weight:900;color:#660094;letter-spacing:.14em;text-transform:uppercase;margin-bottom:5px;}
        .admin-title {font-size:28px;font-weight:950;color:#23152F;line-height:1.1;}
        .admin-note {font-size:12px;color:#667085;margin-top:6px;line-height:1.45;max-width:900px;}
        .admin-card {
            background:#FFFFFF;border:1px solid #E6E8EF;border-radius:16px;padding:14px;
            box-shadow:0 8px 22px rgba(16,24,40,.055);font-family:Arial,sans-serif;margin-bottom:12px;
        }
        .admin-card-title {font-size:14px;font-weight:950;color:#23152F;margin-bottom:4px;}
        .admin-card-note {font-size:11px;color:#667085;line-height:1.4;margin-bottom:8px;}
        .admin-pill {display:inline-block;background:#F4EAF8;color:#660094;border:1px solid #E7D4F1;border-radius:999px;padding:5px 9px;font-size:10px;font-weight:900;margin:2px 4px 2px 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _toml_list(items):
    if not items:
        return "[]"
    quoted = ",\n  ".join([f'"{x}"' for x in items])
    return "[\n  " + quoted + "\n]"


def render_admin_page():
    _admin_css()

    if not is_admin():
        st.error("You do not have permission to access the Administrator Panel.")
        return

    st.markdown(
        """
        <div class="admin-hero">
            <div class="admin-eyebrow">EU SEE control tower</div>
            <div class="admin-title">🔐 Administrator Panel</div>
            <div class="admin-note">
                Admin access is controlled through <code>.streamlit/secrets.toml</code>. Firebase handles login identity; this panel reads the authenticated email and checks whether it is listed as an administrator.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_email = get_current_email()
    current_role = get_current_role().title()
    admin_emails = get_admin_emails()
    power_users = get_power_user_emails()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Current user", current_email or "Not detected")
    with c2:
        st.metric("Current role", current_role)
    with c3:
        st.metric("Configured admins", len(admin_emails))

    tabs = st.tabs(["👤 Access Lists", "🔐 Permissions", "🌍 Data Scope", "🧠 AI & Features", "📋 Setup Export"])

    with tabs[0]:
        st.markdown('<div class="admin-card"><div class="admin-card-title">Admin emails</div><div class="admin-card-note">These emails receive full administrator access after Firebase login.</div></div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Admin email": admin_emails}), use_container_width=True, hide_index=True)

        st.markdown('<div class="admin-card"><div class="admin-card-title">Power users / analysts</div><div class="admin-card-note">These users receive elevated analytical access without admin controls.</div></div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Power user email": power_users}), use_container_width=True, hide_index=True)

        st.info("To add or remove users, edit `.streamlit/secrets.toml`, then restart/redeploy the app.")

    with tabs[1]:
        st.markdown('<div class="admin-card"><div class="admin-card-title">Permission catalogue</div><div class="admin-card-note">Admins receive all permissions. Analysts and viewers receive limited permissions defined in authz.py.</div></div>', unsafe_allow_html=True)
        rows = []
        for key, label in PERMISSION_LABELS.items():
            rows.append({"Permission key": key, "Label": label, "Current user allowed": "Yes" if has_permission(key) else "No"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown('<div class="admin-card"><div class="admin-card-title">Optional data-scope restrictions</div><div class="admin-card-note">Use this when a partner should only see selected regions, countries, or years.</div></div>', unsafe_allow_html=True)
        st.code("[access_scope.\"partner@example.org\"]\nregions = [\"Africa\"]\ncountries = [\"Kenya\", \"Ethiopia\"]\nyears = [2024, 2025]", language="toml")

    with tabs[3]:
        st.markdown('<div class="admin-card"><div class="admin-card-title">Feature control model</div><div class="admin-card-note">For now, feature access is controlled by role permissions in authz.py. Firestore can be added later for live editing.</div></div>', unsafe_allow_html=True)
        feature_rows = [
            {"Feature": "Negative Alerts", "Permission": "view_negative_alerts"},
            {"Feature": "AI Copilot", "Permission": "use_ai_copilot"},
            {"Feature": "Downloads", "Permission": "download_data"},
            {"Feature": "Country counts", "Permission": "view_country_counts"},
        ]
        st.dataframe(pd.DataFrame(feature_rows), use_container_width=True, hide_index=True)

    with tabs[4]:
        st.markdown('<div class="admin-card"><div class="admin-card-title">Copy-ready secrets.toml block</div><div class="admin-card-note">Paste/edit this in Streamlit secrets. Use exact Firebase login emails.</div></div>', unsafe_allow_html=True)
        toml = f"[auth]\nadmin_emails = {_toml_list(admin_emails or ['your-admin-email@example.org'])}\n\npower_users = {_toml_list(power_users or ['analyst@example.org'])}\n"
        st.code(toml, language="toml")
        st.caption(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
