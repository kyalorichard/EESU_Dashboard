from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st

try:
    from authz import (
        ALL_PERMISSION_KEYS,
        DEFAULT_FEATURE_FLAGS,
        DEFAULT_ROLE_PERMISSIONS,
        delete_managed_user,
        get_admin_emails,
        get_current_email,
        get_current_role,
        get_global_feature_flag,
        get_managed_user,
        get_power_user_emails,
        has_permission,
        is_admin,
        load_access_config,
        update_feature_flags,
        upsert_managed_user,
    )
except Exception:
    ALL_PERMISSION_KEYS = ["view_dashboard", "view_overview", "view_negative_alerts", "view_map", "view_manual", "view_country_counts", "use_ai_copilot", "download_data"]
    DEFAULT_FEATURE_FLAGS = {}
    DEFAULT_ROLE_PERMISSIONS = {"viewer": ["view_dashboard", "view_overview"]}
    def get_current_email(): return st.session_state.get("email", "").lower().strip()
    def get_admin_emails(): return [e.lower().strip() for e in st.secrets.get("auth", {}).get("admin_emails", [])]
    def get_power_user_emails(): return [e.lower().strip() for e in st.secrets.get("auth", {}).get("power_users", [])]
    def is_admin(): return get_current_email() in get_admin_emails()
    def get_current_role(): return "admin" if is_admin() else "viewer"
    def has_permission(permission): return is_admin()
    def load_access_config(): return {"users": {}, "feature_flags": {}, "audit_log": []}
    def get_managed_user(email=None): return {}
    def get_global_feature_flag(flag): return True
    def upsert_managed_user(*args, **kwargs): return True
    def delete_managed_user(*args, **kwargs): return True
    def update_feature_flags(*args, **kwargs): return True

PERMISSION_LABELS = {
    "view_dashboard": "View dashboard",
    "view_overview": "View Overview tab",
    "view_negative_alerts": "View Negative Alerts tab",
    "view_map": "View Visualization Map tab",
    "view_manual": "View User Manual / documentation",
    "view_country_counts": "View country counts and detailed coverage KPIs",
    "use_ai_copilot": "Use AI Copilot / chatbot",
    "download_data": "Download/export data",
    "view_admin_summary": "View admin-facing summary cards",
}

FEATURE_LABELS = {
    "dashboard_enabled": "Dashboard access enabled",
    "overview_enabled": "Overview tab enabled",
    "negative_alerts_enabled": "Negative Alerts tab enabled",
    "map_enabled": "Visualization Map tab enabled",
    "manual_enabled": "User Manual tab enabled",
    "ai_copilot_enabled": "AI Copilot enabled",
    "downloads_enabled": "Data exports enabled",
    "country_counts_enabled": "Country-count KPIs enabled",
}

ROLE_LABELS = {
    "analyst": "Analyst / power user",
    "viewer": "Standard viewer",
    "restricted": "Restricted viewer",
}


def _admin_css():
    st.markdown(
        """
        <style>
        .eusee-admin-nav-card {
            margin: 12px 0 10px 0;
            padding: 13px;
            border-radius: 17px;
            background: radial-gradient(circle at 100% 0%, rgba(102,0,148,.15), transparent 34%), linear-gradient(135deg,#FFFFFF 0%,#F7ECFB 100%);
            border: 1px solid rgba(102,0,148,.16);
            box-shadow: 0 12px 28px rgba(16,24,40,.08);
            font-family: Arial, sans-serif;
        }
        .eusee-admin-nav-eyebrow {font-size:9px;font-weight:950;color:#660094;letter-spacing:.13em;text-transform:uppercase;margin-bottom:4px;}
        .eusee-admin-nav-title {font-size:14px;font-weight:950;color:#23152F;line-height:1.15;}
        .eusee-admin-nav-note {font-size:10.5px;color:#667085;line-height:1.35;margin-top:5px;}
        .eusee-admin-nav-user {margin-top:8px;padding:6px 9px;border-radius:999px;background:#EFFBFE;color:#008CAA;border:1px solid rgba(0,140,170,.18);font-size:10px;font-weight:900;width:fit-content;}
        div[data-testid="stSidebar"] .stRadio > div {gap: 6px !important;}
        div[data-testid="stSidebar"] .stRadio label {font-size: 12px !important;font-weight: 900 !important;color:#344054 !important;}
        .admin-hero {
            background: radial-gradient(circle at 100% 0%, rgba(102,0,148,.10), transparent 30%), linear-gradient(135deg,#FFFFFF 0%,#F7ECFB 100%);
            border:1px solid rgba(102,0,148,.16); border-radius:20px; padding:18px 20px;
            box-shadow:0 14px 34px rgba(16,24,40,.08); margin-bottom:14px; font-family:Arial,sans-serif;
        }
        .admin-eyebrow {font-size:10px;font-weight:900;color:#660094;letter-spacing:.14em;text-transform:uppercase;margin-bottom:5px;}
        .admin-title {font-size:28px;font-weight:950;color:#23152F;line-height:1.1;}
        .admin-note {font-size:12px;color:#667085;margin-top:6px;line-height:1.45;max-width:950px;}
        .admin-card {background:#FFFFFF;border:1px solid #E6E8EF;border-radius:16px;padding:14px;box-shadow:0 8px 22px rgba(16,24,40,.055);font-family:Arial,sans-serif;margin-bottom:12px;}
        .admin-card-title {font-size:14px;font-weight:950;color:#23152F;margin-bottom:4px;}
        .admin-card-note {font-size:11px;color:#667085;line-height:1.4;margin-bottom:8px;}
        .admin-pill {display:inline-block;background:#F4EAF8;color:#660094;border:1px solid #E7D4F1;border-radius:999px;padding:5px 9px;font-size:10px;font-weight:900;margin:2px 4px 2px 0;}
        .admin-warning-card {background:#FFFCED;border:1px solid #F8E9A1;border-radius:14px;padding:11px 13px;color:#7A5A00;font-size:11px;font-weight:800;line-height:1.4;margin:6px 0 12px 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_admin_sidebar_navigation():
    """Premium sidebar navigation card. Returns either 'Dashboard' or 'Admin'."""
    _admin_css()
    email = get_current_email() or "admin"
    role = get_current_role().title()
    st.sidebar.markdown(
        f"""
        <div class="eusee-admin-nav-card">
            <div class="eusee-admin-nav-eyebrow">Admin control tower</div>
            <div class="eusee-admin-nav-title">🔐 Dashboard access manager</div>
            <div class="eusee-admin-nav-note">Control what other authenticated users can see across tabs, KPIs, AI tools, downloads, and data scope.</div>
            <div class="eusee-admin-nav-user">{role} · {email}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.sidebar.radio(
        "Workspace",
        ["Dashboard", "Admin"],
        index=0,
        horizontal=True,
        key="eusee_admin_workspace_nav",
        label_visibility="collapsed",
    )


def _permissions_for_role(role: str) -> Dict[str, bool]:
    perms = DEFAULT_ROLE_PERMISSIONS.get(role, [])
    if perms == "ALL":
        return {p: True for p in ALL_PERMISSION_KEYS}
    return {p: p in perms for p in ALL_PERMISSION_KEYS}


def _parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def _toml_list(items):
    if not items:
        return "[]"
    quoted = ",\n  ".join([f'"{x}"' for x in items])
    return "[\n  " + quoted + "\n]"


def _users_table(config):
    rows = []
    for email, profile in sorted(config.get("users", {}).items()):
        perms = profile.get("permissions", {})
        scope = profile.get("data_scope", {})
        rows.append({
            "Email": email,
            "Role": profile.get("role", "viewer"),
            "Active": profile.get("active", True),
            "Allowed features": sum(1 for v in perms.values() if v),
            "Regions": ", ".join(scope.get("regions", []) or []),
            "Countries": ", ".join(scope.get("countries", []) or []),
            "Years": ", ".join(map(str, scope.get("years", []) or [])),
        })
    return pd.DataFrame(rows)


def render_admin_page():
    _admin_css()

    if not is_admin():
        st.error("You do not have permission to access the Administrator Panel.")
        return

    config = load_access_config()
    current_email = get_current_email()
    current_role = get_current_role().title()
    admin_emails = get_admin_emails()
    power_users = get_power_user_emails()

    st.markdown(
        """
        <div class="admin-hero">
            <div class="admin-eyebrow">EU SEE control tower</div>
            <div class="admin-title">🔐 Administrator Panel</div>
            <div class="admin-note">
                Firebase handles login identity. Admin access is granted through <code>.streamlit/secrets.toml</code>, while this panel lets admins define what other authenticated users can see in the dashboard.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Current user", current_email or "Not detected")
    with c2:
        st.metric("Current role", current_role)
    with c3:
        st.metric("Managed users", len(config.get("users", {})))
    with c4:
        st.metric("Configured admins", len(admin_emails))

    tabs = st.tabs([
        "👥 User visibility",
        "🎛️ Global feature toggles",
        "🌍 Data scope",
        "📋 Current rules",
        "⚙️ Setup export",
    ])

    with tabs[0]:
        st.markdown(
            '<div class="admin-card"><div class="admin-card-title">Select what a user can see</div><div class="admin-card-note">Add a user by email, assign a role template, then fine-tune permissions. Admin emails remain controlled only through secrets.toml.</div></div>',
            unsafe_allow_html=True,
        )
        users_df = _users_table(config)
        if not users_df.empty:
            st.dataframe(users_df, use_container_width=True, hide_index=True)
        else:
            st.info("No managed non-admin users yet. Add one below.")

        existing_emails = sorted(config.get("users", {}).keys())
        selected_existing = st.selectbox("Load existing user", ["Create new user"] + existing_emails, key="admin_existing_user")
        loaded = {} if selected_existing == "Create new user" else config.get("users", {}).get(selected_existing, {})

        user_email = st.text_input("User email", value=loaded.get("email", "") or ("" if selected_existing == "Create new user" else selected_existing), placeholder="user@example.org", key="admin_user_email")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            role = st.selectbox("Role template", list(ROLE_LABELS.keys()), format_func=lambda r: ROLE_LABELS.get(r, r), index=list(ROLE_LABELS.keys()).index(loaded.get("role", "viewer")) if loaded.get("role", "viewer") in ROLE_LABELS else 1, key="admin_user_role")
        with col_b:
            active = st.toggle("Active user", value=bool(loaded.get("active", True)), key="admin_user_active")

        base_permissions = loaded.get("permissions") or _permissions_for_role(role)
        st.markdown('<div class="admin-card"><div class="admin-card-title">Visible dashboard sections and actions</div><div class="admin-card-note">Turn on only the parts this user should access.</div></div>', unsafe_allow_html=True)
        perm_cols = st.columns(2)
        permissions = {}
        for i, perm in enumerate(ALL_PERMISSION_KEYS):
            with perm_cols[i % 2]:
                permissions[perm] = st.checkbox(PERMISSION_LABELS.get(perm, perm), value=bool(base_permissions.get(perm, False)), key=f"perm_{perm}")

        st.markdown('<div class="admin-card"><div class="admin-card-title">Data visibility scope</div><div class="admin-card-note">Leave blank to allow all available records for the selected permissions.</div></div>', unsafe_allow_html=True)
        scope = loaded.get("data_scope", {})
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            regions_txt = st.text_input("Allowed regions", value=", ".join(scope.get("regions", []) or []), placeholder="Africa, Asia and the Pacific", key="scope_regions")
        with sc2:
            countries_txt = st.text_input("Allowed countries", value=", ".join(scope.get("countries", []) or []), placeholder="Kenya, Ethiopia", key="scope_countries")
        with sc3:
            years_txt = st.text_input("Allowed years", value=", ".join(map(str, scope.get("years", []) or [])), placeholder="2024, 2025", key="scope_years")

        data_scope = {
            "regions": _parse_csv_list(regions_txt),
            "countries": _parse_csv_list(countries_txt),
            "years": [int(y) for y in _parse_csv_list(years_txt) if str(y).isdigit()],
        }

        save_col, del_col = st.columns([1, 1])
        with save_col:
            if st.button("💾 Save user visibility", type="primary", use_container_width=True):
                upsert_managed_user(user_email, role, active, permissions, data_scope, actor_email=current_email)
                st.success("User visibility saved. The user will receive these permissions after login.")
                st.rerun()
        with del_col:
            if selected_existing != "Create new user" and st.button("🗑️ Remove managed user", use_container_width=True):
                delete_managed_user(selected_existing, actor_email=current_email)
                st.warning("Managed user rule removed.")
                st.rerun()

    with tabs[1]:
        st.markdown('<div class="admin-card"><div class="admin-card-title">Global feature toggles</div><div class="admin-card-note">These switches control whether features are available to non-admin users at all. Admins keep full access.</div></div>', unsafe_allow_html=True)
        current_flags = {k: get_global_feature_flag(k) for k in DEFAULT_FEATURE_FLAGS}
        new_flags = {}
        fc1, fc2 = st.columns(2)
        for i, (flag, default_val) in enumerate(DEFAULT_FEATURE_FLAGS.items()):
            with (fc1 if i % 2 == 0 else fc2):
                new_flags[flag] = st.toggle(FEATURE_LABELS.get(flag, flag), value=bool(current_flags.get(flag, default_val)), key=f"feature_{flag}")
        if st.button("💾 Save global feature toggles", type="primary"):
            update_feature_flags(new_flags, actor_email=current_email)
            st.success("Global feature toggles updated.")
            st.rerun()

    with tabs[2]:
        st.markdown('<div class="admin-card"><div class="admin-card-title">How data scope works</div><div class="admin-card-note">A managed user can be limited by region, country, and year. This is applied before dashboard filters, so users only see records inside their assigned scope.</div></div>', unsafe_allow_html=True)
        scope_rows = []
        for email, profile in config.get("users", {}).items():
            scope = profile.get("data_scope", {})
            scope_rows.append({
                "Email": email,
                "Regions": ", ".join(scope.get("regions", []) or ["All"]),
                "Countries": ", ".join(scope.get("countries", []) or ["All"]),
                "Years": ", ".join(map(str, scope.get("years", []) or ["All"])),
            })
        st.dataframe(pd.DataFrame(scope_rows), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.markdown('<div class="admin-card"><div class="admin-card-title">Current access rules</div><div class="admin-card-note">Review admin emails, power users, managed users, and the current permission catalogue.</div></div>', unsafe_allow_html=True)
        st.subheader("Admin emails from secrets.toml")
        st.dataframe(pd.DataFrame({"Admin email": admin_emails}), use_container_width=True, hide_index=True)
        st.subheader("Power users from secrets.toml")
        st.dataframe(pd.DataFrame({"Power user email": power_users}), use_container_width=True, hide_index=True)
        st.subheader("Permission catalogue")
        st.dataframe(pd.DataFrame([{"Permission key": k, "Label": PERMISSION_LABELS.get(k, k), "Current user allowed": "Yes" if has_permission(k) else "No"} for k in ALL_PERMISSION_KEYS]), use_container_width=True, hide_index=True)

    with tabs[4]:
        st.markdown('<div class="admin-card"><div class="admin-card-title">Copy-ready secrets.toml block</div><div class="admin-card-note">Use this for admin elevation only. Other users can be controlled from this panel.</div></div>', unsafe_allow_html=True)
        toml = f"[auth]\nadmin_emails = {_toml_list(admin_emails or ['your-admin-email@example.org'])}\n\npower_users = {_toml_list(power_users or ['analyst@example.org'])}\n"
        st.code(toml, language="toml")
        st.markdown('<div class="admin-warning-card">Managed user visibility is stored in <strong>admin_access_config.json</strong>. On Render/Docker, map <code>/exports</code> as persistent storage if you want these admin settings to survive redeploys.</div>', unsafe_allow_html=True)
        st.caption(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
