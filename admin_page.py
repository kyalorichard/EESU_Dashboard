"""Full administrator console for the EU SEE Dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
import pandas as pd
from datetime import datetime

try:
    from authz import (
        get_current_email,
        get_role,
        is_admin,
        has_permission,
        get_admin_emails,
        get_power_users,
    )
except ImportError:
    def get_current_email():
        return st.session_state.get("email", "").lower().strip()

    def get_admin_emails():
        return [e.lower() for e in st.secrets.get("auth", {}).get("admin_emails", [])]

    def get_power_users():
        return [e.lower() for e in st.secrets.get("auth", {}).get("power_users", [])]

    def is_admin():
        return get_current_email() in get_admin_emails()

    def get_role():
        if is_admin():
            return "admin"
        if get_current_email() in get_power_users():
            return "analyst"
        if get_current_email():
            return "viewer"
        return "guest"

    def has_permission(permission):
        return is_admin()
from authz import (
    DEFAULT_DATA_SCOPE,
    DEFAULT_FEATURE_TOGGLES,
    DEFAULT_PERMISSIONS,
    ROLE_PRESETS,
    apply_data_scope,
    available_values,
    current_user_email,
    delete_user,
    export_dir,
    get_current_user_access,
    has_permission,
    load_access_config,
    permissions_for_role,
    read_audit_log,
    save_access_config,
    set_feature_toggle,
    upsert_user,
)

PERMISSION_LABELS = {
    "view_dashboard": "View dashboard",
    "view_overview": "View Overview tab",
    "view_negative_alerts": "View Negative Alerts tab",
    "view_visualization_map": "View Visualization Map tab",
    "view_user_manual": "View User Manual tab",
    "view_country_counts": "See country counts",
    "view_detailed_tables": "See detailed tables",
    "download_data": "Download/export data",
    "use_ai_copilot": "Use AI Copilot",
    "manage_users": "Manage users",
    "manage_system_settings": "Manage system settings",
    "view_usage_analytics": "View usage analytics",
    "upload_replace_data": "Upload/replace data",
}

FEATURE_LABELS = {
    "overview_tab": "Overview tab",
    "negative_alerts_tab": "Negative Alerts tab",
    "visualization_map_tab": "Visualization Map tab",
    "user_manual_tab": "User Manual tab",
    "ai_copilot": "AI Copilot",
    "csv_exports": "CSV/data exports",
    "country_counts": "Country counts",
    "detailed_tables": "Detailed tables",
    "admin_console": "Admin console",
}

ROLE_DESCRIPTIONS = {
    "guest": "Public mode with basic dashboard access only.",
    "viewer": "Can view standard dashboard content but cannot export data or use advanced tools.",
    "analyst": "Can use analytical views and AI support, but exports remain restricted.",
    "privileged": "Full analytical access including downloads and AI Copilot.",
    "admin": "Full system access, user management, permissions, and settings.",
}


def _inject_admin_css() -> None:
    st.markdown(
        """
        <style>
        .admin-hero {
            background: radial-gradient(circle at 90% 5%, rgba(0,140,170,.12), transparent 28%), linear-gradient(135deg, #FFFFFF 0%, #F7ECFB 58%, #EFFBFE 100%);
            border: 1px solid rgba(102,0,148,.16);
            border-radius: 24px;
            padding: 22px 24px;
            box-shadow: 0 16px 38px rgba(16,24,40,.09);
            font-family: Arial, sans-serif;
            margin-bottom: 16px;
        }
        .admin-eyebrow { font-size:10px; font-weight:900; color:#660094; letter-spacing:.14em; text-transform:uppercase; margin-bottom:5px; }
        .admin-title { font-size:30px; font-weight:950; color:#23152F; line-height:1.1; letter-spacing:-.02em; }
        .admin-note { font-size:12px; color:#667085; max-width:920px; line-height:1.45; margin-top:7px; }
        .admin-card { background:#FFFFFF; border:1px solid #E6E8EF; border-radius:18px; padding:16px; box-shadow:0 10px 24px rgba(16,24,40,.06); margin-bottom:14px; font-family:Arial, sans-serif; }
        .admin-mini-grid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:10px; margin:12px 0 4px 0; }
        .admin-mini-kpi { background:#FFFFFF; border:1px solid #E6E8EF; border-radius:16px; padding:12px 13px; box-shadow:0 8px 18px rgba(16,24,40,.05); }
        .admin-mini-kpi span { display:block; font-size:10px; color:#667085; font-weight:900; margin-bottom:4px; text-transform:uppercase; letter-spacing:.06em; }
        .admin-mini-kpi strong { display:block; font-size:24px; color:#23152F; font-weight:950; line-height:1; }
        .admin-small { font-size:11px; color:#667085; line-height:1.35; }
        .admin-pill { display:inline-flex; border-radius:999px; background:#F4EAF8; color:#660094; border:1px solid #E7D4F1; padding:5px 9px; font-size:10px; font-weight:900; margin-right:6px; margin-top:6px; }
        .admin-section-title { font-size:15px; font-weight:950; color:#23152F; margin-bottom:4px; }
        .admin-section-note { font-size:11px; color:#667085; line-height:1.35; margin-bottom:10px; }
        div[data-testid="stTabs"] [role="tablist"] { gap:8px; }
        @media (max-width: 900px) { .admin-mini-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _users_dataframe() -> pd.DataFrame:
    users = load_access_config().get("users", {})
    rows = []
    for email, cfg in users.items():
        perms = cfg.get("permissions", {})
        scope = cfg.get("data_scope", {}) or {}
        rows.append(
            {
                "email": email,
                "name": cfg.get("name", ""),
                "role": cfg.get("role", "viewer"),
                "active": bool(cfg.get("active", True)),
                "regions": ", ".join(scope.get("regions", []) or ["All"]),
                "countries": ", ".join(scope.get("countries", [])[:5]) + (" ..." if len(scope.get("countries", []) or []) > 5 else "") if scope.get("countries") else "All",
                "permissions_on": sum(1 for v in perms.values() if v),
                "updated_at_utc": cfg.get("updated_at_utc", ""),
            }
        )
    return pd.DataFrame(rows)


def _dashboard_values(df: pd.DataFrame | None) -> Dict[str, List[str]]:
    if df is None:
        df = pd.DataFrame()
    return {
        "regions": available_values(df, "region"),
        "countries": available_values(df, "alert-country"),
        "years": available_values(df, "year"),
        "alert_impacts": available_values(df, "alert-impact"),
        "alert_types": available_values(df, "alert-type"),
    }


def _render_kpis(users_df: pd.DataFrame, config: Dict) -> None:
    total_users = len(users_df)
    active_users = int(users_df["active"].sum()) if not users_df.empty and "active" in users_df.columns else 0
    admins = int((users_df["role"] == "admin").sum()) if not users_df.empty and "role" in users_df.columns else 0
    enabled_features = sum(1 for v in config.get("feature_toggles", {}).values() if v)
    st.markdown(
        f"""
        <div class="admin-mini-grid">
            <div class="admin-mini-kpi"><span>Total users</span><strong>{total_users:,}</strong></div>
            <div class="admin-mini-kpi"><span>Active users</span><strong>{active_users:,}</strong></div>
            <div class="admin-mini-kpi"><span>Admins</span><strong>{admins:,}</strong></div>
            <div class="admin-mini-kpi"><span>Features on</span><strong>{enabled_features:,}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_overview(users_df: pd.DataFrame, config: Dict) -> None:
    st.markdown("<div class='admin-card'>", unsafe_allow_html=True)
    st.markdown("<div class='admin-section-title'>Current users</div>", unsafe_allow_html=True)
    st.markdown("<div class='admin-section-note'>Review the accounts currently controlled by the dashboard access file.</div>", unsafe_allow_html=True)
    if users_df.empty:
        st.info("No users have been configured yet. Add the first user in the User Manager tab.")
    else:
        st.dataframe(users_df, use_container_width=True, hide_index=True, height=280)
        st.download_button(
            "Download user access table (.csv)",
            data=users_df.to_csv(index=False).encode("utf-8"),
            file_name="eusee_user_access_table.csv",
            mime="text/csv",
            use_container_width=True,
            key="admin_download_users_table",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_user_manager(df: pd.DataFrame | None) -> None:
    values = _dashboard_values(df)
    users = load_access_config().get("users", {})
    existing_emails = sorted(users.keys())

    st.markdown("<div class='admin-card'>", unsafe_allow_html=True)
    st.markdown("<div class='admin-section-title'>Add or update user</div>", unsafe_allow_html=True)
    st.markdown("<div class='admin-section-note'>Assign a role, then optionally override permissions and data scope.</div>", unsafe_allow_html=True)

    selected_existing = st.selectbox(
        "Load existing user",
        ["Create new user"] + existing_emails,
        key="admin_existing_user_select",
    )
    existing = users.get(selected_existing, {}) if selected_existing != "Create new user" else {}

    c1, c2, c3 = st.columns([1.25, 1.15, .75])
    with c1:
        email = st.text_input("User email / username", value=selected_existing if selected_existing != "Create new user" else "", placeholder="name@example.org", key="admin_user_email")
    with c2:
        name = st.text_input("Display name", value=existing.get("name", ""), placeholder="Full name", key="admin_user_name")
    with c3:
        role_options = list(ROLE_PRESETS.keys())
        current_role = existing.get("role", "viewer")
        role = st.selectbox("Role", role_options, index=role_options.index(current_role) if current_role in role_options else 1, key="admin_user_role")

    active = st.toggle("User is active", value=bool(existing.get("active", True)), key="admin_user_active")
    notes = st.text_area("Admin notes", value=existing.get("notes", ""), height=80, placeholder="Optional note about access reason, partner organization, expiry, etc.", key="admin_user_notes")

    with st.expander("Permission overrides", expanded=True):
        st.markdown(f"<div class='admin-small'><strong>{role}</strong>: {ROLE_DESCRIPTIONS.get(role, '')}</div>", unsafe_allow_html=True)
        base = permissions_for_role(role)
        existing_perms = existing.get("permissions", {}) or {}
        perm_values = {}
        cols = st.columns(3)
        for i, perm in enumerate(DEFAULT_PERMISSIONS.keys()):
            with cols[i % 3]:
                perm_values[perm] = st.checkbox(
                    PERMISSION_LABELS.get(perm, perm),
                    value=bool(existing_perms.get(perm, base.get(perm, False))),
                    key=f"admin_perm_{perm}",
                )

    with st.expander("Data access scope", expanded=True):
        st.markdown("<div class='admin-small'>Leave a list empty to allow all values for that dimension.</div>", unsafe_allow_html=True)
        existing_scope = existing.get("data_scope", {}) or {}
        sc1, sc2 = st.columns(2)
        with sc1:
            allowed_regions = st.multiselect("Allowed regions", values["regions"], default=existing_scope.get("regions", []), key="admin_scope_regions")
            allowed_years = st.multiselect("Allowed years", values["years"], default=[str(x) for x in existing_scope.get("years", [])], key="admin_scope_years")
            allowed_impacts = st.multiselect("Allowed alert impacts", values["alert_impacts"], default=existing_scope.get("alert_impacts", []), key="admin_scope_impacts")
        with sc2:
            allowed_countries = st.multiselect("Allowed countries", values["countries"], default=existing_scope.get("countries", []), key="admin_scope_countries")
            allowed_types = st.multiselect("Allowed alert types", values["alert_types"], default=existing_scope.get("alert_types", []), key="admin_scope_types")

        preview_scope = {
            "regions": allowed_regions,
            "countries": allowed_countries,
            "years": allowed_years,
            "alert_impacts": allowed_impacts,
            "alert_types": allowed_types,
        }
        if df is not None and not df.empty:
            preview_df = apply_data_scope(df, {"data_scope": preview_scope})
            st.info(f"This scope would allow {len(preview_df):,} records from the currently loaded dataset.")

    save_col, delete_col = st.columns([1, 1])
    with save_col:
        if st.button("Save user access", type="primary", use_container_width=True, key="admin_save_user"):
            try:
                upsert_user(
                    email=email,
                    name=name,
                    role=role,
                    active=active,
                    permissions=perm_values,
                    data_scope=preview_scope,
                    notes=notes,
                )
                st.success("User access saved.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save user access: {exc}")
    with delete_col:
        if st.button("Delete user", use_container_width=True, key="admin_delete_user"):
            if not email:
                st.warning("Enter the user email / username to delete.")
            elif email.strip().lower() == current_user_email():
                st.warning("You cannot delete your own active administrator record from this panel.")
            else:
                delete_user(email)
                st.success("User deleted if it existed.")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def _render_feature_toggles() -> None:
    config = load_access_config()
    toggles = config.get("feature_toggles", DEFAULT_FEATURE_TOGGLES.copy())
    st.markdown("<div class='admin-card'>", unsafe_allow_html=True)
    st.markdown("<div class='admin-section-title'>Global feature toggles</div>", unsafe_allow_html=True)
    st.markdown("<div class='admin-section-note'>Disable a feature globally even if a user role has permission for it.</div>", unsafe_allow_html=True)

    cols = st.columns(3)
    changed = {}
    for i, key in enumerate(DEFAULT_FEATURE_TOGGLES.keys()):
        with cols[i % 3]:
            changed[key] = st.toggle(FEATURE_LABELS.get(key, key), value=bool(toggles.get(key, DEFAULT_FEATURE_TOGGLES[key])), key=f"feature_{key}")

    if st.button("Save feature toggles", type="primary", use_container_width=True, key="admin_save_features"):
        for key, value in changed.items():
            config.setdefault("feature_toggles", {})[key] = bool(value)
        save_access_config(config)
        st.success("Feature toggles saved.")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _render_roles() -> None:
    rows = []
    for role, perms in ROLE_PRESETS.items():
        rows.append({"role": role, "description": ROLE_DESCRIPTIONS.get(role, ""), **{PERMISSION_LABELS.get(k, k): v for k, v in perms.items()}})
    df = pd.DataFrame(rows)
    st.markdown("<div class='admin-card'>", unsafe_allow_html=True)
    st.markdown("<div class='admin-section-title'>Role presets</div>", unsafe_allow_html=True)
    st.markdown("<div class='admin-section-note'>These are the default permission bundles applied when a role is selected.</div>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True, height=310)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_audit() -> None:
    audit = read_audit_log(limit=300)
    st.markdown("<div class='admin-card'>", unsafe_allow_html=True)
    st.markdown("<div class='admin-section-title'>Audit log</div>", unsafe_allow_html=True)
    st.markdown("<div class='admin-section-note'>Tracks admin changes such as user updates and feature toggle changes.</div>", unsafe_allow_html=True)
    if audit.empty:
        st.info("No admin audit events recorded yet.")
    else:
        st.dataframe(audit.sort_values("timestamp_utc", ascending=False), use_container_width=True, hide_index=True, height=360)
        st.download_button("Download audit log (.csv)", data=audit.to_csv(index=False).encode("utf-8"), file_name="eusee_admin_audit_log.csv", mime="text/csv", use_container_width=True, key="admin_download_audit")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_system() -> None:
    config = load_access_config()
    st.markdown("<div class='admin-card'>", unsafe_allow_html=True)
    st.markdown("<div class='admin-section-title'>System settings and files</div>", unsafe_allow_html=True)
    st.markdown("<div class='admin-section-note'>File-based control settings for the current deployment.</div>", unsafe_allow_html=True)
    st.code(f"Access control file: {export_dir() / 'user_access_control.json'}\nAudit log file: {export_dir() / 'admin_audit_log.jsonl'}")

    raw_json = json.dumps(config, indent=2, sort_keys=True)
    st.download_button("Download access configuration (.json)", data=raw_json.encode("utf-8"), file_name="user_access_control.json", mime="application/json", use_container_width=True, key="admin_download_config")

    uploaded = st.file_uploader("Restore access configuration JSON", type=["json"], key="admin_restore_config")
    if uploaded is not None:
        try:
            restored = json.loads(uploaded.read().decode("utf-8"))
            if st.button("Apply restored configuration", type="primary", use_container_width=True, key="admin_apply_restore"):
                save_access_config(restored)
                st.success("Configuration restored.")
                st.rerun()
        except Exception as exc:
            st.error(f"Invalid JSON file: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_admin_page(df: pd.DataFrame | None = None) -> None:
    _inject_admin_css()

    if not has_permission("manage_users"):
        st.error("You do not have administrator access for this page.")
        if st.button("← Back to dashboard", use_container_width=True):
            st.session_state.admin_view = False
            st.rerun()
        return

    access = get_current_user_access()
    config = load_access_config()
    users_df = _users_dataframe()

    st.markdown(
        f"""
        <div class="admin-hero">
            <div class="admin-eyebrow">Administrator console</div>
            <div class="admin-title">Control user visibility, permissions, and data access</div>
            <div class="admin-note">
                Manage what each user can see in the EU SEE Dashboard, including tabs, exports, AI Copilot,
                country counts, detailed tables, and country/region/year-level data scope.
            </div>
            <div style="margin-top:10px;">
                <span class="admin-pill">Signed in: {access.get('name','Admin')}</span>
                <span class="admin-pill">Role: {access.get('role','admin')}</span>
                <span class="admin-pill">Storage: exports/user_access_control.json</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav1, nav2 = st.columns([1, 1])
    with nav1:
        if st.button("← Back to dashboard", use_container_width=True, key="admin_back_to_dashboard"):
            st.session_state.admin_view = False
            st.rerun()
    with nav2:
        if st.button("Reload admin console", use_container_width=True, key="admin_reload_rules"):
            st.rerun()

    _render_kpis(users_df, config)

    tabs = st.tabs(["Overview", "User Manager", "Feature Toggles", "Roles", "Audit Log", "System"])
    with tabs[0]:
        _render_overview(users_df, config)
    with tabs[1]:
        _render_user_manager(df)
    with tabs[2]:
        _render_feature_toggles()
    with tabs[3]:
        _render_roles()
    with tabs[4]:
        _render_audit()
    with tabs[5]:
        _render_system()
