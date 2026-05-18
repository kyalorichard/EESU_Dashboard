# admin_page.py
from __future__ import annotations

import json
from datetime import datetime
import pandas as pd
import streamlit as st

from authz import (
    get_access_config_path,
    get_admin_emails,
    get_current_email,
    get_current_role,
    get_privileged_domains,
    has_permission,
    is_admin,
    load_access_config,
    save_access_config,
    default_access_config,
    reset_access_config,
)

FEATURE_LABELS = {
    "view_dashboard": "Dashboard access",
    "view_overview": "Overview tab",
    "view_coverage_monitored_countries": "Summary cards",
    "view_monitored_countries_value": "Monitored Countries value",
    "view_maps": "Visualization Map",
    "view_negative_alerts": "Negative Alerts tab",
    "view_analytical_flow_panel": "Analytical Flow Panels (Heatmaps / Sankey)",
    "view_data_table": "Summary data preview",
    "download_data": "CSV/XLSX downloads",
    "use_ai_copilot": "AI Copilot",
    "view_user_manual": "User manual",
    "view_admin_page": "Admin page",
}

GUEST_LOCKED_FEATURES = {
    "download_data",
    "use_ai_copilot",
    "view_admin_page",
}


def inject_admin_css():
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 0.75rem !important;
        max-width: 1500px !important;
    }

    .admin-hero {
        background: linear-gradient(135deg,#ffffff 0%,#f8fafc 100%);
        border: 1px solid #e6e8ef;
        border-radius: 18px;
        padding: 22px 24px;
        box-shadow: 0 10px 28px rgba(16,24,40,.06);
        margin-bottom: 16px;
    }

    .admin-eyebrow {
        font-size: 10px;
        font-weight: 900;
        color: #344054;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .admin-title {
        font-size: 26px;
        font-weight: 900;
        color: #101828;
        line-height: 1.1;
        margin-bottom: 6px;
    }

    .admin-subtitle {
        font-size: 13px;
        color: #667085;
        line-height: 1.45;
        max-width: 980px;
    }

    .admin-info {
        background: #eff8ff;
        border: 1px solid #b2ddff;
        border-radius: 12px;
        padding: 11px 13px;
        color: #175cd3;
        font-size: 12px;
        margin: 12px 0 14px 0;
    }

    .admin-footer {
        text-align: center;
        font-size: 11px;
        color: #98a2b3;
        margin-top: 16px;
    }

    .stButton > button {
        border-radius: 11px !important;
        font-weight: 800 !important;
        height: 42px !important;
    }

    .stDownloadButton > button {
        border-radius: 11px !important;
        font-weight: 800 !important;
        height: 42px !important;
    }

    div[data-baseweb="select"] > div {
        border-radius: 11px !important;
        min-height: 42px !important;
    }

    div[data-testid="stExpander"] {
        border-radius: 14px !important;
        border: 1px solid #e6e8ef !important;
        box-shadow: 0 6px 18px rgba(16,24,40,.04) !important;
        overflow: hidden !important;
    }

    div[data-testid="stExpander"] summary {
        font-weight: 900 !important;
        color: #101828 !important;
        background: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_admin_sidebar_navigation():
    if not is_admin():
        return "Dashboard"

    st.sidebar.markdown("""
    <div style="
        margin-top:14px;
        padding:14px;
        border-radius:16px;
        background:#ffffff;
        border:1px solid #e6e8ef;
        box-shadow:0 8px 20px rgba(16,24,40,.05);
        font-family:Arial,sans-serif;
    ">
        <div style="font-size:9px;font-weight:900;color:#667085;letter-spacing:.12em;text-transform:uppercase;">
            Workspace
        </div>
        <div style="font-size:14px;font-weight:900;color:#101828;margin-top:5px;">
            ⚙️ Administration
        </div>
        <div style="font-size:11px;color:#667085;line-height:1.4;margin-top:5px;">
            Manage visibility, data scope and access roles.
        </div>
    </div>
    """, unsafe_allow_html=True)

    return st.sidebar.radio(
        "Admin navigation",
        ["Dashboard", "Admin"],
        index=0,
        key="admin_navigation_choice",
        label_visibility="collapsed",
    )


def _safe_role_config(config: dict, role: str) -> dict:
    config.setdefault(role, {})
    config[role].setdefault("features", {})
    config[role].setdefault("regions", [])
    config[role].setdefault("countries", [])
    config[role].setdefault("years", [])

    removed_permissions = {
        "view_public_summary",
        "view_country_counts",
        "view_negative_relationship_intelligence",
    }

    for removed_permission in removed_permissions:
        config[role]["features"].pop(removed_permission, None)

    for key in FEATURE_LABELS:
        config[role]["features"].setdefault(key, False)

    return config


def _render_header():
    st.markdown(f"""
    <div class="admin-hero">
        <div class="admin-eyebrow">Admin workspace</div>
        <div class="admin-title">EU SEE Dashboard Administration</div>
        <div class="admin-subtitle">
            Configure access roles, visibility settings, data scope and dashboard governance.
            <br>
            Saved config: <code>{get_access_config_path()}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_role_summary(role: str):
    with st.container(border=True):
        st.markdown("### 👥 Role summary")
        st.caption("Current role")
        st.info(role.capitalize())
        st.caption("Last updated")
        st.write(datetime.now().strftime("%b %d, %Y %I:%M %p"))
        st.caption("Updated by")
        st.write("Admin")


def _render_help_card():
    with st.container(border=True):
        st.markdown("### ❔ Need help?")
        st.caption(
            "Use this page to control what each access role can view or download. "
            "Save changes after editing permissions."
        )
        st.button("View documentation", use_container_width=True)


def _render_visibility_tab(config: dict):
    left, right = st.columns([4.3, 1.35])

    with left:
        with st.container(border=True):
            st.markdown("### Configure role")
            st.caption("Define what content and features this role can see and access.")

            role = st.selectbox(
                "Configure role",
                ["guest", "viewer", "privileged"],
                index=0,
                key="admin_visibility_role",
            )

            config = _safe_role_config(config, role)
            features = config[role]["features"]

            st.markdown("""
            <div class="admin-info">
                ℹ️ Click Save after changes. Settings persist after logout only if this config path is on persistent storage.
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)

            left_features = [
                "view_overview",
                "view_negative_alerts",
                "view_analytical_flow_panel",
                "download_data",
                "view_user_manual",
            ]

            right_features = [
                "view_dashboard",
                "view_coverage_monitored_countries",
                "view_monitored_countries_value",
                "view_maps",
                "view_data_table",
                "use_ai_copilot",
                "view_admin_page",
            ]

            with c1:
                with st.expander("Content visibility", expanded=True):
                    for key in left_features:
                        features[key] = st.checkbox(
                            FEATURE_LABELS[key],
                            value=bool(features.get(key, False)),
                            key=f"persist_feature_{role}_{key}",
                            disabled=(role == "guest" and key in GUEST_LOCKED_FEATURES),
                        )

            with c2:
                with st.expander("Additional permissions", expanded=True):
                    for key in right_features:
                        features[key] = st.checkbox(
                            FEATURE_LABELS[key],
                            value=bool(features.get(key, False)),
                            key=f"persist_feature_{role}_{key}",
                            disabled=(role == "guest" and key in GUEST_LOCKED_FEATURES),
                        )

            config[role]["features"] = features

            save_col, reset_col = st.columns(2)

            with save_col:
                if st.button("💾 Save visibility settings", type="primary", use_container_width=True):
                    if save_access_config(config):
                        st.success("Visibility settings saved.")
                        st.rerun()

            with reset_col:
                if st.button("↩ Reset all roles to defaults", use_container_width=True):
                    if save_access_config(default_access_config()):
                        st.success("Defaults restored.")
                        st.rerun()

            st.download_button(
                "⬇️ Download access config (JSON)",
                data=json.dumps(config, indent=2),
                file_name="eusee_access_config.json",
                mime="application/json",
                use_container_width=True,
            )

    with right:
        _render_role_summary(role)
        _render_help_card()


def _render_scope_tab(config: dict, data=None):
    role = st.selectbox(
        "Configure data scope for role",
        ["guest", "viewer", "privileged"],
        index=0,
        key="scope_role",
    )

    config = _safe_role_config(config, role)

    st.info("Leave selections empty to allow all available values for that role.")

    regions, countries, years = [], [], []

    if data is not None and not getattr(data, "empty", True):
        if "region" in data.columns:
            regions = sorted(data["region"].dropna().astype(str).unique())

        if "alert-country" in data.columns:
            countries = sorted(data["alert-country"].dropna().astype(str).unique())

        if "year" in data.columns:
            years = sorted([int(y) for y in data["year"].dropna().unique()])

    c1, c2, c3 = st.columns(3)

    with c1:
        config[role]["regions"] = st.multiselect(
            "Allowed regions",
            regions,
            default=[x for x in config[role].get("regions", []) if x in regions],
            key=f"persist_regions_{role}",
        )

    with c2:
        config[role]["countries"] = st.multiselect(
            "Allowed countries",
            countries,
            default=[x for x in config[role].get("countries", []) if x in countries],
            key=f"persist_countries_{role}",
        )

    with c3:
        config[role]["years"] = st.multiselect(
            "Allowed years",
            years,
            default=[x for x in config[role].get("years", []) if x in years],
            key=f"persist_years_{role}",
        )

    if st.button("💾 Save data scope", use_container_width=True, type="primary"):
        if save_access_config(config):
            st.success("Data scope saved.")
            st.rerun()

    with st.expander("Current role scope JSON", expanded=False):
        st.json(config[role])


def _render_users_tab():
    rows = []

    for email in get_admin_emails():
        rows.append({
            "identity": email,
            "role": "admin",
            "source": "[auth].admin_emails",
        })

    for domain in get_privileged_domains():
        rows.append({
            "identity": f"*@{domain}",
            "role": "privileged",
            "source": "[access].privileged_domains",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No admin emails or privileged domains configured.")

    st.code("""
[auth]
admin_emails = ["admin@example.org"]

[access]
privileged_domains = ["example.org", "example.org"]

[access_control]
config_path = "/exports/eusee_access_config.json"
""", language="toml")


def _render_diagnostics_tab():
    st.json({
        "current_email": get_current_email(),
        "current_role": get_current_role(),
        "is_admin": is_admin(),
        "config_path": str(get_access_config_path()),
        "can_download": has_permission("download_data"),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "session_email_verified": st.session_state.get("email_verified"),
        "session_user": st.session_state.get("user"),
    })

    st.subheader("Access config reset")

    st.warning(
        "Use this if guest permissions do not update after saving. "
        "It deletes the stale JSON and rebuilds it using the current permission keys."
    )

    if st.button(
        "🧹 Reset access config and rebuild permissions",
        type="primary",
        use_container_width=True,
    ):
        if reset_access_config():
            st.success("Access config reset successfully. Refresh the app, then configure Guest visibility again.")
            st.rerun()

    st.subheader("Loaded access config")
    st.json(load_access_config())


def render_admin_page(data=None):
    if not is_admin():
        st.error("Access restricted. This page is only available to configured admin emails.")
        st.stop()

    inject_admin_css()

    config = load_access_config()

    for role_name in ["guest", "viewer", "privileged"]:
        config = _safe_role_config(config, role_name)

    _render_header()

    tab_visibility, tab_scope, tab_users, tab_system = st.tabs(
        ["Visibility", "Data scope", "Users", "Diagnostics"]
    )

    with tab_visibility:
        _render_visibility_tab(config)

    with tab_scope:
        _render_scope_tab(config, data=data)

    with tab_users:
        _render_users_tab()

    with tab_system:
        _render_diagnostics_tab()

    st.markdown("""
    <div class="admin-footer">
        © 2026 EU SEE Project. All rights reserved.
    </div>
    """, unsafe_allow_html=True)
