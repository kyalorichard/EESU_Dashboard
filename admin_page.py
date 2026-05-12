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
    "view_public_summary": "Public summary",
    "view_dashboard": "Dashboard access",
    "view_overview": "Overview tab",
    "view_coverage_monitored_countries": "Coverage / Monitored Countries",
    "view_country_counts": "Country count KPIs",
    "view_maps": "Visualization Map",
    "view_negative_alerts": "Negative Alerts tab",
    "view_negative_relationship_intelligence": "Negative events relationship intelligence",
    "view_analytical_flow_panel": "Analytical Flow Panels (Heatmaps / Sankey)",
    "view_data_table": "Summary data preview",
    "download_data": "CSV/XLSX downloads",
    "use_ai_copilot": "AI Copilot",
    "view_user_manual": "User manual",
    "view_admin_page": "Admin page",
}


# =========================================================
# PROFESSIONAL ADMIN CSS
# =========================================================
def inject_admin_css():

    st.markdown("""
    <style>

    .admin-shell * {
        font-family: Arial, sans-serif !important;
    }

    .admin-shell {
        margin-top: 4px;
    }

    .admin-hero {
        background:
            radial-gradient(circle at top right, rgba(102,0,148,.08), transparent 28%),
            linear-gradient(135deg,#FFFFFF 0%,#F7ECFB 62%,#EFFBFE 100%);
        border: 1px solid rgba(102,0,148,.14);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 14px 36px rgba(16,24,40,.07);
        margin-bottom: 16px;
    }

    .admin-eyebrow {
        font-size: 10px;
        font-weight: 900;
        color: #660094;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .admin-title {
        font-size: 32px;
        font-weight: 950;
        color: #23152F;
        line-height: 1.05;
        letter-spacing: -.03em;
    }

    .admin-subtitle {
        margin-top: 8px;
        font-size: 12px;
        color: #667085;
        line-height: 1.45;
        max-width: 900px;
    }

    .admin-card {
        background: #FFFFFF;
        border: 1px solid #E6E8EF;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 10px 26px rgba(16,24,40,.05);
        margin-bottom: 16px;
    }

    .admin-card-title {
        font-size: 18px;
        font-weight: 900;
        color: #23152F;
        margin-bottom: 6px;
    }

    .admin-card-note {
        font-size: 11px;
        color: #667085;
        line-height: 1.4;
        margin-bottom: 16px;
    }

    .admin-section-title {
        font-size: 13px;
        font-weight: 900;
        color: #344054;
        margin-bottom: 12px;
    }

    .admin-info-banner {
        background: #EFF8FF;
        border: 1px solid #B2DDFF;
        border-radius: 12px;
        padding: 12px 14px;
        color: #175CD3;
        font-size: 11px;
        margin-top: 12px;
        margin-bottom: 18px;
    }

    .admin-right-card {
        background: linear-gradient(180deg,#FFFFFF 0%,#FAFAFC 100%);
        border: 1px solid #E6E8EF;
        border-radius: 18px;
        padding: 16px;
        box-shadow: 0 8px 20px rgba(16,24,40,.05);
        margin-bottom: 16px;
    }

    .admin-right-title {
        font-size: 15px;
        font-weight: 900;
        color: #23152F;
        margin-bottom: 12px;
    }

    .admin-role-badge {
        display: inline-flex;
        padding: 6px 10px;
        border-radius: 999px;
        background: #F4EAF8;
        border: 1px solid #E7D4F1;
        color: #660094;
        font-size: 11px;
        font-weight: 900;
        margin-top: 6px;
    }

    .admin-meta-label {
        font-size: 10px;
        color: #667085;
        margin-top: 14px;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: .06em;
        font-weight: 800;
    }

    .admin-meta-value {
        font-size: 13px;
        color: #23152F;
        font-weight: 700;
    }

    .stButton > button {
        border-radius: 12px !important;
        height: 42px !important;
        font-weight: 900 !important;
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        min-height: 42px !important;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 1rem !important;
    }

    .admin-footer {
        text-align: center;
        color: #98A2B3;
        font-size: 11px;
        margin-top: 14px;
    }

    </style>
    """, unsafe_allow_html=True)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
def render_admin_sidebar_navigation():

    if not is_admin():
        return "Dashboard"

    st.sidebar.markdown("""
    <div style="
        margin-top:14px;
        padding:14px;
        border-radius:18px;
        background:linear-gradient(135deg,#FFFFFF 0%,#F4EAF8 100%);
        border:1px solid rgba(102,0,148,.15);
        box-shadow:0 10px 22px rgba(16,24,40,.06);
    ">
        <div style="
            font-size:9px;
            font-weight:900;
            color:#660094;
            letter-spacing:.13em;
            text-transform:uppercase;
        ">
            Admin workspace
        </div>

        <div style="
            font-size:15px;
            font-weight:900;
            color:#23152F;
            margin-top:5px;
        ">
            🔐 Administration
        </div>

        <div style="
            font-size:11px;
            color:#667085;
            margin-top:5px;
            line-height:1.4;
        ">
            Configure visibility, permissions and data scope.
        </div>
    </div>
    """, unsafe_allow_html=True)

    return st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Admin"],
        index=0,
        key="admin_navigation_choice",
        label_visibility="collapsed",
    )


# =========================================================
# MAIN ADMIN PAGE
# =========================================================
def render_admin_page(data=None):

    if not is_admin():
        st.error("Access restricted.")
        st.stop()

    inject_admin_css()

    config = load_access_config()

    st.markdown('<div class="admin-shell">', unsafe_allow_html=True)

    # =====================================================
    # HERO
    # =====================================================
    st.markdown(f"""
    <div class="admin-hero">

        <div class="admin-eyebrow">
            Admin workspace
        </div>

        <div class="admin-title">
            EU SEE Dashboard Administration
        </div>

        <div class="admin-subtitle">
            Configure access roles, visibility settings and manage dashboard
            permissions and data governance.
            <br><br>
            Saved config:
            <code>{get_access_config_path()}</code>
        </div>

    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # TABS
    # =====================================================
    tab_visibility, tab_scope, tab_users, tab_system = st.tabs(
        ["Visibility", "Data scope", "Users", "Diagnostics"]
    )

    # =====================================================
    # VISIBILITY TAB
    # =====================================================
    with tab_visibility:

        left, right = st.columns([4.2, 1.3])

        # -------------------------------------------------
        # LEFT PANEL
        # -------------------------------------------------
        with left:

            st.markdown("""
            <div class="admin-card">
                <div class="admin-card-title">
                    Configure role
                </div>

                <div class="admin-card-note">
                    Define what content and features this role can see and access.
                </div>
            """, unsafe_allow_html=True)

            role = st.selectbox(
                "Role",
                ["guest", "viewer", "privileged"],
                label_visibility="collapsed",
            )

            st.markdown("""
            <div class="admin-info-banner">
                ℹ️ Click Save after changes. Settings persist after logout only
                if the config path is on persistent storage.
            </div>
            """, unsafe_allow_html=True)

            features = config[role]["features"]

            col1, col2 = st.columns(2)

            LEFT_FEATURES = [
                "view_public_summary",
                "view_overview",
                "view_country_counts",
                "view_negative_alerts",
                "view_analytical_flow_panel",
                "download_data",
                "view_user_manual",
            ]

            RIGHT_FEATURES = [
                "view_dashboard",
                "view_coverage_monitored_countries",
                "view_maps",
                "view_negative_relationship_intelligence",
                "view_data_table",
                "use_ai_copilot",
                "view_admin_page",
            ]

            with col1:

                st.markdown("""
                <div class="admin-section-title">
                    Content visibility
                </div>
                """, unsafe_allow_html=True)

                for key in LEFT_FEATURES:

                    features[key] = st.checkbox(
                        FEATURE_LABELS[key],
                        value=bool(features.get(key, False)),
                        key=f"persist_feature_{role}_{key}",
                        disabled=(
                            role == "guest"
                            and key in [
                                "download_data",
                                "use_ai_copilot",
                                "view_admin_page",
                            ]
                        ),
                    )

            with col2:

                st.markdown("""
                <div class="admin-section-title">
                    Additional permissions
                </div>
                """, unsafe_allow_html=True)

                for key in RIGHT_FEATURES:

                    features[key] = st.checkbox(
                        FEATURE_LABELS[key],
                        value=bool(features.get(key, False)),
                        key=f"persist_feature_{role}_{key}",
                        disabled=(
                            role == "guest"
                            and key in [
                                "download_data",
                                "use_ai_copilot",
                                "view_admin_page",
                            ]
                        ),
                    )

            config[role]["features"] = features

            save_col, reset_col = st.columns(2)

            with save_col:

                if st.button(
                    "💾 Save visibility settings",
                    use_container_width=True,
                    type="primary",
                ):

                    if save_access_config(config):
                        st.success("Visibility settings saved.")
                        st.rerun()

            with reset_col:

                if st.button(
                    "↩ Reset all roles to defaults",
                    use_container_width=True,
                ):

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

            st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------------------------
        # RIGHT PANEL
        # -------------------------------------------------
        with right:

            st.markdown(f"""
            <div class="admin-right-card">

                <div class="admin-right-title">
                    Role summary
                </div>

                <div class="admin-meta-label">
                    Current role
                </div>

                <div class="admin-role-badge">
                    {role.capitalize()}
                </div>

                <div class="admin-meta-label">
                    Last updated
                </div>

                <div class="admin-meta-value">
                    {datetime.now().strftime("%b %d, %Y %I:%M %p")}
                </div>

                <div class="admin-meta-label">
                    Updated by
                </div>

                <div class="admin-meta-value">
                    Admin
                </div>

            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="admin-right-card">

                <div class="admin-right-title">
                    Need help?
                </div>

                <div style="
                    font-size:11px;
                    color:#667085;
                    line-height:1.5;
                    margin-bottom:16px;
                ">
                    Read the documentation or contact support for assistance.
                </div>

            </div>
            """, unsafe_allow_html=True)

            st.button(
                "View documentation",
                use_container_width=True,
            )

    # =====================================================
    # DATA SCOPE TAB
    # =====================================================
    with tab_scope:

        role = st.selectbox(
            "Configure data scope for role",
            ["guest", "viewer", "privileged"],
            key="scope_role",
        )

        st.caption(
            "Leave selections empty to allow all available values."
        )

        regions, countries, years = [], [], []

        if data is not None and not getattr(data, "empty", True):

            if "region" in data.columns:
                regions = sorted(
                    data["region"].dropna().astype(str).unique()
                )

            if "alert-country" in data.columns:
                countries = sorted(
                    data["alert-country"].dropna().astype(str).unique()
                )

            if "year" in data.columns:
                years = sorted(
                    [int(y) for y in data["year"].dropna().unique()]
                )

        col1, col2, col3 = st.columns(3)

        with col1:
            config[role]["regions"] = st.multiselect(
                "Allowed regions",
                regions,
                default=[
                    x for x in config[role].get("regions", [])
                    if x in regions
                ],
                key=f"persist_regions_{role}",
            )

        with col2:
            config[role]["countries"] = st.multiselect(
                "Allowed countries",
                countries,
                default=[
                    x for x in config[role].get("countries", [])
                    if x in countries
                ],
                key=f"persist_countries_{role}",
            )

        with col3:
            config[role]["years"] = st.multiselect(
                "Allowed years",
                years,
                default=[
                    x for x in config[role].get("years", [])
                    if x in years
                ],
                key=f"persist_years_{role}",
            )

        if st.button(
            "💾 Save data scope",
            type="primary",
            use_container_width=True,
        ):

            if save_access_config(config):
                st.success("Data scope saved.")
                st.rerun()

    # =====================================================
    # USERS TAB
    # =====================================================
    with tab_users:

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

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.warning(
                "No admin emails or privileged domains configured."
            )

        st.code("""
[auth]
admin_emails = ["admin@example.org"]

[access]
privileged_domains = ["icarda.org", "cgiar.org"]

[access_control]
config_path = "/exports/eusee_access_config.json"
""", language="toml")

    # =====================================================
    # DIAGNOSTICS TAB
    # =====================================================
    with tab_system:

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
            "Use this if guest permissions do not update after saving."
        )

        if st.button(
            "🧹 Reset access config and rebuild permissions",
            type="primary",
            use_container_width=True,
        ):

            if reset_access_config():
                st.success(
                    "Access config reset successfully."
                )
                st.rerun()

        st.subheader("Loaded access config")

        st.json(load_access_config())

    st.markdown("""
    <div class="admin-footer">
        © 2026 EU SEE Project. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
