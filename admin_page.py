# admin_page.py
"""
EU SEE Dashboard Admin Page.

Admin access is controlled by [auth].admin_emails in .streamlit/secrets.toml.
This UI lets admins review configured users and set runtime visibility templates.
For persistent multi-user controls, connect the same settings to Firestore.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from authz import (
    get_current_email,
    get_admin_emails,
    get_power_user_emails,
    get_current_role,
    is_admin,
)


def render_admin_sidebar_navigation():
    """Compatibility helper; app.py now uses a premium toggle instead."""
    if not is_admin():
        return "Dashboard"
    return "Admin" if st.session_state.get("admin_mode", False) else "Dashboard"


def _admin_css():
    st.markdown(
        """
        <style>
        .admin-hero {
            background:
                radial-gradient(circle at 95% 5%, rgba(255,219,88,.22), transparent 30%),
                linear-gradient(135deg, #FFFFFF 0%, #F7ECFB 62%, #EFFBFE 100%);
            border: 1px solid rgba(102,0,148,.16);
            border-radius: 22px;
            padding: 20px 22px;
            box-shadow: 0 16px 38px rgba(16,24,40,.08);
            font-family: Arial, sans-serif;
            margin-bottom: 16px;
        }
        .admin-eyebrow { font-size: 10px; font-weight: 950; color: #660094; letter-spacing: .14em; text-transform: uppercase; margin-bottom: 6px; }
        .admin-title { font-size: 28px; font-weight: 950; color: #23152F; letter-spacing: -.03em; line-height: 1.05; font-family: Arial Black, Arial, sans-serif; }
        .admin-note { font-size: 12px; color: #667085; max-width: 850px; line-height: 1.45; margin-top: 8px; }
        .admin-pill-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
        .admin-pill { border-radius: 999px; padding: 6px 10px; font-size: 10.5px; font-weight: 900; background: #F4EAF8; color: #660094; border: 1px solid #E7D4F1; }
        .admin-section-card { background: #FFFFFF; border: 1px solid #E6E8EF; border-radius: 18px; padding: 15px; box-shadow: 0 10px 24px rgba(16,24,40,.055); margin-bottom: 14px; font-family: Arial, sans-serif; }
        .admin-section-title { font-size: 15px; font-weight: 950; color: #23152F; margin-bottom: 4px; }
        .admin-section-subtitle { font-size: 11px; color: #667085; line-height: 1.35; margin-bottom: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _feature_labels():
    return {
        "view_dashboard": "Dashboard access",
        "view_overview": "Overview tab",
        "view_negative_alerts": "Negative Alerts tab",
        "view_maps": "Maps and spatial views",
        "view_data_table": "Data preview tables",
        "view_country_counts": "Country count KPIs",
        "download_data": "CSV/data downloads",
        "use_ai_copilot": "AI Copilot",
        "view_admin_page": "Admin page",
    }


def _load_runtime_config():
    if "admin_runtime_config" not in st.session_state:
        st.session_state["admin_runtime_config"] = {
            "viewer": {
                "features": {
                    "view_dashboard": True,
                    "view_overview": True,
                    "view_negative_alerts": False,
                    "view_maps": True,
                    "view_data_table": False,
                    "view_country_counts": False,
                    "download_data": False,
                    "use_ai_copilot": False,
                    "view_admin_page": False,
                },
                "regions": [],
                "countries": [],
                "years": [],
            },
            "analyst": {
                "features": {
                    "view_dashboard": True,
                    "view_overview": True,
                    "view_negative_alerts": True,
                    "view_maps": True,
                    "view_data_table": True,
                    "view_country_counts": True,
                    "download_data": True,
                    "use_ai_copilot": True,
                    "view_admin_page": False,
                },
                "regions": [],
                "countries": [],
                "years": [],
            },
        }
    return st.session_state["admin_runtime_config"]


def render_admin_page(data=None):
    _admin_css()

    if not is_admin():
        st.error("Access restricted. This page is available only to configured admin emails.")
        st.stop()

    current_email = get_current_email()
    admin_emails = get_admin_emails()
    power_users = get_power_user_emails()
    role = get_current_role()

    st.markdown(
        f"""
        <div class="admin-hero">
            <div class="admin-eyebrow">Admin workspace</div>
            <div class="admin-title">EU SEE Dashboard Administration</div>
            <div class="admin-note">
                Manage dashboard visibility, role-level access, and data-scope settings.
                Admin access is controlled by <code>[auth].admin_emails</code> in Streamlit secrets.
            </div>
            <div class="admin-pill-row">
                <span class="admin-pill">Signed in: {current_email or "unknown"}</span>
                <span class="admin-pill">Role: {role}</span>
                <span class="admin-pill">Admin emails: {len(admin_emails)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_users, tab_visibility, tab_scope, tab_system = st.tabs(["👤 Users", "🎛️ Visibility", "🌍 Data scope", "⚙️ System"])

    with tab_users:
        st.markdown(
            """
            <div class="admin-section-card">
                <div class="admin-section-title">Configured Access</div>
                <div class="admin-section-subtitle">
                    Admin and analyst users are read from secrets. To change them permanently,
                    edit <code>.streamlit/secrets.toml</code> and redeploy.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        rows = []
        for email in admin_emails:
            rows.append({"email": email, "role": "admin", "active": True, "source": "secrets.toml"})
        for email in power_users:
            rows.append({"email": email, "role": "analyst", "active": True, "source": "secrets.toml"})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No admin or power-user emails are configured in secrets.")
        st.info("For live user creation/editing without redeploying, store user profiles in Firestore.")

    with tab_visibility:
        cfg = _load_runtime_config()
        labels = _feature_labels()
        st.markdown(
            """
            <div class="admin-section-card">
                <div class="admin-section-title">Select what other users can see</div>
                <div class="admin-section-subtitle">
                    Configure runtime templates for viewer and analyst roles. Admins always retain full access.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        role_to_edit = st.selectbox("Role to configure", ["viewer", "analyst"], index=0)
        feature_state = cfg[role_to_edit]["features"]
        c1, c2 = st.columns(2)
        for i, key in enumerate(labels.keys()):
            with (c1 if i % 2 == 0 else c2):
                feature_state[key] = st.checkbox(labels[key], value=bool(feature_state.get(key, False)), key=f"admin_feature_{role_to_edit}_{key}")
        cfg[role_to_edit]["features"] = feature_state
        st.session_state["admin_runtime_config"] = cfg
        st.download_button(
            "⬇️ Download visibility config JSON",
            data=json.dumps(cfg, indent=2),
            file_name="eusee_visibility_config.json",
            mime="application/json",
            use_container_width=True,
        )

    with tab_scope:
        cfg = _load_runtime_config()
        role_to_scope = st.selectbox("Role scope to configure", ["viewer", "analyst"], index=0, key="scope_role")
        all_regions, all_countries, all_years = [], [], []
        if data is not None and not getattr(data, "empty", True):
            if "region" in data.columns:
                all_regions = sorted([x for x in data["region"].dropna().astype(str).unique() if x])
            if "alert-country" in data.columns:
                all_countries = sorted([x for x in data["alert-country"].dropna().astype(str).unique() if x])
            if "year" in data.columns:
                all_years = sorted([int(x) for x in data["year"].dropna().unique() if pd.notna(x)])
        st.markdown(
            """
            <div class="admin-section-card">
                <div class="admin-section-title">Data-scope controls</div>
                <div class="admin-section-subtitle">Leave selections empty to allow all values for that role.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cfg[role_to_scope]["regions"] = st.multiselect("Allowed regions", all_regions, default=cfg[role_to_scope].get("regions", []), key=f"scope_regions_{role_to_scope}")
        cfg[role_to_scope]["countries"] = st.multiselect("Allowed countries", all_countries, default=cfg[role_to_scope].get("countries", []), key=f"scope_countries_{role_to_scope}")
        cfg[role_to_scope]["years"] = st.multiselect("Allowed years", all_years, default=cfg[role_to_scope].get("years", []), key=f"scope_years_{role_to_scope}")
        st.session_state["admin_runtime_config"] = cfg
        st.json(cfg[role_to_scope])

    with tab_system:
        st.markdown(
            """
            <div class="admin-section-card">
                <div class="admin-section-title">System diagnostics</div>
                <div class="admin-section-subtitle">Use this to confirm why the Admin toggle appears or does not appear.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.json({
            "current_email": current_email,
            "is_admin": is_admin(),
            "role": role,
            "admin_email_count": len(admin_emails),
            "power_user_count": len(power_users),
            "session_email": st.session_state.get("email"),
            "session_user": st.session_state.get("user"),
            "session_email_verified": st.session_state.get("email_verified"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })
        st.code('[auth]\nadmin_emails = ["your-login-email@example.com"]', language="toml")
