# admin_page.py
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from authz import (
    get_admin_emails,
    get_current_email,
    get_current_role,
    get_privileged_domains,
    is_admin,
    load_visibility_config,
    save_visibility_config,
)

PERMISSION_LABELS = {
    "view_public_summary": "Public summary access",
    "view_overview": "Overview tab",
    "view_negative_alerts": "Negative Alerts tab",
    "view_map": "Visualization Map tab",
    "view_manual": "User Manual tab",
    "view_data_table": "Data preview tables",
    "view_country_counts": "Country-count KPIs",
    "download_data": "CSV/data downloads",
    "use_ai_copilot": "AI Copilot",
    "view_admin_page": "Admin page",
}

EDITABLE_ROLES = ["guest", "viewer", "privileged"]


def render_admin_sidebar_navigation():
    st.sidebar.markdown(
        f"""
        <div style="
            margin-top:14px;
            padding:14px;
            border-radius:17px;
            background:radial-gradient(circle at 92% 0%,rgba(255,219,88,.22),transparent 30%),linear-gradient(135deg,#FFFFFF 0%,#F4EAF8 72%,#EFFBFE 100%);
            border:1px solid rgba(102,0,148,.18);
            box-shadow:0 12px 28px rgba(16,24,40,.08);
            font-family:Arial,sans-serif;">
            <div style="font-size:9px;font-weight:950;color:#660094;letter-spacing:.13em;text-transform:uppercase;">Admin workspace</div>
            <div style="font-size:14px;font-weight:950;color:#23152F;margin-top:4px;">🔐 Administration</div>
            <div style="font-size:10.8px;color:#667085;line-height:1.35;margin-top:5px;">Manage guest, viewer, and privileged visibility.</div>
            <div style="margin-top:8px;border-radius:999px;padding:5px 8px;background:#FFFFFF;border:1px solid #E7D4F1;color:#660094;font-size:9.6px;font-weight:900;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{get_current_email()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.sidebar.radio(
        "Admin navigation",
        ["Dashboard", "Admin"],
        index=0,
        key="admin_navigation_radio",
        label_visibility="collapsed",
    )


def _admin_css():
    st.markdown(
        """
        <style>
        .admin-hero {background:radial-gradient(circle at 95% 5%,rgba(255,219,88,.22),transparent 30%),linear-gradient(135deg,#FFFFFF 0%,#F7ECFB 62%,#EFFBFE 100%);border:1px solid rgba(102,0,148,.16);border-radius:22px;padding:20px 22px;box-shadow:0 16px 38px rgba(16,24,40,.08);font-family:Arial,sans-serif;margin-bottom:16px;}
        .admin-eyebrow {font-size:10px;font-weight:950;color:#660094;letter-spacing:.14em;text-transform:uppercase;margin-bottom:6px;}
        .admin-title {font-size:28px;font-weight:950;color:#23152F;letter-spacing:-.03em;line-height:1.05;font-family:Arial Black,Arial,sans-serif;}
        .admin-note {font-size:12px;color:#667085;max-width:850px;line-height:1.45;margin-top:8px;}
        .admin-pill-row {display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;}
        .admin-pill {border-radius:999px;padding:6px 10px;font-size:10.5px;font-weight:900;background:#F4EAF8;color:#660094;border:1px solid #E7D4F1;}
        .admin-card {background:#FFFFFF;border:1px solid #E6E8EF;border-radius:18px;padding:15px;box-shadow:0 10px 24px rgba(16,24,40,.055);margin-bottom:14px;font-family:Arial,sans-serif;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _options_from_data(data, col):
    if data is None or getattr(data, "empty", True) or col not in data.columns:
        return []
    vals = data[col].dropna().unique().tolist()
    return sorted(vals, key=lambda x: str(x).lower())


def render_admin_page(data=None):
    if not is_admin():
        st.error("Access restricted. This page is available only to configured admin emails.")
        st.stop()

    _admin_css()
    cfg = load_visibility_config()

    st.markdown(
        f"""
        <div class="admin-hero">
            <div class="admin-eyebrow">Admin workspace</div>
            <div class="admin-title">EU SEE Dashboard Administration</div>
            <div class="admin-note">Configure what guest, viewer, and privileged users can see. Admin access is still restricted to specific emails in <code>[auth].admin_emails</code>.</div>
            <div class="admin-pill-row">
                <span class="admin-pill">Signed in: {get_current_email()}</span>
                <span class="admin-pill">Role: {get_current_role()}</span>
                <span class="admin-pill">Admin emails: {len(get_admin_emails())}</span>
                <span class="admin-pill">Privileged domains: {len(get_privileged_domains())}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_visibility, tab_scope, tab_users, tab_system = st.tabs(["🎛️ Visibility", "🌍 Data scope", "👤 Access lists", "⚙️ System"])

    with tab_visibility:
        st.markdown('<div class="admin-card"><b>Select what each role can see</b><br><span style="color:#667085;font-size:12px;">Guest = not logged in. Viewer = logged in but non-approved domain. Privileged = logged in from approved domain.</span></div>', unsafe_allow_html=True)
        role = st.selectbox("Role to configure", EDITABLE_ROLES, index=0)
        role_perms = cfg["permissions"].setdefault(role, {})
        c1, c2 = st.columns(2)
        for i, (perm, label) in enumerate(PERMISSION_LABELS.items()):
            if perm == "view_admin_page":
                continue
            with (c1 if i % 2 == 0 else c2):
                role_perms[perm] = st.checkbox(label, value=bool(role_perms.get(perm, False)), key=f"perm_{role}_{perm}")
        cfg["permissions"][role] = role_perms
        if st.button("💾 Save visibility settings", use_container_width=True):
            if save_visibility_config(cfg):
                st.success("Visibility settings saved.")

    with tab_scope:
        st.markdown('<div class="admin-card"><b>Optional data scope</b><br><span style="color:#667085;font-size:12px;">Leave blank to allow all values for that role.</span></div>', unsafe_allow_html=True)
        role = st.selectbox("Role to scope", EDITABLE_ROLES, index=0, key="scope_role")
        scope = cfg["data_scope"].setdefault(role, {"regions": [], "countries": [], "years": []})
        regions = _options_from_data(data, "region")
        countries = _options_from_data(data, "alert-country")
        years = _options_from_data(data, "year")
        scope["regions"] = st.multiselect("Allowed regions", regions, default=[x for x in scope.get("regions", []) if x in regions], key=f"regions_{role}")
        scope["countries"] = st.multiselect("Allowed countries", countries, default=[x for x in scope.get("countries", []) if x in countries], key=f"countries_{role}")
        scope["years"] = st.multiselect("Allowed years", years, default=[x for x in scope.get("years", []) if x in years], key=f"years_{role}")
        cfg["data_scope"][role] = scope
        if st.button("💾 Save data scope", use_container_width=True):
            if save_visibility_config(cfg):
                st.success("Data scope saved.")

    with tab_users:
        rows = []
        for email in get_admin_emails():
            rows.append({"email/domain": email, "type": "admin email", "source": "secrets.toml"})
        for domain in get_privileged_domains():
            rows.append({"email/domain": domain, "type": "privileged domain", "source": "secrets.toml"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.info("To change admin emails or privileged domains permanently, edit .streamlit/secrets.toml and redeploy.")

    with tab_system:
        st.json({
            "current_email": get_current_email(),
            "current_role": get_current_role(),
            "config_saved_at": datetime.utcnow().isoformat() + "Z",
            "config_path": "admin_visibility_config.json",
            "permissions": cfg.get("permissions", {}),
            "data_scope": cfg.get("data_scope", {}),
        })
        st.download_button("⬇️ Download current access config", data=json.dumps(cfg, indent=2), file_name="admin_visibility_config.json", mime="application/json", use_container_width=True)
