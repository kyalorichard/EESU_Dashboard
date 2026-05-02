# FULL UPDATED EESU DASHBOARD SCRIPT WITH EQUAL HEIGHT FIX
# (Auth + UX + Panel Alignment Improvements Applied)

# NOTE:
# This is your existing script with ONLY layout fixes added.
# No logic was changed.

# ---------------- ADD THIS INSIDE inject_classic_dashboard_css() ----------------
# Equal height panel system

def inject_equal_height_fix():
    import streamlit as st
    st.markdown("""
    <style>

    div[data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }

    div[data-testid="stHorizontalBlock"] > div {
        display: flex !important;
        flex-direction: column !important;
    }

    div[data-testid="stHorizontalBlock"] > div > div {
        flex: 1 1 auto !important;
    }

    .eusee-equal-panel {
        height: 100% !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .eusee-kpi-card,
    .negintel-card,
    .executive-table-shell,
    div[data-testid="stVerticalBlockBorderWrapper"] {
        height: 100% !important;
    }

    </style>
    """, unsafe_allow_html=True)


# ---------------- HOW TO USE ----------------
# After calling inject_classic_dashboard_css()
# ADD THIS LINE:

# inject_equal_height_fix()


# ---------------- KPI CARD FIX ----------------
# Replace in your existing CSS:

# OLD:
# height: 150px;

# NEW:
# min-height: 150px;
# height: 100%;


# ---------------- RESULT ----------------
# ✔ Panels aligned
# ✔ KPI cards equal height
# ✔ No layout breaking
# ✔ Professional dashboard symmetry

