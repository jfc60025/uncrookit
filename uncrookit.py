"""
Uncrookit — Cook County Property Tax Equity Tool
================================================
Compares a Subject PIN against the 5 most similar neighboring properties
based on price per square foot to surface tax uniformity violations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sodapy import Socrata
from typing import Optional
import re

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Uncrookit · Cook County Tax Equity",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS  (editorial / utility aesthetic)
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');

/* ── base ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0f14; color: #e8e3d9; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #13161f;
    border-right: 1px solid #2a2d38;
}
[data-testid="stSidebar"] label { color: #9b98a0 !important; font-size: 0.78rem; letter-spacing: .06em; text-transform: uppercase; }

/* ── headings ── */
h1 { font-family: 'Syne', sans-serif; font-size: 2.6rem !important; font-weight: 800;
     background: linear-gradient(135deg, #f5c842 0%, #f08030 100%);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0 !important; }
h2, h3 { font-family: 'Syne', sans-serif; color: #f5c842; }

/* ── metric cards ── */
.metric-card {
    background: #181b25;
    border: 1px solid #2a2d38;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #f5c842, #f08030);
}
.metric-label { font-size: 0.72rem; letter-spacing: .08em; text-transform: uppercase; color: #6b6876; margin-bottom: .3rem; }
.metric-value { font-family: 'DM Mono', monospace; font-size: 1.8rem; font-weight: 500; color: #f5c842; }
.metric-sub   { font-size: 0.78rem; color: #6b6876; margin-top: .2rem; }

/* ── savings card (highlight) ── */
.savings-card {
    background: linear-gradient(135deg, #1a1f0e 0%, #1c2010 100%);
    border: 1px solid #4a7a20;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative; overflow: hidden;
}
.savings-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #7ec832, #4a7a20);
}
.savings-value { font-family: 'DM Mono', monospace; font-size: 1.8rem; font-weight: 500; color: #7ec832; }

/* ── warning card ── */
.warn-card {
    background: #1a1510;
    border: 1px solid #7a4a20;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}

/* ── data table ── */
.stDataFrame { border: 1px solid #2a2d38 !important; border-radius: 8px !important; }
[data-testid="stDataFrame"] th { background: #181b25 !important; color: #9b98a0 !important;
    font-family: 'DM Mono', monospace; font-size: 0.72rem; letter-spacing: .06em; }
[data-testid="stDataFrame"] td { font-family: 'DM Mono', monospace; font-size: 0.82rem; }

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #f5c842, #f08030);
    color: #0d0f14; font-weight: 700; border: none; border-radius: 8px;
    padding: .6rem 1.4rem; font-family: 'DM Sans', sans-serif;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .85; }

/* ── dividers ── */
hr { border-color: #2a2d38; }

/* ── tags ── */
.tag { display: inline-block; background: #1e2130; border: 1px solid #2a2d38;
       border-radius: 4px; padding: .15rem .5rem; font-size: 0.72rem;
       font-family: 'DM Mono', monospace; color: #9b98a0; margin-right: .3rem; }

/* ── section header ── */
.section-header {
    border-left: 3px solid #f5c842;
    padding-left: .75rem;
    margin: 1.5rem 0 1rem;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #e8e3d9;
    letter-spacing: .03em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CONSTANTS & MAPPINGS
# ──────────────────────────────────────────────
SOCRATA_DOMAIN = "data.cookcountyil.gov"
CHAR_DATASET   = "x54s-btds"   # Improvements / Characteristics
VAL_DATASET    = "uzyt-m557"   # Assessed Values
ADDR_DATASET   = "3723-97qp"   # Parcel Addresses

PROPERTY_CLASS_MAP = {
    "202": "Single-Family Residence",
    "203": "Two-Family Residence (2-Flat)",
    "204": "Three-Family Residence (3-Flat)",
    "205": "Four-Family Residence (4-Flat)",
    "206": "Five-or-More-Family Residence",
    "207": "Single-Family w/ Commercial",
    "208": "Two-Family w/ Commercial",
    "209": "Coach House",
    "210": "Condo Unit",
    "211": "Condo Apt (>6 stories)",
    "212": "Townhouse",
    "278": "Incentive Residential",
    "295": "Residential Leaseback",
    "299": "Misc. Residential",
    "300": "Vacant Land – Residential",
}

GARAGE_MAP = {
    "0": "Not Recorded",
    "1": "Frame",
    "2": "Masonry",
    "3": "Frame/Masonry",
    "4": "Loft",
    "5": "Attached",
    "6": "Detached",
    "7": "Built-In",
    "8": "Carport",
}

BASEMENT_MAP = {
    "0": "Not Recorded",
    "1": "Full",
    "2": "Slab",
    "3": "Partial",
    "4": "Crawl",
    "5": "Walk-Out",
}

EXTERIOR_MAP = {
    "1": "Wood",
    "2": "Masonry",
    "3": "Wood/Masonry",
    "4": "Stucco",
}

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def fmt_currency(val: float) -> str:
    return f"${val:,.0f}"

def fmt_rate(val: float) -> str:
    return f"${val:.2f}/SF"

def safe_map(code, mapping: dict, fallback: str = "Not Recorded") -> str:
    if pd.isna(code) or str(code).strip() in ("", "0", "nan"):
        return fallback
    return mapping.get(str(int(float(code))), str(code))

def normalize_pin(pin: str) -> str:
    """Strip dashes/spaces from PIN."""
    return re.sub(r"[\s\-]", "", pin.strip())

def pin_prefix(pin: str, length: int = 7) -> str:
    """Return township+section prefix (first 7 digits)."""
    return normalize_pin(pin)[:length]

# ──────────────────────────────────────────────
# SCHEMA DISCOVERY
# ──────────────────────────────────────────────
ADDR_COLUMN_CANDIDATES = [
    "property_address", "address", "addr",
    "prop_address", "mail_address", "site_address",
]

@st.cache_data(ttl=3600, show_spinner=False)
def discover_address_column(client: Socrata) -> Optional[str]:
    """
    Try each candidate column name against the address dataset.
    Returns the first that succeeds, or None.
    """
    for col in ADDR_COLUMN_CANDIDATES:
        try:
            results = client.get(ADDR_DATASET, limit=1, select=col)
            if results:
                return col
        except Exception:
            continue
    return None

# ──────────────────────────────────────────────
# DATA FETCHERS
# ──────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_characteristics(pin: str) -> Optional[pd.DataFrame]:
    """Fetch building characteristics for a single PIN."""
    try:
        client = Socrata(SOCRATA_DOMAIN, None)
        # Single-quote PIN in SOQL to avoid Bad Request
        results = client.get(
            CHAR_DATASET,
            where=f"pin = '{pin}'",
            limit=5,
        )
        return pd.DataFrame.from_records(results) if results else None
    except Exception as e:
        st.error(f"Characteristics API error: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_assessed_value(pin: str) -> Optional[pd.DataFrame]:
    """Fetch most-recent assessed value for a PIN."""
    try:
        client = Socrata(SOCRATA_DOMAIN, None)
        results = client.get(
            VAL_DATASET,
            where=f"pin = '{pin}'",
            order="tax_year DESC",
            limit=5,
        )
        return pd.DataFrame.from_records(results) if results else None
    except Exception as e:
        st.error(f"Values API error: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_neighborhood_chars(prefix: str, sf: float) -> Optional[pd.DataFrame]:
    """
    Fetch characteristics for all PINs sharing the same township/section prefix.
    Filter to within ±15% of subject square footage.
    """
    low_sf  = sf * 0.85
    high_sf = sf * 1.15
    try:
        client = Socrata(SOCRATA_DOMAIN, None)
        results = client.get(
            CHAR_DATASET,
            where=(
                f"pin like '{prefix}%' "
                f"AND bldg_sf >= '{low_sf:.0f}' "
                f"AND bldg_sf <= '{high_sf:.0f}'"
            ),
            limit=500,
        )
        return pd.DataFrame.from_records(results) if results else None
    except Exception as e:
        st.error(f"Neighborhood chars API error: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_neighborhood_values(pins: list) -> Optional[pd.DataFrame]:
    """Fetch assessed values for a list of PINs (batch)."""
    if not pins:
        return None
    # Build strict single-quoted IN list
    quoted = ", ".join(f"'{p}'" for p in pins)
    try:
        client = Socrata(SOCRATA_DOMAIN, None)
        results = client.get(
            VAL_DATASET,
            where=f"pin IN ({quoted})",
            order="tax_year DESC",
            limit=2000,
        )
        return pd.DataFrame.from_records(results) if results else None
    except Exception as e:
        st.error(f"Neighborhood values API error: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_address(pin: str) -> str:
    """Fetch address with schema-discovery fallback."""
    try:
        client = Socrata(SOCRATA_DOMAIN, None)
        addr_col = discover_address_column(client)
        if addr_col is None:
            return "Address unavailable"
        results = client.get(
            ADDR_DATASET,
            where=f"pin = '{pin}'",
            select=addr_col,
            limit=1,
        )
        if results:
            return results[0].get(addr_col, "Address unavailable")
    except Exception:
        pass
    return "Address unavailable"

# ──────────────────────────────────────────────
# CORE ANALYSIS ENGINE
# ──────────────────────────────────────────────
def run_analysis(
    subject_pin: str,
    current_assessment: float,
    subject_sf: float,
) -> dict:
    """
    Main analysis pipeline.
    Returns a dict with all results or raises on fatal error.
    """
    result = {}

    # 1. Subject characteristics
    with st.spinner("Fetching subject property characteristics…"):
        char_df = fetch_characteristics(subject_pin)
    if char_df is None or char_df.empty:
        raise ValueError(f"No characteristics found for PIN {subject_pin}. Check the PIN and try again.")

    subj_char = char_df.iloc[0]
    result["char"] = subj_char

    # 2. Subject assessed value
    with st.spinner("Fetching subject assessed value…"):
        val_df = fetch_assessed_value(subject_pin)

    if val_df is not None and not val_df.empty:
        subj_val = val_df.iloc[0]
        result["official_assessment"] = float(subj_val.get("assessed_value", current_assessment) or current_assessment)
        result["tax_year"] = subj_val.get("tax_year", "N/A")
    else:
        result["official_assessment"] = current_assessment
        result["tax_year"] = "N/A"

    # 3. Address
    with st.spinner("Looking up address…"):
        result["address"] = fetch_address(subject_pin)

    # 4. Neighborhood search
    prefix = pin_prefix(subject_pin)
    with st.spinner(f"Searching neighborhood (prefix {prefix}, ±15% SF)…"):
        nbr_chars = fetch_neighborhood_chars(prefix, subject_sf)

    if nbr_chars is None or nbr_chars.empty:
        raise ValueError("No neighboring properties found. Try a different PIN or provide SF manually.")

    # Drop subject from neighbors
    nbr_chars = nbr_chars[nbr_chars["pin"] != subject_pin].copy()

    # 5. Neighborhood values
    nbr_pins = nbr_chars["pin"].unique().tolist()
    with st.spinner(f"Fetching values for {len(nbr_pins)} neighboring properties…"):
        nbr_vals = fetch_neighborhood_values(nbr_pins)

    if nbr_vals is None or nbr_vals.empty:
        raise ValueError("Could not retrieve assessed values for neighbors.")

    # Keep most-recent year per PIN
    if "tax_year" in nbr_vals.columns:
        nbr_vals["tax_year"] = pd.to_numeric(nbr_vals["tax_year"], errors="coerce")
        nbr_vals = (
            nbr_vals.sort_values("tax_year", ascending=False)
            .groupby("pin", as_index=False)
            .first()
        )

    # 6. Merge chars + values
    merged = nbr_chars.merge(nbr_vals[["pin", "assessed_value"]], on="pin", how="inner")

    # Coerce numerics
    for col in ["bldg_sf", "assessed_value"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.dropna(subset=["bldg_sf", "assessed_value"])
    merged = merged[merged["bldg_sf"] > 0]
    merged = merged[merged["assessed_value"] > 0]

    if merged.empty:
        raise ValueError("No valid comparable properties after merging characteristics and values.")

    # 7. Rate calculation
    merged["bldg_assess_per_sf"] = merged["assessed_value"] / merged["bldg_sf"]
    merged_sorted = merged.sort_values("bldg_assess_per_sf").reset_index(drop=True)
    top5 = merged_sorted.head(5).copy()

    result["top5"]            = top5
    result["all_neighbors"]   = merged_sorted
    result["avg_rate"]        = top5["bldg_assess_per_sf"].mean()
    result["justified_value"] = result["avg_rate"] * subject_sf
    result["potential_savings"] = max(0, current_assessment - result["justified_value"])
    result["subject_rate"]    = current_assessment / subject_sf if subject_sf > 0 else np.nan
    result["subject_sf"]      = subject_sf
    result["subject_pin"]     = subject_pin
    result["current_assessment"] = current_assessment

    return result

# ──────────────────────────────────────────────
# UI COMPONENTS
# ──────────────────────────────────────────────
def render_metric(label: str, value: str, sub: str = "", savings: bool = False):
    card_class = "savings-card" if savings else "metric-card"
    val_class  = "savings-value" if savings else "metric-value"
    st.markdown(f"""
    <div class="{card_class}">
        <div class="metric-label">{label}</div>
        <div class="{val_class}">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)

def render_neighbor_table(top5: pd.DataFrame, show_raw: bool):
    display = top5[["pin", "bldg_sf", "assessed_value", "bldg_assess_per_sf"]].copy()
    display.columns = ["PIN", "Building SF", "Assessed Value", "Rate ($/SF)"]

    if not show_raw:
        display["Building SF"]    = display["Building SF"].apply(lambda x: f"{x:,.0f}")
        display["Assessed Value"] = display["Assessed Value"].apply(lambda x: fmt_currency(x))
        display["Rate ($/SF)"]    = display["Rate ($/SF)"].apply(lambda x: f"${x:.2f}")

    # Optional extra columns if present
    for col, label, mapping in [
        ("garage_indicator", "Garage",   GARAGE_MAP),
        ("basement",         "Basement", BASEMENT_MAP),
        ("exterior",         "Exterior", EXTERIOR_MAP),
    ]:
        if col in top5.columns:
            if show_raw:
                display[label] = top5[col]
            else:
                display[label] = top5[col].apply(lambda x: safe_map(x, mapping))

    st.dataframe(display, use_container_width=True, hide_index=True)

def render_subject_details(char, show_raw: bool):
    cols = st.columns(4)
    fields = [
        ("Property Class", "class", PROPERTY_CLASS_MAP),
        ("Year Built",     "year_built", None),
        ("Garage",         "garage_indicator", GARAGE_MAP),
        ("Basement",       "basement", BASEMENT_MAP),
        ("Exterior",       "exterior", EXTERIOR_MAP),
        ("Rooms",          "num_rooms", None),
        ("Bedrooms",       "num_bedrooms", None),
        ("Full Baths",     "num_bathrooms_full", None),
    ]
    rendered = 0
    for label, key, mapping in fields:
        raw = char.get(key, None)
        if raw is None:
            continue
        display_val = str(raw) if (show_raw or mapping is None) else safe_map(raw, mapping)
        with cols[rendered % 4]:
            st.markdown(f"""
            <div class="metric-card" style="padding:.8rem 1rem;margin-bottom:.6rem;">
                <div class="metric-label">{label}</div>
                <div style="font-family:'DM Mono',monospace;font-size:1rem;color:#e8e3d9;">{display_val}</div>
            </div>""", unsafe_allow_html=True)
        rendered += 1

# ──────────────────────────────────────────────
# CSV EXPORT
# ──────────────────────────────────────────────
def build_export_csv(result: dict) -> bytes:
    top5 = result["top5"].copy()
    export_cols = ["pin", "bldg_sf", "assessed_value", "bldg_assess_per_sf"]
    available   = [c for c in export_cols if c in top5.columns]
    export = top5[available].rename(columns={
        "pin": "Comparable PIN",
        "bldg_sf": "Building SF",
        "assessed_value": "Assessed Value",
        "bldg_assess_per_sf": "Rate ($/SF)",
    })
    export.loc["SUBJECT"] = {
        "Comparable PIN":  result["subject_pin"],
        "Building SF":     result["subject_sf"],
        "Assessed Value":  result["current_assessment"],
        "Rate ($/SF)":     result["subject_rate"],
    }
    export.loc["AVG RATE"]  = {"Rate ($/SF)": result["avg_rate"]}
    export.loc["JUSTIFIED"] = {"Assessed Value": result["justified_value"]}
    export.loc["SAVINGS"]   = {"Assessed Value": result["potential_savings"]}
    return export.to_csv(index=True).encode("utf-8")

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Uncrookit")
    st.markdown("<p style='color:#6b6876;font-size:.85rem;margin-top:-.5rem;'>Cook County Property Tax Equity</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### Subject Property")
    subject_pin = st.text_input(
        "Property Index Number (PIN)",
        placeholder="e.g. 10-12-345-006-0000",
        help="Enter the 14-digit Cook County PIN (dashes optional).",
    )
    current_assessment = st.number_input(
        "Your Current Assessment ($)",
        min_value=0,
        step=1000,
        value=0,
        format="%d",
        help="Find this on your assessment notice.",
    )
    subject_sf = st.number_input(
        "Building Square Footage",
        min_value=0,
        step=50,
        value=0,
        format="%d",
        help="Above-grade finished square footage.",
    )

    st.divider()
    st.markdown("### Display Options")
    show_raw = st.toggle(
        "Show Raw County Codes",
        value=False,
        help="Toggle between human-readable labels and raw data codes.",
    )

    st.divider()
    analyze_btn = st.button("⚖️ Analyze Equity", use_container_width=True)

    st.markdown("""
    <p style='color:#3a3d4a;font-size:.72rem;margin-top:1.5rem;'>
    Data sourced from Cook County Open Data via Socrata API.<br>
    Not legal advice. For informational use only.
    </p>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MAIN PANEL
# ──────────────────────────────────────────────
st.markdown("# ⚖️ Uncrookit")
st.markdown(
    "<p style='color:#9b98a0;margin-top:-.5rem;margin-bottom:1.5rem;'>"
    "Uncover property tax inequity in Cook County, IL — powered by open assessment data."
    "</p>",
    unsafe_allow_html=True,
)

if not analyze_btn:
    # Landing state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">How It Works</div>
            <div style="color:#e8e3d9;font-size:.9rem;line-height:1.7;margin-top:.5rem;">
                Enter your PIN, current assessment, and building SF. 
                We find the 5 most similar nearby properties and 
                calculate whether you're assessed fairly.
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">The Math</div>
            <div style="color:#e8e3d9;font-size:.9rem;line-height:1.7;margin-top:.5rem;">
                <span style="font-family:'DM Mono',monospace;color:#f5c842;">
                Justified = Avg Top-5 Rate × Your SF
                </span><br>
                The "rate" is Assessment ÷ Square Footage. 
                Lower rate neighbors reveal your overassessment.
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Data Sources</div>
            <div style="color:#e8e3d9;font-size:.9rem;line-height:1.7;margin-top:.5rem;">
                <span class="tag">x54s-btds</span> Characteristics<br>
                <span class="tag">uzyt-m557</span> Assessed Values<br>
                <span class="tag">3723-97qp</span> Addresses
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-card" style="margin-top:1.5rem;">
        <span style="color:#f5c842;">ℹ️</span>
        <span style="color:#9b98a0;font-size:.85rem;">
        Enter your property details in the sidebar and click <strong style="color:#e8e3d9;">Analyze Equity</strong> to begin.
        </span>
    </div>
    """, unsafe_allow_html=True)

else:
    # Validate inputs
    errors = []
    if not subject_pin:
        errors.append("Please enter a PIN.")
    if current_assessment <= 0:
        errors.append("Please enter your current assessment.")
    if subject_sf <= 0:
        errors.append("Please enter the building square footage.")

    if errors:
        for e in errors:
            st.warning(e)
    else:
        clean_pin = normalize_pin(subject_pin)
        try:
            result = run_analysis(clean_pin, float(current_assessment), float(subject_sf))

            # ── Property header ──
            addr = result.get("address", "Address unavailable")
            st.markdown(f"""
            <div style="margin-bottom:1.5rem;">
                <div style="font-family:'Syne',sans-serif;font-size:1.3rem;color:#e8e3d9;">{addr}</div>
                <div style="font-family:'DM Mono',monospace;font-size:.85rem;color:#6b6876;margin-top:.2rem;">
                    PIN {clean_pin} &nbsp;·&nbsp; Tax Year {result['tax_year']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Key metrics ──
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric(
                    "Your Assessment",
                    fmt_currency(result["current_assessment"]),
                    f"{fmt_rate(result['subject_rate'])}",
                )
            with m2:
                render_metric(
                    "Justified Assessment",
                    fmt_currency(result["justified_value"]),
                    f"Avg top-5 rate: {fmt_rate(result['avg_rate'])}",
                )
            with m3:
                render_metric(
                    "Avg Neighbor Rate",
                    fmt_rate(result["avg_rate"]),
                    f"vs your {fmt_rate(result['subject_rate'])}",
                )
            with m4:
                render_metric(
                    "Potential Over-Assessment",
                    fmt_currency(result["potential_savings"]),
                    "if appealed to justified value",
                    savings=result["potential_savings"] > 0,
                )

            # ── Equity verdict ──
            ratio = result["current_assessment"] / result["justified_value"] if result["justified_value"] > 0 else 1.0
            if ratio > 1.15:
                verdict_color = "#f08030"
                verdict_icon  = "🔴"
                verdict_text  = f"Your property appears <strong>over-assessed by {(ratio-1)*100:.1f}%</strong> relative to comparable neighbors. You may have grounds for an appeal."
            elif ratio > 1.05:
                verdict_color = "#f5c842"
                verdict_icon  = "🟡"
                verdict_text  = f"Your assessment is <strong>moderately above</strong> comparable neighbors ({(ratio-1)*100:.1f}% over). Consider reviewing further."
            else:
                verdict_color = "#7ec832"
                verdict_icon  = "🟢"
                verdict_text  = "Your assessment appears <strong>equitable</strong> relative to comparable neighbors."

            st.markdown(f"""
            <div style="background:#181b25;border:1px solid {verdict_color};border-radius:12px;padding:1rem 1.4rem;margin:1rem 0;">
                <span style="font-size:1.1rem;">{verdict_icon}</span>
                <span style="color:#e8e3d9;font-size:.95rem;margin-left:.5rem;">{verdict_text}</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Subject property details ──
            st.markdown('<div class="section-header">Subject Property Details</div>', unsafe_allow_html=True)
            render_subject_details(result["char"], show_raw)

            # ── Top 5 comparables ──
            st.markdown('<div class="section-header">Top 5 Most Favorable Comparable Properties</div>', unsafe_allow_html=True)
            st.caption(
                f"Sorted by lowest assessment rate ($/SF) · "
                f"Filtered to ±15% of your {subject_sf:,.0f} SF · "
                f"{len(result['all_neighbors'])} total neighbors found"
            )
            render_neighbor_table(result["top5"], show_raw)

            # ── Uniformity chart ──
            st.markdown('<div class="section-header">Assessment Rate Distribution</div>', unsafe_allow_html=True)
            chart_data = result["all_neighbors"][["pin", "bldg_assess_per_sf"]].copy()
            chart_data = chart_data.rename(columns={"bldg_assess_per_sf": "Rate ($/SF)"})
            # Add subject as reference line via annotation
            subject_row = pd.DataFrame({
                "pin": [f"YOU ({clean_pin})"],
                "Rate ($/SF)": [result["subject_rate"]],
            })
            chart_combined = pd.concat([chart_data, subject_row], ignore_index=True)
            st.bar_chart(
                chart_combined.set_index("pin")["Rate ($/SF)"],
                use_container_width=True,
                height=220,
                color="#f5c842",
            )
            st.caption("Yellow bars = neighbors. Your rate appears at the far right.")

            # ── All neighbors table (expandable) ──
            with st.expander(f"View All {len(result['all_neighbors'])} Comparable Properties"):
                render_neighbor_table(result["all_neighbors"].head(100), show_raw)

            # ── Export ──
            st.markdown('<div class="section-header">Appeal Evidence Export</div>', unsafe_allow_html=True)
            csv_bytes = build_export_csv(result)
            st.download_button(
                label="📥 Download Appeal Evidence CSV",
                data=csv_bytes,
                file_name=f"uncrookit_appeal_{clean_pin}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(
                "CSV includes the 5 comparables, their rates, your subject property, "
                "the justified assessment, and potential savings — ready for your appeal packet."
            )

        except ValueError as ve:
            st.markdown(f"""
            <div class="warn-card">
                <span style="color:#f08030;font-size:1.1rem;">⚠️</span>
                <span style="color:#e8e3d9;margin-left:.5rem;">{ve}</span>
            </div>
            """, unsafe_allow_html=True)
        except Exception as ex:
            st.error(f"Unexpected error: {ex}")
            st.caption("If this persists, check your network connection or try a different PIN.")
