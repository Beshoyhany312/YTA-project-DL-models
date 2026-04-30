import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from fpdf import FPDF
from datetime import datetime

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DL Electricity Predictor Suite",
    page_icon="⚡",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

[data-testid="stTabs"] [role="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important; font-weight: 700 !important;
    letter-spacing: 0.05em; padding: 0.6rem 1.1rem !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    border-bottom: 3px solid #a78bfa !important; color: #a78bfa !important;
}
[data-testid="stForm"] {
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 14px; padding: 1.5rem;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #161b22, #1c2128);
    border: 1px solid #30363d; border-radius: 12px; padding: 1.1rem 1.4rem;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.85rem !important; font-weight: 700 !important;
    color: #a78bfa !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important; text-transform: uppercase;
    letter-spacing: 0.1em; color: #8b949e !important;
}
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #a78bfa, #ec4899) !important;
    color: #0d1117 !important; font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important; font-size: 0.9rem !important;
    border: none !important; border-radius: 10px !important; width: 100% !important;
}
[data-testid="stFormSubmitButton"] button:hover { opacity: 0.85 !important; }
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #28a745, #20c997) !important;
    color: white !important; font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important; border: none !important;
    border-radius: 10px !important; width: 100% !important;
}
hr { border-color: #21262d; }
.model-badge {
    display: inline-block; background: #161b22; border: 1px solid #30363d;
    border-radius: 20px; padding: 0.25rem 0.9rem; font-size: 0.72rem;
    color: #8b949e; font-family: 'Space Mono', monospace;
    margin-right: 0.4rem; letter-spacing: 0.05em;
}
.result-card {
    background: #161b22; border: 1px solid #30363d; border-left: 4px solid;
    border-radius: 12px; padding: 1rem 1.4rem; margin-top: 0.8rem;
}
.warning-banner {
    background: rgba(255,75,75,0.1); border: 2px solid #ff4b4b;
    border-radius: 12px; padding: 1rem 1.4rem;
    text-align: center; margin-top: 0.8rem;
}
.default-note {
    font-size: 0.78rem; color: #8b949e; font-style: italic;
    margin-top: -0.4rem; margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model 1: Multi-output LSTM → home electricity (kWh + EGP)
LSTM_MULTI_PATH    = os.path.join(BASE_DIR, "multi_output_lstm_model.keras")
FEATURE_SCALER     = os.path.join(BASE_DIR, "feature_scaler.joblib")

# Model 2: Single-output LSTM → smart city site (USD)
LSTM_SITE_PATH     = os.path.join(BASE_DIR, "lstm_model.keras")
SITE_SCALER        = os.path.join(BASE_DIR, "scaler.joblib")

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ── DATASET MEANS (from feature_scaler.joblib) ─────────────────────────────────
MEANS = {
    'number_of_air_conditioners':     2.01,
    'ac_power_hp':                    2.25,
    'number_of_refrigerators':        1.51,
    'number_of_televisions':          1.49,
    'number_of_fans':                 1.99,
    'number_of_computers':            1.00,
    'average_daily_usage_hours':      6.49,
    'house_size_m2':                  132.15,
    'washing_machine_usage_per_week': 3.01,
}

# SITE MEANS (from scaler.joblib)
SITE_MEANS = {
    'Site Area (square meters)':         2743.0,
    'Water Consumption (liters/day)':    3461.7,
    'Recycling Rate (%)':                49.9,
    'Utilisation Rate (%)':              64.7,
    'Air Quality Index (AQI)':           98.8,
    'Issue Resolution Time (hours)':     36.5,
    'Resident Count (number of people)': 84.7,
}

# ── LOADERS ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Multi-Output LSTM…")
def load_multi_lstm():
    missing = [p for p in (LSTM_MULTI_PATH, FEATURE_SCALER) if not os.path.exists(p)]
    if missing:
        st.warning(f"Missing files: {[os.path.basename(p) for p in missing]}")
        return None, None
    try:
        from tensorflow import keras
        model  = keras.models.load_model(LSTM_MULTI_PATH, compile=False)
        scaler = joblib.load(FEATURE_SCALER)
        return model, scaler
    except Exception as e:
        st.error(f"Failed to load Multi-Output LSTM: {e}")
        return None, None

@st.cache_resource(show_spinner="Loading Site LSTM…")
def load_site_lstm():
    missing = [p for p in (LSTM_SITE_PATH, SITE_SCALER) if not os.path.exists(p)]
    if missing:
        st.warning(f"Missing files: {[os.path.basename(p) for p in missing]}")
        return None, None
    try:
        from tensorflow import keras
        model  = keras.models.load_model(LSTM_SITE_PATH, compile=False)
        scaler = joblib.load(SITE_SCALER)
        return model, scaler
    except Exception as e:
        st.error(f"Failed to load Site LSTM: {e}")
        return None, None

# ── PREPROCESSING ──────────────────────────────────────────────────────────────
HOME_COLS = [
    'number_of_air_conditioners', 'ac_power_hp', 'number_of_refrigerators',
    'number_of_televisions', 'number_of_fans', 'number_of_computers',
    'average_daily_usage_hours', 'house_size_m2', 'has_water_heater',
    'washing_machine_usage_per_week', 'season_winter',
    'insulation_quality_low', 'insulation_quality_medium'
]

SITE_COLS = [
    'Site Area (square meters)', 'Water Consumption (liters/day)',
    'Recycling Rate (%)', 'Utilisation Rate (%)', 'Air Quality Index (AQI)',
    'Issue Resolution Time (hours)', 'Resident Count (number of people)',
    'Structure Type_Industrial', 'Structure Type_Mixed-use', 'Structure Type_Residential'
]

def preprocess_home(raw: dict, scaler) -> np.ndarray:
    df = pd.DataFrame([raw])
    if 'season' in df.columns:
        df['season_winter'] = (df['season'].str.lower() == 'winter').astype(int)
        df.drop(columns=['season'], inplace=True)
    else:
        df['season_winter'] = 0
    if 'insulation_quality' in df.columns:
        df['insulation_quality_low']    = (df['insulation_quality'].str.lower() == 'low').astype(int)
        df['insulation_quality_medium'] = (df['insulation_quality'].str.lower() == 'medium').astype(int)
        df.drop(columns=['insulation_quality'], inplace=True)
    else:
        df['insulation_quality_low'] = df['insulation_quality_medium'] = 0
    for col in HOME_COLS:
        if col not in df.columns:
            df[col] = 0
    scaled = scaler.transform(df[HOME_COLS])
    # LSTM expects shape (samples, timesteps=1, features)
    return scaled.reshape(scaled.shape[0], 1, scaled.shape[1])

def preprocess_site(raw: dict, scaler) -> np.ndarray:
    df = pd.DataFrame([raw], columns=SITE_COLS)
    scaled = scaler.transform(df)
    return scaled.reshape(scaled.shape[0], 1, scaled.shape[1])

# ── HELPERS ────────────────────────────────────────────────────────────────────
def monthly_trend(base: float) -> list:
    return [round(base * (1 + 0.25 * np.cos((i - 6) / 1.9)), 2) for i in range(12)]

def plot_trend(values: list, currency: str, color: str) -> str:
    """Renders a responsive SVG area chart — no external library needed."""
    w, h, pad_l, pad_r, pad_t, pad_b = 700, 280, 55, 20, 20, 40
    chart_w = w - pad_l - pad_r
    chart_h = h - pad_t - pad_b
    min_v, max_v = min(values), max(values)
    span = max_v - min_v if max_v != min_v else 1
    def cx(i):   return pad_l + i * chart_w / (len(values) - 1)
    def cy(v):   return pad_t + chart_h - (v - min_v) / span * chart_h
    pts  = " ".join(f"{cx(i):.1f},{cy(v):.1f}" for i, v in enumerate(values))
    area = pts + f" {cx(len(values)-1):.1f},{pad_t+chart_h} {cx(0):.1f},{pad_t+chart_h}"
    grid_lines = ""
    for k in range(5):
        val2 = min_v + k * span / 4
        y    = cy(val2)
        grid_lines += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-pad_r}" y2="{y:.1f}" stroke="#21262d" stroke-width="1"/>'
        grid_lines += f'<text x="{pad_l-6}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="#8b949e">{val2:,.0f}</text>'
    x_labels = ""
    for i, m in enumerate(MONTHS):
        x_labels += f'<text x="{cx(i):.1f}" y="{h-6}" text-anchor="middle" font-size="10" fill="#8b949e">{m}</text>'
    dots = ""
    for i, v in enumerate(values):
        dots += f'<circle cx="{cx(i):.1f}" cy="{cy(v):.1f}" r="4" fill="{color}" stroke="#0d1117" stroke-width="2"/>'
    r2, g2, b2 = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    return f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#0d1117;border-radius:10px;"><defs><linearGradient id="ag" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="rgba({r2},{g2},{b2},0.35)"/><stop offset="100%" stop-color="rgba({r2},{g2},{b2},0.02)"/></linearGradient></defs>{grid_lines}<polygon points="{area}" fill="url(#ag)"/><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>{dots}{x_labels}</svg>'

def zone_egp(val):
    if val < 300:   return "#a78bfa", "Low Consumption Zone",      "Great efficiency — below average for this configuration."
    elif val < 700: return "#FFD700", "Moderate Consumption Zone", "Typical range for a household of this size."
    else:           return "#FF6B6B", "High Consumption Zone",     "Consider energy-saving measures or appliance upgrades."

def zone_usd(val):
    if val < 500:    return "#a78bfa", "Low Consumption Zone",      "This site has a very efficient electricity footprint."
    elif val < 1500: return "#FFD700", "Moderate Consumption Zone", "Typical range for sites of this type and size."
    else:            return "#FF6B6B", "High Consumption Zone",     "Consider energy audits or efficiency improvements."

# ── PDF GENERATORS (no emojis — latin-1 safe) ─────────────────────────────────
def generate_pdf_home(inputs: dict, kwh: float, bill: float, ex_rate: float, budget: float) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(80, 30, 160)
    pdf.cell(0, 12, "Home Electricity Prediction Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Model: Multi-Output LSTM", ln=True, align='C')
    pdf.ln(5)

    # Results
    pdf.set_fill_color(240, 230, 255)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(80, 30, 160)
    pdf.cell(0, 9, "Prediction Results", ln=True, fill=True)
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(95, 9, f"Monthly Consumption: {kwh:,.2f} kWh",  border=1)
    pdf.cell(95, 9, f"Monthly Bill: {bill:,.2f} EGP",        border=1, ln=True)
    pdf.cell(95, 9, f"Annual Bill: {bill*12:,.2f} EGP",      border=1)
    pdf.cell(95, 9, f"USD Equivalent: ${bill/ex_rate:,.2f}", border=1, ln=True)
    pdf.cell(95, 9, f"Daily Average: {bill/30:,.2f} EGP",    border=1)
    pdf.cell(95, 9, f"Exchange Rate: 1 USD = {ex_rate} EGP", border=1, ln=True)
    pdf.ln(4)

    # Budget status
    pdf.set_font("Arial", 'B', 11)
    if bill > budget:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 9, f"STATUS: OVER BUDGET  (Bill: {bill:,.1f} EGP > Limit: {budget:,} EGP)", ln=True)
    else:
        pdf.set_text_color(0, 140, 0)
        pdf.cell(0, 9, f"STATUS: WITHIN BUDGET  (Bill: {bill:,.1f} EGP <= Limit: {budget:,} EGP)", ln=True)
    pdf.ln(4)

    # Monthly forecast table
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(80, 30, 160)
    pdf.cell(0, 9, "12-Month Cost Forecast (EGP)", ln=True)
    pdf.set_font("Arial", '', 9)
    pdf.set_text_color(30, 30, 30)
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    trend  = monthly_trend(bill)
    col_w  = 190 / 6
    for row in range(2):
        for col in range(6):
            idx = row * 6 + col
            pdf.cell(col_w, 8, f"{months[idx]}: {trend[idx]:,.1f}", border=1)
        pdf.ln()
    pdf.ln(4)

    # Inputs
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(80, 30, 160)
    pdf.cell(0, 9, "Input Data", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(30, 30, 30)
    for k, v in inputs.items():
        pdf.cell(100, 8, str(k), border='LTB')
        pdf.cell(90,  8, str(v), border='RTB', ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(0, 8, "Smart Electricity Prediction Suite  |  Multi-Output LSTM + Site LSTM", ln=True, align='C')
    return pdf.output(dest='S').encode('latin-1')

def generate_pdf_site(inputs: dict, cost_usd: float, ex_rate: float, budget: float) -> bytes:
    cost_egp  = cost_usd * ex_rate
    budget_usd = budget / ex_rate
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(80, 30, 160)
    pdf.cell(0, 12, "Smart City Site Electricity Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Model: Site LSTM", ln=True, align='C')
    pdf.ln(5)

    pdf.set_fill_color(240, 230, 255)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(80, 30, 160)
    pdf.cell(0, 9, "Prediction Results", ln=True, fill=True)
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(95, 9, f"Monthly Cost: ${cost_usd:,.2f} USD",      border=1)
    pdf.cell(95, 9, f"EGP Equivalent: {cost_egp:,.2f} EGP",    border=1, ln=True)
    pdf.cell(95, 9, f"Annual Estimate: ${cost_usd*12:,.2f}",    border=1)
    pdf.cell(95, 9, f"Daily Average: ${cost_usd/30:,.2f}",      border=1, ln=True)
    pdf.cell(95, 9, f"Exchange Rate: 1 USD = {ex_rate} EGP",   border=1, ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", 'B', 11)
    if cost_usd > budget_usd:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 9, f"STATUS: OVER BUDGET  (${cost_usd:,.2f} > ${budget_usd:,.2f})", ln=True)
    elif cost_usd > 500:
        pdf.set_text_color(180, 120, 0)
        pdf.cell(0, 9, "STATUS: MODERATE CONSUMPTION", ln=True)
    else:
        pdf.set_text_color(0, 140, 0)
        pdf.cell(0, 9, "STATUS: LOW / EFFICIENT", ln=True)
    pdf.ln(4)

    # Forecast table
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(80, 30, 160)
    pdf.cell(0, 9, "12-Month Cost Forecast (USD)", ln=True)
    pdf.set_font("Arial", '', 9)
    pdf.set_text_color(30, 30, 30)
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    trend  = monthly_trend(cost_usd)
    col_w  = 190 / 6
    for row in range(2):
        for col in range(6):
            idx = row * 6 + col
            pdf.cell(col_w, 8, f"{months[idx]}: ${trend[idx]:,.1f}", border=1)
        pdf.ln()
    pdf.ln(4)

    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(80, 30, 160)
    pdf.cell(0, 9, "Site Input Data", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(30, 30, 30)
    for k, v in inputs.items():
        pdf.cell(110, 8, str(k), border='LTB')
        pdf.cell(80,  8, str(v), border='RTB', ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(0, 8, "Smart Electricity Prediction Suite  |  Multi-Output LSTM + Site LSTM", ln=True, align='C')
    return pdf.output(dest='S').encode('latin-1')

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    ex_rate    = st.number_input("USD -> EGP Exchange Rate", min_value=1.0, value=48.5, step=0.5)
    budget_egp = st.number_input("Monthly Budget Alert (EGP)", min_value=0, value=1000, step=50)

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1.5rem 0 0.5rem 0;">
    <span class="model-badge">DEEP LEARNING SUITE</span>
    <span class="model-badge">Multi-Output LSTM</span>
    <span class="model-badge">Site LSTM</span>
    <h1 style="margin-top:0.8rem; font-size:2rem; color:#e6edf3; line-height:1.3;">
        Electricity Cost &<br><span style="color:#a78bfa;">Energy Predictor</span>
    </h1>
    <p style="color:#8b949e; font-size:0.9rem; margin-top:0.4rem;">
        Two LSTM models · PDF report · 12-month forecast · Budget alert<br>
        Inputs default to dataset averages — only fill in what you know.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "Multi-Output LSTM  —  Home Electricity (kWh + EGP)",
    "Site LSTM  —  Smart City Site (USD)"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Multi-Output LSTM  (home: kWh + EGP)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <span class="model-badge">Multi-Output LSTM</span>
    <span class="model-badge">Dual Output: kWh + EGP</span>
    <p class="default-note" style="margin-top:0.6rem;">
        All inputs default to dataset means. Leave unchanged if unsure.
    </p>
    """, unsafe_allow_html=True)

    lstm_multi, scaler_home = load_multi_lstm()

    with st.form("lstm_home_form"):
        st.markdown("#### Home and Appliances")
        c1, c2, c3 = st.columns(3)
        with c1:
            h_ac      = st.number_input("Air Conditioners",   min_value=0,   value=int(round(MEANS['number_of_air_conditioners'])))
            h_ac_hp   = st.number_input("AC Power (HP)",      min_value=0.0, step=0.5, value=round(MEANS['ac_power_hp'], 1))
            h_fridge  = st.number_input("Refrigerators",      min_value=0,   value=int(round(MEANS['number_of_refrigerators'])))
            h_tv      = st.number_input("Televisions",        min_value=0,   value=int(round(MEANS['number_of_televisions'])))
        with c2:
            h_fans    = st.number_input("Fans",               min_value=0,   value=int(round(MEANS['number_of_fans'])))
            h_pc      = st.number_input("Computers/Laptops",  min_value=0,   value=int(round(MEANS['number_of_computers'])))
            h_washing = st.number_input("Washing (times/wk)", min_value=0,   max_value=20, value=int(round(MEANS['washing_machine_usage_per_week'])))
            h_heater  = st.selectbox("Water Heater?", ["Yes", "No"], index=0)
        with c3:
            h_hours   = st.slider("Daily Usage Hours", 0.0, 24.0, round(MEANS['average_daily_usage_hours'], 1))
            h_house   = st.number_input("House Size (m2)", min_value=10.0, value=round(MEANS['house_size_m2'], 0))
            h_season  = st.selectbox("Season", ["Summer", "Winter"], index=0)
            h_insul   = st.selectbox("Insulation Quality", ["High", "Medium", "Low"], index=0)

        home_submit = st.form_submit_button("Predict with Multi-Output LSTM", use_container_width=True)

    if home_submit:
        if lstm_multi is None:
            st.error("Multi-Output LSTM not loaded. Check multi_output_lstm_model.keras and feature_scaler.joblib are in the repo.")
        else:
            raw_home = {
                'number_of_air_conditioners':     h_ac,
                'ac_power_hp':                    h_ac_hp,
                'number_of_refrigerators':        h_fridge,
                'number_of_televisions':          h_tv,
                'number_of_fans':                 h_fans,
                'number_of_computers':            h_pc,
                'average_daily_usage_hours':      h_hours,
                'season':                         h_season.lower(),
                'house_size_m2':                  float(h_house),
                'insulation_quality':             h_insul.lower(),
                'has_water_heater':               1 if h_heater == "Yes" else 0,
                'washing_machine_usage_per_week': h_washing,
            }
            with st.spinner("Running Multi-Output LSTM..."):
                try:
                    X       = preprocess_home(raw_home, scaler_home)
                    preds   = lstm_multi.predict(X, verbose=0)
                    kwh     = float(preds[0].flatten()[0])
                    bill    = float(preds[1].flatten()[0])

                    # Budget alert
                    if bill > budget_egp:
                        st.markdown(f"""
                        <div class="warning-banner">
                            <h3 style="color:#ff4b4b; margin:0;">Over Budget Alert</h3>
                            <p style="color:#ff4b4b; margin:0.3rem 0 0 0;">
                                Predicted bill <b>{bill:,.1f} EGP</b> exceeds your budget of <b>{budget_egp:,} EGP</b>
                            </p>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.success(f"Within budget! Predicted bill: {bill:,.1f} EGP (limit: {budget_egp:,} EGP)")

                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Consumption",   f"{kwh:,.1f} kWh")
                    m2.metric("Monthly Bill",  f"{bill:,.1f} EGP")
                    m3.metric("Annual Bill",   f"{bill*12:,.0f} EGP")
                    m4.metric("Daily Cost",    f"{bill/30:,.1f} EGP")

                    color, zone_label, zone_note = zone_egp(bill)
                    st.markdown(f"""
                    <div class="result-card" style="border-left-color:{color};">
                        <span style="font-family:'Space Mono',monospace; font-weight:700;
                                     color:{color}; font-size:0.95rem;">{zone_label}</span><br>
                        <span style="color:#8b949e; font-size:0.85rem;">{zone_note}</span>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    chart_col, pdf_col = st.columns([3, 1])
                    with chart_col:
                        st.markdown("#### 12-Month Cost Forecast")
                        st.markdown(plot_trend(monthly_trend(bill), "EGP", "#a78bfa"), unsafe_allow_html=True)
                    with pdf_col:
                        st.markdown("#### Download Report")
                        st.markdown("<br>", unsafe_allow_html=True)
                        human_in = {
                            "Air Conditioners":   h_ac,    "AC Power (HP)":      h_ac_hp,
                            "Refrigerators":      h_fridge,"Televisions":         h_tv,
                            "Fans":               h_fans,  "Computers":           h_pc,
                            "Daily Hours":        h_hours, "House Size (m2)":     h_house,
                            "Season":             h_season,"Insulation":          h_insul,
                            "Water Heater":       h_heater,"Washing (times/wk)":  h_washing,
                        }
                        pdf_bytes = generate_pdf_home(human_in, kwh, bill, ex_rate, budget_egp)
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"home_lstm_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                        st.markdown(f"""
                        <div style="margin-top:0.8rem;font-size:0.78rem;color:#8b949e;text-align:center;">
                            Rate used:<br><b style="color:#e6edf3;">1 USD = {ex_rate} EGP</b>
                        </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.exception(e)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Site LSTM  (smart city: USD)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <span class="model-badge">Site LSTM</span>
    <span class="model-badge">Output: USD/month</span>
    <p class="default-note" style="margin-top:0.6rem;">
        Smart-city site model. Inputs default to dataset means. Leave unchanged if unsure.
    </p>
    """, unsafe_allow_html=True)

    lstm_site, scaler_site = load_site_lstm()

    with st.form("lstm_site_form"):
        st.markdown("#### Site Information")
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            s_area     = st.number_input("Site Area (m2)",             min_value=100,   value=int(round(SITE_MEANS['Site Area (square meters)'])),         step=50)
            s_resident = st.number_input("Resident Count",              min_value=1,     value=int(round(SITE_MEANS['Resident Count (number of people)'])), step=1)
            s_util     = st.slider("Utilisation Rate (%)",              0, 100,          int(round(SITE_MEANS['Utilisation Rate (%)'])))
        with lc2:
            s_water    = st.number_input("Water Consumption (L/day)",   min_value=0.0,   value=round(SITE_MEANS['Water Consumption (liters/day)'], 0),      step=50.0)
            s_recycle  = st.slider("Recycling Rate (%)",                0, 100,          int(round(SITE_MEANS['Recycling Rate (%)'])))
            s_aqi      = st.number_input("Air Quality Index (AQI)",     min_value=0,     max_value=500, value=int(round(SITE_MEANS['Air Quality Index (AQI)'])))
        with lc3:
            s_issue    = st.number_input("Issue Resolution Time (hrs)", min_value=0,     value=int(round(SITE_MEANS['Issue Resolution Time (hours)'])),     step=1)
            s_struct   = st.selectbox("Structure Type", ["Commercial", "Industrial", "Mixed-use", "Residential"], index=0)

        site_submit = st.form_submit_button("Predict with Site LSTM", use_container_width=True)

    if site_submit:
        if lstm_site is None:
            st.error("Site LSTM not loaded. Check lstm_model.keras and scaler.joblib are in the repo.")
        else:
            site_input = {
                'Site Area (square meters)':          s_area,
                'Water Consumption (liters/day)':     s_water,
                'Recycling Rate (%)':                 s_recycle,
                'Utilisation Rate (%)':               s_util,
                'Air Quality Index (AQI)':            s_aqi,
                'Issue Resolution Time (hours)':      s_issue,
                'Resident Count (number of people)':  s_resident,
                'Structure Type_Industrial':          int(s_struct == "Industrial"),
                'Structure Type_Mixed-use':           int(s_struct == "Mixed-use"),
                'Structure Type_Residential':         int(s_struct == "Residential"),
            }
            with st.spinner("Running Site LSTM..."):
                try:
                    X    = preprocess_site(site_input, scaler_site)
                    cost = float(lstm_site.predict(X, verbose=0).flatten()[0])
                    cost_egp   = cost * ex_rate
                    budget_usd = budget_egp / ex_rate

                    if cost > budget_usd:
                        st.markdown(f"""
                        <div class="warning-banner">
                            <h3 style="color:#ff4b4b; margin:0;">Over Budget Alert</h3>
                            <p style="color:#ff4b4b; margin:0.3rem 0 0 0;">
                                Predicted cost <b>${cost:,.2f}</b> exceeds your budget of <b>${budget_usd:,.2f}</b>
                            </p>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.success(f"Within budget! Predicted cost: ${cost:,.2f} (limit: ${budget_usd:,.2f})")

                    lm1, lm2, lm3, lm4 = st.columns(4)
                    lm1.metric("Monthly Cost",    f"${cost:,.2f}")
                    lm2.metric("In EGP",          f"{cost_egp:,.1f}")
                    lm3.metric("Annual Estimate", f"${cost*12:,.2f}")
                    lm4.metric("Daily Average",   f"${cost/30:,.2f}")

                    color, zone_label, zone_note = zone_usd(cost)
                    st.markdown(f"""
                    <div class="result-card" style="border-left-color:{color};">
                        <span style="font-family:'Space Mono',monospace; font-weight:700;
                                     color:{color}; font-size:0.95rem;">{zone_label}</span><br>
                        <span style="color:#8b949e; font-size:0.85rem;">{zone_note}</span>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    chart_col2, pdf_col2 = st.columns([3, 1])
                    with chart_col2:
                        st.markdown("#### 12-Month Cost Forecast")
                        st.markdown(plot_trend(monthly_trend(cost), "USD", "#ec4899"), unsafe_allow_html=True)
                    with pdf_col2:
                        st.markdown("#### Download Report")
                        st.markdown("<br>", unsafe_allow_html=True)
                        pdf_bytes2 = generate_pdf_site(site_input, cost, ex_rate, budget_egp)
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_bytes2,
                            file_name=f"site_lstm_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                        st.markdown(f"""
                        <div style="margin-top:0.8rem;font-size:0.78rem;color:#8b949e;text-align:center;">
                            Rate used:<br><b style="color:#e6edf3;">1 USD = {ex_rate} EGP</b>
                        </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.exception(e)

st.divider()
st.markdown(
    "<p style='text-align:center;color:#30363d;font-size:0.78rem;font-family:Space Mono,monospace;'>"
    "Multi-Output LSTM + Site LSTM  |  Smart Energy Analytics</p>",
    unsafe_allow_html=True,
)
