"""
dashboard/app.py  —  Qystas Smart Pricing Engine
Professional UI — Full Dynamic Brand Experience
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from core.curve_fitting import get_curve_data, get_bracket_affordability
from core.optimizer import PricingOptimizer, ProductInput
from core.ml_model import load_churn_model

# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Qystas — Smart Pricing Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════
# FULL CSS + HTML INJECTION
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Cairo:wght@300;400;600;700;900&display=swap');

:root {
  --navy:  #0B2545;
  --navy2: #133A6A;
  --slate: #4A6FA5;
  --gold:  #C9A84C;
  --gold2: #E8C96A;
  --bg:    #F0F4F8;
  --bg2:   #E2EAF4;
  --white: #FFFFFF;
  --text:  #1A2E4A;
  --muted: #7A8FA6;
  --green: #1E8C5A;
  --red:   #C0392B;
  --amber: #E67E22;
  --r:     14px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
  background: var(--bg) !important;
  font-family: 'Cairo', 'Inter', sans-serif !important;
}

/* ── hide streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

[data-testid="stAppViewContainer"] > .main > .block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ═══════════════════════════════
   NAVBAR
═══════════════════════════════ */
.q-nav {
  background: var(--navy);
  height: 66px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 48px;
  position: sticky;
  top: 0;
  z-index: 1000;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 2px 24px rgba(0,0,0,0.25);
}

.q-nav-left { display: flex; align-items: center; gap: 14px; }

.q-nav-icon {
  width: 38px; height: 38px;
  background: var(--navy2);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 20px;
  border: 1px solid rgba(201,168,76,0.25);
  box-shadow: 0 0 0 1px rgba(201,168,76,0.1);
}

.q-nav-brand { display: flex; flex-direction: column; gap: 1px; }

.q-nav-name {
  font-family: 'Inter', sans-serif;
  font-size: 21px;
  font-weight: 900;
  color: #fff;
  letter-spacing: -0.8px;
  line-height: 1;
}

.q-nav-name b { color: var(--gold); font-weight: 900; }

.q-nav-sub {
  font-family: 'Inter', sans-serif;
  font-size: 9px;
  font-weight: 600;
  color: rgba(255,255,255,0.3);
  letter-spacing: 2.5px;
  text-transform: uppercase;
}

.q-nav-right { display: flex; align-items: center; gap: 10px; }

.q-badge {
  padding: 5px 13px;
  border-radius: 100px;
  font-family: 'Inter', sans-serif;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.8px;
  text-transform: uppercase;
}

.q-badge-outline {
  border: 1px solid rgba(255,255,255,0.12);
  color: rgba(255,255,255,0.35);
}

.q-badge-gold {
  background: var(--gold);
  color: var(--navy);
}

/* ═══════════════════════════════
   TICKER
═══════════════════════════════ */
.q-ticker {
  background: linear-gradient(90deg, var(--gold) 0%, var(--gold2) 50%, var(--gold) 100%);
  overflow: hidden;
  height: 36px;
  display: flex;
  align-items: center;
}

.q-ticker-inner {
  display: inline-flex;
  animation: qtick 32s linear infinite;
  white-space: nowrap;
  align-items: center;
}

.q-tick-item {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 0 30px;
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 800;
  color: var(--navy);
  letter-spacing: 0.2px;
}

.q-tick-dot {
  width: 4px; height: 4px;
  background: rgba(11,37,69,0.25);
  border-radius: 50%;
  flex-shrink: 0;
}

@keyframes qtick {
  0%   { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}

/* ═══════════════════════════════
   HERO
═══════════════════════════════ */
.q-hero {
  background: linear-gradient(135deg, #07192E 0%, var(--navy) 40%, var(--navy2) 75%, #1A4A8A 100%);
  padding: 72px 48px 80px;
  position: relative;
  overflow: hidden;
}

.q-hero-glow-1 {
  position: absolute;
  top: -120px; right: -120px;
  width: 500px; height: 500px;
  background: radial-gradient(circle, rgba(201,168,76,0.09) 0%, transparent 65%);
  pointer-events: none;
}

.q-hero-glow-2 {
  position: absolute;
  bottom: -80px; left: 8%;
  width: 340px; height: 340px;
  background: radial-gradient(circle, rgba(74,111,165,0.18) 0%, transparent 65%);
  pointer-events: none;
}

.q-hero-grid {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none;
}

.q-hero-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(201,168,76,0.12);
  border: 1px solid rgba(201,168,76,0.3);
  padding: 6px 16px;
  border-radius: 100px;
  margin-bottom: 22px;
  position: relative;
}

.q-hero-chip-dot {
  width: 7px; height: 7px;
  background: var(--gold);
  border-radius: 50%;
  animation: q-pulse 2.2s ease infinite;
}

@keyframes q-pulse {
  0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 0 0 rgba(201,168,76,0.4); }
  50% { opacity: 0.7; transform: scale(0.85); box-shadow: 0 0 0 6px rgba(201,168,76,0); }
}

.q-hero-chip-txt {
  font-family: 'Inter', sans-serif;
  font-size: 10px;
  font-weight: 700;
  color: var(--gold);
  letter-spacing: 1.8px;
  text-transform: uppercase;
}

.q-hero-h1 {
  font-family: 'Inter', sans-serif;
  font-size: clamp(32px, 4.5vw, 58px);
  font-weight: 900;
  color: #fff;
  line-height: 1.06;
  letter-spacing: -2px;
  margin-bottom: 16px;
  max-width: 660px;
  position: relative;
}

.q-hero-h1 .q-gold { color: var(--gold); }

.q-hero-p {
  font-family: 'Cairo', sans-serif;
  font-size: 15px;
  font-weight: 400;
  color: rgba(255,255,255,0.5);
  line-height: 1.8;
  max-width: 500px;
  margin-bottom: 52px;
  position: relative;
}

.q-hero-stats {
  display: flex;
  align-items: stretch;
  gap: 0;
  flex-wrap: wrap;
  position: relative;
}

.q-stat {
  padding: 0 40px 0 0;
  margin-right: 40px;
  border-right: 1px solid rgba(255,255,255,0.08);
}

.q-stat:last-child { border-right: none; margin-right: 0; padding-right: 0; }

.q-stat-val {
  font-family: 'Inter', sans-serif;
  font-size: 38px;
  font-weight: 900;
  color: var(--gold);
  letter-spacing: -2px;
  line-height: 1;
}

.q-stat-lbl {
  font-family: 'Cairo', sans-serif;
  font-size: 11px;
  color: rgba(255,255,255,0.35);
  margin-top: 5px;
}

/* ═══════════════════════════════
   BODY WRAP
═══════════════════════════════ */
.q-body {
  padding: 36px 48px 56px;
  display: flex;
  flex-direction: column;
  gap: 28px;
}

/* ═══════════════════════════════
   TABS
═══════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background: var(--white) !important;
  border-radius: 13px !important;
  padding: 6px !important;
  gap: 4px !important;
  border: 1px solid var(--bg2) !important;
  box-shadow: 0 2px 8px rgba(11,37,69,0.06) !important;
  width: fit-content !important;
}

.stTabs [data-baseweb="tab"] {
  border-radius: 9px !important;
  font-family: 'Cairo', sans-serif !important;
  font-size: 13px !important;
  font-weight: 700 !important;
  color: var(--muted) !important;
  padding: 10px 22px !important;
  background: transparent !important;
  border: none !important;
  letter-spacing: 0.2px !important;
  transition: all .15s !important;
}

.stTabs [aria-selected="true"] {
  background: var(--navy) !important;
  color: #fff !important;
  box-shadow: 0 2px 10px rgba(11,37,69,0.3) !important;
}

.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ═══════════════════════════════
   KPI CARDS (HTML)
═══════════════════════════════ */
.q-kpi-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.q-kpi {
  background: var(--white);
  border: 1px solid var(--bg2);
  border-radius: var(--r);
  padding: 22px 24px 18px;
  box-shadow: 0 1px 4px rgba(11,37,69,0.04);
  transition: box-shadow .2s, transform .2s;
  cursor: default;
  position: relative;
  overflow: hidden;
}

.q-kpi::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  border-radius: var(--r) var(--r) 0 0;
}

.q-kpi.kpi-navy::before  { background: var(--navy); }
.q-kpi.kpi-gold::before  { background: var(--gold); }
.q-kpi.kpi-green::before { background: var(--green); }
.q-kpi.kpi-red::before   { background: var(--red); }

.q-kpi:hover {
  box-shadow: 0 8px 24px rgba(11,37,69,0.10);
  transform: translateY(-3px);
}

.q-kpi-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 14px;
}

.q-kpi-lbl {
  font-family: 'Cairo', sans-serif;
  font-size: 11px;
  font-weight: 700;
  color: var(--muted);
  line-height: 1.4;
  max-width: 120px;
}

.q-kpi-ico {
  width: 34px; height: 34px;
  border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  font-size: 16px;
  flex-shrink: 0;
}

.q-kpi-ico.bg-navy  { background: rgba(11,37,69,0.07); }
.q-kpi-ico.bg-gold  { background: rgba(201,168,76,0.12); }
.q-kpi-ico.bg-green { background: rgba(30,140,90,0.1); }
.q-kpi-ico.bg-red   { background: rgba(192,57,43,0.1); }

.q-kpi-val {
  font-family: 'Inter', sans-serif;
  font-size: 30px;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -1.2px;
  line-height: 1;
  margin-bottom: 8px;
}

.q-kpi-delta {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 100px;
}

.d-up   { background: rgba(30,140,90,0.10);  color: var(--green); }
.d-down { background: rgba(192,57,43,0.10);  color: var(--red); }
.d-warn { background: rgba(230,126,34,0.10); color: var(--amber); }
.d-info { background: rgba(74,111,165,0.10); color: var(--slate); }

.q-kpi-bar { height: 5px; background: var(--bg2); border-radius: 100px; margin-top: 12px; overflow: hidden; }
.q-kpi-fill { height: 100%; border-radius: 100px; transition: width 1.2s cubic-bezier(.4,0,.2,1); }

/* ═══════════════════════════════
   SECTION HEADERS
═══════════════════════════════ */
.q-sec-hd {
  font-family: 'Inter', sans-serif;
  font-size: 20px;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.5px;
  margin-bottom: 4px;
}

.q-sec-sub {
  font-family: 'Cairo', sans-serif;
  font-size: 13px;
  color: var(--muted);
  margin-bottom: 24px;
  line-height: 1.6;
}

/* ═══════════════════════════════
   INSIGHT BOX
═══════════════════════════════ */
.q-insight {
  background: rgba(11,37,69,0.03);
  border: 1px solid var(--bg2);
  border-right: 4px solid var(--navy);
  border-radius: 10px;
  padding: 14px 18px;
  font-family: 'Cairo', sans-serif;
  font-size: 13px;
  color: var(--text);
  line-height: 1.75;
  margin-top: 16px;
}

.q-insight.gold  { border-right-color: var(--gold); }
.q-insight.green { border-right-color: var(--green); background: rgba(30,140,90,0.03); }
.q-insight.red   { border-right-color: var(--red);   background: rgba(192,57,43,0.03); }
.q-insight.amber { border-right-color: var(--amber); background: rgba(230,126,34,0.03); }

/* ═══════════════════════════════
   REC CARDS
═══════════════════════════════ */
.q-rec {
  border-radius: 13px;
  padding: 20px 22px;
  border: 1.5px solid;
  margin-bottom: 14px;
}

.q-rec.success { background: rgba(30,140,90,0.04);  border-color: rgba(30,140,90,0.2); }
.q-rec.warning { background: rgba(230,126,34,0.04); border-color: rgba(230,126,34,0.2); }
.q-rec.danger  { background: rgba(192,57,43,0.04);  border-color: rgba(192,57,43,0.2); }

.q-rec-title {
  font-family: 'Cairo', sans-serif;
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 12px;
  display: flex; align-items: center; gap: 6px;
}

.q-rec-title.success { color: var(--green); }
.q-rec-title.warning { color: var(--amber); }
.q-rec-title.danger  { color: var(--red); }

.q-rec-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 7px 0;
  border-bottom: 1px solid rgba(0,0,0,0.05);
}
.q-rec-row:last-child { border-bottom: none; }

.q-rec-k {
  font-family: 'Cairo', sans-serif;
  font-size: 11px; font-weight: 600;
  color: var(--muted);
}

.q-rec-v {
  font-family: 'Inter', sans-serif;
  font-size: 16px; font-weight: 800;
  color: var(--text);
  letter-spacing: -0.5px;
}

.q-rec-v.gold  { color: var(--gold); }
.q-rec-v.green { color: var(--green); }
.q-rec-v.red   { color: var(--red); }

/* ═══════════════════════════════
   STREAMLIT OVERRIDES
═══════════════════════════════ */
[data-testid="stMetric"] {
  background: var(--white) !important;
  border: 1px solid var(--bg2) !important;
  border-radius: var(--r) !important;
  padding: 20px 22px !important;
  box-shadow: 0 1px 4px rgba(11,37,69,0.04) !important;
}

[data-testid="stMetricLabel"] p {
  font-family: 'Cairo', sans-serif !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  color: var(--muted) !important;
}

[data-testid="stMetricValue"] {
  font-family: 'Inter', sans-serif !important;
  font-size: 26px !important;
  font-weight: 800 !important;
  color: var(--text) !important;
  letter-spacing: -0.8px !important;
}

[data-testid="stMetricDelta"] {
  font-family: 'Inter', sans-serif !important;
  font-size: 11px !important;
  font-weight: 700 !important;
}

[data-testid="stDataFrame"] {
  border-radius: var(--r) !important;
  border: 1px solid var(--bg2) !important;
  overflow: hidden !important;
}

[data-testid="stAlert"] {
  border-radius: 12px !important;
  font-family: 'Cairo', sans-serif !important;
  font-size: 13px !important;
}

.stButton > button {
  background: var(--navy) !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  font-family: 'Cairo', sans-serif !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  padding: 13px 28px !important;
  letter-spacing: 0.3px !important;
  transition: all .2s !important;
  box-shadow: 0 2px 8px rgba(11,37,69,0.2) !important;
}

.stButton > button:hover {
  background: var(--navy2) !important;
  box-shadow: 0 6px 20px rgba(11,37,69,0.3) !important;
  transform: translateY(-2px) !important;
}

.stRadio label { font-family: 'Cairo', sans-serif !important; font-weight: 600 !important; }
.stSlider label { font-family: 'Cairo', sans-serif !important; font-weight: 600 !important; }
.stSelectbox label { font-family: 'Cairo', sans-serif !important; font-weight: 600 !important; }
.stNumberInput label { font-family: 'Cairo', sans-serif !important; font-weight: 600 !important; }
.stMarkdown p { font-family: 'Cairo', sans-serif !important; }
.stSpinner p { font-family: 'Cairo', sans-serif !important; color: var(--navy) !important; }

/* ═══════════════════════════════
   FOOTER
═══════════════════════════════ */
.q-footer {
  background: var(--navy);
  padding: 28px 48px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 16px;
  margin-top: 24px;
  border-top: 1px solid rgba(255,255,255,0.05);
}

.q-footer-left { display: flex; align-items: center; gap: 14px; }

.q-footer-icon {
  width: 36px; height: 36px;
  background: var(--navy2);
  border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px;
  border: 1px solid rgba(201,168,76,0.2);
}

.q-footer-brand-name {
  font-family: 'Inter', sans-serif;
  font-size: 16px;
  font-weight: 800;
  color: #fff;
  letter-spacing: -0.5px;
}

.q-footer-brand-name b { color: var(--gold); }

.q-footer-copy {
  font-family: 'Cairo', sans-serif;
  font-size: 11px;
  color: rgba(255,255,255,0.28);
  margin-top: 2px;
}

.q-footer-right { display: flex; gap: 8px; flex-wrap: wrap; }

.q-footer-pill {
  padding: 5px 14px;
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 100px;
  font-family: 'Inter', sans-serif;
  font-size: 9px;
  font-weight: 700;
  color: rgba(255,255,255,0.28);
  letter-spacing: 1.2px;
  text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════
st.markdown("""
<div class="q-nav">
  <div class="q-nav-left">
    <div class="q-nav-icon">⚖️</div>
    <div class="q-nav-brand">
      <div class="q-nav-name">Qys<b>tas</b></div>
      <div class="q-nav-sub">Smart Pricing Engine</div>
    </div>
  </div>
  <div class="q-nav-right">
    <span class="q-badge q-badge-outline">ML POWERED</span>
    <span class="q-badge q-badge-outline">EGYPT 2026</span>
    <span class="q-badge q-badge-gold">v1.0</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TICKER
# ══════════════════════════════════════════════
TICK_ITEMS = [
    ("📈", "تضخم تراكمي حضري: 2.47×"),
    ("💰", "القوة الشرائية الحقيقية: 40.5%"),
    ("🍞", "ميزانية الأكل الإجبارية: 28.63%"),
    ("📊", "وسيط الدخل 2026: 138,739 ج/سنة"),
    ("⚖️", "Qystas — محرك التسعير الذكي"),
    ("🌍", "تضخم تراكمي ريفي: 2.53×"),
    ("📉", "القوة الشرائية الريفية: 39.49%"),
    ("🤖", "دقة الموديل AUC: 0.74"),
]

SEP = '<span class="q-tick-dot"></span>'
items_html = f" {SEP} ".join(
    f'<span class="q-tick-item">{ic} {tx}</span>'
    for ic, tx in TICK_ITEMS * 2
)
st.markdown(f"""
<div class="q-ticker">
  <div class="q-ticker-inner">{items_html}</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════
st.markdown("""
<div class="q-hero">
  <div class="q-hero-grid"></div>
  <div class="q-hero-glow-1"></div>
  <div class="q-hero-glow-2"></div>

  <div class="q-hero-chip">
    <span class="q-hero-chip-dot"></span>
    <span class="q-hero-chip-txt">بيانات توزيع الدخل المصري 2020 – 2026</span>
  </div>

  <div class="q-hero-h1">
    قرار التسعير الصح<br>في <span class="q-gold">ثوانٍ</span>
  </div>

  <div class="q-hero-p">
    نظام ذكاء اصطناعي يحسب لك الوزن الأمثل ونسبة المقاطعة المتوقعة —
    بناءً على الدخل الحقيقي لزبائنك اليوم.
  </div>

  <div class="q-hero-stats">
    <div class="q-stat">
      <div class="q-stat-val">2.47×</div>
      <div class="q-stat-lbl">التضخم التراكمي</div>
    </div>
    <div class="q-stat">
      <div class="q-stat-val">9</div>
      <div class="q-stat-lbl">فئة دخلية محللة</div>
    </div>
    <div class="q-stat">
      <div class="q-stat-val">0.74</div>
      <div class="q-stat-lbl">دقة الموديل AUC</div>
    </div>
    <div class="q-stat">
      <div class="q-stat-val">24K</div>
      <div class="q-stat-lbl">سيناريو تدريب</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# LOAD RESOURCES
# ══════════════════════════════════════════════
@st.cache_resource
def get_optimizer():
    return PricingOptimizer()

@st.cache_resource
def get_model():
    return load_churn_model()

@st.cache_data
def get_curves(area):
    return get_curve_data(area)

optimizer = get_optimizer()
model     = get_model()

# ══════════════════════════════════════════════
# BODY
# ══════════════════════════════════════════════
st.markdown('<div class="q-body">', unsafe_allow_html=True)

# ── KPI ROW ──
st.markdown("""
<div class="q-kpi-row">
  <div class="q-kpi kpi-navy">
    <div class="q-kpi-top">
      <div class="q-kpi-lbl">وسيط الدخل 2026</div>
      <div class="q-kpi-ico bg-gold">💰</div>
    </div>
    <div class="q-kpi-val">138K</div>
    <div class="q-kpi-delta d-up">▲ +146.8% اسمي</div>
    <div class="q-kpi-bar"><div class="q-kpi-fill" style="width:65%;background:var(--gold)"></div></div>
  </div>
  <div class="q-kpi kpi-red">
    <div class="q-kpi-top">
      <div class="q-kpi-lbl">القوة الشرائية الحقيقية</div>
      <div class="q-kpi-ico bg-red">📉</div>
    </div>
    <div class="q-kpi-val">40.5%</div>
    <div class="q-kpi-delta d-down">▼ انخفاض حقيقي</div>
    <div class="q-kpi-bar"><div class="q-kpi-fill" style="width:40.5%;background:var(--red)"></div></div>
  </div>
  <div class="q-kpi kpi-green">
    <div class="q-kpi-top">
      <div class="q-kpi-lbl">دقة موديل المقاطعة</div>
      <div class="q-kpi-ico bg-green">🤖</div>
    </div>
    <div class="q-kpi-val">0.74</div>
    <div class="q-kpi-delta d-info">AUC Score</div>
    <div class="q-kpi-bar"><div class="q-kpi-fill" style="width:74%;background:var(--green)"></div></div>
  </div>
  <div class="q-kpi kpi-gold">
    <div class="q-kpi-top">
      <div class="q-kpi-lbl">سيناريوهات التدريب</div>
      <div class="q-kpi-ico bg-navy">📊</div>
    </div>
    <div class="q-kpi-val">24K</div>
    <div class="q-kpi-delta d-info">سيناريو محاكاة</div>
    <div class="q-kpi-bar"><div class="q-kpi-fill" style="width:85%;background:var(--navy)"></div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊  رادار انكماش الطبقات",
    "🏭  بيانات المنتج",
    "🎯  التوصية الذكية",
])

# ─────────────────────────────────
# TAB 1
# ─────────────────────────────────
with tab1:
    st.markdown('<div class="q-sec-hd">رادار انكماش الطبقات</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="q-sec-sub">مقارنة منحنى توزيع الدخل 2020 vs 2026 — '
        'الإزاحة لليمين لا تعني تحسناً، بل ارتفاع اسمي مقابل انخفاض حقيقي في القوة الشرائية.</div>',
        unsafe_allow_html=True,
    )

    area_choice = st.radio("المنطقة:", ["Urban", "Rural"], horizontal=True, key="r1")
    data = get_curves(area_choice)
    x   = np.array(data["x"])
    p20 = np.array(data["pdf_2020"])
    p26 = np.array(data["pdf_2026"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=p20,
        fill="tozeroy", fillcolor="rgba(74,111,165,0.13)",
        line=dict(color="#4A6FA5", width=2.5),
        name="توزيع 2020",
        hovertemplate="دخل: %{x:,.0f} ج<extra>2020</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p26,
        fill="tozeroy", fillcolor="rgba(192,57,43,0.10)",
        line=dict(color="#C0392B", width=2.5),
        name="توزيع 2026",
        hovertemplate="دخل: %{x:,.0f} ج<extra>2026</extra>",
    ))
    fig.add_vline(x=data["median_2020"], line_dash="dot", line_color="#4A6FA5", line_width=1.5,
                  annotation_text=f"وسيط 2020: {data['median_2020']:,.0f}",
                  annotation_font_size=11, annotation_font_color="#4A6FA5")
    fig.add_vline(x=data["median_2026"], line_dash="dot", line_color="#C0392B", line_width=1.5,
                  annotation_text=f"وسيط 2026: {data['median_2026']:,.0f}",
                  annotation_font_size=11, annotation_font_color="#C0392B")
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="الدخل السنوي (جنيه)", tickformat=",", range=[0, 420_000],
                   gridcolor="#F0F4F8", title_font=dict(family="Cairo", size=12, color="#7A8FA6")),
        yaxis=dict(title="كثافة الاحتمال", gridcolor="#F0F4F8",
                   title_font=dict(family="Cairo", size=12, color="#7A8FA6")),
        legend=dict(x=0.72, y=0.97, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#E2EAF4", borderwidth=1, font=dict(family="Cairo", size=12)),
        height=400, margin=dict(t=24, b=24, l=8, r=8),
        hovermode="x unified", font=dict(family="Cairo"),
    )
    st.plotly_chart(fig, use_container_width=True)

    pwr  = 40.50 if area_choice == "Urban" else 39.49
    food = 28.63 if area_choice == "Urban" else 36.39

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("وسيط الدخل 2020",         f"{data['median_2020']:,.0f} ج/سنة")
    c2.metric("وسيط الدخل 2026",         f"{data['median_2026']:,.0f} ج/سنة",
              delta=f"+{data['shift_pct']:.1f}% إزاحة اسمية")
    c3.metric("القوة الشرائية الحقيقية", f"{pwr}%",
              delta=f"-{100-pwr:.1f}%", delta_color="inverse")
    c4.metric("ميزانية الأكل الإجبارية", f"{food}%")

    st.markdown(
        f'<div class="q-insight gold">'
        f'💡 <strong>التفسير:</strong> وسيط الدخل ارتفع اسمياً +{data["shift_pct"]:.1f}% '
        f'لكن القوة الشرائية الحقيقية بقت {pwr}% بس — '
        f'زبونك بيكسب أكتر بالأرقام لكن يقدر يشتري أقل.</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────
# TAB 2
# ─────────────────────────────────
with tab2:
    st.markdown('<div class="q-sec-hd">بيانات المنتج</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="q-sec-sub">أدخل بيانات منتجك وسيحلل النظام موقعه في السوق ويحسب التوصية المثلى.</div>',
        unsafe_allow_html=True,
    )

    with st.form("product_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📦 بيانات المنتج**")
            current_price   = st.number_input("السعر الحالي (جنيه)", min_value=1.0, max_value=500.0, value=25.0, step=0.5)
            current_weight  = st.number_input("الوزن الحالي (جرام)", min_value=10.0, max_value=5000.0, value=100.0, step=5.0)
            cost_per_gram   = st.number_input("تكلفة الإنتاج / جرام", min_value=0.01, max_value=10.0, value=0.18, step=0.01,
                                               help="شاملة الخامات + التشغيل + التعبئة")
            new_price_input = st.number_input("السعر الجديد المقترح", min_value=1.0, max_value=500.0, value=30.0, step=0.5)
        with c2:
            st.markdown("**⚙️ معاملات التحليل**")
            area_sel      = st.selectbox("المنطقة المستهدفة", ["Urban", "Rural"])
            target_margin = st.slider("هامش الربح المستهدف %", 5, 60, 30, 5) / 100
            purchase_freq = st.slider("تكرار الشراء (مرات/شهر)", 1, 20, 4)
            current_cost   = cost_per_gram * current_weight
            current_margin = (current_price - current_cost) / current_price * 100
            st.markdown(
                f'<div class="q-insight" style="margin-top:12px">'
                f'<strong>ملخص:</strong><br>'
                f'تكلفة الإنتاج: <strong>{current_cost:.2f} ج</strong> &nbsp;|&nbsp; '
                f'هامش الربح: <strong>{current_margin:.1f}%</strong><br>'
                f'سعر الجرام: <strong>{current_price/current_weight:.3f} ج/ج</strong>'
                f'</div>', unsafe_allow_html=True,
            )
        st.form_submit_button("⚙️ تحليل وإنتاج التوصية الذكية", use_container_width=True, type="primary")

    if st.session_state.get("FormSubmitter:product_form-⚙️ تحليل وإنتاج التوصية الذكية", False) or "product_data" in st.session_state:
        try:
            st.session_state["product_data"] = {
                "current_price": current_price, "current_weight_g": current_weight,
                "cost_per_gram": cost_per_gram, "area": area_sel,
                "target_margin": target_margin, "purchase_freq": purchase_freq,
                "new_price": new_price_input,
            }
        except Exception:
            pass

    if "product_data" in st.session_state:
        pd_data  = st.session_state["product_data"]
        segments = get_bracket_affordability(pd_data["current_price"], pd_data["area"], pd_data["purchase_freq"])
        seg_df   = pd.DataFrame(segments)
        st.markdown("---")
        st.markdown("**خريطة القدرة الشرائية لكل فئة — بأسعار اليوم**")
        fig_aff = go.Figure(go.Bar(
            x=seg_df["bracket"],
            y=seg_df["price_burden_pct"],
            marker_color=["#1E8C5A" if r else "#C0392B" for r in seg_df["affordable"]],
            text=[f"{v:.1f}%" for v in seg_df["price_burden_pct"]],
            textposition="outside",
        ))
        fig_aff.add_hline(y=15, line_dash="dot", line_color="#E67E22", line_width=2,
                          annotation_text="عتبة 15%", annotation_font_color="#E67E22")
        fig_aff.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(tickangle=-30, gridcolor="#F0F4F8"),
            yaxis=dict(title="العبء السعري %", gridcolor="#F0F4F8"),
            height=300, margin=dict(t=24, b=80, l=8, r=8), font=dict(family="Cairo"),
        )
        st.plotly_chart(fig_aff, use_container_width=True)

        seg_df["الحالة"]          = seg_df["affordable"].apply(lambda x: "✅ في المتناول" if x else "🔴 خارج المتناول")
        seg_df["الدخل المتاح/شهر"] = seg_df["monthly_disposable"].apply(lambda v: f"{v:,.0f} ج")
        st.dataframe(
            seg_df[["bracket","population_pct","الدخل المتاح/شهر","price_burden_pct","الحالة"]].rename(
                columns={"bracket":"الفئة الدخلية","population_pct":"% السكان","price_burden_pct":"عبء سعري %"}
            ), use_container_width=True, hide_index=True,
        )

# ─────────────────────────────────
# TAB 3
# ─────────────────────────────────
with tab3:
    st.markdown('<div class="q-sec-hd">التوصية الذكية</div>', unsafe_allow_html=True)

    if "product_data" not in st.session_state:
        st.warning("👈 أدخل بيانات المنتج في تبويب «بيانات المنتج» أولاً.")
        st.stop()

    pd_data = st.session_state["product_data"]
    product = ProductInput(
        current_price=pd_data["current_price"], current_weight_g=pd_data["current_weight_g"],
        cost_per_gram=pd_data["cost_per_gram"], area=pd_data["area"],
        purchase_freq=pd_data["purchase_freq"], target_margin=pd_data["target_margin"],
    )

    with st.spinner("جاري التحليل بالموديل..."):
        w_rec  = optimizer.find_optimal_weight(product)
        c_pred = optimizer.predict_market_churn(product, pd_data["new_price"])

    # ── A ──
    st.markdown("### 🅐 الوزن الأمثل")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("الوزن الحالي",      f"{product.current_weight_g:.0f} جرام")
    c2.metric("الوزن الأمثل",      f"{w_rec.optimal_weight_g} جرام",
              delta=f"-{w_rec.weight_reduction_pct}%", delta_color="inverse")
    c3.metric("هامش الربح الجديد", f"{w_rec.new_margin_pct}%")
    c4.metric("سعر الجرام الجديد", f"{w_rec.price_per_gram_new} ج/ج")

    color = "green" if w_rec.feasible else "red"
    st.markdown(f'<div class="q-insight {color}"><strong>{w_rec.warning}</strong></div>',
                unsafe_allow_html=True)

    fig_w = go.Figure(go.Bar(
        x=["الوزن الحالي", "الوزن الأمثل"],
        y=[product.current_weight_g, w_rec.optimal_weight_g],
        marker_color=["#4A6FA5", "#1E8C5A"],
        text=[f"{product.current_weight_g:.0f}g", f"{w_rec.optimal_weight_g}g"],
        textposition="outside", width=0.4,
    ))
    fig_w.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="الوزن (جرام)", gridcolor="#F0F4F8"),
        height=250, margin=dict(t=20, b=20, l=8, r=8), font=dict(family="Cairo"),
    )
    st.plotly_chart(fig_w, use_container_width=True)
    st.divider()

    # ── B ──
    st.markdown(f"### 🅑 تنبؤ المقاطعة — رفع السعر من {product.current_price:.0f} لـ {c_pred.new_price:.0f} ج")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("نسبة المقاطعة",        f"{c_pred.weighted_churn_pct}%")
    c2.metric("السكان المعرضون للخطر", f"{c_pred.at_risk_population_pct}%")
    c3.metric("احتمال المقاطعة ML",    f"{c_pred.ml_churn_prob}%")
    c4.metric("مستوى الخطر",           c_pred.risk_level)

    risk_color = {"HIGH":"red","MEDIUM":"amber","LOW":"green"}[c_pred.risk_level]
    st.markdown(f'<div class="q-insight {risk_color}"><strong>{c_pred.recommendation}</strong></div>',
                unsafe_allow_html=True)

    seg_df = pd.DataFrame(c_pred.segments_detail)
    fig_s  = go.Figure()
    fig_s.add_trace(go.Bar(
        x=seg_df["bracket"], y=seg_df["price_burden_pct"],
        marker_color=["#C0392B" if r else "#1E8C5A" for r in seg_df["at_risk"]],
        name="العبء السعري %",
        text=[f"{v:.1f}%" for v in seg_df["price_burden_pct"]], textposition="outside",
    ))
    fig_s.add_trace(go.Scatter(
        x=seg_df["bracket"], y=seg_df["churn_threshold_pct"],
        mode="lines+markers", name="عتبة المقاطعة",
        line=dict(color="#E67E22", width=2.5, dash="dash"),
        marker=dict(size=7, color="#E67E22"),
    ))
    fig_s.add_trace(go.Bar(
        x=seg_df["bracket"], y=[v*100 for v in seg_df["ml_churn_prob"]],
        name="ML احتمال %", marker_color="rgba(74,111,165,0.45)", opacity=0.9,
    ))
    fig_s.update_layout(
        barmode="overlay", plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickangle=-30, gridcolor="#F0F4F8"),
        yaxis=dict(title="%", gridcolor="#F0F4F8"),
        legend=dict(x=0.65, y=0.97, bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#E2EAF4", borderwidth=1, font=dict(family="Cairo", size=11)),
        height=380, margin=dict(t=20, b=90, l=8, r=8), font=dict(family="Cairo"),
    )
    st.plotly_chart(fig_s, use_container_width=True)

    seg_show = seg_df[["bracket","population_pct","monthly_disposable",
                        "price_burden_pct","churn_threshold_pct","at_risk","ml_churn_prob"]].copy()
    seg_show.columns = ["الفئة الدخلية","% السكان","دخل متاح/شهر",
                         "عبء سعري %","عتبة المقاطعة %","في خطر؟","ML احتمال"]
    seg_show["في خطر؟"]   = seg_show["في خطر؟"].apply(lambda x: "🔴 نعم" if x else "🟢 لا")
    seg_show["ML احتمال"] = seg_show["ML احتمال"].apply(lambda x: f"{x:.0%}")
    st.dataframe(seg_show, use_container_width=True, hide_index=True)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════
st.markdown("""
<div class="q-footer">
  <div class="q-footer-left">
    <div class="q-footer-icon">⚖️</div>
    <div>
      <div class="q-footer-brand-name">Qys<b>tas</b></div>
      <div class="q-footer-copy">© 2026 Qystas Smart Pricing Engine — مبني على بيانات توزيع الدخل المصري</div>
    </div>
  </div>
  <div class="q-footer-right">
    <span class="q-footer-pill">ML POWERED</span>
    <span class="q-footer-pill">EGYPT 2026</span>
    <span class="q-footer-pill">LOG-NORMAL</span>
    <span class="q-footer-pill">GRADIENT BOOSTING</span>
  </div>
</div>
""", unsafe_allow_html=True)
