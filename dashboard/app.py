"""
dashboard/app.py  —  Qystas Smart Pricing Engine
=================================================
تشغيل:
  cd pricing_engine
  streamlit run dashboard/app.py
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
# إعداد الصفحة
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Qystas — Smart Pricing Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════
# CSS الكامل
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Cairo:wght@300;400;600;700;900&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

:root {
  --navy:    #0B2545;
  --navy2:   #133A6A;
  --slate:   #4A6FA5;
  --gold:    #C9A84C;
  --gold2:   #E8C96A;
  --bg:      #F0F4F8;
  --bg2:     #E2EAF4;
  --white:   #FFFFFF;
  --text:    #1A2E4A;
  --muted:   #7A8FA6;
  --green:   #1E8C5A;
  --red:     #C0392B;
  --amber:   #E67E22;
  --radius:  14px;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: 'Cairo', 'Inter', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { visibility: hidden !important; }

/* Main content padding */
[data-testid="stAppViewContainer"] > .main > .block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ── Navbar ── */
.qystas-nav {
  background: var(--navy);
  padding: 0 40px;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 999;
  border-bottom: 1px solid rgba(255,255,255,0.07);
}

.nav-logo {
  display: flex;
  align-items: center;
  gap: 12px;
}

.nav-logo-name {
  font-family: 'Inter', sans-serif;
  font-size: 20px;
  font-weight: 900;
  color: #fff;
  letter-spacing: -0.5px;
  line-height: 1;
}

.nav-logo-name .tas { color: var(--gold); }

.nav-tagline {
  font-family: 'Inter', sans-serif;
  font-size: 9px;
  font-weight: 500;
  color: rgba(255,255,255,0.35);
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-top: 3px;
}

.nav-pills {
  display: flex;
  gap: 8px;
}

.nav-pill {
  padding: 5px 14px;
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 100px;
  font-family: 'Inter', sans-serif;
  font-size: 10px;
  font-weight: 600;
  color: rgba(255,255,255,0.4);
  letter-spacing: 0.8px;
}

/* ── Ticker ── */
.ticker-wrap {
  background: var(--gold);
  overflow: hidden;
  padding: 7px 0;
}

.ticker-track {
  display: inline-flex;
  animation: ticker-scroll 28s linear infinite;
  white-space: nowrap;
}

.ticker-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 0 28px;
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 700;
  color: var(--navy);
  letter-spacing: 0.3px;
}

.ticker-sep {
  width: 4px; height: 4px;
  background: var(--navy);
  border-radius: 50%;
  opacity: 0.3;
  flex-shrink: 0;
}

@keyframes ticker-scroll {
  0%   { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}

/* ── Hero ── */
.hero {
  background: linear-gradient(135deg, var(--navy) 0%, var(--navy2) 55%, #1A4A8A 100%);
  padding: 56px 40px 64px;
  position: relative;
  overflow: hidden;
}

.hero::before {
  content: '';
  position: absolute;
  top: -100px; right: -100px;
  width: 420px; height: 420px;
  background: radial-gradient(circle, rgba(201,168,76,0.10) 0%, transparent 65%);
  pointer-events: none;
}

.hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(201,168,76,0.12);
  border: 1px solid rgba(201,168,76,0.28);
  padding: 5px 14px;
  border-radius: 100px;
  margin-bottom: 20px;
}

.hero-dot {
  width: 6px; height: 6px;
  background: var(--gold);
  border-radius: 50%;
  animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

.hero-eyebrow-text {
  font-family: 'Inter', sans-serif;
  font-size: 10px;
  font-weight: 700;
  color: var(--gold);
  letter-spacing: 1.5px;
  text-transform: uppercase;
}

.hero-h1 {
  font-family: 'Inter', sans-serif;
  font-size: clamp(30px, 4vw, 52px);
  font-weight: 900;
  color: #fff;
  line-height: 1.08;
  letter-spacing: -1.5px;
  margin-bottom: 14px;
  max-width: 620px;
}

.hero-h1 .gold { color: var(--gold); }

.hero-p {
  font-family: 'Cairo', sans-serif;
  font-size: 15px;
  color: rgba(255,255,255,0.55);
  line-height: 1.75;
  max-width: 480px;
  margin-bottom: 44px;
}

.hero-kpis {
  display: flex;
  gap: 0;
  flex-wrap: wrap;
}

.hero-kpi {
  padding: 0 36px 0 0;
  margin-right: 36px;
  border-right: 1px solid rgba(255,255,255,0.1);
}

.hero-kpi:last-child { border-right: none; }

.hero-kpi-val {
  font-family: 'Inter', sans-serif;
  font-size: 34px;
  font-weight: 800;
  color: var(--gold);
  letter-spacing: -1.5px;
  line-height: 1;
}

.hero-kpi-lbl {
  font-family: 'Cairo', sans-serif;
  font-size: 11px;
  color: rgba(255,255,255,0.4);
  margin-top: 4px;
}

/* ── Dashboard wrap ── */
.dash {
  padding: 32px 40px 48px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--white) !important;
  border-radius: 12px !important;
  padding: 6px !important;
  border: 1px solid var(--bg2) !important;
  gap: 4px !important;
  box-shadow: 0 1px 4px rgba(11,37,69,0.06) !important;
  width: fit-content !important;
}

.stTabs [data-baseweb="tab"] {
  border-radius: 9px !important;
  font-family: 'Cairo', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
  padding: 9px 20px !important;
  background: transparent !important;
  border: none !important;
}

.stTabs [aria-selected="true"] {
  background: var(--navy) !important;
  color: white !important;
  box-shadow: 0 2px 8px rgba(11,37,69,0.25) !important;
}

.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Cards ── */
.card {
  background: var(--white);
  border-radius: var(--radius);
  border: 1px solid var(--bg2);
  box-shadow: 0 1px 4px rgba(11,37,69,0.04);
  overflow: hidden;
}

.card-hd {
  padding: 18px 24px 14px;
  border-bottom: 1px solid var(--bg2);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.card-title {
  font-family: 'Cairo', sans-serif;
  font-size: 14px;
  font-weight: 700;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: 9px;
}

.card-icon {
  width: 30px; height: 30px;
  border-radius: 8px;
  background: rgba(11,37,69,0.06);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  flex-shrink: 0;
}

/* ── KPI Cards ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.kpi {
  background: var(--white);
  border-radius: var(--radius);
  border: 1px solid var(--bg2);
  padding: 20px 22px 18px;
  box-shadow: 0 1px 3px rgba(11,37,69,0.04);
  transition: box-shadow .2s, transform .2s;
}

.kpi:hover {
  box-shadow: 0 6px 20px rgba(11,37,69,0.09);
  transform: translateY(-2px);
}

.kpi-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.kpi-label {
  font-family: 'Cairo', sans-serif;
  font-size: 11px;
  font-weight: 700;
  color: var(--muted);
  letter-spacing: 0.2px;
}

.kpi-badge {
  width: 32px; height: 32px;
  border-radius: 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
}

.kpi-val {
  font-family: 'Inter', sans-serif;
  font-size: 28px;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -1px;
  line-height: 1;
  margin-bottom: 6px;
}

.kpi-delta {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 700;
  padding: 3px 9px;
  border-radius: 100px;
}

.delta-up   { background: rgba(30,140,90,0.1);  color: var(--green); }
.delta-down { background: rgba(192,57,43,0.1);  color: var(--red); }
.delta-warn { background: rgba(230,126,34,0.1); color: var(--amber); }
.delta-info { background: rgba(74,111,165,0.1); color: var(--slate); }

.kpi-bar {
  height: 5px;
  background: var(--bg2);
  border-radius: 100px;
  margin-top: 10px;
  overflow: hidden;
}

.kpi-fill {
  height: 100%;
  border-radius: 100px;
}

/* ── Rec Cards ── */
.rec {
  border-radius: 12px;
  padding: 18px 20px;
  border: 1.5px solid;
  margin-bottom: 12px;
}

.rec.success { background: rgba(30,140,90,0.05);  border-color: rgba(30,140,90,0.2); }
.rec.warning { background: rgba(230,126,34,0.05); border-color: rgba(230,126,34,0.2); }
.rec.danger  { background: rgba(192,57,43,0.05);  border-color: rgba(192,57,43,0.2); }

.rec-title {
  font-family: 'Cairo', sans-serif;
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 10px;
}

.rec-title.success { color: var(--green); }
.rec-title.warning { color: var(--amber); }
.rec-title.danger  { color: var(--red); }

.rec-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 5px 0;
  border-bottom: 1px solid rgba(0,0,0,0.04);
}

.rec-row:last-child { border-bottom: none; }

.rec-k {
  font-family: 'Cairo', sans-serif;
  font-size: 11px;
  color: var(--muted);
  font-weight: 600;
}

.rec-v {
  font-family: 'Inter', sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.3px;
}

.rec-v.gold  { color: var(--gold); }
.rec-v.green { color: var(--green); }
.rec-v.red   { color: var(--red); }

/* ── Form Inputs ── */
.stNumberInput input,
.stSelectbox select,
.stSlider {
  font-family: 'Inter', sans-serif !important;
}

/* ── Streamlit metric override ── */
[data-testid="stMetric"] {
  background: var(--white) !important;
  border: 1px solid var(--bg2) !important;
  border-radius: var(--radius) !important;
  padding: 18px 20px !important;
  box-shadow: 0 1px 3px rgba(11,37,69,0.04) !important;
}

[data-testid="stMetricLabel"] {
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

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  border-radius: var(--radius) !important;
  border: 1px solid var(--bg2) !important;
  overflow: hidden !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--navy) !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  font-family: 'Cairo', sans-serif !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  padding: 12px 24px !important;
  transition: all .2s !important;
  letter-spacing: 0.3px !important;
}

.stButton > button:hover {
  background: var(--navy2) !important;
  box-shadow: 0 4px 16px rgba(11,37,69,0.25) !important;
  transform: translateY(-1px) !important;
}

/* ── Info / Warning boxes ── */
[data-testid="stAlert"] {
  border-radius: 12px !important;
  font-family: 'Cairo', sans-serif !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
  font-family: 'Cairo', sans-serif !important;
  color: var(--navy) !important;
}

/* ── Footer ── */
.qystas-footer {
  background: var(--navy);
  padding: 24px 40px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 16px;
  margin-top: 16px;
}

.footer-brand {
  display: flex;
  align-items: center;
  gap: 10px;
}

.footer-copy {
  font-family: 'Cairo', sans-serif;
  font-size: 12px;
  color: rgba(255,255,255,0.3);
}

.footer-pills {
  display: flex;
  gap: 8px;
}

.footer-pill {
  padding: 4px 12px;
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 100px;
  font-family: 'Inter', sans-serif;
  font-size: 9px;
  font-weight: 600;
  color: rgba(255,255,255,0.3);
  letter-spacing: 1px;
}

/* ── Divider ── */
hr { border-color: var(--bg2) !important; }

/* ── Radio ── */
.stRadio label {
  font-family: 'Cairo', sans-serif !important;
  font-weight: 600 !important;
}

/* Section headers inside tabs */
.section-hd {
  font-family: 'Cairo', sans-serif;
  font-size: 18px;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 4px;
}

.section-sub {
  font-family: 'Cairo', sans-serif;
  font-size: 13px;
  color: var(--muted);
  margin-bottom: 20px;
  line-height: 1.6;
}

.insight-box {
  background: rgba(11,37,69,0.04);
  border: 1px solid var(--bg2);
  border-right: 4px solid var(--navy);
  border-radius: 10px;
  padding: 14px 18px;
  font-family: 'Cairo', sans-serif;
  font-size: 13px;
  color: var(--text);
  line-height: 1.7;
  margin-top: 16px;
}

.insight-box.gold-border { border-right-color: var(--gold); }
.insight-box.green-border { border-right-color: var(--green); }
.insight-box.red-border   { border-right-color: var(--red); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# الـ SVG Logo
# ══════════════════════════════════════════════
LOGO_SVG = """
<svg width="36" height="36" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="28" cy="28" r="28" fill="#133A6A"/>
  <rect x="14" y="26" width="28" height="3" rx="1.5" fill="#C9A84C"/>
  <rect x="26.5" y="18" width="3" height="20" rx="1.5" fill="#C9A84C"/>
  <path d="M14 27 Q11 33 18 33 Q25 33 22 27" fill="none" stroke="#E8C96A" stroke-width="1.8" stroke-linecap="round"/>
  <path d="M34 27 Q31 31 38 31 Q45 31 42 27" fill="none" stroke="#E8C96A" stroke-width="1.8" stroke-linecap="round" opacity="0.6"/>
  <path d="M39 22 L39 17 M37 19 L39 17 L41 19" stroke="#C9A84C" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

# ══════════════════════════════════════════════
# Navbar
# ══════════════════════════════════════════════
st.markdown(f"""
<div class="qystas-nav">
  <div class="nav-logo">
    {LOGO_SVG}
    <div>
      <div class="nav-logo-name">Qys<span class="tas">tas</span></div>
      <div class="nav-tagline">Smart Pricing Engine</div>
    </div>
  </div>
  <div class="nav-pills">
    <span class="nav-pill">ML POWERED</span>
    <span class="nav-pill">EGYPT 2026</span>
    <span class="nav-pill">v1.0</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# Ticker
# ══════════════════════════════════════════════
ticker_items = [
    "📈 تضخم تراكمي حضري: 2.47×",
    "💰 القوة الشرائية الحقيقية: 40.5%",
    "🍞 ميزانية الأكل الإجبارية: 28.63%",
    "📊 وسيط الدخل 2026: 138,739 ج/سنة",
    "⚖️ Qystas — محرك التسعير الذكي",
    "🌍 تضخم تراكمي ريفي: 2.53×",
    "📉 القوة الشرائية الريفية: 39.49%",
]

def make_ticker(items):
    sep = '<span class="ticker-sep"></span>'
    inner = f" {sep} ".join(
        f'<span class="ticker-item">{i}</span>' for i in items * 2
    )
    return f"""
    <div class="ticker-wrap">
      <div class="ticker-track">{inner}</div>
    </div>
    """

st.markdown(make_ticker(ticker_items), unsafe_allow_html=True)

# ══════════════════════════════════════════════
# Hero
# ══════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">
    <span class="hero-dot"></span>
    <span class="hero-eyebrow-text">بيانات توزيع الدخل المصري 2020 – 2026</span>
  </div>
  <div class="hero-h1">
    قرار التسعير الصح<br>في <span class="gold">ثوانٍ</span>
  </div>
  <div class="hero-p">
    نظام ذكاء اصطناعي يحسب لك الوزن الأمثل ونسبة المقاطعة المتوقعة —
    بناءً على الدخل الحقيقي لزبائنك اليوم.
  </div>
  <div class="hero-kpis">
    <div class="hero-kpi">
      <div class="hero-kpi-val">2.47×</div>
      <div class="hero-kpi-lbl">التضخم التراكمي</div>
    </div>
    <div class="hero-kpi">
      <div class="hero-kpi-val">9</div>
      <div class="hero-kpi-lbl">فئة دخلية محللة</div>
    </div>
    <div class="hero-kpi">
      <div class="hero-kpi-val">0.74</div>
      <div class="hero-kpi-lbl">دقة الموديل AUC</div>
    </div>
    <div class="hero-kpi">
      <div class="hero-kpi-val">24K</div>
      <div class="hero-kpi-lbl">سيناريو تدريب</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# تحميل الـ Resources
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
# Dashboard Body
# ══════════════════════════════════════════════
st.markdown('<div class="dash">', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📊  رادار انكماش الطبقات",
    "🏭  بيانات المنتج",
    "🎯  التوصية الذكية",
])

# ═══════════════════════════════════════
# TAB 1 — رادار انكماش الطبقات
# ═══════════════════════════════════════
with tab1:
    st.markdown('<div class="section-hd">رادار انكماش الطبقات</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">مقارنة منحنى توزيع الدخل بين 2020 و2026 — '
        'الإزاحة لليمين لا تعني تحسناً، بل ارتفاع اسمي مقابل انخفاض حقيقي في القوة الشرائية.</div>',
        unsafe_allow_html=True,
    )

    area_choice = st.radio("المنطقة:", ["Urban", "Rural"], horizontal=True, key="r1")
    data = get_curves(area_choice)
    x   = np.array(data["x"])
    p20 = np.array(data["pdf_2020"])
    p26 = np.array(data["pdf_2026"])

    # ── الرسم ──
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=p20,
        fill="tozeroy", fillcolor="rgba(74,111,165,0.13)",
        line=dict(color="#4A6FA5", width=2.5),
        name="توزيع 2020 (سنة الأساس)",
        hovertemplate="دخل: %{x:,.0f} ج<br>كثافة: %{y:.6f}<extra>2020</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p26,
        fill="tozeroy", fillcolor="rgba(192,57,43,0.10)",
        line=dict(color="#C0392B", width=2.5),
        name="توزيع 2026 (أسعار اليوم)",
        hovertemplate="دخل: %{x:,.0f} ج<br>كثافة: %{y:.6f}<extra>2026</extra>",
    ))
    fig.add_vline(
        x=data["median_2020"], line_dash="dot", line_color="#4A6FA5", line_width=1.5,
        annotation_text=f"وسيط 2020<br>{data['median_2020']:,.0f} ج",
        annotation_font_size=11, annotation_font_color="#4A6FA5",
    )
    fig.add_vline(
        x=data["median_2026"], line_dash="dot", line_color="#C0392B", line_width=1.5,
        annotation_text=f"وسيط 2026<br>{data['median_2026']:,.0f} ج",
        annotation_font_size=11, annotation_font_color="#C0392B",
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            title="الدخل السنوي (جنيه)",
            tickformat=",",
            range=[0, 420_000],
            gridcolor="#F0F4F8",
            title_font=dict(family="Cairo", size=12, color="#7A8FA6"),
        ),
        yaxis=dict(
            title="كثافة الاحتمال",
            gridcolor="#F0F4F8",
            title_font=dict(family="Cairo", size=12, color="#7A8FA6"),
        ),
        legend=dict(
            x=0.72, y=0.97,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E2EAF4",
            borderwidth=1,
            font=dict(family="Cairo", size=12),
        ),
        height=400,
        margin=dict(t=24, b=24, l=16, r=16),
        hovermode="x unified",
        font=dict(family="Cairo"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── KPIs ──
    pwr  = 40.50 if area_choice == "Urban" else 39.49
    food = 28.63 if area_choice == "Urban" else 36.39

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("وسيط الدخل 2020",         f"{data['median_2020']:,.0f} ج/سنة")
    c2.metric("وسيط الدخل 2026",         f"{data['median_2026']:,.0f} ج/سنة",
              delta=f"+{data['shift_pct']:.1f}% إزاحة اسمية")
    c3.metric("القوة الشرائية الحقيقية", f"{pwr}%",
              delta=f"-{100-pwr:.1f}% مقارنة بالاسمي", delta_color="inverse")
    c4.metric("ميزانية الأكل الإجبارية", f"{food}%")

    st.markdown(
        f'<div class="insight-box gold-border">'
        f'💡 <strong>تفسير المنحنيين:</strong> وسيط 2020 كان {data["median_2020"]:,.0f} جنيه سنوياً، '
        f'وارتفع اسمياً لـ {data["median_2026"]:,.0f} جنيه في 2026 — زيادة {data["shift_pct"]:.1f}%. '
        f'لكن بعد خصم أثر التضخم، القوة الشرائية الحقيقية بقت {pwr}% بس من الدخل الاسمي. '
        f'يعني زبونك الوسطي بيكسب أكتر بالأرقام لكن يقدر يشتري أقل.</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════
# TAB 2 — بيانات المنتج
# ═══════════════════════════════════════
with tab2:
    st.markdown('<div class="section-hd">بيانات المنتج</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">أدخل بيانات منتجك وسيقوم النظام بتحليل موقعه في السوق '
        'وحساب التوصية المثلى.</div>',
        unsafe_allow_html=True,
    )

    with st.form("product_form"):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**📦 بيانات المنتج**")
            current_price   = st.number_input("السعر الحالي (جنيه)",
                                               min_value=1.0, max_value=500.0, value=25.0, step=0.5)
            current_weight  = st.number_input("الوزن الحالي (جرام)",
                                               min_value=10.0, max_value=5000.0, value=100.0, step=5.0)
            cost_per_gram   = st.number_input("تكلفة الإنتاج / جرام",
                                               min_value=0.01, max_value=10.0, value=0.18, step=0.01,
                                               help="شاملة الخامات + التشغيل + التعبئة")
            new_price_input = st.number_input("السعر الجديد المقترح (لحساب المقاطعة)",
                                               min_value=1.0, max_value=500.0, value=30.0, step=0.5)

        with c2:
            st.markdown("**⚙️ معاملات التحليل**")
            area_sel      = st.selectbox("المنطقة المستهدفة", ["Urban", "Rural"])
            target_margin = st.slider("هامش الربح المستهدف %", 5, 60, 30, 5) / 100
            purchase_freq = st.slider("تكرار الشراء (مرات/شهر)", 1, 20, 4)

            current_cost   = cost_per_gram * current_weight
            current_margin = (current_price - current_cost) / current_price * 100

            st.markdown("---")
            st.markdown(f"""
            <div class="insight-box">
              <strong>ملخص الإدخال:</strong><br>
              تكلفة الإنتاج الحالية: <strong>{current_cost:.2f} جنيه</strong><br>
              هامش الربح الحالي: <strong>{current_margin:.1f}%</strong><br>
              سعر الجرام الحالي: <strong>{current_price/current_weight:.3f} ج/ج</strong>
            </div>
            """, unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "⚙️ تحليل وإنتاج التوصية الذكية",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        st.session_state["product_data"] = {
            "current_price":    current_price,
            "current_weight_g": current_weight,
            "cost_per_gram":    cost_per_gram,
            "area":             area_sel,
            "target_margin":    target_margin,
            "purchase_freq":    purchase_freq,
            "new_price":        new_price_input,
        }
        st.success("✅ تم الحفظ — انتقل لتبويب التوصية الذكية")

    # ── جدول القدرة الشرائية ──
    if "product_data" in st.session_state:
        pd_data  = st.session_state["product_data"]
        segments = get_bracket_affordability(
            pd_data["current_price"], pd_data["area"], pd_data["purchase_freq"]
        )
        seg_df = pd.DataFrame(segments)

        st.markdown("---")
        st.markdown("**خريطة القدرة الشرائية لكل فئة — بأسعار اليوم**")

        # Bar chart للـ affordability
        colors_bar = ["#1E8C5A" if r else "#C0392B" for r in seg_df["affordable"]]
        fig_aff = go.Figure(go.Bar(
            x=seg_df["bracket"],
            y=seg_df["price_burden_pct"],
            marker_color=colors_bar,
            text=[f"{v:.1f}%" for v in seg_df["price_burden_pct"]],
            textposition="outside",
            hovertemplate="الفئة: %{x}<br>العبء السعري: %{y:.1f}%<extra></extra>",
        ))
        fig_aff.add_hline(
            y=15, line_dash="dot", line_color="#E67E22", line_width=2,
            annotation_text="عتبة 15%", annotation_font_color="#E67E22",
        )
        fig_aff.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(tickangle=-30, gridcolor="#F0F4F8"),
            yaxis=dict(title="العبء السعري %", gridcolor="#F0F4F8"),
            height=300, margin=dict(t=24, b=80, l=16, r=16),
            font=dict(family="Cairo"),
        )
        st.plotly_chart(fig_aff, use_container_width=True)

        seg_df["الحالة"] = seg_df["affordable"].apply(lambda x: "✅ في المتناول" if x else "🔴 خارج المتناول")
        seg_df["الدخل المتاح/شهر"] = seg_df["monthly_disposable"].apply(lambda v: f"{v:,.0f} ج")
        st.dataframe(
            seg_df[["bracket","population_pct","الدخل المتاح/شهر","price_burden_pct","الحالة"]].rename(
                columns={"bracket":"الفئة الدخلية","population_pct":"% من السكان","price_burden_pct":"عبء سعري %"}
            ),
            use_container_width=True, hide_index=True,
        )

# ═══════════════════════════════════════
# TAB 3 — التوصية الذكية
# ═══════════════════════════════════════
with tab3:
    st.markdown('<div class="section-hd">التوصية الذكية</div>', unsafe_allow_html=True)

    if "product_data" not in st.session_state:
        st.warning("👈 أدخل بيانات المنتج في تبويب «بيانات المنتج» أولاً.")
        st.stop()

    pd_data = st.session_state["product_data"]
    product = ProductInput(
        current_price    = pd_data["current_price"],
        current_weight_g = pd_data["current_weight_g"],
        cost_per_gram    = pd_data["cost_per_gram"],
        area             = pd_data["area"],
        purchase_freq    = pd_data["purchase_freq"],
        target_margin    = pd_data["target_margin"],
    )

    with st.spinner("جاري التحليل بالموديل..."):
        w_rec  = optimizer.find_optimal_weight(product)
        c_pred = optimizer.predict_market_churn(product, pd_data["new_price"])

    # ── A: الوزن الأمثل ──
    st.markdown("### 🅐 توصية الوزن الأمثل")
    st.markdown(
        '<div class="section-sub">للحفاظ على السعر النفسي للزبون مع ضمان هامش الربح المستهدف</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("الوزن الحالي",       f"{product.current_weight_g:.0f} جرام")
    c2.metric("الوزن الأمثل",       f"{w_rec.optimal_weight_g} جرام",
              delta=f"-{w_rec.weight_reduction_pct}%", delta_color="inverse")
    c3.metric("هامش الربح الجديد",  f"{w_rec.new_margin_pct}%")
    c4.metric("سعر الجرام الجديد",  f"{w_rec.price_per_gram_new} ج/ج")

    border = "green" if w_rec.feasible else "red"
    st.markdown(
        f'<div class="insight-box {border}-border"><strong>{w_rec.warning}</strong></div>',
        unsafe_allow_html=True,
    )

    # ── Bar مقارنة الوزن ──
    fig_w = go.Figure(go.Bar(
        x=["الوزن الحالي", "الوزن الأمثل"],
        y=[product.current_weight_g, w_rec.optimal_weight_g],
        marker_color=["#4A6FA5", "#1E8C5A"],
        text=[f"{product.current_weight_g:.0f}g", f"{w_rec.optimal_weight_g}g"],
        textposition="outside",
        width=0.45,
    ))
    fig_w.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="الوزن (جرام)", gridcolor="#F0F4F8"),
        height=260, margin=dict(t=20, b=20, l=16, r=16),
        font=dict(family="Cairo"),
    )
    st.plotly_chart(fig_w, use_container_width=True)

    st.divider()

    # ── B: تنبؤ المقاطعة ──
    st.markdown("### 🅑 تنبؤ نسبة المقاطعة")
    st.markdown(
        f'<div class="section-sub">لو رفعت السعر من {product.current_price:.0f} '
        f'لـ {c_pred.new_price:.0f} جنيه (+{c_pred.price_increase_pct}%)</div>',
        unsafe_allow_html=True,
    )

    risk_border = {"HIGH":"red","MEDIUM":"warning","LOW":"green"}[c_pred.risk_level]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("نسبة المقاطعة المتوقعة", f"{c_pred.weighted_churn_pct}%")
    c2.metric("السكان المعرضون للخطر",  f"{c_pred.at_risk_population_pct}%")
    c3.metric("احتمال المقاطعة ML",     f"{c_pred.ml_churn_prob}%")
    c4.metric("مستوى الخطر",            c_pred.risk_level)

    st.markdown(
        f'<div class="insight-box {risk_border}-border"><strong>{c_pred.recommendation}</strong></div>',
        unsafe_allow_html=True,
    )

    # ── Segment chart ──
    st.markdown("**تفاصيل التأثير على كل فئة دخلية**")
    seg_df = pd.DataFrame(c_pred.segments_detail)

    fig_s = go.Figure()
    fig_s.add_trace(go.Bar(
        x=seg_df["bracket"],
        y=seg_df["price_burden_pct"],
        marker_color=["#C0392B" if r else "#1E8C5A" for r in seg_df["at_risk"]],
        name="العبء السعري %",
        text=[f"{v:.1f}%" for v in seg_df["price_burden_pct"]],
        textposition="outside",
    ))
    fig_s.add_trace(go.Scatter(
        x=seg_df["bracket"],
        y=seg_df["churn_threshold_pct"],
        mode="lines+markers",
        name="عتبة المقاطعة",
        line=dict(color="#E67E22", width=2.5, dash="dash"),
        marker=dict(size=7, color="#E67E22"),
    ))
    fig_s.add_trace(go.Bar(
        x=seg_df["bracket"],
        y=[v * 100 for v in seg_df["ml_churn_prob"]],
        name="ML احتمال المقاطعة %",
        marker_color="rgba(74,111,165,0.5)",
        opacity=0.8,
    ))
    fig_s.update_layout(
        barmode="overlay",
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickangle=-30, gridcolor="#F0F4F8"),
        yaxis=dict(title="%", gridcolor="#F0F4F8"),
        legend=dict(x=0.68, y=0.97, font=dict(family="Cairo", size=11),
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2EAF4", borderwidth=1),
        height=380, margin=dict(t=20, b=90, l=16, r=16),
        font=dict(family="Cairo"),
    )
    st.plotly_chart(fig_s, use_container_width=True)

    # ── جدول مفصل ──
    seg_show = seg_df[[
        "bracket","population_pct","monthly_disposable",
        "price_burden_pct","churn_threshold_pct","at_risk","ml_churn_prob"
    ]].copy()
    seg_show.columns = [
        "الفئة الدخلية","% السكان","دخل متاح/شهر",
        "عبء سعري %","عتبة المقاطعة %","في خطر؟","ML احتمال"
    ]
    seg_show["في خطر؟"]  = seg_show["في خطر؟"].apply(lambda x: "🔴 نعم" if x else "🟢 لا")
    seg_show["ML احتمال"] = seg_show["ML احتمال"].apply(lambda x: f"{x:.0%}")
    st.dataframe(seg_show, use_container_width=True, hide_index=True)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════
st.markdown(f"""
<div class="qystas-footer">
  <div class="footer-brand">
    {LOGO_SVG}
    <span class="footer-copy">© 2026 Qystas — Smart Pricing Engine. مبني على بيانات توزيع الدخل المصري.</span>
  </div>
  <div class="footer-pills">
    <span class="footer-pill">ML POWERED</span>
    <span class="footer-pill">EGYPT 2026</span>
    <span class="footer-pill">LOG-NORMAL</span>
    <span class="footer-pill">GRADIENT BOOSTING</span>
  </div>
</div>
""", unsafe_allow_html=True)
