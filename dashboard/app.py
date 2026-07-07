"""
Qystas — Smart Pricing Engine
World-class UI. 1700-line version. Dataframe fix applied.
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

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Qystas — Smart Pricing Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────
# DESIGN SYSTEM + CSS
# KEY FIX: no CSS touches iframe / stDataFrame internals
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ═══════════════════════════════════════
   TOKENS
═══════════════════════════════════════ */
:root {
  --void:     #060912;
  --surface:  #0D1117;
  --raised:   #161B27;
  --border:   rgba(255,255,255,0.06);
  --border-2: rgba(255,255,255,0.10);
  --iris:     #7C3AED;
  --iris-2:   #9D5FF5;
  --iris-glow:rgba(124,58,237,0.35);
  --jade:     #06D6A0;
  --jade-glow:rgba(6,214,160,0.25);
  --crimson:  #FF4757;
  --crim-glow:rgba(255,71,87,0.25);
  --amber:    #FFB020;
  --snow:     #F8FAFC;
  --mist:     #8B9AB3;
  --dim:      #4A5568;
  --r-sm:     8px;
  --r-md:     14px;
  --r-lg:     20px;
  --r-xl:     28px;
  --ease:     cubic-bezier(0.4, 0, 0.2, 1);
}

/* ═══════════════════════════════════════
   RESET — scoped only to known containers
   NEVER use * { color } — breaks iframe
═══════════════════════════════════════ */
.q-nav, .q-ticker-wrap, .q-hero, .q-body,
.q-kpi, .q-callout, .q-rec, .q-footer {
  box-sizing: border-box;
}

/* ── App shell background only ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
  background: var(--void) !important;
  font-family: 'Inter', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="collapsedControl"],
.stDeployButton { display: none !important; }

[data-testid="stAppViewContainer"] > .main > .block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb { background: var(--raised); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--iris); }

/* ═══════════════════════════════════════
   TOPBAR
═══════════════════════════════════════ */
.q-nav {
  position: sticky; top: 0; z-index: 999;
  height: 64px;
  background: rgba(6,9,18,0.88);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 44px;
}
.q-nav-logo { display: flex; align-items: center; gap: 13px; }
.q-nav-mark {
  width: 38px; height: 38px;
  background: linear-gradient(135deg, var(--iris), var(--iris-2));
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 19px;
  box-shadow: 0 0 22px var(--iris-glow);
  flex-shrink: 0;
}
.q-nav-name {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 22px; font-weight: 700;
  color: var(--snow); letter-spacing: -0.8px; line-height: 1;
}
.q-nav-name b { color: var(--iris-2); }
.q-nav-sub {
  font-size: 9px; font-weight: 600; color: var(--dim);
  letter-spacing: 2.2px; text-transform: uppercase; margin-top: 2px;
}
.q-nav-right { display: flex; align-items: center; gap: 8px; }
.q-chip {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 5px 13px; border-radius: 100px;
  font-size: 10px; font-weight: 700;
  letter-spacing: 0.7px; text-transform: uppercase;
  transition: all 0.2s var(--ease);
}
.chip-ghost { border: 1px solid var(--border-2); color: var(--dim); background: transparent; }
.chip-ghost:hover { border-color: var(--iris); color: var(--iris-2); background: rgba(124,58,237,0.08); }
.chip-live {
  background: rgba(6,214,160,0.10); border: 1px solid rgba(6,214,160,0.22); color: var(--jade);
}
.chip-live::before {
  content: ''; width: 6px; height: 6px; background: var(--jade); border-radius: 50%;
  animation: live 1.8s ease infinite;
}
@keyframes live { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
.chip-iris { background: linear-gradient(135deg,var(--iris),var(--iris-2)); color:#fff; box-shadow:0 2px 12px var(--iris-glow); }
.chip-amber { background:rgba(255,176,32,0.12); border:1px solid rgba(255,176,32,0.2); color:var(--amber); }

/* ═══════════════════════════════════════
   TICKER
═══════════════════════════════════════ */
.q-ticker-wrap {
  height: 38px; overflow: hidden;
  background: linear-gradient(90deg,rgba(124,58,237,0.14),rgba(124,58,237,0.07),rgba(124,58,237,0.14));
  border-bottom: 1px solid rgba(124,58,237,0.18);
  display: flex; align-items: center;
}
.q-ticker-scroll {
  display: inline-flex;
  animation: qtick 36s linear infinite;
  white-space: nowrap; align-items: center;
}
.q-tick {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 0 30px; font-size: 11px; font-weight: 600; color: var(--mist);
}
.q-tick b { color: var(--iris-2); font-weight: 700; }
.q-tick-sep { width: 3px; height: 3px; background: var(--dim); border-radius: 50%; opacity: 0.4; flex-shrink:0; }
@keyframes qtick { 0%{transform:translateX(0);} 100%{transform:translateX(-50%);} }

/* ═══════════════════════════════════════
   HERO
═══════════════════════════════════════ */
.q-hero {
  position: relative; overflow: hidden;
  padding: 72px 44px 88px; background: var(--void);
}
.q-hero-bg { position: absolute; inset: 0; pointer-events: none; }
.q-hero-orb-1 {
  position: absolute; top: -140px; left: -100px;
  width: 560px; height: 560px;
  background: radial-gradient(circle,rgba(124,58,237,0.11) 0%,transparent 65%);
  border-radius: 50%;
}
.q-hero-orb-2 {
  position: absolute; bottom: -180px; right: -80px;
  width: 460px; height: 460px;
  background: radial-gradient(circle,rgba(6,214,160,0.07) 0%,transparent 65%);
  border-radius: 50%;
}
.q-hero-grid {
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.017) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.017) 1px, transparent 1px);
  background-size: 54px 54px;
}
.q-hero-grid::after {
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(to bottom, transparent 55%, var(--void) 100%);
}
.q-hero-eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(124,58,237,0.09); border: 1px solid rgba(124,58,237,0.24);
  padding: 6px 16px; border-radius: 100px; margin-bottom: 26px; position: relative;
}
.q-hero-eyebrow-dot {
  width: 7px; height: 7px; background: var(--iris-2); border-radius: 50%;
  box-shadow: 0 0 9px var(--iris-glow); animation: edot 2.2s ease infinite;
}
@keyframes edot { 0%,100%{transform:scale(1);opacity:1;} 50%{transform:scale(0.75);opacity:0.45;} }
.q-hero-eyebrow-text {
  font-size: 10px; font-weight: 700; color: var(--iris-2);
  letter-spacing: 1.5px; text-transform: uppercase;
}
.q-hero-h1 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: clamp(38px,5.5vw,72px); font-weight: 700;
  color: var(--snow); line-height: 1.04; letter-spacing: -2.5px;
  margin-bottom: 18px; max-width: 700px; position: relative;
}
.q-hero-h1-accent {
  background: linear-gradient(135deg,var(--iris-2) 0%,#A78BFA 50%,var(--jade) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.q-hero-sub {
  font-size: 17px; color: var(--mist); line-height: 1.72;
  max-width: 510px; margin-bottom: 52px; position: relative;
}
.q-hero-stats { display: flex; align-items: stretch; gap: 0; position: relative; }
.q-hero-stat { padding: 0 44px 0 0; margin-right: 44px; border-right: 1px solid var(--border); }
.q-hero-stat:last-child { border-right: none; margin-right: 0; padding-right: 0; }
.q-hero-stat-val {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 40px; font-weight: 700; letter-spacing: -2px; line-height: 1;
}
.q-hero-stat-val.iris { background:linear-gradient(135deg,var(--iris-2),#A78BFA); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.q-hero-stat-val.jade { background:linear-gradient(135deg,var(--jade),#34D399); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.q-hero-stat-val.white { background:linear-gradient(135deg,#fff 60%,rgba(255,255,255,0.5)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.q-hero-stat-label { font-size: 11px; font-weight: 500; color: var(--dim); margin-top: 6px; }

/* ═══════════════════════════════════════
   BODY
═══════════════════════════════════════ */
.q-body { padding: 36px 44px 60px; background: var(--void); display: flex; flex-direction: column; gap: 28px; }

/* ── Section labels ── */
.q-section-label { font-size: 10px; font-weight: 700; color: var(--dim); letter-spacing: 2px; text-transform: uppercase; margin-bottom: 5px; }
.q-section-title { font-family:'Space Grotesk',sans-serif; font-size: 22px; font-weight: 600; color: var(--snow); letter-spacing: -0.6px; margin-bottom: 6px; }
.q-section-sub { font-size: 13px; color: var(--mist); line-height: 1.65; margin-bottom: 24px; max-width: 580px; }

/* ═══════════════════════════════════════
   KPI GRID
═══════════════════════════════════════ */
.q-kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; }
.q-kpi {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-lg);
  padding: 22px 22px 18px; position: relative; overflow: hidden;
  transition: border-color .25s var(--ease), box-shadow .25s var(--ease), transform .25s var(--ease);
  cursor: default;
}
.q-kpi::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; border-radius:var(--r-lg) var(--r-lg) 0 0; }
.kpi-iris::before  { background:linear-gradient(90deg,var(--iris),var(--iris-2)); }
.kpi-jade::before  { background:linear-gradient(90deg,var(--jade),#34D399); }
.kpi-crim::before  { background:linear-gradient(90deg,var(--crimson),#FF6B78); }
.kpi-amber::before { background:linear-gradient(90deg,var(--amber),#FCD34D); }
.q-kpi:hover { transform: translateY(-4px); }
.kpi-iris:hover  { border-color:rgba(124,58,237,0.35); box-shadow:0 18px 50px rgba(124,58,237,0.10); }
.kpi-jade:hover  { border-color:rgba(6,214,160,0.28);  box-shadow:0 18px 50px rgba(6,214,160,0.07); }
.kpi-crim:hover  { border-color:rgba(255,71,87,0.28);   box-shadow:0 18px 50px rgba(255,71,87,0.07); }
.kpi-amber:hover { border-color:rgba(255,176,32,0.28);  box-shadow:0 18px 50px rgba(255,176,32,0.07); }
.q-kpi-glow { position:absolute; top:-50px; right:-50px; width:140px; height:140px; border-radius:50%; opacity:0.5; pointer-events:none; transition:opacity .3s; }
.kpi-iris  .q-kpi-glow { background:radial-gradient(circle,rgba(124,58,237,0.18) 0%,transparent 70%); }
.kpi-jade  .q-kpi-glow { background:radial-gradient(circle,rgba(6,214,160,0.14) 0%,transparent 70%); }
.kpi-crim  .q-kpi-glow { background:radial-gradient(circle,rgba(255,71,87,0.14) 0%,transparent 70%); }
.kpi-amber .q-kpi-glow { background:radial-gradient(circle,rgba(255,176,32,0.14) 0%,transparent 70%); }
.q-kpi:hover .q-kpi-glow { opacity:1; }
.q-kpi-top { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:14px; }
.q-kpi-label { font-size:10px; font-weight:600; color:var(--dim); line-height:1.45; max-width:110px; }
.q-kpi-icon { width:34px; height:34px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:15px; flex-shrink:0; }
.ico-iris  { background:rgba(124,58,237,0.14); }
.ico-jade  { background:rgba(6,214,160,0.11); }
.ico-crim  { background:rgba(255,71,87,0.11); }
.ico-amber { background:rgba(255,176,32,0.11); }
.q-kpi-val { font-family:'Space Grotesk',sans-serif; font-size:30px; font-weight:700; color:var(--snow); letter-spacing:-1.5px; line-height:1; margin-bottom:8px; }
.q-kpi-delta { display:inline-flex; align-items:center; gap:3px; font-size:10px; font-weight:700; padding:3px 9px; border-radius:100px; }
.d-iris  { background:rgba(124,58,237,0.12); color:var(--iris-2); }
.d-jade  { background:rgba(6,214,160,0.10);  color:var(--jade); }
.d-crim  { background:rgba(255,71,87,0.10);  color:var(--crimson); }
.d-amber { background:rgba(255,176,32,0.10); color:var(--amber); }
.q-kpi-bar { height:2px; background:var(--raised); border-radius:100px; margin-top:12px; overflow:hidden; }
.q-kpi-fill { height:100%; border-radius:100px; }

/* ═══════════════════════════════════════
   TABS
═══════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important; border-radius: var(--r-md) !important;
  padding: 5px !important; gap: 3px !important; border: 1px solid var(--border) !important;
  width: fit-content !important; box-shadow: inset 0 1px 0 rgba(255,255,255,0.04) !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px !important; font-family: 'Inter',sans-serif !important;
  font-size: 13px !important; font-weight: 600 !important; color: var(--dim) !important;
  padding: 9px 20px !important; background: transparent !important; border: none !important;
  transition: all 0.2s var(--ease) !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--mist) !important; background: var(--raised) !important; }
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg,var(--iris),var(--iris-2)) !important;
  color: #fff !important; box-shadow: 0 4px 16px var(--iris-glow) !important;
}
.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ═══════════════════════════════════════
   CALLOUT
═══════════════════════════════════════ */
.q-callout {
  border-radius: var(--r-md); padding: 15px 18px; margin-top: 14px;
  border: 1px solid; display: flex; gap: 11px; align-items: flex-start;
  font-family: 'Inter',sans-serif; font-size: 13px; line-height: 1.65;
}
.call-iris  { background:rgba(124,58,237,0.06); border-color:rgba(124,58,237,0.2); color:#C4B5FD; }
.call-jade  { background:rgba(6,214,160,0.05);  border-color:rgba(6,214,160,0.2);  color:#6EE7C7; }
.call-crim  { background:rgba(255,71,87,0.06);   border-color:rgba(255,71,87,0.2);   color:#FCA5A5; }
.call-amber { background:rgba(255,176,32,0.06);  border-color:rgba(255,176,32,0.2);  color:#FCD34D; }
.call-ico   { font-size:15px; flex-shrink:0; margin-top:1px; }

/* ═══════════════════════════════════════
   STREAMLIT NATIVE — safe overrides only
   (no color on * selector, no iframe touch)
═══════════════════════════════════════ */
[data-testid="stMetric"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r-lg) !important; padding: 20px 22px !important;
  transition: transform .2s var(--ease), box-shadow .2s var(--ease) !important;
}
[data-testid="stMetric"]:hover { transform: translateY(-3px) !important; box-shadow: 0 16px 44px rgba(0,0,0,0.3) !important; }
[data-testid="stMetricLabel"] p {
  font-family: 'Inter',sans-serif !important; font-size: 10px !important;
  font-weight: 700 !important; color: var(--dim) !important;
  letter-spacing: 0.5px !important; text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Space Grotesk',sans-serif !important; font-size: 26px !important;
  font-weight: 700 !important; color: var(--snow) !important; letter-spacing: -1px !important;
}
[data-testid="stMetricDelta"] > div {
  font-family: 'Inter',sans-serif !important; font-size: 11px !important; font-weight: 700 !important;
}

/* ── Alert ── */
[data-testid="stAlert"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important; color: var(--mist) !important;
  font-family: 'Inter',sans-serif !important;
}

/* ── Button ── */
.stButton > button {
  background: linear-gradient(135deg,var(--iris),var(--iris-2)) !important;
  color: white !important; border: none !important; border-radius: var(--r-md) !important;
  font-family: 'Inter',sans-serif !important; font-size: 14px !important;
  font-weight: 700 !important; padding: 13px 28px !important;
  box-shadow: 0 4px 18px var(--iris-glow) !important;
  transition: all 0.2s var(--ease) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important; box-shadow: 0 8px 28px var(--iris-glow) !important;
  filter: brightness(1.08) !important;
}

/* ── Form labels ── */
.stRadio > label, div[role="radiogroup"] label {
  font-family: 'Inter',sans-serif !important; font-size: 13px !important;
  font-weight: 600 !important; color: var(--mist) !important;
}
.stSlider label, .stSelectbox label, .stNumberInput label {
  font-family: 'Inter',sans-serif !important; font-size: 11px !important;
  font-weight: 700 !important; color: var(--dim) !important;
  letter-spacing: 0.5px !important; text-transform: uppercase !important;
}
.stSelectbox [data-baseweb="select"] {
  background: var(--raised) !important; border-radius: 8px !important;
  border-color: var(--border-2) !important;
}
.stSelectbox [data-baseweb="select"] > div {
  background: var(--raised) !important; color: var(--snow) !important;
  font-family: 'Inter',sans-serif !important;
}
.stNumberInput input {
  background: var(--raised) !important; border-color: var(--border-2) !important;
  color: var(--snow) !important; font-family: 'Space Grotesk',sans-serif !important;
  font-size: 15px !important; font-weight: 500 !important; border-radius: 8px !important;
}
.stSpinner > div { border-top-color: var(--iris) !important; }
.stSpinner p { font-family: 'Inter',sans-serif !important; color: var(--mist) !important; }
.stMarkdown h3 { font-family: 'Space Grotesk',sans-serif !important; color: var(--snow) !important; }

/* ── Vertical block bg ── */
[data-testid="stVerticalBlock"] { background: var(--void) !important; }

/* ── hr ── */
hr { border-color: var(--border) !important; margin: 24px 0 !important; }

/* ═══════════════════════════════════════
   DATAFRAME — THE FIX
   Do NOT set color/background on the
   dataframe container or its children.
   Only set the outer wrapper border/radius.
═══════════════════════════════════════ */
[data-testid="stDataFrame"] {
  border-radius: var(--r-md) !important;
  overflow: hidden !important;
  border: 1px solid var(--border-2) !important;
}

/* ═══════════════════════════════════════
   FOOTER
═══════════════════════════════════════ */
.q-footer {
  background: var(--surface); border-top: 1px solid var(--border);
  padding: 26px 44px; display: flex; align-items: center;
  justify-content: space-between; flex-wrap: wrap; gap: 14px; margin-top: 28px;
}
.q-footer-brand { display: flex; align-items: center; gap: 12px; }
.q-footer-mark {
  width: 32px; height: 32px;
  background: linear-gradient(135deg,var(--iris),var(--iris-2));
  border-radius: 8px; display: flex; align-items: center; justify-content: center;
  font-size: 15px; box-shadow: 0 0 14px var(--iris-glow);
}
.q-footer-name { font-family:'Space Grotesk',sans-serif; font-size:15px; font-weight:700; color:var(--snow); letter-spacing:-0.3px; }
.q-footer-name b { color:var(--iris-2); }
.q-footer-copy { font-size:10px; color:var(--dim); margin-top:2px; }
.q-footer-pills { display:flex; gap:7px; flex-wrap:wrap; }
.q-footer-pill {
  padding:4px 12px; border:1px solid var(--border); border-radius:100px;
  font-size:9px; font-weight:700; color:var(--dim);
  letter-spacing:1.2px; text-transform:uppercase; transition:all .2s var(--ease); cursor:default;
}
.q-footer-pill:hover { border-color:var(--iris); color:var(--iris-2); }
.q-div { height:1px; background:var(--border); margin:28px 0; }

/* ═══════════════════════════════════════
   RESPONSIVE — TABLET (≤ 1024px)
═══════════════════════════════════════ */
@media (max-width: 1024px) {
  /* Nav */
  .q-nav { padding: 0 24px; }
  .chip-ghost, .chip-amber { display: none; }

  /* Hero */
  .q-hero { padding: 52px 24px 64px; }
  .q-hero-h1 { letter-spacing: -1.5px; }
  .q-hero-sub { font-size: 15px; }
  .q-hero-stats { gap: 0; flex-wrap: wrap; }
  .q-hero-stat { padding: 0 28px 0 0; margin-right: 28px; }
  .q-hero-stat-val { font-size: 30px; }

  /* Body */
  .q-body { padding: 24px 24px 48px; gap: 20px; }

  /* KPI grid → 2×2 */
  .q-kpi-grid { grid-template-columns: repeat(2,1fr); gap: 12px; }
}

/* ═══════════════════════════════════════
   RESPONSIVE — MOBILE (≤ 640px)
═══════════════════════════════════════ */
@media (max-width: 640px) {
  /* ── Nav ── */
  .q-nav { padding: 0 16px; height: 56px; }
  .q-nav-sub { display: none; }
  .q-nav-name { font-size: 18px; }
  .q-nav-mark { width: 32px; height: 32px; font-size: 16px; }
  .chip-ghost, .chip-amber { display: none; }
  .chip-live { padding: 4px 10px; font-size: 9px; }
  .chip-iris { padding: 4px 10px; font-size: 9px; }

  /* ── Ticker ── */
  .q-ticker-wrap { height: 32px; }
  .q-tick { font-size: 10px; padding: 0 18px; gap: 5px; }

  /* ── Hero ── */
  .q-hero { padding: 36px 16px 48px; }
  .q-hero-eyebrow { padding: 5px 12px; margin-bottom: 18px; }
  .q-hero-eyebrow-text { font-size: 9px; letter-spacing: 1px; }
  .q-hero-h1 {
    font-size: clamp(28px, 8vw, 42px);
    letter-spacing: -1px;
    margin-bottom: 14px;
  }
  .q-hero-sub { font-size: 14px; line-height: 1.65; margin-bottom: 36px; max-width: 100%; }

  /* Hero stats → 2×2 grid */
  .q-hero-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 20px 0; }
  .q-hero-stat {
    padding: 0; margin: 0;
    border-right: none !important;
  }
  .q-hero-stat:nth-child(odd) { border-right: 1px solid var(--border) !important; padding-right: 16px; }
  .q-hero-stat-val { font-size: 28px; }
  .q-hero-stat-label { font-size: 10px; }

  /* ── Body ── */
  .q-body { padding: 16px 16px 40px; gap: 16px; }

  /* ── KPI → single column ── */
  .q-kpi-grid { grid-template-columns: 1fr 1fr; gap: 10px; }
  .q-kpi { padding: 16px 16px 14px; border-radius: var(--r-md); }
  .q-kpi-val { font-size: 22px; }
  .q-kpi-label { font-size: 9px; max-width: 90px; }
  .q-kpi-icon { width: 28px; height: 28px; font-size: 13px; }
  .q-kpi-delta { font-size: 9px; padding: 2px 7px; }

  /* ── Section headers ── */
  .q-section-title { font-size: 18px; }
  .q-section-sub { font-size: 12px; }

  /* ── Callout ── */
  .q-callout { font-size: 12px; padding: 12px 14px; gap: 9px; }
  .call-ico { font-size: 13px; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] { width: 100% !important; }
  .stTabs [data-baseweb="tab"] {
    font-size: 11px !important;
    padding: 8px 12px !important;
    flex: 1 !important;
    justify-content: center !important;
    text-align: center !important;
  }

  /* ── Metrics ── */
  [data-testid="stMetric"] { padding: 14px 16px !important; }
  [data-testid="stMetricValue"] { font-size: 20px !important; }
  [data-testid="stMetricLabel"] p { font-size: 9px !important; }

  /* ── Charts: reduce height on mobile ── */
  .js-plotly-plot { max-height: 280px; }

  /* ── Footer ── */
  .q-footer { padding: 20px 16px; flex-direction: column; align-items: flex-start; gap: 12px; }
  .q-footer-pills { display: none; }
  .q-footer-copy { font-size: 10px; }

  /* ── Streamlit column fix on mobile ──
     Streamlit columns stack on very small screens but
     we nudge the gap so they don't feel cramped          */
  [data-testid="stHorizontalBlock"] { gap: 8px !important; }

  /* ── Number inputs full width ── */
  .stNumberInput { width: 100% !important; }
  .stSelectbox  { width: 100% !important; }

  /* ── Button ── */
  .stButton > button { font-size: 13px !important; padding: 11px 20px !important; }
}

/* ═══════════════════════════════════════
   RESPONSIVE — VERY SMALL (≤ 380px)
═══════════════════════════════════════ */
@media (max-width: 380px) {
  .q-kpi-grid { grid-template-columns: 1fr; }
  .q-hero-stats { grid-template-columns: 1fr; gap: 16px; }
  .q-hero-stat { border-right: none !important; padding-right: 0 !important; }
  .q-hero-stat-val { font-size: 32px; }
  .stTabs [data-baseweb="tab"] { font-size: 10px !important; padding: 7px 8px !important; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# PLOTLY SHARED THEME
# ─────────────────────────────────────────────────────────
CHART = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#8B9AB3"),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.07)",
        zeroline=False, tickfont=dict(family="Inter", size=11, color="#4A5568"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.07)",
        zeroline=False, tickfont=dict(family="Inter", size=11, color="#4A5568"),
    ),
    margin=dict(t=24, b=20, l=8, r=8),
    hoverlabel=dict(
        bgcolor="rgba(22,27,39,0.96)", bordercolor="rgba(124,58,237,0.4)",
        font=dict(family="Inter", size=12, color="#F8FAFC"),
    ),
    legend=dict(
        bgcolor="rgba(13,17,23,0.88)", bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1, font=dict(family="Inter", size=12, color="#8B9AB3"),
    ),
)

def chart(**overrides):
    base = {**CHART}
    base.update(overrides)
    return base

# ─────────────────────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="q-nav">
  <div class="q-nav-logo">
    <div class="q-nav-mark">⚖️</div>
    <div>
      <div class="q-nav-name">Qys<b>tas</b></div>
      <div class="q-nav-sub">Smart Pricing Engine</div>
    </div>
  </div>
  <div class="q-nav-right">
    <span class="q-chip chip-live">Live Model</span>
    <span class="q-chip chip-ghost">Egypt 2026</span>
    <span class="q-chip chip-amber">ML Powered</span>
    <span class="q-chip chip-iris">v1.0</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# TICKER
# ─────────────────────────────────────────────────────────
TICKS = [
    ("Urban Inflation",         "2.47×"),
    ("Real Purchasing Power",   "40.5%"),
    ("Food Budget (Urban)",     "28.63%"),
    ("Income Median 2026",      "138,739 EGP"),
    ("Model AUC Score",         "0.74"),
    ("Rural Inflation",         "2.53×"),
    ("Rural Purchasing Power",  "39.49%"),
    ("Training Scenarios",      "24,480"),
    ("Food Budget (Rural)",     "36.39%"),
    ("Income Brackets",         "9 analysed"),
]
SEP = '<span class="q-tick-sep"></span>'
ticks_html = f" {SEP} ".join(
    f'<span class="q-tick">{lbl}: <b>{val}</b></span>'
    for lbl, val in TICKS
) * 2

st.markdown(f"""
<div class="q-ticker-wrap">
  <div class="q-ticker-scroll">{ticks_html}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
.q-hero-logo-block{display:flex;flex-direction:column;align-items:center;gap:14px;flex-shrink:0;}
.q-hero-logo-mark{width:130px;height:130px;background:linear-gradient(135deg,#7C3AED,#9D5FF5);border-radius:32px;display:flex;align-items:center;justify-content:center;font-size:64px;box-shadow:0 0 60px rgba(124,58,237,0.45),0 0 120px rgba(124,58,237,0.15);animation:logo-float 4s ease-in-out infinite;}
@keyframes logo-float{0%,100%{transform:translateY(0);}50%{transform:translateY(-8px);}}
.q-hero-logo-name{font-family:'Space Grotesk',sans-serif;font-size:42px;font-weight:700;color:#F8FAFC;letter-spacing:-1.5px;line-height:1;}
.q-hero-logo-name b{color:#9D5FF5;}
.q-hero-logo-tag{font-size:11px;font-weight:600;color:#4A5568;letter-spacing:2px;text-transform:uppercase;}
@media(max-width:768px){.q-hero-logo-block{display:none;}}
</style>

<div class="q-hero">
  <div class="q-hero-bg">
    <div class="q-hero-grid"></div>
    <div class="q-hero-orb-1"></div>
    <div class="q-hero-orb-2"></div>
  </div>
  <div style="display:flex;align-items:center;justify-content:space-between;gap:32px;position:relative;">
    <div style="flex:1;min-width:0;">
      <div class="q-hero-eyebrow">
        <span class="q-hero-eyebrow-dot"></span>
        <span class="q-hero-eyebrow-text">Egyptian Income Distribution · 2020 – 2026</span>
      </div>
      <div class="q-hero-h1">
        Price smarter.<br><span class="q-hero-h1-accent">Win the market.</span>
      </div>
      <div class="q-hero-sub">
        AI-powered pricing intelligence for Egyptian manufacturers — grounded in real income
        distribution data so every decision reflects how your customers actually live today.
      </div>
      <div class="q-hero-stats">
        <div class="q-hero-stat">
          <div class="q-hero-stat-val iris">2.47×</div>
          <div class="q-hero-stat-label">Cumulative Urban Inflation</div>
        </div>
        <div class="q-hero-stat">
          <div class="q-hero-stat-val jade">40.5%</div>
          <div class="q-hero-stat-label">Real Purchasing Power</div>
        </div>
        <div class="q-hero-stat">
          <div class="q-hero-stat-val white">24K</div>
          <div class="q-hero-stat-label">Training Scenarios</div>
        </div>
        <div class="q-hero-stat">
          <div class="q-hero-stat-val iris">0.74</div>
          <div class="q-hero-stat-label">Model AUC Score</div>
        </div>
      </div>
    </div>
    <div class="q-hero-logo-block">
      <div class="q-hero-logo-mark">⚖️</div>
      <div class="q-hero-logo-name">Qys<b>tas</b></div>
      <div class="q-hero-logo-tag">Smart Pricing Engine</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_optimizer():
    return PricingOptimizer()

@st.cache_resource(show_spinner=False)
def get_model():
    return load_churn_model()

@st.cache_data(show_spinner=False)
def get_curves(area):
    return get_curve_data(area)

with st.spinner("Initializing AI engine..."):
    optimizer = get_optimizer()
    model     = get_model()

# ─────────────────────────────────────────────────────────
# BODY
# ─────────────────────────────────────────────────────────
st.markdown('<div class="q-body">', unsafe_allow_html=True)

# ── KPI ROW ──
st.markdown("""
<div class="q-kpi-grid">
  <div class="q-kpi kpi-iris">
    <div class="q-kpi-glow"></div>
    <div class="q-kpi-top">
      <div class="q-kpi-label">Income Median 2026 (Nominal)</div>
      <div class="q-kpi-icon ico-iris">💰</div>
    </div>
    <div class="q-kpi-val">138K <span style="font-size:14px;color:var(--dim)">EGP</span></div>
    <div class="q-kpi-delta d-iris">▲ +146.8% nominal shift</div>
    <div class="q-kpi-bar"><div class="q-kpi-fill" style="width:65%;background:linear-gradient(90deg,#7C3AED,#9D5FF5)"></div></div>
  </div>
  <div class="q-kpi kpi-jade">
    <div class="q-kpi-glow"></div>
    <div class="q-kpi-top">
      <div class="q-kpi-label">Real Purchasing Power (Urban)</div>
      <div class="q-kpi-icon ico-jade">📉</div>
    </div>
    <div class="q-kpi-val">40.5<span style="font-size:18px;color:var(--dim)">%</span></div>
    <div class="q-kpi-delta d-crim">▼ vs nominal income</div>
    <div class="q-kpi-bar"><div class="q-kpi-fill" style="width:40%;background:linear-gradient(90deg,#06D6A0,#34D399)"></div></div>
  </div>
  <div class="q-kpi kpi-crim">
    <div class="q-kpi-glow"></div>
    <div class="q-kpi-top">
      <div class="q-kpi-label">Churn Model AUC Score</div>
      <div class="q-kpi-icon ico-crim">🤖</div>
    </div>
    <div class="q-kpi-val">0.74</div>
    <div class="q-kpi-delta d-iris">5-fold cross-validation</div>
    <div class="q-kpi-bar"><div class="q-kpi-fill" style="width:74%;background:linear-gradient(90deg,#FF4757,#FF6B78)"></div></div>
  </div>
  <div class="q-kpi kpi-amber">
    <div class="q-kpi-glow"></div>
    <div class="q-kpi-top">
      <div class="q-kpi-label">Synthetic Training Scenarios</div>
      <div class="q-kpi-icon ico-amber">📊</div>
    </div>
    <div class="q-kpi-val">24,480</div>
    <div class="q-kpi-delta d-amber">economic simulations</div>
    <div class="q-kpi-bar"><div class="q-kpi-fill" style="width:88%;background:linear-gradient(90deg,#FFB020,#FCD34D)"></div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📡  Income Radar",
    "⚙️  Product Input",
    "🎯  AI Recommendation",
])

# ═══════════════════════════════════════════════
# TAB 1 — INCOME RADAR
# ═══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="q-section-label">Macro Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="q-section-title">Income Distribution Radar</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="q-section-sub">Log-Normal fit on 9 income brackets — 2020 vs 2026. '
        'The rightward shift is nominal only; real purchasing power fell to 40.5¢ per earned pound.</div>',
        unsafe_allow_html=True,
    )

    col_ctrl, _ = st.columns([1, 3])
    with col_ctrl:
        area_choice = st.radio("Region", ["Urban", "Rural"], horizontal=True, key="r1")

    data = get_curves(area_choice)
    x, p20, p26 = np.array(data["x"]), np.array(data["pdf_2020"]), np.array(data["pdf_2026"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=p20, fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
        line=dict(color="#7C3AED", width=2.5), name="2020 Distribution",
        hovertemplate="<b>Income:</b> %{x:,.0f} EGP<extra>2020</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p26, fill="tozeroy", fillcolor="rgba(255,71,87,0.07)",
        line=dict(color="#FF4757", width=2.5), name="2026 Distribution",
        hovertemplate="<b>Income:</b> %{x:,.0f} EGP<extra>2026</extra>",
    ))
    fig.add_vline(x=data["median_2020"], line_dash="dot", line_color="#7C3AED", line_width=1.5,
                  annotation_text=f" 2020 Median  {data['median_2020']:,.0f} EGP",
                  annotation_font_size=11, annotation_font_color="#9D5FF5")
    fig.add_vline(x=data["median_2026"], line_dash="dot", line_color="#FF4757", line_width=1.5,
                  annotation_text=f" 2026 Median  {data['median_2026']:,.0f} EGP",
                  annotation_font_size=11, annotation_font_color="#FF6B78")
    fig.update_layout(
        **chart(
            height=420, hovermode="x unified",
            xaxis=dict(**CHART["xaxis"], title="Annual Household Income (EGP)",
                       tickformat=",", range=[0, 420_000],
                       title_font=dict(family="Inter", size=12, color="#8B9AB3")),
            yaxis=dict(**CHART["yaxis"], title="Probability Density",
                       title_font=dict(family="Inter", size=12, color="#8B9AB3")),
            legend=dict(**CHART["legend"], x=0.72, y=0.97),
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    pwr  = 40.50 if area_choice == "Urban" else 39.49
    food = 28.63 if area_choice == "Urban" else 36.39

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median 2020",           f"{data['median_2020']:,.0f} EGP/yr")
    c2.metric("Median 2026",           f"{data['median_2026']:,.0f} EGP/yr",
              delta=f"+{data['shift_pct']:.1f}% nominal")
    c3.metric("Real Purchasing Power", f"{pwr}%",
              delta=f"−{100-pwr:.1f}% lost", delta_color="inverse")
    c4.metric("Mandatory Food Budget", f"{food}%")

    st.markdown(
        f'<div class="q-callout call-iris"><span class="call-ico">⚡</span>'
        f'<div><strong>Key insight:</strong> The 2026 curve shifted +{data["shift_pct"]:.1f}% rightward '
        f'due to inflation (×{2.47 if area_choice == "Urban" else 2.53}). '
        f'Real purchasing power is <strong>{pwr}%</strong> of nominal. '
        f'Nominal-only pricing overestimates affordability by <strong>{100-pwr:.0f}%</strong>.</div></div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════
# TAB 2 — PRODUCT INPUT
# ═══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="q-section-label">Product Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="q-section-title">Define Your Product</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="q-section-sub">Enter your product parameters. The engine maps its position '
        'against the 2026 income distribution and computes optimal recommendations.</div>',
        unsafe_allow_html=True,
    )

    with st.form("product_form"):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**📦 Product parameters**")
            current_price  = st.number_input("Current price (EGP)",         min_value=1.0,  max_value=500.0,  value=25.0, step=0.5)
            current_weight = st.number_input("Current weight (grams)",       min_value=10.0, max_value=5000.0, value=100.0, step=5.0)
            cost_per_gram  = st.number_input("Production cost / gram (EGP)", min_value=0.01, max_value=10.0,   value=0.18, step=0.01,
                                              help="Raw materials + manufacturing + packaging")
            new_price_in   = st.number_input("Proposed new price (EGP)",     min_value=1.0,  max_value=500.0,  value=30.0, step=0.5,
                                              help="Used to predict churn if you raise price without changing weight")
        with c2:
            st.markdown("**⚙️ Analysis parameters**")
            area_sel      = st.selectbox("Target region", ["Urban", "Rural"])
            target_margin = st.slider("Target profit margin (%)", 5, 60, 30, 5) / 100
            purchase_freq = st.slider("Monthly purchase frequency", 1, 20, 4)

            cur_cost   = cost_per_gram * current_weight
            cur_margin = (current_price - cur_cost) / current_price * 100
            inc_pct    = (new_price_in - current_price) / current_price * 100
            st.markdown(
                f'<div class="q-callout call-iris" style="margin-top:12px">'
                f'<span class="call-ico">📊</span>'
                f'<div style="font-size:12px;line-height:1.85">'
                f'Production cost: <strong>{cur_cost:.2f} EGP</strong><br>'
                f'Current margin: <strong>{cur_margin:.1f}%</strong><br>'
                f'Price/gram: <strong>{current_price/current_weight:.3f} EGP/g</strong><br>'
                f'Proposed increase: <strong>+{inc_pct:.1f}%</strong>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button(
            "⚡ Run AI Analysis", use_container_width=True, type="primary",
        )

    if submitted:
        st.session_state["pd"] = {
            "current_price":    current_price,
            "current_weight_g": current_weight,
            "cost_per_gram":    cost_per_gram,
            "area":             area_sel,
            "target_margin":    target_margin,
            "purchase_freq":    purchase_freq,
            "new_price":        new_price_in,
        }
        st.success("✅ Analysis complete — check the AI Recommendation tab")

    # ── Affordability map ──
    if "pd" in st.session_state:
        pdd  = st.session_state["pd"]
        segs = get_bracket_affordability(pdd["current_price"], pdd["area"], pdd["purchase_freq"])
        sdf  = pd.DataFrame(segs)

        st.markdown('<div class="q-div"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="q-section-label">Market Affordability</div>'
            f'<div class="q-section-title">Purchasing Reach at {pdd["current_price"]:.0f} EGP</div>',
            unsafe_allow_html=True,
        )

        fig_aff = go.Figure(go.Bar(
            x=sdf["bracket"], y=sdf["price_burden_pct"],
            marker=dict(
                color=["#06D6A0" if r else "#FF4757" for r in sdf["affordable"]],
                opacity=0.82, line=dict(width=0),
            ),
            text=[f"{v:.1f}%" for v in sdf["price_burden_pct"]],
            textposition="outside", textfont=dict(family="Inter", size=11, color="#8B9AB3"),
            hovertemplate="<b>%{x}</b><br>Burden: <b>%{y:.1f}%</b><extra></extra>",
        ))
        fig_aff.add_hline(y=15, line_dash="dot", line_color="#FFB020", line_width=2,
                          annotation_text="Affordability threshold 15%",
                          annotation_font_color="#FFB020", annotation_font_size=11)
        fig_aff.update_layout(**chart(
            height=300,
            xaxis=dict(**CHART["xaxis"], tickangle=-30),
            yaxis=dict(**CHART["yaxis"], title="Price Burden %",
                       title_font=dict(family="Inter", size=12, color="#8B9AB3")),
        ))
        st.plotly_chart(fig_aff, use_container_width=True)

        # ── DATAFRAME — plain, no custom CSS on it ──
        sdf_show = sdf[["bracket","population_pct","monthly_disposable","price_burden_pct"]].copy()
        sdf_show["Status"] = sdf["affordable"].apply(lambda x: "✅ Affordable" if x else "🔴 Out of reach")
        sdf_show.columns   = ["Income Bracket", "Pop %", "Disposable/mo (EGP)", "Burden %", "Status"]
        sdf_show["Pop %"]              = sdf_show["Pop %"].round(2)
        sdf_show["Disposable/mo (EGP)"]= sdf_show["Disposable/mo (EGP)"].round(0).astype(int)
        sdf_show["Burden %"]           = sdf_show["Burden %"].round(1)
        st.dataframe(sdf_show, use_container_width=True, hide_index=True, height=340)

# ═══════════════════════════════════════════════
# TAB 3 — AI RECOMMENDATION
# ═══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="q-section-label">AI Engine Output</div>', unsafe_allow_html=True)
    st.markdown('<div class="q-section-title">Smart Recommendation</div>', unsafe_allow_html=True)

    if "pd" not in st.session_state:
        st.markdown(
            '<div class="q-callout call-amber"><span class="call-ico">⚠️</span>'
            '<div>No product configured yet. Go to <strong>Product Input</strong> and run the analysis first.</div></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    pdd = st.session_state["pd"]
    product = ProductInput(
        current_price    = pdd["current_price"],
        current_weight_g = pdd["current_weight_g"],
        cost_per_gram    = pdd["cost_per_gram"],
        area             = pdd["area"],
        purchase_freq    = pdd["purchase_freq"],
        target_margin    = pdd["target_margin"],
    )

    with st.spinner("Running gradient boosting inference..."):
        w_rec  = optimizer.find_optimal_weight(product)
        c_pred = optimizer.predict_market_churn(product, pdd["new_price"])

    # ── Strategy A: Optimal Weight ──────────────────────────
    st.markdown("### ⚖️ Strategy A — Optimal Weight Adjustment")
    st.markdown(
        '<div class="q-section-sub">Hold the psychological price point. '
        'Reduce weight to hit your margin target.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Weight",   f"{product.current_weight_g:.0f}g")
    c2.metric("Optimal Weight",   f"{w_rec.optimal_weight_g}g",
              delta=f"−{w_rec.weight_reduction_pct}%", delta_color="inverse")
    c3.metric("New Margin",       f"{w_rec.new_margin_pct}%")
    c4.metric("New Price / Gram", f"{w_rec.price_per_gram_new} EGP/g",
              delta=f"+{w_rec.price_per_gram_new - w_rec.price_per_gram_old:.3f}")

    cc = "call-jade" if w_rec.feasible else "call-crim"
    ci = "✅" if w_rec.feasible else "🚨"
    st.markdown(
        f'<div class="q-callout {cc}"><span class="call-ico">{ci}</span>'
        f'<div><strong>Assessment:</strong> {w_rec.warning}</div></div>',
        unsafe_allow_html=True,
    )

    # Weight bar chart
    fig_w = go.Figure(go.Bar(
        x=["Current Weight", "Optimal Weight"],
        y=[product.current_weight_g, w_rec.optimal_weight_g],
        marker=dict(
            color=["rgba(124,58,237,0.6)", "rgba(6,214,160,0.72)"], line=dict(width=0),
        ),
        text=[f"{product.current_weight_g:.0f}g", f"{w_rec.optimal_weight_g}g"],
        textposition="outside", textfont=dict(family="Space Grotesk", size=15, color="#F8FAFC"),
        width=0.38,
    ))
    fig_w.update_layout(**chart(
        height=260,
        yaxis=dict(**CHART["yaxis"], title="Weight (grams)",
                   title_font=dict(family="Inter", size=12, color="#8B9AB3")),
    ))
    st.plotly_chart(fig_w, use_container_width=True)

    st.markdown('<div class="q-div"></div>', unsafe_allow_html=True)

    # ── Strategy B: Churn Prediction ────────────────────────
    risk_call = {"HIGH":"call-crim","MEDIUM":"call-amber","LOW":"call-jade"}[c_pred.risk_level]
    risk_icon = {"HIGH":"🚨","MEDIUM":"⚠️","LOW":"✅"}[c_pred.risk_level]

    st.markdown(
        f"### 📈 Strategy B — Price Raise to {c_pred.new_price:.0f} EGP (+{c_pred.price_increase_pct}%)"
    )
    st.markdown(
        '<div class="q-section-sub">Hold weight constant. Predict customer churn '
        'across every income bracket.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weighted Churn",       f"{c_pred.weighted_churn_pct}%")
    c2.metric("At-Risk Population",   f"{c_pred.at_risk_population_pct}%")
    c3.metric("ML Churn Probability", f"{c_pred.ml_churn_prob}%")
    c4.metric("Risk Level",           c_pred.risk_level)

    st.markdown(
        f'<div class="q-callout {risk_call}"><span class="call-ico">{risk_icon}</span>'
        f'<div><strong>Decision signal:</strong> {c_pred.recommendation}</div></div>',
        unsafe_allow_html=True,
    )

    # ── 3 simple charts for factory owners ──
    sdf2 = pd.DataFrame(c_pred.segments_detail)
    brackets  = [b.replace(",","").replace(" ","") for b in sdf2["bracket"]]
    b_short   = [b if len(b) <= 12 else b[:10]+"…" for b in sdf2["bracket"]]

    ch1, ch2, ch3 = st.columns(3, gap="small")

    # Chart 1: هل الزبون يقدر يشتري؟ — ✅ أو 🔴
    with ch1:
        st.markdown(
            '<div class="q-section-label" style="text-align:center;margin-bottom:8px">هل يقدر يشتري؟</div>',
            unsafe_allow_html=True,
        )
        fig1 = go.Figure(go.Bar(
            x=b_short,
            y=sdf2["price_burden_pct"],
            marker=dict(
                color=["#FF4757" if r else "#06D6A0" for r in sdf2["at_risk"]],
                opacity=0.88, line=dict(width=0),
            ),
            text=["🔴 غالي" if r else "✅ مناسب" for r in sdf2["at_risk"]],
            textposition="outside",
            textfont=dict(family="Inter", size=11, color="#F8FAFC"),
            hovertemplate="<b>%{x}</b><br>عبء السعر: %{y:.1f}%<extra></extra>",
        ))
        fig1.add_hline(
            y=15, line_dash="dot", line_color="#FFB020", line_width=2,
            annotation_text="حد المقاطعة", annotation_font_color="#FFB020",
            annotation_font_size=11,
        )
        fig1.update_layout(**chart(
            height=300, title=dict(text="عبء السعر على كل فئة", font=dict(family="Space Grotesk", size=13, color="#8B9AB3"), x=0.5),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.07)", zeroline=False, tickangle=-45, tickfont=dict(family="Inter", size=9, color="#4A5568")),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.07)", zeroline=False, title="% من الدخل المتاح", title_font=dict(family="Inter", size=11, color="#8B9AB3"), tickfont=dict(family="Inter", size=11, color="#4A5568")),
            margin=dict(t=36, b=60, l=8, r=8),
            showlegend=False,
        ))
        st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: نسبة المقاطعة المتوقعة — gauge بسيط لكل فئة
    with ch2:
        st.markdown(
            '<div class="q-section-label" style="text-align:center;margin-bottom:8px">احتمال المقاطعة</div>',
            unsafe_allow_html=True,
        )
        ml_probs = [v * 100 for v in sdf2["ml_churn_prob"]]
        bar_colors = []
        for p in ml_probs:
            if p >= 50:
                bar_colors.append("#FF4757")
            elif p >= 25:
                bar_colors.append("#FFB020")
            else:
                bar_colors.append("#06D6A0")

        fig2 = go.Figure(go.Bar(
            x=b_short,
            y=ml_probs,
            marker=dict(color=bar_colors, opacity=0.88, line=dict(width=0)),
            text=[f"{p:.0f}%" for p in ml_probs],
            textposition="outside",
            textfont=dict(family="Space Grotesk", size=12, color="#F8FAFC"),
            hovertemplate="<b>%{x}</b><br>احتمال المقاطعة: %{y:.0f}%<extra></extra>",
        ))
        fig2.update_layout(**chart(
            height=300, title=dict(text="% احتمال ترك المنتج", font=dict(family="Space Grotesk", size=13, color="#8B9AB3"), x=0.5),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.07)", zeroline=False, tickangle=-45, tickfont=dict(family="Inter", size=9, color="#4A5568")),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.07)", zeroline=False, title="%", range=[0, 105], title_font=dict(family="Inter", size=11, color="#8B9AB3"), tickfont=dict(family="Inter", size=11, color="#4A5568")),
            margin=dict(t=36, b=60, l=8, r=8),
            showlegend=False,
        ))
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: حجم كل فئة في السوق — pie chart بسيط
    with ch3:
        st.markdown(
            '<div class="q-section-label" style="text-align:center;margin-bottom:8px">حجم كل فئة في السوق</div>',
            unsafe_allow_html=True,
        )
        pop_vals = [round(v * 100, 2) for v in sdf2["population_pct"]]
        pie_colors = ["#FF4757" if r else "#7C3AED" for r in sdf2["at_risk"]]
        fig3 = go.Figure(go.Pie(
            labels=b_short,
            values=pop_vals,
            marker=dict(colors=pie_colors, line=dict(color=["#060912"]*len(b_short), width=2)),
            textinfo="percent",
            textfont=dict(family="Inter", size=11, color="#F8FAFC"),
            hovertemplate="<b>%{label}</b><br>%{value:.2f}% من السوق<extra></extra>",
            hole=0.45,
            sort=False,
        ))
        fig3.update_layout(**chart(
            height=300, title=dict(text="🔴 = في خطر  🟣 = آمن", font=dict(family="Space Grotesk", size=13, color="#8B9AB3"), x=0.5),
            showlegend=False,
            margin=dict(t=36, b=20, l=8, r=8),
        ))
        st.plotly_chart(fig3, use_container_width=True)

    # ── DATAFRAME — clean, no CSS interference ──
    st.markdown("**Segment detail breakdown**")
    sdf2_show = sdf2[[
        "bracket","population_pct","monthly_disposable",
        "price_burden_pct","churn_threshold_pct","at_risk","ml_churn_prob",
    ]].copy()
    sdf2_show.columns = [
        "Bracket","Pop %","Disposable/mo (EGP)",
        "Burden %","Threshold %","At Risk","ML Prob",
    ]
    sdf2_show["At Risk"]             = sdf2_show["At Risk"].apply(lambda x: "🔴 Yes" if x else "🟢 No")
    sdf2_show["ML Prob"]             = sdf2_show["ML Prob"].apply(lambda x: f"{x:.0%}")
    sdf2_show["Disposable/mo (EGP)"] = sdf2_show["Disposable/mo (EGP)"].round(0).astype(int)
    sdf2_show["Pop %"]               = sdf2_show["Pop %"].round(2)
    sdf2_show["Burden %"]            = sdf2_show["Burden %"].round(1)
    sdf2_show["Threshold %"]         = sdf2_show["Threshold %"].round(1)
    st.dataframe(sdf2_show, use_container_width=True, hide_index=True, height=340)

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="q-footer">
  <div class="q-footer-brand">
    <div class="q-footer-mark">⚖️</div>
    <div>
      <div class="q-footer-name">Qys<b>tas</b></div>
      <div class="q-footer-copy">© 2026 Qystas Smart Pricing Engine · Egyptian Income Distribution 2020–2026</div>
    </div>
  </div>
  <div class="q-footer-pills">
    <span class="q-footer-pill">ML Powered</span>
    <span class="q-footer-pill">Egypt 2026</span>
    <span class="q-footer-pill">Log-Normal</span>
    <span class="q-footer-pill">Gradient Boosting</span>
    <span class="q-footer-pill">Open Source</span>
    <span class="q-footer-pill">AUC 0.74</span>
  </div>
</div>
""", unsafe_allow_html=True)
