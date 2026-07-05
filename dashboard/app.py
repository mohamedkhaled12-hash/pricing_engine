"""
Qystas — Smart Pricing Engine
World-class UI. Production-ready. WWDC-level design.
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
   RESET + BASE
═══════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stVerticalBlock"] {
  background: var(--void) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--snow) !important;
}

/* Hide all Streamlit UI chrome */
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

/* ═══════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb { background: var(--raised); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--iris); }

/* ═══════════════════════════════════════
   TOPBAR
═══════════════════════════════════════ */
.q-topbar {
  position: sticky;
  top: 0;
  z-index: 999;
  height: 60px;
  background: rgba(6,9,18,0.85);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 40px;
  transition: background 0.3s var(--ease);
}

.q-logo {
  display: flex;
  align-items: center;
  gap: 12px;
}

.q-logo-mark {
  width: 34px; height: 34px;
  background: linear-gradient(135deg, var(--iris), var(--iris-2));
  border-radius: 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 17px;
  box-shadow: 0 0 20px var(--iris-glow);
  flex-shrink: 0;
}

.q-logo-text {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 18px;
  font-weight: 700;
  color: var(--snow);
  letter-spacing: -0.5px;
}

.q-logo-text span { color: var(--iris-2); }

.q-logo-sub {
  font-family: 'Inter', sans-serif;
  font-size: 10px;
  font-weight: 500;
  color: var(--dim);
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-top: 1px;
}

.q-topbar-right {
  display: flex;
  align-items: center;
  gap: 8px;
}

.q-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border-radius: 100px;
  font-family: 'Inter', sans-serif;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  transition: all 0.2s var(--ease);
}

.q-chip-ghost {
  border: 1px solid var(--border-2);
  color: var(--dim);
  background: transparent;
}

.q-chip-ghost:hover {
  border-color: var(--iris);
  color: var(--iris-2);
  background: rgba(124,58,237,0.08);
}

.q-chip-iris {
  background: linear-gradient(135deg, var(--iris), var(--iris-2));
  color: white;
  box-shadow: 0 2px 12px var(--iris-glow);
}

.q-chip-live {
  background: rgba(6,214,160,0.1);
  border: 1px solid rgba(6,214,160,0.25);
  color: var(--jade);
}

.q-chip-live::before {
  content: '';
  width: 6px; height: 6px;
  background: var(--jade);
  border-radius: 50%;
  animation: live-pulse 1.8s ease infinite;
}

@keyframes live-pulse {
  0%, 100% { opacity: 1; box-shadow: 0 0 0 0 var(--jade-glow); }
  50% { opacity: 0.6; box-shadow: 0 0 0 4px transparent; }
}

/* ═══════════════════════════════════════
   TICKER
═══════════════════════════════════════ */
.q-ticker-wrap {
  height: 38px;
  background: linear-gradient(90deg,
    rgba(124,58,237,0.15) 0%,
    rgba(124,58,237,0.08) 50%,
    rgba(124,58,237,0.15) 100%);
  border-bottom: 1px solid rgba(124,58,237,0.2);
  overflow: hidden;
  display: flex;
  align-items: center;
}

.q-ticker-scroll {
  display: inline-flex;
  animation: scroll-left 36s linear infinite;
  white-space: nowrap;
  align-items: center;
}

.q-tick {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 0 32px;
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 600;
  color: var(--mist);
  letter-spacing: 0.2px;
}

.q-tick-val { color: var(--iris-2); font-weight: 700; }
.q-tick-sep { width: 3px; height: 3px; background: var(--dim); border-radius: 50%; opacity: 0.5; }

@keyframes scroll-left {
  0%   { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}

/* ═══════════════════════════════════════
   HERO
═══════════════════════════════════════ */
.q-hero {
  position: relative;
  overflow: hidden;
  padding: 80px 48px 96px;
  background: var(--void);
}

.q-hero-bg {
  position: absolute;
  inset: 0;
  pointer-events: none;
  overflow: hidden;
}

.q-hero-orb-1 {
  position: absolute;
  top: -180px; left: -100px;
  width: 600px; height: 600px;
  background: radial-gradient(circle, rgba(124,58,237,0.12) 0%, transparent 65%);
  border-radius: 50%;
}

.q-hero-orb-2 {
  position: absolute;
  bottom: -200px; right: -80px;
  width: 500px; height: 500px;
  background: radial-gradient(circle, rgba(6,214,160,0.07) 0%, transparent 65%);
  border-radius: 50%;
}

.q-hero-grid {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px);
  background-size: 56px 56px;
}

.q-hero-grid::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(to bottom, transparent 60%, var(--void) 100%);
}

.q-hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(124,58,237,0.1);
  border: 1px solid rgba(124,58,237,0.25);
  padding: 6px 16px;
  border-radius: 100px;
  margin-bottom: 28px;
  position: relative;
}

.q-hero-eyebrow-dot {
  width: 7px; height: 7px;
  background: var(--iris-2);
  border-radius: 50%;
  box-shadow: 0 0 8px var(--iris-glow);
  animation: iris-pulse 2s ease infinite;
}

@keyframes iris-pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(0.8); opacity: 0.5; }
}

.q-hero-eyebrow-text {
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 600;
  color: var(--iris-2);
  letter-spacing: 1.2px;
  text-transform: uppercase;
}

.q-hero-h1 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: clamp(36px, 5.5vw, 72px);
  font-weight: 700;
  color: var(--snow);
  line-height: 1.04;
  letter-spacing: -2.5px;
  margin-bottom: 20px;
  max-width: 700px;
  position: relative;
}

.q-hero-h1-accent {
  background: linear-gradient(135deg, var(--iris-2) 0%, #A78BFA 50%, var(--jade) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.q-hero-sub {
  font-family: 'Inter', sans-serif;
  font-size: 17px;
  font-weight: 400;
  color: var(--mist);
  line-height: 1.7;
  max-width: 520px;
  margin-bottom: 56px;
  position: relative;
}

.q-hero-stats {
  display: flex;
  align-items: stretch;
  gap: 0;
  position: relative;
}

.q-hero-stat {
  padding: 0 48px 0 0;
  margin-right: 48px;
  border-right: 1px solid var(--border);
}

.q-hero-stat:last-child {
  border-right: none;
  padding-right: 0;
  margin-right: 0;
}

.q-hero-stat-val {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 42px;
  font-weight: 700;
  letter-spacing: -2px;
  line-height: 1;
  background: linear-gradient(135deg, #fff 60%, rgba(255,255,255,0.55) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.q-hero-stat-val.iris {
  background: linear-gradient(135deg, var(--iris-2), #A78BFA);
  -webkit-background-clip: text;
  background-clip: text;
}

.q-hero-stat-val.jade {
  background: linear-gradient(135deg, var(--jade), #34D399);
  -webkit-background-clip: text;
  background-clip: text;
}

.q-hero-stat-label {
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 500;
  color: var(--dim);
  margin-top: 6px;
  letter-spacing: 0.2px;
}

/* ═══════════════════════════════════════
   BODY CONTAINER
═══════════════════════════════════════ */
.q-body {
  padding: 40px 48px 64px;
  display: flex;
  flex-direction: column;
  gap: 32px;
  background: var(--void);
}

/* ═══════════════════════════════════════
   SECTION HEADERS
═══════════════════════════════════════ */
.q-section-label {
  font-family: 'Inter', sans-serif;
  font-size: 10px;
  font-weight: 700;
  color: var(--dim);
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-bottom: 6px;
}

.q-section-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 24px;
  font-weight: 600;
  color: var(--snow);
  letter-spacing: -0.8px;
  margin-bottom: 6px;
}

.q-section-sub {
  font-family: 'Inter', sans-serif;
  font-size: 14px;
  color: var(--mist);
  line-height: 1.65;
  margin-bottom: 28px;
  max-width: 600px;
}

/* ═══════════════════════════════════════
   KPI GRID
═══════════════════════════════════════ */
.q-kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.q-kpi {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 24px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s var(--ease), box-shadow 0.3s var(--ease), transform 0.3s var(--ease);
  cursor: default;
}

.q-kpi:hover {
  transform: translateY(-4px);
  box-shadow: 0 20px 60px rgba(0,0,0,0.4);
}

.q-kpi.iris-kpi:hover { border-color: rgba(124,58,237,0.4); box-shadow: 0 20px 60px rgba(124,58,237,0.12); }
.q-kpi.jade-kpi:hover { border-color: rgba(6,214,160,0.3);  box-shadow: 0 20px 60px rgba(6,214,160,0.08); }
.q-kpi.crim-kpi:hover { border-color: rgba(255,71,87,0.3);   box-shadow: 0 20px 60px rgba(255,71,87,0.08); }
.q-kpi.amber-kpi:hover{ border-color: rgba(255,176,32,0.3);  box-shadow: 0 20px 60px rgba(255,176,32,0.08);}

.q-kpi-glow {
  position: absolute;
  top: -60px; right: -60px;
  width: 160px; height: 160px;
  border-radius: 50%;
  opacity: 0.5;
  pointer-events: none;
  transition: opacity 0.3s;
}

.iris-kpi .q-kpi-glow  { background: radial-gradient(circle, rgba(124,58,237,0.2) 0%, transparent 70%); }
.jade-kpi .q-kpi-glow  { background: radial-gradient(circle, rgba(6,214,160,0.15) 0%, transparent 70%); }
.crim-kpi .q-kpi-glow  { background: radial-gradient(circle, rgba(255,71,87,0.15) 0%, transparent 70%); }
.amber-kpi .q-kpi-glow { background: radial-gradient(circle, rgba(255,176,32,0.15) 0%, transparent 70%); }

.q-kpi:hover .q-kpi-glow { opacity: 1; }

.q-kpi-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 16px;
}

.q-kpi-label {
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 600;
  color: var(--dim);
  letter-spacing: 0.3px;
  line-height: 1.4;
  max-width: 120px;
}

.q-kpi-icon {
  width: 36px; height: 36px;
  border-radius: var(--r-sm);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  flex-shrink: 0;
}

.q-kpi-icon.iris  { background: rgba(124,58,237,0.15); }
.q-kpi-icon.jade  { background: rgba(6,214,160,0.12); }
.q-kpi-icon.crim  { background: rgba(255,71,87,0.12); }
.q-kpi-icon.amber { background: rgba(255,176,32,0.12); }

.q-kpi-val {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 32px;
  font-weight: 700;
  color: var(--snow);
  letter-spacing: -1.5px;
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
  padding: 3px 9px;
  border-radius: 100px;
}

.delta-up   { background: rgba(6,214,160,0.1);  color: var(--jade); }
.delta-down { background: rgba(255,71,87,0.1);   color: var(--crimson); }
.delta-warn { background: rgba(255,176,32,0.1);  color: var(--amber); }
.delta-iris { background: rgba(124,58,237,0.12); color: var(--iris-2); }

.q-kpi-bar {
  height: 3px;
  background: var(--raised);
  border-radius: 100px;
  margin-top: 14px;
  overflow: hidden;
}

.q-kpi-fill {
  height: 100%;
  border-radius: 100px;
  transition: width 1.4s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ═══════════════════════════════════════
   GLASS CARD
═══════════════════════════════════════ */
.q-glass {
  background: rgba(13,17,23,0.8);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: var(--r-xl);
  overflow: hidden;
  box-shadow: 0 4px 40px rgba(0,0,0,0.3);
}

.q-card-header {
  padding: 20px 28px 18px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.q-card-title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: 'Space Grotesk', sans-serif;
  font-size: 15px;
  font-weight: 600;
  color: var(--snow);
  letter-spacing: -0.3px;
}

.q-card-icon {
  width: 32px; height: 32px;
  background: var(--raised);
  border-radius: var(--r-sm);
  border: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
}

.q-card-body { padding: 24px 28px; }

/* ═══════════════════════════════════════
   TABS
═══════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-radius: var(--r-md) !important;
  padding: 5px !important;
  gap: 3px !important;
  border: 1px solid var(--border) !important;
  width: fit-content !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04) !important;
}

.stTabs [data-baseweb="tab"] {
  border-radius: 10px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--dim) !important;
  padding: 9px 20px !important;
  background: transparent !important;
  border: none !important;
  letter-spacing: 0.1px !important;
  transition: all 0.2s var(--ease) !important;
}

.stTabs [data-baseweb="tab"]:hover {
  color: var(--mist) !important;
  background: var(--raised) !important;
}

.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--iris), var(--iris-2)) !important;
  color: white !important;
  box-shadow: 0 4px 16px var(--iris-glow) !important;
}

.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

.stTabs [data-testid="stMarkdownContainer"] p {
  font-family: 'Inter', sans-serif !important;
  color: var(--mist) !important;
}

/* ═══════════════════════════════════════
   INSIGHT CALLOUT
═══════════════════════════════════════ */
.q-callout {
  border-radius: var(--r-md);
  padding: 16px 20px;
  margin-top: 16px;
  border: 1px solid;
  display: flex;
  gap: 12px;
  align-items: flex-start;
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  line-height: 1.65;
}

.q-callout.iris  { background: rgba(124,58,237,0.07); border-color: rgba(124,58,237,0.2); color: #C4B5FD; }
.q-callout.jade  { background: rgba(6,214,160,0.06);  border-color: rgba(6,214,160,0.2);  color: #6EE7C7; }
.q-callout.crim  { background: rgba(255,71,87,0.07);  border-color: rgba(255,71,87,0.2);   color: #FCA5A5; }
.q-callout.amber { background: rgba(255,176,32,0.07); border-color: rgba(255,176,32,0.2);  color: #FCD34D; }

.q-callout-icon { font-size: 16px; flex-shrink: 0; margin-top: 1px; }

/* ═══════════════════════════════════════
   REC CARDS
═══════════════════════════════════════ */
.q-rec {
  border-radius: var(--r-lg);
  padding: 22px 24px;
  border: 1px solid;
  margin-bottom: 16px;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease);
}

.q-rec:hover {
  transform: translateY(-2px);
}

.q-rec.jade-rec {
  background: rgba(6,214,160,0.05);
  border-color: rgba(6,214,160,0.2);
}

.q-rec.jade-rec:hover {
  box-shadow: 0 12px 40px rgba(6,214,160,0.08);
}

.q-rec.crim-rec {
  background: rgba(255,71,87,0.05);
  border-color: rgba(255,71,87,0.2);
}

.q-rec.crim-rec:hover {
  box-shadow: 0 12px 40px rgba(255,71,87,0.08);
}

.q-rec.amber-rec {
  background: rgba(255,176,32,0.05);
  border-color: rgba(255,176,32,0.2);
}

.q-rec-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 14px;
}

.q-rec-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  border-radius: 100px;
  font-family: 'Inter', sans-serif;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.8px;
  text-transform: uppercase;
}

.badge-jade  { background: rgba(6,214,160,0.15);  color: var(--jade); }
.badge-crim  { background: rgba(255,71,87,0.15);   color: var(--crimson); }
.badge-amber { background: rgba(255,176,32,0.15);  color: var(--amber); }

.q-rec-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 14px;
  font-weight: 600;
  color: var(--snow);
  letter-spacing: -0.2px;
}

.q-rec-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 8px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}

.q-rec-row:last-child { border-bottom: none; }

.q-rec-key {
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  font-weight: 500;
  color: var(--dim);
}

.q-rec-val {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 18px;
  font-weight: 600;
  color: var(--snow);
  letter-spacing: -0.5px;
}

.q-rec-val.jade  { color: var(--jade); }
.q-rec-val.crim  { color: var(--crimson); }
.q-rec-val.amber { color: var(--amber); }
.q-rec-val.iris  { color: var(--iris-2); }

/* ═══════════════════════════════════════
   RISK GAUGE (HTML/SVG)
═══════════════════════════════════════ */
.q-gauge-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 20px;
}

/* ═══════════════════════════════════════
   STREAMLIT NATIVE OVERRIDES
═══════════════════════════════════════ */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-lg) !important;
  padding: 22px 24px !important;
  transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease) !important;
}

[data-testid="stMetric"]:hover {
  transform: translateY(-3px) !important;
  box-shadow: 0 16px 48px rgba(0,0,0,0.3) !important;
}

[data-testid="stMetricLabel"] p {
  font-family: 'Inter', sans-serif !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  color: var(--dim) !important;
  letter-spacing: 0.3px !important;
  text-transform: uppercase !important;
}

[data-testid="stMetricValue"] {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 28px !important;
  font-weight: 700 !important;
  color: var(--snow) !important;
  letter-spacing: -1px !important;
}

[data-testid="stMetricDelta"] svg { display: none !important; }

[data-testid="stMetricDelta"] > div {
  font-family: 'Inter', sans-serif !important;
  font-size: 11px !important;
  font-weight: 700 !important;
}

[data-testid="stDataFrame"] {
  border-radius: var(--r-md) !important;
  border: 1px solid var(--border) !important;
  background: var(--surface) !important;
  overflow: hidden !important;
}

[data-testid="stDataFrame"] * {
  font-family: 'Inter', sans-serif !important;
  background: var(--surface) !important;
  color: var(--mist) !important;
}

[data-testid="stAlert"] {
  background: var(--surface) !important;
  border-radius: var(--r-md) !important;
  border: 1px solid var(--border) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--mist) !important;
}

.stButton > button {
  background: linear-gradient(135deg, var(--iris), var(--iris-2)) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--r-md) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  padding: 13px 28px !important;
  letter-spacing: 0.2px !important;
  box-shadow: 0 4px 20px var(--iris-glow) !important;
  transition: all 0.2s var(--ease) !important;
  position: relative !important;
  overflow: hidden !important;
}

.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 30px var(--iris-glow) !important;
  filter: brightness(1.1) !important;
}

.stButton > button:active {
  transform: translateY(0) !important;
}

.stRadio > label {
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--mist) !important;
}

.stRadio [data-testid="stMarkdownContainer"] p {
  color: var(--mist) !important;
}

div[role="radiogroup"] label {
  font-family: 'Inter', sans-serif !important;
  color: var(--mist) !important;
}

.stSlider label,
.stSelectbox label,
.stNumberInput label {
  font-family: 'Inter', sans-serif !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  color: var(--dim) !important;
  letter-spacing: 0.3px !important;
  text-transform: uppercase !important;
}

.stSelectbox [data-baseweb="select"] {
  background: var(--raised) !important;
  border-radius: var(--r-sm) !important;
  border-color: var(--border) !important;
}

.stSelectbox [data-baseweb="select"] * {
  background: var(--raised) !important;
  color: var(--snow) !important;
  font-family: 'Inter', sans-serif !important;
}

.stNumberInput input {
  background: var(--raised) !important;
  border-color: var(--border) !important;
  color: var(--snow) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 15px !important;
  font-weight: 500 !important;
  border-radius: var(--r-sm) !important;
}

.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
  font-family: 'Space Grotesk', sans-serif !important;
  background: var(--iris) !important;
  color: white !important;
}

.stSpinner > div {
  border-top-color: var(--iris) !important;
}

.stSpinner p {
  font-family: 'Inter', sans-serif !important;
  color: var(--mist) !important;
  font-size: 13px !important;
}

.stMarkdown h3 {
  font-family: 'Space Grotesk', sans-serif !important;
  color: var(--snow) !important;
  letter-spacing: -0.5px !important;
}

hr {
  border-color: var(--border) !important;
  margin: 24px 0 !important;
}

/* ═══════════════════════════════════════
   FOOTER
═══════════════════════════════════════ */
.q-footer {
  background: var(--surface);
  border-top: 1px solid var(--border);
  padding: 28px 48px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 16px;
  margin-top: 32px;
}

.q-footer-brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.q-footer-mark {
  width: 30px; height: 30px;
  background: linear-gradient(135deg, var(--iris), var(--iris-2));
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  box-shadow: 0 0 12px var(--iris-glow);
}

.q-footer-name {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: var(--snow);
  letter-spacing: -0.3px;
}

.q-footer-name span { color: var(--iris-2); }

.q-footer-copy {
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  color: var(--dim);
  margin-top: 2px;
}

.q-footer-pills {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.q-footer-pill {
  padding: 4px 12px;
  border: 1px solid var(--border);
  border-radius: 100px;
  font-family: 'Inter', sans-serif;
  font-size: 9px;
  font-weight: 700;
  color: var(--dim);
  letter-spacing: 1.2px;
  text-transform: uppercase;
  transition: all 0.2s var(--ease);
}

.q-footer-pill:hover {
  border-color: var(--iris);
  color: var(--iris-2);
}

/* ═══════════════════════════════════════
   DIVIDER
═══════════════════════════════════════ */
.q-divider {
  height: 1px;
  background: var(--border);
  margin: 32px 0;
}

/* ═══════════════════════════════════════
   SCORE RING (SVG inline)
═══════════════════════════════════════ */
.q-score-ring-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# JS: Count-up animation + smooth interactions
# ─────────────────────────────────────────────────────────
st.markdown("""
<script>
(function() {
  // Count-up for KPI values when they appear
  function countUp(el, target, duration, isFloat, suffix) {
    let start = 0;
    const step = target / (duration / 16);
    const timer = setInterval(() => {
      start += step;
      if (start >= target) { start = target; clearInterval(timer); }
      el.textContent = (isFloat ? start.toFixed(2) : Math.floor(start).toLocaleString()) + (suffix || '');
    }, 16);
  }

  // Intersection Observer for KPI cards
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target.querySelector('.q-kpi-val');
        if (el && !el.dataset.counted) {
          el.dataset.counted = true;
          const raw = parseFloat(el.dataset.val || el.textContent.replace(/[^0-9.]/g, ''));
          const suffix = el.dataset.suffix || '';
          const isFloat = el.dataset.float === 'true';
          countUp(el, raw, 1200, isFloat, suffix);
        }
        // Animate progress bars
        entry.target.querySelectorAll('.q-kpi-fill').forEach(bar => {
          const w = bar.dataset.width;
          if (w) { setTimeout(() => bar.style.width = w + '%', 200); }
        });
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.3 });

  // Run after DOM ready
  setTimeout(() => {
    document.querySelectorAll('.q-kpi').forEach(el => observer.observe(el));
  }, 800);
})();
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="q-topbar">
  <div class="q-logo">
    <div class="q-logo-mark">⚖️</div>
    <div>
      <div class="q-logo-text">Qys<span>tas</span></div>
      <div class="q-logo-sub">Smart Pricing Engine</div>
    </div>
  </div>
  <div class="q-topbar-right">
    <span class="q-chip q-chip-live">Live Model</span>
    <span class="q-chip q-chip-ghost">Egypt 2026</span>
    <span class="q-chip q-chip-iris">v1.0</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# TICKER
# ─────────────────────────────────────────────────────────
TICKS = [
    ("Inflation Factor", "2.47×", "حضر"),
    ("Purchasing Power", "40.5%", "حقيقي"),
    ("Food Budget Share", "28.63%", "حضر"),
    ("Income Median 2026", "138,739 ج", "سنوياً"),
    ("Model AUC", "0.74", "CV Score"),
    ("Rural Inflation", "2.53×", "ريف"),
    ("Rural Purchasing Power", "39.49%", "حقيقي"),
    ("Training Scenarios", "24,480", "سيناريو"),
]

sep = '<span class="q-tick-sep"></span>'
ticks_html = (sep + " ").join(
    f'<span class="q-tick">{label}: <span class="q-tick-val">{val}</span> <span style="color:var(--dim);font-size:10px">{note}</span></span>'
    for label, val, note in TICKS
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
<div class="q-hero">
  <div class="q-hero-bg">
    <div class="q-hero-grid"></div>
    <div class="q-hero-orb-1"></div>
    <div class="q-hero-orb-2"></div>
  </div>

  <div class="q-hero-eyebrow">
    <span class="q-hero-eyebrow-dot"></span>
    <span class="q-hero-eyebrow-text">Egyptian Income Distribution · 2020 – 2026</span>
  </div>

  <div class="q-hero-h1">
    Price smarter.<br><span class="q-hero-h1-accent">Win the market.</span>
  </div>

  <div class="q-hero-sub">
    AI-powered pricing intelligence for Egyptian manufacturers —
    built on real income distribution data so every decision is grounded in how your customers actually live.
  </div>

  <div class="q-hero-stats">
    <div class="q-hero-stat">
      <div class="q-hero-stat-val iris">2.47×</div>
      <div class="q-hero-stat-label">Cumulative Inflation</div>
    </div>
    <div class="q-hero-stat">
      <div class="q-hero-stat-val jade">40.5%</div>
      <div class="q-hero-stat-label">Real Purchasing Power</div>
    </div>
    <div class="q-hero-stat">
      <div class="q-hero-stat-val">24K</div>
      <div class="q-hero-stat-label">Training Scenarios</div>
    </div>
    <div class="q-hero-stat">
      <div class="q-hero-stat-val iris">0.74</div>
      <div class="q-hero-stat-label">Model AUC Score</div>
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

  <div class="q-kpi iris-kpi">
    <div class="q-kpi-glow"></div>
    <div class="q-kpi-top">
      <div class="q-kpi-label">Income Median<br>2026 (Nominal)</div>
      <div class="q-kpi-icon iris">💰</div>
    </div>
    <div class="q-kpi-val" data-val="138" data-suffix="K">138K</div>
    <div class="q-kpi-delta delta-iris">▲ +146.8% nominal shift</div>
    <div class="q-kpi-bar"><div class="q-kpi-fill" data-width="65" style="width:0%;background:linear-gradient(90deg,#7C3AED,#9D5FF5)"></div></div>
  </div>

  <div class="q-kpi jade-kpi">
    <div class="q-kpi-glow"></div>
    <div class="q-kpi-top">
      <div class="q-kpi-label">Real Purchasing<br>Power (Urban)</div>
      <div class="q-kpi-icon jade">📉</div>
    </div>
    <div class="q-kpi-val" data-val="40.5" data-float="true" data-suffix="%">40.5%</div>
    <div class="q-kpi-delta delta-down">▼ vs nominal income</div>
    <div class="q-kpi-bar"><div class="q-kpi-fill" data-width="40" style="width:0%;background:linear-gradient(90deg,#06D6A0,#34D399)"></div></div>
  </div>

  <div class="q-kpi crim-kpi">
    <div class="q-kpi-glow"></div>
    <div class="q-kpi-top">
      <div class="q-kpi-label">Model Accuracy<br>AUC Score</div>
      <div class="q-kpi-icon crim">🤖</div>
    </div>
    <div class="q-kpi-val" data-val="0.74" data-float="true">0.74</div>
    <div class="q-kpi-delta delta-iris">5-fold cross-validation</div>
    <div class="q-kpi-bar"><div class="q-kpi-fill" data-width="74" style="width:0%;background:linear-gradient(90deg,#FF4757,#FF6B78)"></div></div>
  </div>

  <div class="q-kpi amber-kpi">
    <div class="q-kpi-glow"></div>
    <div class="q-kpi-top">
      <div class="q-kpi-label">Training<br>Scenarios</div>
      <div class="q-kpi-icon amber">📊</div>
    </div>
    <div class="q-kpi-val" data-val="24480" data-suffix="">24,480</div>
    <div class="q-kpi-delta delta-warn">synthetic simulations</div>
    <div class="q-kpi-bar"><div class="q-kpi-fill" data-width="88" style="width:0%;background:linear-gradient(90deg,#FFB020,#FCD34D)"></div></div>
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
        '<div class="q-section-sub">Log-Normal fit on 2020 and 2026 brackets reveals the real shift — '
        'nominal incomes rose by +146% while real purchasing power collapsed to 40.5¢ per earned pound.</div>',
        unsafe_allow_html=True,
    )

    area_col, _ = st.columns([1, 3])
    with area_col:
        area_choice = st.radio("Region:", ["Urban", "Rural"], horizontal=True, key="r1")

    data = get_curves(area_choice)
    x, p20, p26 = np.array(data["x"]), np.array(data["pdf_2020"]), np.array(data["pdf_2026"])

    # Premium chart
    fig = go.Figure()

    # Fill area 2020
    fig.add_trace(go.Scatter(
        x=x, y=p20,
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.08)",
        line=dict(color="#7C3AED", width=2),
        name="2020 Distribution",
        hovertemplate="<b>Income:</b> %{x:,.0f} EGP<br><b>Density:</b> %{y:.6f}<extra>2020</extra>",
    ))

    # Fill area 2026
    fig.add_trace(go.Scatter(
        x=x, y=p26,
        fill="tozeroy",
        fillcolor="rgba(255,71,87,0.07)",
        line=dict(color="#FF4757", width=2),
        name="2026 Distribution",
        hovertemplate="<b>Income:</b> %{x:,.0f} EGP<br><b>Density:</b> %{y:.6f}<extra>2026</extra>",
    ))

    # Median lines
    fig.add_vline(x=data["median_2020"], line_dash="dot", line_color="#7C3AED", line_width=1.5,
                  annotation_text=f"  2020 Median<br>  {data['median_2020']:,.0f} EGP",
                  annotation_font_size=11, annotation_font_color="#9D5FF5")
    fig.add_vline(x=data["median_2026"], line_dash="dot", line_color="#FF4757", line_width=1.5,
                  annotation_text=f"  2026 Median<br>  {data['median_2026']:,.0f} EGP",
                  annotation_font_size=11, annotation_font_color="#FF6B78")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.0)",
        xaxis=dict(
            title="Annual Household Income (EGP)",
            tickformat=",",
            range=[0, 420_000],
            gridcolor="rgba(255,255,255,0.04)",
            linecolor="rgba(255,255,255,0.08)",
            tickfont=dict(family="Inter", size=11, color="#4A5568"),
            title_font=dict(family="Inter", size=12, color="#8B9AB3"),
            zeroline=False,
        ),
        yaxis=dict(
            title="Probability Density",
            gridcolor="rgba(255,255,255,0.04)",
            linecolor="rgba(255,255,255,0.08)",
            tickfont=dict(family="Inter", size=11, color="#4A5568"),
            title_font=dict(family="Inter", size=12, color="#8B9AB3"),
            zeroline=False,
        ),
        legend=dict(
            x=0.70, y=0.96,
            bgcolor="rgba(13,17,23,0.85)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(family="Inter", size=12, color="#8B9AB3"),
        ),
        height=420,
        margin=dict(t=20, b=20, l=8, r=8),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(22,27,39,0.95)",
            bordercolor="rgba(124,58,237,0.4)",
            font=dict(family="Inter", size=12, color="#F8FAFC"),
        ),
        font=dict(family="Inter"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    pwr  = 40.50 if area_choice == "Urban" else 39.49
    food = 28.63 if area_choice == "Urban" else 36.39

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median Income 2020",   f"{data['median_2020']:,.0f} EGP/yr")
    c2.metric("Median Income 2026",   f"{data['median_2026']:,.0f} EGP/yr",
              delta=f"+{data['shift_pct']:.1f}% nominal")
    c3.metric("Real Purchasing Power", f"{pwr}%",
              delta=f"−{100-pwr:.1f}% lost to inflation", delta_color="inverse")
    c4.metric("Food Budget (Mandatory)", f"{food}%")

    st.markdown(
        f'<div class="q-callout iris"><span class="q-callout-icon">⚡</span>'
        f'<div><strong>Key insight:</strong> The rightward shift of the 2026 curve '
        f'(+{data["shift_pct"]:.1f}% nominal) is entirely explained by inflation (2.47×). '
        f'Your customers earn more numbers but command {pwr}% of 2020\'s real purchasing power. '
        f'Price decisions made on nominal data alone will overestimate affordability by {100-pwr:.0f}%.</div></div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════
# TAB 2 — PRODUCT INPUT
# ═══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="q-section-label">Product Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="q-section-title">Define Your Product</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="q-section-sub">Enter your product parameters. The engine will map your product\'s '
        'position against the 2026 income distribution and compute optimal recommendations.</div>',
        unsafe_allow_html=True,
    )

    with st.form("product_form"):
        c1, c2 = st.columns([1, 1], gap="large")

        with c1:
            st.markdown("**Current product parameters**")
            current_price  = st.number_input("Current price (EGP)", min_value=1.0, max_value=500.0, value=25.0, step=0.5)
            current_weight = st.number_input("Current weight (grams)", min_value=10.0, max_value=5000.0, value=100.0, step=5.0)
            cost_per_gram  = st.number_input("Production cost per gram (EGP)", min_value=0.01, max_value=10.0, value=0.18, step=0.01,
                                              help="Include raw materials, manufacturing, and packaging")
            new_price_in   = st.number_input("Proposed new price (EGP)", min_value=1.0, max_value=500.0, value=30.0, step=0.5,
                                              help="Used to predict churn if you raise price without changing weight")

        with c2:
            st.markdown("**Analysis parameters**")
            area_sel      = st.selectbox("Target region", ["Urban", "Rural"])
            target_margin = st.slider("Target profit margin (%)", 5, 60, 30, 5) / 100
            purchase_freq = st.slider("Monthly purchase frequency", 1, 20, 4)

            # Live summary
            cur_cost   = cost_per_gram * current_weight
            cur_margin = (current_price - cur_cost) / current_price * 100
            inc = ((new_price_in - current_price) / current_price * 100)

            st.markdown(f"""
            <div class="q-callout iris" style="margin-top:16px">
              <span class="q-callout-icon">📊</span>
              <div style="font-size:12px;line-height:1.8">
                <strong>Current economics:</strong><br>
                Production cost: <strong>{cur_cost:.2f} EGP</strong> &nbsp;·&nbsp;
                Margin: <strong>{cur_margin:.1f}%</strong><br>
                Price/gram: <strong>{current_price/current_weight:.3f} EGP/g</strong> &nbsp;·&nbsp;
                Proposed increase: <strong>+{inc:.1f}%</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

        submitted = st.form_submit_button("⚡ Run AI Analysis", use_container_width=True, type="primary")

    if submitted:
        st.session_state["product_data"] = {
            "current_price": current_price, "current_weight_g": current_weight,
            "cost_per_gram": cost_per_gram, "area": area_sel,
            "target_margin": target_margin, "purchase_freq": purchase_freq,
            "new_price": new_price_in,
        }
        st.success("✅ Analysis queued — view results in the AI Recommendation tab")

    # Affordability map
    if "product_data" in st.session_state:
        pd_data  = st.session_state["product_data"]
        segments = get_bracket_affordability(pd_data["current_price"], pd_data["area"], pd_data["purchase_freq"])
        seg_df   = pd.DataFrame(segments)

        st.markdown('<div class="q-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="q-section-label">Market Affordability</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="q-section-title">Purchasing Reach at {pd_data["current_price"]:.0f} EGP</div>',
            unsafe_allow_html=True,
        )

        fig_aff = go.Figure()
        fig_aff.add_trace(go.Bar(
            x=seg_df["bracket"],
            y=seg_df["price_burden_pct"],
            marker=dict(
                color=["#06D6A0" if r else "#FF4757" for r in seg_df["affordable"]],
                opacity=0.85,
                line=dict(width=0),
            ),
            text=[f"{v:.1f}%" for v in seg_df["price_burden_pct"]],
            textposition="outside",
            textfont=dict(family="Inter", size=11, color="#8B9AB3"),
            hovertemplate="<b>%{x}</b><br>Burden: <b>%{y:.1f}%</b><extra></extra>",
        ))
        fig_aff.add_hline(y=15, line_dash="dot", line_color="#FFB020", line_width=2,
                          annotation_text="Affordability threshold 15%",
                          annotation_font_color="#FFB020", annotation_font_size=11)
        fig_aff.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickangle=-30, gridcolor="rgba(255,255,255,0.04)",
                       tickfont=dict(family="Inter", size=10, color="#4A5568"), zeroline=False),
            yaxis=dict(title="Price Burden %", gridcolor="rgba(255,255,255,0.04)",
                       tickfont=dict(family="Inter", size=11, color="#4A5568"),
                       title_font=dict(family="Inter", size=12, color="#8B9AB3"), zeroline=False),
            height=300, margin=dict(t=24, b=80, l=8, r=8),
            font=dict(family="Inter"),
            hoverlabel=dict(bgcolor="rgba(22,27,39,0.95)", bordercolor="rgba(124,58,237,0.4)",
                            font=dict(family="Inter", size=12, color="#F8FAFC")),
        )
        st.plotly_chart(fig_aff, use_container_width=True)

        seg_df["Status"]       = seg_df["affordable"].apply(lambda x: "✅ Affordable" if x else "🔴 Out of reach")
        seg_df["Monthly Disp"] = seg_df["monthly_disposable"].apply(lambda v: f"{v:,.0f} EGP")
        st.dataframe(
            seg_df[["bracket","population_pct","Monthly Disp","price_burden_pct","Status"]].rename(
                columns={"bracket":"Income Bracket","population_pct":"Pop %","price_burden_pct":"Burden %"}
            ),
            use_container_width=True, hide_index=True,
        )

# ═══════════════════════════════════════════════
# TAB 3 — AI RECOMMENDATION
# ═══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="q-section-label">AI Engine Output</div>', unsafe_allow_html=True)
    st.markdown('<div class="q-section-title">Smart Recommendation</div>', unsafe_allow_html=True)

    if "product_data" not in st.session_state:
        st.markdown("""
        <div class="q-callout amber">
          <span class="q-callout-icon">⚠️</span>
          <div>No product configured yet. Go to <strong>Product Input</strong> and run the analysis first.</div>
        </div>
        """, unsafe_allow_html=True)
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

    with st.spinner("Running gradient boosting inference..."):
        w_rec  = optimizer.find_optimal_weight(product)
        c_pred = optimizer.predict_market_churn(product, pd_data["new_price"])

    # ── SCENARIO A: Optimal Weight ──────────────────────────
    st.markdown("### Strategy A — Optimal Weight")
    st.markdown(
        '<div class="q-section-sub" style="margin-bottom:20px">'
        'Hold the psychological price point. Adjust weight to hit your margin target.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Weight",   f"{product.current_weight_g:.0f}g")
    c2.metric("Optimal Weight",   f"{w_rec.optimal_weight_g}g",
              delta=f"−{w_rec.weight_reduction_pct}%", delta_color="inverse")
    c3.metric("New Margin",       f"{w_rec.new_margin_pct}%")
    c4.metric("New Price/Gram",   f"{w_rec.price_per_gram_new} EGP/g")

    callout_class = "jade" if w_rec.feasible else "crim"
    st.markdown(
        f'<div class="q-callout {callout_class}"><span class="q-callout-icon">{"✅" if w_rec.feasible else "🚨"}</span>'
        f'<div><strong>Assessment:</strong> {w_rec.warning}</div></div>',
        unsafe_allow_html=True,
    )

    # Weight comparison chart
    fig_w = go.Figure()
    fig_w.add_trace(go.Bar(
        x=["Current Weight", "Optimal Weight"],
        y=[product.current_weight_g, w_rec.optimal_weight_g],
        marker=dict(
            color=["rgba(124,58,237,0.6)", "rgba(6,214,160,0.7)"],
            line=dict(width=0),
        ),
        text=[f"{product.current_weight_g:.0f}g", f"{w_rec.optimal_weight_g}g"],
        textposition="outside",
        textfont=dict(family="Space Grotesk", size=14, color="#F8FAFC"),
        width=0.4,
    ))
    fig_w.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(family="Inter", size=12, color="#8B9AB3"), zeroline=False),
        yaxis=dict(title="Weight (grams)", gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(family="Inter", size=11, color="#4A5568"),
                   title_font=dict(family="Inter", size=12, color="#8B9AB3"), zeroline=False),
        height=260, margin=dict(t=24, b=20, l=8, r=8),
        font=dict(family="Inter"),
        hoverlabel=dict(bgcolor="rgba(22,27,39,0.95)", bordercolor="rgba(124,58,237,0.4)",
                        font=dict(family="Inter", size=12, color="#F8FAFC")),
    )
    st.plotly_chart(fig_w, use_container_width=True)

    st.markdown('<div class="q-divider"></div>', unsafe_allow_html=True)

    # ── SCENARIO B: Churn Prediction ────────────────────────
    risk_colors = {"HIGH": "crim", "MEDIUM": "amber", "LOW": "jade"}
    risk_icons  = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "✅"}
    rc          = risk_colors[c_pred.risk_level]
    ri          = risk_icons[c_pred.risk_level]

    st.markdown(
        f"### Strategy B — Price Raise to {c_pred.new_price:.0f} EGP (+{c_pred.price_increase_pct}%)"
    )
    st.markdown(
        '<div class="q-section-sub" style="margin-bottom:20px">'
        'Hold weight constant. Predict customer churn across every income bracket.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Churn",     f"{c_pred.weighted_churn_pct}%")
    c2.metric("At-Risk Population",  f"{c_pred.at_risk_population_pct}%")
    c3.metric("ML Churn Probability", f"{c_pred.ml_churn_prob}%")
    c4.metric("Risk Level",           c_pred.risk_level)

    st.markdown(
        f'<div class="q-callout {rc}"><span class="q-callout-icon">{ri}</span>'
        f'<div><strong>Decision signal:</strong> {c_pred.recommendation}</div></div>',
        unsafe_allow_html=True,
    )

    # Segment breakdown chart
    seg_df = pd.DataFrame(c_pred.segments_detail)

    fig_s = go.Figure()
    fig_s.add_trace(go.Bar(
        x=seg_df["bracket"],
        y=seg_df["price_burden_pct"],
        marker=dict(
            color=["rgba(255,71,87,0.7)" if r else "rgba(6,214,160,0.6)" for r in seg_df["at_risk"]],
            line=dict(width=0),
        ),
        name="Price Burden %",
        text=[f"{v:.1f}%" for v in seg_df["price_burden_pct"]],
        textposition="outside",
        textfont=dict(family="Inter", size=10, color="#8B9AB3"),
        hovertemplate="<b>%{x}</b><br>Burden: %{y:.1f}%<extra>Burden</extra>",
    ))
    fig_s.add_trace(go.Scatter(
        x=seg_df["bracket"],
        y=seg_df["churn_threshold_pct"],
        mode="lines+markers",
        name="Churn Threshold",
        line=dict(color="#FFB020", width=2, dash="dash"),
        marker=dict(size=8, color="#FFB020", symbol="circle"),
        hovertemplate="<b>%{x}</b><br>Threshold: %{y:.1f}%<extra>Threshold</extra>",
    ))
    fig_s.add_trace(go.Scatter(
        x=seg_df["bracket"],
        y=[v * 100 for v in seg_df["ml_churn_prob"]],
        mode="lines+markers",
        name="ML Churn Probability %",
        line=dict(color="#9D5FF5", width=2),
        marker=dict(size=6, color="#9D5FF5"),
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.06)",
        hovertemplate="<b>%{x}</b><br>ML Churn Prob: %{y:.1f}%<extra>ML Model</extra>",
    ))
    fig_s.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-30, gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(family="Inter", size=10, color="#4A5568"), zeroline=False),
        yaxis=dict(title="%", gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(family="Inter", size=11, color="#4A5568"),
                   title_font=dict(family="Inter", size=12, color="#8B9AB3"), zeroline=False),
        legend=dict(x=0.60, y=0.97, bgcolor="rgba(13,17,23,0.9)",
                    bordercolor="rgba(255,255,255,0.08)", borderwidth=1,
                    font=dict(family="Inter", size=11, color="#8B9AB3")),
        height=400, margin=dict(t=20, b=90, l=8, r=8),
        font=dict(family="Inter"),
        hoverlabel=dict(bgcolor="rgba(22,27,39,0.95)", bordercolor="rgba(124,58,237,0.4)",
                        font=dict(family="Inter", size=12, color="#F8FAFC")),
    )
    st.plotly_chart(fig_s, use_container_width=True)

    # Detailed table
    seg_show = seg_df[[
        "bracket","population_pct","monthly_disposable",
        "price_burden_pct","churn_threshold_pct","at_risk","ml_churn_prob",
    ]].copy()
    seg_show.columns = ["Bracket","Pop %","Monthly Disposable","Burden %","Threshold %","At Risk","ML Prob"]
    seg_show["At Risk"]  = seg_show["At Risk"].apply(lambda x: "🔴 Yes" if x else "🟢 No")
    seg_show["ML Prob"]  = seg_show["ML Prob"].apply(lambda x: f"{x:.0%}")
    st.dataframe(seg_show, use_container_width=True, hide_index=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="q-footer">
  <div class="q-footer-brand">
    <div class="q-footer-mark">⚖️</div>
    <div>
      <div class="q-footer-name">Qys<span>tas</span></div>
      <div class="q-footer-copy">© 2026 Qystas Smart Pricing Engine · Egyptian Income Distribution Data</div>
    </div>
  </div>
  <div class="q-footer-pills">
    <span class="q-footer-pill">ML Powered</span>
    <span class="q-footer-pill">Egypt 2026</span>
    <span class="q-footer-pill">Log-Normal</span>
    <span class="q-footer-pill">Gradient Boosting</span>
    <span class="q-footer-pill">Open Source</span>
  </div>
</div>
""", unsafe_allow_html=True)
