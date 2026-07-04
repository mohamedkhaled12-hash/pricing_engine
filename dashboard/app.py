"""
dashboard/app.py
================
الـ Dashboard الكامل — 3 شاشات مع تصميم Qystas Premium و Dark/Light Mode.

تشغيل:
  cd pricing_engine
  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from core.curve_fitting import get_curve_data, get_bracket_affordability
from core.optimizer import PricingOptimizer, ProductInput
from core.ml_model import load_churn_model, train_churn_model

# ─────────────────────────────────────────────
# إعداد الصفحة
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Qystas | محرك التسعير الذكي",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# القائمة الجانبية (Sidebar) - إعدادات الثيم
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ إعدادات العرض")
    dark_mode = st.toggle("🌙 تفعيل الوضع الليلي (Dark Mode)", value=False)

# ─────────────────────────────────────────────
# تحديد متغيرات الألوان بناءً على الثيم
# ─────────────────────────────────────────────
if dark_mode:
    # ألوان الوضع الليلي
    theme_colors = """
    --bg:      #0B1121;
    --bg2:     #1F2937;
    --white:   #111827;
    --text:    #F3F4F6;
    --muted:   #9CA3AF;
    --navy:    #1E3A8A;
    --gold:    #EAB308;
    --green:   #10B981;
    --red:     #EF4444;
    --amber:   #F59E0B;
    """
    chart_text_color = "#F3F4F6"
    chart_grid_color = "#1F2937"
else:
    # ألوان وضع النهار
    theme_colors = """
    --bg:      #F4F7F9;
    --bg2:     #E2EAF4;
    --white:   #FFFFFF;
    --text:    #1A2E4A;
    --muted:   #7A8FA6;
    --navy:    #0B2545;
    --gold:    #C9A84C;
    --green:   #1E8C5A;
    --red:     #C0392B;
    --amber:   #D68910;
    """
    chart_text_color = "#1A2E4A"
    chart_grid_color = "#E2EAF4"

# ─────────────────────────────────────────────
# حقن الـ CSS المخصص الخاص بـ Qystas وتصحيح الـ Labels
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Cairo:wght@300;400;600;700;900&display=swap');

  :root {{
    {theme_colors}
  }}

  /* Overrides for Streamlit App Background */
  .stApp {{
      background-color: var(--bg);
      font-family: 'Cairo', 'Inter', sans-serif;
      color: var(--text);
  }}
  
  /* Hide Streamlit Header */
  header {{visibility: hidden;}}

  /* ══════════════════════════════════════════════════
     إصلاح مشكلة تباين الألوان (Labels & Markdown)
  ══════════════════════════════════════════════════ */
  div[data-testid="stWidgetLabel"] p, 
  .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
  .stText p, label p {{
      color: var(--text) !important;
      font-weight: 700 !important;
  }}
  
  /* إصلاح ألوان نصوص التنبيهات (Success, Warning, etc) */
  div[data-testid="stAlert"] {{
      background-color: var(--white) !important;
      border: 1px solid var(--bg2) !important;
      color: var(--text) !important;
  }}
  div[data-testid="stAlert"] p {{
      color: var(--text) !important;
  }}

  /* Customizing Streamlit Tabs */
  .stTabs [data-baseweb="tab-list"] {{
      background-color: var(--white);
      padding: 8px;
      border-radius: 14px;
      border: 1px solid var(--bg2);
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
      gap: 8px;
  }}
  .stTabs [data-baseweb="tab"] {{
      padding: 12px 24px;
      border-radius: 10px;
      font-family: 'Cairo', sans-serif;
      font-size: 15px;
      font-weight: 700;
      color: var(--muted);
      border: none;
      background: transparent;
      transition: all 0.2s ease;
  }}
  .stTabs [aria-selected="true"] {{
      background-color: var(--navy) !important;
      color: #FFFFFF !important;
      box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
  }}
  
  /* Customizing Streamlit Inputs */
  .stNumberInput input, .stSelectbox div[data-baseweb="select"] {{
      background-color: var(--white) !important;
      border: 2px solid var(--bg2) !important;
      border-radius: 10px !important;
      font-family: 'Inter', sans-serif;
      font-weight: 600;
      color: var(--text) !important;
      transition: border-color 0.2s;
  }}
  .stNumberInput input:focus {{ border-color: var(--gold) !important; }}
  
  .stButton button {{
      background-color: var(--navy) !important;
      color: #FFFFFF !important;
      border-radius: 12px !important;
      font-family: 'Cairo', sans-serif !important;
      font-weight: 700 !important;
      font-size: 16px !important;
      padding: 12px 24px !important;
      border: none !important;
      transition: all 0.3s !important;
      box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
  }}
  .stButton button:hover {{
      background-color: var(--gold) !important;
      color: var(--navy) !important;
      transform: translateY(-2px) !important;
      box-shadow: 0 6px 16px rgba(201,168,76,0.4) !important;
  }}

  /* ══════════════════════════════
     Qystas Custom HTML Elements
  ══════════════════════════════ */
  .logo-showcase {{
    background: var(--navy);
    padding: 32px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 32px;
    border-radius: 0 0 24px 24px;
    margin-top: -60px;
  }}
  .logo-main {{ display: flex; align-items: center; gap: 16px; }}
  .logo-icon {{ width: 56px; height: 56px; flex-shrink: 0; }}
  .logo-text-group {{ display: flex; flex-direction: column; gap: 2px; }}
  .logo-name {{ font-family: 'Inter', sans-serif; font-size: 28px; font-weight: 900; color: var(--gold); letter-spacing: -0.5px; line-height: 1; text-transform: uppercase;}}
  .logo-tagline {{ font-family: 'Cairo', sans-serif; font-size: 12px; font-weight: 600; color: #FFFFFF; letter-spacing: 2px; text-transform: uppercase; line-height: 1; opacity: 0.8;}}

  /* Ticker */
  .ticker {{ background: var(--gold); padding: 10px 0; overflow: hidden; position: relative; margin-top: 16px; border-radius: 8px;}}
  .ticker-track {{ display: flex; gap: 0; animation: ticker 25s linear infinite; white-space: nowrap; }}
  .ticker-item {{ display: inline-flex; align-items: center; gap: 8px; padding: 0 32px; font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 700; color: #000000; }}
  .ticker-dot {{ width: 6px; height: 6px; background: #000000; border-radius: 50%; opacity: 0.5; }}
  @keyframes ticker {{ 0% {{ transform: translateX(0); }} 100% {{ transform: translateX(-50%); }} }}

  /* Hero */
  .hero {{
    background: linear-gradient(135deg, var(--navy) 0%, #1A4A8A 100%);
    padding: 48px 40px;
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    margin-top: 16px;
    margin-bottom: 32px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
  }}
  .hero-eyebrow {{ display: inline-flex; align-items: center; gap: 8px; background: rgba(201,168,76,0.15); border: 1px solid rgba(201,168,76,0.3); padding: 6px 14px; border-radius: 100px; margin-bottom: 24px; }}
  .hero-eyebrow-dot {{ width: 6px; height: 6px; background: var(--gold); border-radius: 50%; animation: pulse 2s infinite; }}
  @keyframes pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.5; transform: scale(0.8); }} }}
  .hero-eyebrow span {{ font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 700; color: var(--gold); letter-spacing: 1.5px; text-transform: uppercase; }}
  .hero-headline {{ font-family: 'Inter', sans-serif; font-size: 42px; font-weight: 900; color: #FFFFFF; line-height: 1.2; margin-bottom: 16px; }}
  .hero-headline .accent {{ color: var(--gold); }}
  .hero-sub {{ font-family: 'Cairo', sans-serif; font-size: 17px; font-weight: 400; color: rgba(255,255,255,0.8); max-width: 650px; margin-bottom: 20px; line-height: 1.6;}}

  /* KPI Cards */
  .kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 32px;}}
  .kpi-card {{ background: var(--white); border-radius: 16px; padding: 24px; border: 1px solid var(--bg2); box-shadow: 0 4px 12px rgba(0,0,0,0.05); display: flex; flex-direction: column; gap: 12px; transition: all 0.3s; border-top: 4px solid var(--navy); }}
  .kpi-card:hover {{ box-shadow: 0 8px 24px rgba(0,0,0,0.12); transform: translateY(-4px); border-top-color: var(--gold);}}
  .kpi-top {{ display: flex; align-items: center; justify-content: space-between; }}
  .kpi-label {{ font-family: 'Cairo', sans-serif; font-size: 14px; font-weight: 700; color: var(--muted); }}
  .kpi-badge {{ width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px; }}
  .kpi-badge.navy {{ background: rgba(11,37,69,0.07); }} .kpi-badge.gold {{ background: rgba(201,168,76,0.15); }} .kpi-badge.green {{ background: rgba(30,140,90,0.15); }} .kpi-badge.red {{ background: rgba(192,57,43,0.15); }}
  .kpi-value {{ font-family: 'Inter', sans-serif; font-size: 32px; font-weight: 900; color: var(--text); line-height: 1; }}
  .kpi-delta {{ display: inline-flex; align-items: center; gap: 4px; font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 100px; width: fit-content;}}
  .kpi-delta.up {{ background: rgba(30,140,90,0.1); color: var(--green); }}
  .kpi-delta.down {{ background: rgba(192,57,43,0.1); color: var(--red); }}
  .kpi-delta.warn {{ background: rgba(214,137,16,0.15); color: var(--amber); }}

  /* Rec Cards */
  .rec-card {{ border-radius: 16px; padding: 24px; margin-bottom: 16px; border: 2px solid; display: flex; flex-direction: column; gap: 12px; background: var(--white); }}
  .rec-card.success {{ border-color: var(--green); box-shadow: 0 8px 20px rgba(30,140,90,0.1); }}
  .rec-card.warning {{ border-color: var(--amber); box-shadow: 0 8px 20px rgba(214,137,16,0.1); }}
  .rec-card.danger {{ border-color: var(--red); box-shadow: 0 8px 20px rgba(192,57,43,0.1); }}
  .rec-title {{ font-family: 'Cairo', sans-serif; font-size: 18px; font-weight: 800; display: flex; align-items: center; gap: 8px; margin-bottom: 8px;}}
  .rec-title.success {{ color: var(--green); }} .rec-title.warning {{ color: var(--amber); }} .rec-title.danger {{ color: var(--red); }}
  .rec-row {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px dashed var(--bg2); }}
  .rec-row:last-child {{ border-bottom: none; }}
  .rec-key {{ font-family: 'Cairo', sans-serif; font-size: 14px; color: var(--muted); font-weight: 700; }}
  .rec-val {{ font-family: 'Inter', sans-serif; font-size: 18px; font-weight: 900; color: var(--text); }}
  
  /* Table Risk Pills */
  .risk-pill {{ display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px; border-radius: 100px; font-size: 12px; font-weight: 800; font-family: 'Cairo', sans-serif; letter-spacing: 0.5px;}}
  .risk-pill.ok {{ background: rgba(30,140,90,0.15); color: var(--green); border: 1px solid rgba(30,140,90,0.3);}}
  .risk-pill.hi {{ background: rgba(192,57,43,0.15); color: var(--red); border: 1px solid rgba(192,57,43,0.3);}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# تحميل الـ Resources
# ─────────────────────────────────────────────
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
model = get_model()

# ─────────────────────────────────────────────
# الهيدر المخصص (Qystas Logo + Ticker + Hero)
# ─────────────────────────────────────────────
custom_header = """<div class="logo-showcase">
<div class="logo-main">
<svg class="logo-icon" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="28" cy="28" r="28" fill="#C9A84C"/>
<rect x="14" y="26" width="28" height="3" rx="1.5" fill="#0B2545"/>
<rect x="26.5" y="18" width="3" height="20" rx="1.5" fill="#0B2545"/>
<path d="M14 27 Q11 33 18 33 Q25 33 22 27" fill="none" stroke="#0B2545" stroke-width="2.5" stroke-linecap="round"/>
<path d="M34 27 Q31 31 38 31 Q45 31 42 27" fill="none" stroke="#0B2545" stroke-width="2.5" stroke-linecap="round" opacity="0.6"/>
<path d="M39 22 L39 17 M37 19 L39 17 L41 19" stroke="#0B2545" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
<div class="logo-text-group">
<div class="logo-name">QYSTAS</div>
<div class="logo-tagline">AI Pricing Intelligence</div>
</div>
</div>
</div>
<div class="ticker">
<div class="ticker-track">
<span class="ticker-item">📈 تضخم تراكمي حضري: 2.47×<div class="ticker-dot"></div></span>
<span class="ticker-item">💰 القوة الشرائية الحقيقية: 40.5%<div class="ticker-dot"></div></span>
<span class="ticker-item">🍞 ميزانية الأكل الإجبارية: 28.63%<div class="ticker-dot"></div></span>
<span class="ticker-item">📊 وسيط الدخل 2026: 138,739 جنيه/سنة<div class="ticker-dot"></div></span>
<span class="ticker-item">⚖️ Qystas AI — نظام التسعير العادل<div class="ticker-dot"></div></span>
<span class="ticker-item">📈 تضخم تراكمي حضري: 2.47×<div class="ticker-dot"></div></span>
<span class="ticker-item">💰 القوة الشرائية الحقيقية: 40.5%<div class="ticker-dot"></div></span>
</div>
</div>
<div class="hero">
<div class="hero-eyebrow">
<div class="hero-eyebrow-dot"></div>
<span>مبني على بيانات توزيع الدخل 2020–2026 (Log-Normal)</span>
</div>
<div class="hero-headline">
قرار التسعير الصح<br>في <span class="accent">ثوانٍ</span>
</div>
<div class="hero-sub">
نظام ذكاء اصطناعي يحسب لك الوزن الأمثل ونسبة المقاطعة المتوقعة (Churn) — بناءً على الدخل الحقيقي لزبائنك اليوم والقوة الشرائية.
</div>
</div>"""

st.markdown(custom_header, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# التبويبات الرئيسية
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 رادار الانكماش والقدرة الشرائية",
    "🏭 بيانات المنتج (Input)",
    "🎯 التوصية الذكية (Qystas AI)"
])

# ═══════════════════════════════════════════════════════
# شاشة ١: رادار انكماش الطبقات
# ═══════════════════════════════════════════════════════
with tab1:
    area_choice = st.radio("تحليل المنطقة المستهدفة:", ["Urban", "Rural"], horizontal=True)
    data = get_curves(area_choice)
    
    pwr = 40.5 if area_choice == "Urban" else 39.49
    food = 28.63 if area_choice == "Urban" else 36.39
    
    kpis_html = f"""<div class="kpi-row">
<div class="kpi-card">
<div class="kpi-top">
<div class="kpi-label">وسيط الدخل {area_choice} (2020)</div>
<div class="kpi-badge navy">🏦</div>
</div>
<div class="kpi-value">{data['median_2020']:,.0f} <span style="font-size:16px;color:var(--muted)">ج</span></div>
<div class="kpi-delta up">سنة الأساس</div>
</div>
<div class="kpi-card">
<div class="kpi-top">
<div class="kpi-label">وسيط الدخل (2026)</div>
<div class="kpi-badge gold">💰</div>
</div>
<div class="kpi-value">{data['median_2026']:,.0f} <span style="font-size:16px;color:var(--muted)">ج</span></div>
<div class="kpi-delta up">▲ +{data['shift_pct']:.1f}% اسمي</div>
</div>
<div class="kpi-card">
<div class="kpi-top">
<div class="kpi-label">القوة الشرائية الحقيقية</div>
<div class="kpi-badge red">📉</div>
</div>
<div class="kpi-value">{pwr}%</div>
<div class="kpi-delta down">▼ انخفاض حقيقي</div>
</div>
<div class="kpi-card">
<div class="kpi-top">
<div class="kpi-label">ميزانية الأكل الإجبارية</div>
<div class="kpi-badge green">🍞</div>
</div>
<div class="kpi-value">{food}%</div>
<div class="kpi-delta warn">ضغط على الميزانية</div>
</div>
</div>"""
    st.markdown(kpis_html, unsafe_allow_html=True)

    x = np.array(data["x"])
    p20 = np.array(data["pdf_2020"])
    p26 = np.array(data["pdf_2026"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=p20, fill="tozeroy", fillcolor="rgba(74,111,165,0.15)",
        line=dict(color="#4A6FA5", width=3), name="توزيع 2020"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p26, fill="tozeroy", fillcolor="rgba(192,57,43,0.15)",
        line=dict(color="#C0392B", width=3), name="توزيع 2026"
    ))
    fig.add_vline(x=data["median_2020"], line_dash="dash", line_color="#4A6FA5", annotation_text=f"وسيط 2020", annotation_position="top right")
    fig.add_vline(x=data["median_2026"], line_dash="dash", line_color="#C0392B", annotation_text=f"وسيط 2026", annotation_position="top right")

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="الدخل السنوي المتاح (جنيه)",
        yaxis_title="كثافة الاحتمال (Log-Normal Distribution)",
        legend=dict(x=0.75, y=0.95, bgcolor="rgba(255,255,255,0.1)", bordercolor=chart_grid_color, borderwidth=1),
        height=450,
        margin=dict(t=30, b=30, l=10, r=10),
        hovermode="x unified",
        font=dict(family="Cairo", color=chart_text_color, size=14)
    )
    fig.update_xaxes(tickformat=",", range=[0, 400_000], showgrid=True, gridcolor=chart_grid_color)
    fig.update_yaxes(showgrid=True, gridcolor=chart_grid_color)
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **تحليل Qystas:** الإزاحة لليمين لا تعني تحسناً — الأرقام ارتفعت بفعل التضخم لكن القوة الشرائية الحقيقية انخفضت بشدة.")

# ═══════════════════════════════════════════════════════
# شاشة ٢: إدخال بيانات المصنع
# ═══════════════════════════════════════════════════════
with tab2:
    with st.container():
        st.markdown("### 🏭 إدخال بيانات المنتج والمستهدفات")
        with st.form("product_form"):
            col1, col2 = st.columns(2)

            with col1:
                current_price    = st.number_input("السعر الحالي للمستهلك (جنيه)", min_value=1.0, max_value=500.0, value=25.0, step=0.5)
                current_weight   = st.number_input("الوزن الحالي (جرام)",  min_value=10.0, max_value=5000.0, value=100.0, step=5.0)
                cost_per_gram    = st.number_input("تكلفة الإنتاج للجرام (جنيه/جرام)", min_value=0.01, max_value=10.0, value=0.18, step=0.01)
                new_price_input  = st.number_input("سعر افتراضي جديد لاختبار المقاطعة (جنيه)", min_value=1.0, max_value=500.0, value=30.0, step=0.5)

            with col2:
                area_sel       = st.selectbox("المنطقة المستهدفة للبيع", ["Urban", "Rural"])
                target_margin  = st.slider("هامش الربح المستهدف (Shrinkflation target) %", min_value=5, max_value=60, value=30, step=5) / 100
                purchase_freq  = st.slider("تكرار شراء المنتج (مرات/شهر)", min_value=1, max_value=20, value=4)
                
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("⚖️ تشغيل محرك Qystas AI", use_container_width=True)

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
        st.success("✅ تم حفظ البيانات وتشغيل النموذج. انتقل لتبويب 'التوصية الذكية' لرؤية النتائج.")

    if "product_data" in st.session_state:
        st.markdown("### 📋 خريطة القدرة الشرائية بأسعار اليوم (Affordability Matrix)")
        pd_data = st.session_state["product_data"]
        segments = get_bracket_affordability(pd_data["current_price"], pd_data["area"], pd_data["purchase_freq"])
        
        table_rows = ""
        for seg in segments:
            risk_class = "ok" if seg['affordable'] else "hi"
            risk_text = "🟢 في المتناول" if seg['affordable'] else "🔴 خارج المتناول"
            color_class = "var(--green)" if seg['affordable'] else "var(--red)"
            bar_width = min(seg['price_burden_pct'], 100)
            
            table_rows += f"""<tr style="border-bottom: 1px solid var(--bg2); transition: background 0.2s;">
<td style="padding: 16px; font-weight:700; color: var(--text); font-size:14px;">{seg['bracket']}</td>
<td style="padding: 16px; font-weight:800; color: var(--text); font-size:14px;">{seg['population_pct']:.2f}%</td>
<td style="padding: 16px; font-family: 'Inter', sans-serif; color: var(--text); font-size:14px; font-weight:600;">{seg['monthly_disposable']:,.0f} ج</td>
<td style="padding: 16px;">
<div style="display:flex; align-items:center; gap:12px;">
<span style="font-weight:900; font-family: 'Inter', sans-serif; color:{color_class}; width:50px; text-align:left; font-size:15px;">{seg['price_burden_pct']:.1f}%</span>
<div style="flex-grow:1; height:8px; background:var(--bg2); border-radius:10px; overflow:hidden;">
<div style="width:{bar_width}%; height:100%; background:{color_class}; border-radius:10px;"></div>
</div>
</div>
</td>
<td style="padding: 16px;"><span class="risk-pill {risk_class}">{risk_text}</span></td>
</tr>"""
            
        full_table = f"""<div style="overflow-x: auto; background: var(--white); border-radius: 16px; border: 1px solid var(--bg2); box-shadow: 0 4px 16px rgba(0,0,0,0.05); margin-top: 16px;">
<table class="seg-table" dir="rtl" style="width: 100%; min-width: 750px; margin: 0; border-collapse: collapse;">
<thead style="background: var(--navy); color: #FFFFFF;">
<tr>
<th style="padding: 18px 16px; text-align: right; font-family: 'Cairo', sans-serif; font-size: 14px; font-weight: 800;">الفئة الدخلية السنوية</th>
<th style="padding: 18px 16px; text-align: right; font-family: 'Cairo', sans-serif; font-size: 14px; font-weight: 800;">% من السكان</th>
<th style="padding: 18px 16px; text-align: right; font-family: 'Cairo', sans-serif; font-size: 14px; font-weight: 800;">الدخل المتاح/شهر</th>
<th style="padding: 18px 16px; text-align: right; font-family: 'Cairo', sans-serif; font-size: 14px; font-weight: 800;">العبء السعري للسلعة %</th>
<th style="padding: 18px 16px; text-align: right; font-family: 'Cairo', sans-serif; font-size: 14px; font-weight: 800;">الحالة السلوكية</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>"""
        st.markdown(full_table, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# شاشة ٣: التوصية الذكية (ML & Optimization)
# ═══════════════════════════════════════════════════════
with tab3:
    if "product_data" not in st.session_state:
        st.warning("⚠️ برجاء إدخال بيانات المصنع في الشاشة السابقة أولاً لتفعيل محرك Qystas.")
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

    with st.spinner("🧠 Qystas AI قيد التشغيل... يتم تحليل البيانات..."):
        weight_rec  = optimizer.find_optimal_weight(product)
        churn_pred  = optimizer.predict_market_churn(product, pd_data["new_price"])

    colA, colB = st.columns(2, gap="large")

    with colA:
        st.markdown("### ⚖️ سيناريو أ: التوصية بالانكماش (Shrinkflation)")
        st.caption("الحفاظ على السعر النفسي للزبون مع تقليل الوزن للوصول للربح المستهدف")
        
        status_class = "success" if weight_rec.feasible else "danger"
        icon = "✅" if weight_rec.feasible else "❌"
        
        rec_a_html = f"""<div class="rec-card {status_class}">
<div class="rec-title {status_class}">{icon} الوزن الأمثل للإنتاج</div>
<div class="rec-row">
<span class="rec-key">الوزن الجديد المقترح</span>
<span class="rec-val" style="color:var(--gold); font-size:24px;">{weight_rec.optimal_weight_g} جرام</span>
</div>
<div class="rec-row">
<span class="rec-key">نسبة التخفيض من الحجم الأصلي</span>
<span class="rec-val" style="color:var(--red)">-{weight_rec.weight_reduction_pct}%</span>
</div>
<div class="rec-row">
<span class="rec-key">هامش الربح المحقق</span>
<span class="rec-val" style="color:var(--green)">{weight_rec.new_margin_pct}%</span>
</div>
<div style="font-size:13px; font-family:'Cairo',sans-serif; font-weight:700; color:var(--text); margin-top:12px; padding:10px; background:var(--bg); border-radius:8px;">
{weight_rec.warning}
</div>
</div>"""
        st.markdown(rec_a_html, unsafe_allow_html=True)
        
        fig_w = px.bar(
            x=["الوزن الأصلي", "وزن Qystas الأمثل"], 
            y=[product.current_weight_g, weight_rec.optimal_weight_g],
            color=["Original", "Optimized"],
            color_discrete_sequence=["#4A6FA5", "#C9A84C"],
            text=[f"{product.current_weight_g}g", f"{weight_rec.optimal_weight_g}g"]
        )
        fig_w.update_layout(height=280, margin=dict(t=20,b=20), showlegend=False, plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Cairo", size=14, color=chart_text_color))
        fig_w.update_traces(textposition='auto', textfont_size=18, textfont_color="white", marker_line_width=0)
        fig_w.update_xaxes(showgrid=False)
        fig_w.update_yaxes(showgrid=True, gridcolor=chart_grid_color)
        st.plotly_chart(fig_w, use_container_width=True)

    with colB:
        st.markdown("### 📈 سيناريو ب: زيادة السعر المباشرة (Price Hike)")
        st.caption(f"تأثير رفع السعر لـ {churn_pred.new_price} ج (+{churn_pred.price_increase_pct}%) دون تغيير الحجم")

        risk_classes = {"HIGH": "danger", "MEDIUM": "warning", "LOW": "success"}
        rc = risk_classes.get(churn_pred.risk_level, "success")
        risk_icon = "🔴" if churn_pred.risk_level == "HIGH" else "⚠️" if churn_pred.risk_level == "MEDIUM" else "🟢"
        
        rec_b_html = f"""<div class="rec-card {rc}">
<div class="rec-title {rc}">{risk_icon} توقعات Qystas ML للمقاطعة</div>
<div class="rec-row">
<span class="rec-key">نسبة المقاطعة المتوقعة (Market Churn)</span>
<span class="rec-val" style="color:var(--red); font-size:24px;">{churn_pred.weighted_churn_pct}%</span>
</div>
<div class="rec-row">
<span class="rec-key">شريحة السكان المعرضة للتسرب</span>
<span class="rec-val">{churn_pred.at_risk_population_pct}%</span>
</div>
<div class="rec-row">
<span class="rec-key">تقييم المخاطرة العام</span>
<span class="rec-val">{churn_pred.risk_level} RISK</span>
</div>
<div style="font-size:13px; font-family:'Cairo',sans-serif; font-weight:700; color:var(--text); margin-top:12px; padding:10px; background:var(--bg); border-radius:8px;">
{churn_pred.recommendation}
</div>
</div>"""
        st.markdown(rec_b_html, unsafe_allow_html=True)
        
        seg_df = pd.DataFrame(churn_pred.segments_detail)
        fig_c = go.Figure()
        colors = ["#C0392B" if r else "#1E8C5A" for r in seg_df["at_risk"]]
        
        fig_c.add_trace(go.Bar(
            x=seg_df["bracket"], y=seg_df["price_burden_pct"],
            marker_color=colors, name="العبء السعري الجديد %"
        ))
        fig_c.add_trace(go.Scatter(
            x=seg_df["bracket"], y=seg_df["churn_threshold_pct"],
            mode="lines+markers", name="عتبة المقاطعة (Threshold)",
            line=dict(color=chart_text_color, width=2, dash="dot"), marker=dict(size=8, color="#C9A84C")
        ))
        fig_c.update_layout(
            height=280, margin=dict(t=20,b=20), plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Cairo", size=12, color=chart_text_color)
        )
        fig_c.update_xaxes(showgrid=False)
        fig_c.update_yaxes(showgrid=True, gridcolor=chart_grid_color)
        st.plotly_chart(fig_c, use_container_width=True)

# ─────────────────────────────────────────────
# الفوتر المخصص (Qystas Footer)
# ─────────────────────────────────────────────
footer_html = """<div style="background:var(--navy); padding: 24px 40px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px; border-radius: 24px 24px 0 0; margin-top: 48px; box-shadow: 0 -4px 16px rgba(0,0,0,0.2);">
<div style="display:flex;align-items:center;gap:16px;">
<svg width="28" height="28" viewBox="0 0 56 56" fill="none">
<circle cx="28" cy="28" r="28" fill="#C9A84C"/>
<rect x="14" y="26" width="28" height="3" rx="1.5" fill="#0B2545"/>
<rect x="26.5" y="18" width="3" height="20" rx="1.5" fill="#0B2545"/>
<path d="M14 27 Q11 33 18 33 Q25 33 22 27" fill="none" stroke="#0B2545" stroke-width="2.5" stroke-linecap="round"/>
<path d="M34 27 Q31 31 38 31 Q45 31 42 27" fill="none" stroke="#0B2545" stroke-width="2.5" stroke-linecap="round" opacity="0.6"/>
<path d="M39 22 L39 17 M37 19 L39 17 L41 19" stroke="#0B2545" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
<span style="font-family:'Cairo',sans-serif; font-size:14px; font-weight:600; color:rgba(255,255,255,0.8);">
© 2026 Qystas AI — نظام التسعير العادل. مبني على بيانات توزيع الدخل.
</span>
</div>
<div style="display:flex; gap:10px;">
<span style="padding: 6px 14px; border: 1px solid rgba(201,168,76,0.4); border-radius: 100px; font-family:'Inter',sans-serif; font-size:11px; font-weight:700; color:var(--gold); letter-spacing:1px;">XGBOOST POWERED</span>
<span style="padding: 6px 14px; border: 1px solid rgba(255,255,255,0.2); border-radius: 100px; font-family:'Inter',sans-serif; font-size:11px; font-weight:700; color:rgba(255,255,255,0.8); letter-spacing:1px;">EGYPT 2026</span>
</div>
</div>"""
st.markdown(footer_html, unsafe_allow_html=True)
