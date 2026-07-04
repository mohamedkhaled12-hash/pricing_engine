"""
dashboard/app.py
================
الـ Dashboard الكامل — 3 شاشات في تبويبات مع تصميم PriceOpt Premium.

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
    page_title="PriceOpt | محرك التسعير الذكي",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# حقن الـ CSS المخصص الخاص بك
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Cairo:wght@300;400;600;700;900&display=swap');

  :root {
    --navy:    #0B2545;
    --navy2:   #133A6A;
    --gold:    #C9A84C;
    --gold2:   #E8C96A;
    --slate:   #4A6FA5;
    --bg:      #F0F4F8;
    --bg2:     #E2EAF4;
    --white:   #FFFFFF;
    --text:    #1A2E4A;
    --muted:   #7A8FA6;
    --green:   #1E8C5A;
    --red:     #C0392B;
    --amber:   #E67E22;
  }

  /* Overrides for Streamlit App Background */
  .stApp {
      background-color: var(--bg);
      font-family: 'Cairo', 'Inter', sans-serif;
      color: var(--text);
  }
  
  /* Hide Streamlit Header */
  header {visibility: hidden;}

  /* Customizing Streamlit Tabs to match your design */
  .stTabs [data-baseweb="tab-list"] {
      background-color: var(--white);
      padding: 6px;
      border-radius: 14px;
      border: 1px solid var(--bg2);
      box-shadow: 0 1px 4px rgba(11,37,69,0.06);
      gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
      padding: 10px 20px;
      border-radius: 10px;
      font-family: 'Cairo', sans-serif;
      font-size: 14px;
      font-weight: 600;
      color: var(--muted);
      border: none;
      background: transparent;
  }
  .stTabs [aria-selected="true"] {
      background-color: var(--navy) !important;
      color: var(--white) !important;
      box-shadow: 0 2px 8px rgba(11,37,69,0.25) !important;
  }
  
  /* Customizing Streamlit Inputs */
  .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
      background-color: var(--bg) !important;
      border: 1.5px solid var(--bg2) !important;
      border-radius: 10px !important;
      font-family: 'Inter', sans-serif;
      color: var(--text) !important;
  }
  .stNumberInput input:focus { border-color: var(--navy) !important; }
  
  .stButton button {
      background-color: var(--navy) !important;
      color: var(--white) !important;
      border-radius: 12px !important;
      font-family: 'Cairo', sans-serif !important;
      font-weight: 700 !important;
      padding: 10px 24px !important;
      border: none !important;
      transition: all 0.2s !important;
  }
  .stButton button:hover {
      background-color: var(--navy2) !important;
      box-shadow: 0 4px 16px rgba(11,37,69,0.25) !important;
      transform: translateY(-1px) !important;
      color: var(--gold) !important;
  }

  /* ══════════════════════════════
     Your Custom HTML Elements
  ══════════════════════════════ */
  .logo-showcase {
    background: var(--navy);
    padding: 32px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 32px;
    border-radius: 0 0 24px 24px;
    margin-top: -60px;
  }
  .logo-main { display: flex; align-items: center; gap: 16px; }
  .logo-icon { width: 56px; height: 56px; flex-shrink: 0; }
  .logo-text-group { display: flex; flex-direction: column; gap: 2px; }
  .logo-name { font-family: 'Inter', sans-serif; font-size: 24px; font-weight: 800; color: var(--white); letter-spacing: -0.5px; line-height: 1; }
  .logo-name span { color: var(--gold); }
  .logo-tagline { font-family: 'Cairo', sans-serif; font-size: 12px; font-weight: 400; color: var(--slate); letter-spacing: 2px; text-transform: uppercase; line-height: 1; }

  /* Ticker */
  .ticker { background: var(--gold); padding: 8px 0; overflow: hidden; position: relative; margin-top: 16px;}
  .ticker-track { display: flex; gap: 0; animation: ticker 22s linear infinite; white-space: nowrap; }
  .ticker-item { display: inline-flex; align-items: center; gap: 8px; padding: 0 32px; font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 600; color: var(--navy); }
  .ticker-dot { width: 5px; height: 5px; background: var(--navy); border-radius: 50%; opacity: 0.4; }
  @keyframes ticker { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }

  /* Hero */
  .hero {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy2) 60%, #1A4A8A 100%);
    padding: 48px 40px;
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    margin-top: 16px;
    margin-bottom: 32px;
  }
  .hero-eyebrow { display: inline-flex; align-items: center; gap: 8px; background: rgba(201,168,76,0.15); border: 1px solid rgba(201,168,76,0.3); padding: 6px 14px; border-radius: 100px; margin-bottom: 24px; }
  .hero-eyebrow-dot { width: 6px; height: 6px; background: var(--gold); border-radius: 50%; animation: pulse 2s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.8); } }
  .hero-eyebrow span { font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 600; color: var(--gold); letter-spacing: 1.5px; text-transform: uppercase; }
  .hero-headline { font-family: 'Inter', sans-serif; font-size: 40px; font-weight: 900; color: var(--white); line-height: 1.1; margin-bottom: 16px; }
  .hero-headline .accent { color: var(--gold); }
  .hero-sub { font-family: 'Cairo', sans-serif; font-size: 16px; font-weight: 400; color: rgba(255,255,255,0.7); max-width: 600px; margin-bottom: 32px; }

  /* KPI Cards */
  .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 32px;}
  .kpi-card { background: var(--white); border-radius: 16px; padding: 20px 22px; border: 1px solid var(--bg2); box-shadow: 0 1px 4px rgba(11,37,69,0.04); display: flex; flex-direction: column; gap: 12px; transition: all 0.2s; }
  .kpi-card:hover { box-shadow: 0 6px 20px rgba(11,37,69,0.10); transform: translateY(-2px); }
  .kpi-top { display: flex; align-items: center; justify-content: space-between; }
  .kpi-label { font-family: 'Cairo', sans-serif; font-size: 13px; font-weight: 600; color: var(--muted); }
  .kpi-badge { width: 32px; height: 32px; border-radius: 9px; display: flex; align-items: center; justify-content: center; font-size: 15px; }
  .kpi-badge.navy { background: rgba(11,37,69,0.07); } .kpi-badge.gold { background: rgba(201,168,76,0.12); } .kpi-badge.green { background: rgba(30,140,90,0.1); } .kpi-badge.red { background: rgba(192,57,43,0.1); }
  .kpi-value { font-family: 'Inter', sans-serif; font-size: 28px; font-weight: 800; color: var(--text); line-height: 1; }
  .kpi-delta { display: inline-flex; align-items: center; gap: 4px; font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 100px; width: fit-content;}
  .kpi-delta.up { background: rgba(30,140,90,0.1); color: var(--green); }
  .kpi-delta.down { background: rgba(192,57,43,0.1); color: var(--red); }
  .kpi-delta.warn { background: rgba(230,126,34,0.1); color: var(--amber); }

  /* Rec Cards */
  .rec-card { border-radius: 14px; padding: 18px; margin-bottom: 16px; border: 1.5px solid; display: flex; flex-direction: column; gap: 10px; background: var(--white); }
  .rec-card.success { border-color: rgba(30,140,90,0.4); box-shadow: 0 4px 12px rgba(30,140,90,0.05); }
  .rec-card.warning { border-color: rgba(230,126,34,0.4); box-shadow: 0 4px 12px rgba(230,126,34,0.05); }
  .rec-card.danger { border-color: rgba(192,57,43,0.4); box-shadow: 0 4px 12px rgba(192,57,43,0.05); }
  .rec-title { font-family: 'Cairo', sans-serif; font-size: 15px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
  .rec-title.success { color: var(--green); } .rec-title.warning { color: var(--amber); } .rec-title.danger { color: var(--red); }
  .rec-row { display: flex; justify-content: space-between; align-items: center; padding: 4px 0; border-bottom: 1px dashed var(--bg2); }
  .rec-row:last-child { border-bottom: none; }
  .rec-key { font-family: 'Cairo', sans-serif; font-size: 13px; color: var(--muted); font-weight: 600; }
  .rec-val { font-family: 'Inter', sans-serif; font-size: 16px; font-weight: 800; color: var(--text); }
  
  /* Tables */
  .seg-table { width: 100%; border-collapse: collapse; background: var(--white); border-radius: 16px; overflow: hidden; box-shadow: 0 1px 4px rgba(11,37,69,0.04); margin-top: 16px;}
  .seg-table th { background: var(--navy); color: var(--white); font-family: 'Cairo', sans-serif; font-size: 12px; font-weight: 700; padding: 12px 16px; text-align: right; }
  .seg-table td { padding: 12px 16px; font-family: 'Inter', sans-serif; font-size: 13px; color: var(--text); border-bottom: 1px solid var(--bg2); font-weight: 500; text-align: right; }
  .risk-pill { display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: 100px; font-size: 11px; font-weight: 700; font-family: 'Inter', sans-serif; }
  .risk-pill.ok { background: rgba(30,140,90,0.1); color: var(--green); }
  .risk-pill.hi { background: rgba(192,57,43,0.1); color: var(--red); }

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
# الهيدر المخصص (Logo + Ticker + Hero)
# ─────────────────────────────────────────────
custom_header = """
<div class="logo-showcase">
  <div class="logo-main">
    <svg class="logo-icon" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="28" cy="28" r="28" fill="#133A6A"/>
      <rect x="14" y="26" width="28" height="3" rx="1.5" fill="#C9A84C"/>
      <rect x="26.5" y="18" width="3" height="20" rx="1.5" fill="#C9A84C"/>
      <path d="M14 27 Q11 33 18 33 Q25 33 22 27" fill="none" stroke="#E8C96A" stroke-width="1.8" stroke-linecap="round"/>
      <path d="M34 27 Q31 31 38 31 Q45 31 42 27" fill="none" stroke="#E8C96A" stroke-width="1.8" stroke-linecap="round" opacity="0.6"/>
      <path d="M39 22 L39 17 M37 19 L39 17 L41 19" stroke="#C9A84C" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <div class="logo-text-group">
      <div class="logo-name">Price<span>Opt</span></div>
      <div class="logo-tagline">Smart Pricing Engine</div>
    </div>
  </div>
</div>

<div class="ticker">
  <div class="ticker-track">
    <span class="ticker-item">📈 تضخم تراكمي حضري: 2.47×<div class="ticker-dot"></div></span>
    <span class="ticker-item">💰 القوة الشرائية الحقيقية: 40.5%<div class="ticker-dot"></div></span>
    <span class="ticker-item">🍞 ميزانية الأكل الإجبارية: 28.63%<div class="ticker-dot"></div></span>
    <span class="ticker-item">📊 وسيط الدخل 2026: 138,739 جنيه/سنة<div class="ticker-dot"></div></span>
    <span class="ticker-item">⚙️ محرك التسعير الذكي — PriceOpt v1.0<div class="ticker-dot"></div></span>
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
</div>
"""
st.markdown(custom_header, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# التبويبات الرئيسية
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 رادار الانكماش والقدرة الشرائية",
    "🏭 بيانات المنتج (Input)",
    "🎯 التوصية الذكية (AI Output)"
])

# ═══════════════════════════════════════════════════════
# شاشة ١: رادار انكماش الطبقات
# ═══════════════════════════════════════════════════════
with tab1:
    area_choice = st.radio("تحليل المنطقة المستهدفة:", ["Urban", "Rural"], horizontal=True)
    data = get_curves(area_choice)
    
    # ── KPIs بطريقة الـ HTML Premium ──
    pwr = 40.5 if area_choice == "Urban" else 39.49
    food = 28.63 if area_choice == "Urban" else 36.39
    
    kpis_html = f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-top">
          <div class="kpi-label">وسيط الدخل {area_choice} (2020)</div>
          <div class="kpi-badge navy">🏦</div>
        </div>
        <div class="kpi-value">{data['median_2020']:,.0f} <span style="font-size:14px;color:var(--muted)">ج</span></div>
        <div class="kpi-delta up">سنة الأساس</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-top">
          <div class="kpi-label">وسيط الدخل (2026)</div>
          <div class="kpi-badge gold">💰</div>
        </div>
        <div class="kpi-value">{data['median_2026']:,.0f} <span style="font-size:14px;color:var(--muted)">ج</span></div>
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
    </div>
    """
    st.markdown(kpis_html, unsafe_allow_html=True)

    # ── الرسم البياني (Plotly) بمحاذاة ألوان البراند ──
    x = np.array(data["x"])
    p20 = np.array(data["pdf_2020"])
    p26 = np.array(data["pdf_2026"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=p20, fill="tozeroy", fillcolor="rgba(74,111,165,0.15)",
        line=dict(color="#4A6FA5", width=2.5), name="توزيع 2020"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p26, fill="tozeroy", fillcolor="rgba(192,57,43,0.12)",
        line=dict(color="#C0392B", width=2.5), name="توزيع 2026"
    ))
    fig.add_vline(x=data["median_2020"], line_dash="dot", line_color="#4A6FA5", annotation_text=f"وسيط 2020", annotation_position="top right")
    fig.add_vline(x=data["median_2026"], line_dash="dot", line_color="#C0392B", annotation_text=f"وسيط 2026", annotation_position="top right")

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="الدخل السنوي المتاح (جنيه)",
        yaxis_title="كثافة الاحتمال (Log-Normal Distribution)",
        legend=dict(x=0.75, y=0.95, bgcolor="rgba(255,255,255,0.8)", bordercolor="#E2EAF4", borderwidth=1),
        height=450,
        margin=dict(t=30, b=30, l=10, r=10),
        hovermode="x unified",
        font=dict(family="Cairo", color="#1A2E4A")
    )
    fig.update_xaxes(tickformat=",", range=[0, 400_000], showgrid=True, gridcolor="#E2EAF4")
    fig.update_yaxes(showgrid=True, gridcolor="#E2EAF4")
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **تحليل:** الإزاحة لليمين لا تعني تحسناً — الأرقام ارتفعت بفعل التضخم لكن القوة الشرائية الحقيقية انخفضت بشدة.")


# ═══════════════════════════════════════════════════════
# شاشة ٢: إدخال بيانات المصنع
# ═══════════════════════════════════════════════════════
with tab2:
    with st.container():
        st.markdown("### 🏭 أدخل بيانات المنتج الحالي لتوليد سيناريوهات التسعير")
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
                submitted = st.form_submit_button("⚙️ تشغيل محرك التسعير (Run XGBoost Engine)", use_container_width=True)

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
        st.success("✅ تم حفظ البيانات وتشغيل الموديل. انتقل لتبويب 'التوصية الذكية' لرؤية النتائج.")

    if "product_data" in st.session_state:
        st.markdown("### 📋 خريطة القدرة الشرائية بأسعار اليوم (Affordability Matrix)")
        pd_data = st.session_state["product_data"]
        segments = get_bracket_affordability(pd_data["current_price"], pd_data["area"], pd_data["purchase_freq"])
        
        # Build Custom HTML Table
        table_rows = ""
        for seg in segments:
            risk_class = "ok" if seg['affordable'] else "hi"
            risk_text = "🟢 آمن" if seg['affordable'] else "🔴 خارج المتناول"
            color_class = "var(--green)" if seg['affordable'] else "var(--red)"
            
            table_rows += f"""
            <tr>
              <td>{seg['bracket']}</td>
              <td style="font-weight:700">{seg['population_pct']:.2f}%</td>
              <td>{seg['monthly_disposable']:,.0f} ج</td>
              <td style="font-weight:800; color:{color_class}">{seg['price_burden_pct']:.1f}%</td>
              <td><span class="risk-pill {risk_class}">{risk_text}</span></td>
            </tr>
            """
            
        full_table = f"""
        <table class="seg-table" dir="rtl">
          <thead>
            <tr>
              <th>الفئة الدخلية السنوية</th>
              <th>% من السكان</th>
              <th>الدخل المتاح/شهر</th>
              <th>العبء السعري للسلعة %</th>
              <th>الحالة السلوكية</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
        """
        st.markdown(full_table, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# شاشة ٣: التوصية الذكية (ML & Optimization)
# ═══════════════════════════════════════════════════════
with tab3:
    if "product_data" not in st.session_state:
        st.warning("⚠️ برجاء إدخال بيانات المصنع في الشاشة السابقة أولاً.")
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

    with st.spinner("🧠 جاري تشغيل XGBoost Optimizer..."):
        weight_rec  = optimizer.find_optimal_weight(product)
        churn_pred  = optimizer.predict_market_churn(product, pd_data["new_price"])

    colA, colB = st.columns(2, gap="large")

    with colA:
        st.markdown("### ⚖️ سيناريو أ: الانكماش (Shrinkflation)")
        st.caption("الحفاظ على السعر النفسي مع تقليل الوزن (موصى به)")
        
        status_class = "success" if weight_rec.feasible else "danger"
        icon = "✅" if weight_rec.feasible else "❌"
        
        rec_a_html = f"""
        <div class="rec-card {status_class}">
          <div class="rec-title {status_class}">{icon} الوزن الأمثل للإنتاج</div>
          <div class="rec-row">
            <span class="rec-key">الوزن الجديد المقترح</span>
            <span class="rec-val" style="color:var(--gold); font-size:22px;">{weight_rec.optimal_weight_g} جرام</span>
          </div>
          <div class="rec-row">
            <span class="rec-key">نسبة التخفيض من الحجم الأصلي</span>
            <span class="rec-val" style="color:var(--red)">-{weight_rec.weight_reduction_pct}%</span>
          </div>
          <div class="rec-row">
            <span class="rec-key">هامش الربح المحقق</span>
            <span class="rec-val" style="color:var(--green)">{weight_rec.new_margin_pct}%</span>
          </div>
          <div style="font-size:12px; color:var(--muted); margin-top:8px;">
            {weight_rec.warning}
          </div>
        </div>
        """
        st.markdown(rec_a_html, unsafe_allow_html=True)
        
        # Visual Bar
        fig_w = px.bar(
            x=["الوزن الأصلي", "الوزن الأمثل للمصنع"], 
            y=[product.current_weight_g, weight_rec.optimal_weight_g],
            color=["Original", "Optimized"],
            color_discrete_sequence=["#4A6FA5", "#2ca02c"],
            text=[f"{product.current_weight_g}g", f"{weight_rec.optimal_weight_g}g"]
        )
        fig_w.update_layout(height=250, margin=dict(t=10,b=10), showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        fig_w.update_traces(textposition='auto', textfont_size=16, textfont_color="white")
        st.plotly_chart(fig_w, use_container_width=True)

    with colB:
        st.markdown("### 📈 سيناريو ب: زيادة السعر المباشرة")
        st.caption(f"لو تم رفع السعر لـ {churn_pred.new_price} ج (+{churn_pred.price_increase_pct}%) دون تغيير الوزن")

        risk_classes = {"HIGH": "danger", "MEDIUM": "warning", "LOW": "success"}
        rc = risk_classes.get(churn_pred.risk_level, "success")
        risk_icon = "🔴" if churn_pred.risk_level == "HIGH" else "⚠️" if churn_pred.risk_level == "MEDIUM" else "🟢"
        
        rec_b_html = f"""
        <div class="rec-card {rc}">
          <div class="rec-title {rc}">{risk_icon} توقعات نموذج Machine Learning</div>
          <div class="rec-row">
            <span class="rec-key">نسبة المقاطعة المتوقعة (Market Churn)</span>
            <span class="rec-val" style="color:var(--red); font-size:22px;">{churn_pred.weighted_churn_pct}%</span>
          </div>
          <div class="rec-row">
            <span class="rec-key">السكان المعرضون لخطر التسرب</span>
            <span class="rec-val">{churn_pred.at_risk_population_pct}%</span>
          </div>
          <div class="rec-row">
            <span class="rec-key">تقييم المخاطرة العام</span>
            <span class="rec-val">{churn_pred.risk_level} RISK</span>
          </div>
          <div style="font-size:12px; color:var(--text); font-weight:600; margin-top:8px;">
            {churn_pred.recommendation}
          </div>
        </div>
        """
        st.markdown(rec_b_html, unsafe_allow_html=True)
        
        # تفصيل الـ Segments Impact chart
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
            line=dict(color="#C9A84C", width=3, dash="dot")
        ))
        fig_c.update_layout(
            height=250, margin=dict(t=10,b=10), plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_c, use_container_width=True)

# ─────────────────────────────────────────────
# الفوتر المخصص
# ─────────────────────────────────────────────
footer_html = """
<div style="background:var(--navy); padding: 24px 40px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px; border-radius: 24px 24px 0 0; margin-top: 40px;">
  <div style="display:flex;align-items:center;gap:12px;">
    <svg width="24" height="24" viewBox="0 0 56 56" fill="none">
      <circle cx="28" cy="28" r="28" fill="#133A6A"/>
      <rect x="14" y="26" width="28" height="3" rx="1.5" fill="#C9A84C"/>
      <rect x="26.5" y="18" width="3" height="20" rx="1.5" fill="#C9A84C"/>
      <path d="M14 27 Q11 33 18 33 Q25 33 22 27" fill="none" stroke="#E8C96A" stroke-width="2" stroke-linecap="round"/>
      <path d="M34 27 Q31 31 38 31 Q45 31 42 27" fill="none" stroke="#E8C96A" stroke-width="2" stroke-linecap="round" opacity="0.6"/>
      <path d="M39 22 L39 17 M37 19 L39 17 L41 19" stroke="#C9A84C" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <span style="font-family:'Cairo',sans-serif; font-size:13px; color:rgba(255,255,255,0.4);">
      © 2026 PriceOpt by Mohamed Khaled Eid — Smart Pricing Engine. مبني على بيانات توزيع الدخل المصري.
    </span>
  </div>
  <div style="display:flex; gap:8px;">
    <span style="padding: 4px 12px; border: 1px solid rgba(255,255,255,0.1); border-radius: 100px; font-family:'Inter',sans-serif; font-size:10px; font-weight:600; color:rgba(255,255,255,0.5);">XGBOOST POWERED</span>
    <span style="padding: 4px 12px; border: 1px solid rgba(255,255,255,0.1); border-radius: 100px; font-family:'Inter',sans-serif; font-size:10px; font-weight:600; color:rgba(255,255,255,0.5);">EGYPT 2026</span>
  </div>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
