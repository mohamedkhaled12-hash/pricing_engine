"""
dashboard/app.py
================
الـ Dashboard الكامل — 3 شاشات في تبويبات.

تشغيل:
  cd pricing_engine
  streamlit run dashboard/app.py

المكتبات المطلوبة:
  pip install streamlit plotly pandas numpy scipy scikit-learn fastapi uvicorn
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
    page_title="محرك التسعير الذكي",
    page_icon="⚙️",
    layout="wide",
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #1f77b4;
    }
    .risk-high   { border-color: #d62728; background: #fff5f5; }
    .risk-medium { border-color: #ff7f0e; background: #fffaf0; }
    .risk-low    { border-color: #2ca02c; background: #f0fff0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# تحميل الـ Resources (مرة واحدة)
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
# Header
# ─────────────────────────────────────────────
st.title("⚙️ محرك التسعير واقتراح الأوزان الذكي")
st.caption("مبني على بيانات توزيع الدخل 2020 vs 2026 — تضخم تراكمي حضري 2.47×")

tab1, tab2, tab3 = st.tabs([
    "📊 شاشة ١ — رادار انكماش الطبقات",
    "🏭 شاشة ٢ — إدخال بيانات المصنع",
    "🎯 شاشة ٣ — التوصية الذكية",
])


# ═══════════════════════════════════════════════════════
# شاشة ١: رادار انكماش الطبقات
# ═══════════════════════════════════════════════════════
with tab1:
    st.subheader("مقارنة منحنى توزيع الدخل: 2020 vs 2026")
    st.info(
        "المنحنى الأزرق = توزيع الدخل الاسمي 2020 (سنة الأساس)  \n"
        "المنحنى الأحمر = توزيع الدخل الاسمي 2026 (بعد التضخم)  \n"
        "**الإزاحة لليمين ≠ تحسن** — الأرقام أكبر لكن القوة الشرائية انخفضت بفعل التضخم."
    )

    area_choice = st.radio("المنطقة:", ["Urban", "Rural"], horizontal=True, key="area_tab1")

    data = get_curves(area_choice)
    x    = np.array(data["x"])
    p20  = np.array(data["pdf_2020"])
    p26  = np.array(data["pdf_2026"])

    fig = go.Figure()

    # منطقة 2020
    fig.add_trace(go.Scatter(
        x=x, y=p20,
        fill="tozeroy", fillcolor="rgba(31,119,180,0.15)",
        line=dict(color="#1f77b4", width=2.5),
        name="توزيع 2020 (سنة الأساس)",
    ))

    # منطقة 2026
    fig.add_trace(go.Scatter(
        x=x, y=p26,
        fill="tozeroy", fillcolor="rgba(214,39,40,0.12)",
        line=dict(color="#d62728", width=2.5),
        name="توزيع 2026 (أسعار اليوم)",
    ))

    # خط الـ Median
    fig.add_vline(x=data["median_2020"], line_dash="dot", line_color="#1f77b4",
                  annotation_text=f"وسيط 2020: {data['median_2020']:,.0f}", annotation_position="top right")
    fig.add_vline(x=data["median_2026"], line_dash="dot", line_color="#d62728",
                  annotation_text=f"وسيط 2026: {data['median_2026']:,.0f}", annotation_position="top right")

    fig.update_layout(
        xaxis_title="الدخل السنوي (جنيه)",
        yaxis_title="كثافة الاحتمال",
        legend=dict(x=0.7, y=0.95),
        height=420,
        margin=dict(t=30, b=30),
        hovermode="x unified",
    )
    fig.update_xaxes(tickformat=",", range=[0, 400_000])
    st.plotly_chart(fig, use_container_width=True)

    # مقاييس مقارنة
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("وسيط الدخل 2020", f"{data['median_2020']:,.0f} ج/سنة")
    col2.metric("وسيط الدخل 2026", f"{data['median_2026']:,.0f} ج/سنة",
                delta=f"+{data['shift_pct']:.1f}% إزاحة اسمية")
    pwr = 40.5 if area_choice == "Urban" else 39.49
    col3.metric("القوة الشرائية الحقيقية", f"{pwr}%",
                delta="-59.5% مقارنة بالاسمي", delta_color="inverse")
    food = 28.63 if area_choice == "Urban" else 36.39
    col4.metric("ميزانية الأكل الإجبارية", f"{food}%")


# ═══════════════════════════════════════════════════════
# شاشة ٢: إدخال بيانات المصنع
# ═══════════════════════════════════════════════════════
with tab2:
    st.subheader("أدخل بيانات منتجك")

    with st.form("product_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📦 بيانات المنتج")
            current_price    = st.number_input("السعر الحالي (جنيه)", min_value=1.0, max_value=500.0, value=25.0, step=0.5)
            current_weight   = st.number_input("الوزن الحالي (جرام)",  min_value=10.0, max_value=5000.0, value=100.0, step=5.0)
            cost_per_gram    = st.number_input("تكلفة الإنتاج / جرام", min_value=0.01, max_value=10.0, value=0.18, step=0.01,
                                               help="شاملة الخامات + التشغيل + التعبئة")
            new_price_input  = st.number_input("السعر الجديد المقترح (لحساب المقاطعة)", min_value=1.0, max_value=500.0,
                                               value=30.0, step=0.5)

        with col2:
            st.markdown("#### ⚙️ معاملات التحليل")
            area_sel       = st.selectbox("المنطقة المستهدفة", ["Urban", "Rural"])
            target_margin  = st.slider("هامش الربح المستهدف %", min_value=5, max_value=60, value=30, step=5) / 100
            purchase_freq  = st.slider("تكرار الشراء (مرات/شهر)", min_value=1, max_value=20, value=4)

            st.markdown("---")
            st.markdown("**ملخص الإدخال:**")
            current_cost    = cost_per_gram * current_weight
            current_margin  = (current_price - current_cost) / current_price * 100
            st.write(f"• تكلفة الإنتاج الحالية: **{current_cost:.2f} جنيه**")
            st.write(f"• هامش الربح الحالي: **{current_margin:.1f}%**")
            st.write(f"• سعر الجرام الحالي: **{current_price/current_weight:.3f} ج/ج**")

        submitted = st.form_submit_button("🔍 تحليل وإنتاج التوصية", use_container_width=True, type="primary")

    # حفظ المدخلات في session state
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
        st.success("✅ تم حفظ البيانات — انتقل لشاشة التوصية الذكية ←")

    # Affordability map
    if "product_data" in st.session_state:
        st.markdown("---")
        st.markdown("#### خريطة القدرة الشرائية لكل فئة (بأسعار اليوم)")
        pd_data = st.session_state["product_data"]
        segments = get_bracket_affordability(
            pd_data["current_price"], pd_data["area"], pd_data["purchase_freq"]
        )
        seg_df = pd.DataFrame(segments)
        seg_df["العبء السعري %"] = seg_df["price_burden_pct"]
        seg_df["الدخل المتاح/شهر"] = seg_df["monthly_disposable"].apply(lambda x: f"{x:,.0f} ج")
        seg_df["الحالة"] = seg_df["affordable"].apply(lambda x: "✅ في المتناول" if x else "🔴 خارج المتناول")

        st.dataframe(
            seg_df[["bracket","population_pct","الدخل المتاح/شهر","العبء السعري %","الحالة"]].rename(
                columns={"bracket": "الفئة الدخلية", "population_pct": "% من السكان"}
            ),
            use_container_width=True, hide_index=True,
        )


# ═══════════════════════════════════════════════════════
# شاشة ٣: التوصية الذكية
# ═══════════════════════════════════════════════════════
with tab3:
    st.subheader("التوصية الذكية")

    if "product_data" not in st.session_state:
        st.warning("👈 أدخل بيانات المنتج في شاشة ٢ أولاً.")
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

    with st.spinner("جاري التحليل..."):
        weight_rec  = optimizer.find_optimal_weight(product)
        churn_pred  = optimizer.predict_market_churn(product, pd_data["new_price"])

    # ── توصية A ──────────────────────────────────────────────────────
    st.markdown("## 🅐 توصية الوزن الأمثل")
    st.caption("للحفاظ على السعر النفسي للزبون مع ضمان هامش الربح المستهدف")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("الوزن الحالي",  f"{product.current_weight_g:.0f} جرام")
    col2.metric("الوزن الأمثل",  f"{weight_rec.optimal_weight_g} جرام",
                delta=f"-{weight_rec.weight_reduction_pct}%", delta_color="inverse")
    col3.metric("هامش الربح الجديد", f"{weight_rec.new_margin_pct}%",
                delta=f"+{weight_rec.new_margin_pct - (pd_data['current_price'] - pd_data['cost_per_gram']*pd_data['current_weight_g'])/pd_data['current_price']*100:.1f}%")
    col4.metric("سعر الجرام الجديد", f"{weight_rec.price_per_gram_new} ج/ج",
                delta=f"+{(weight_rec.price_per_gram_new - weight_rec.price_per_gram_old):.3f}")

    risk_class = "risk-low" if weight_rec.feasible else "risk-high"
    st.markdown(f"""
    <div class="metric-card {risk_class}">
        <strong>{weight_rec.warning}</strong>
    </div>
    """, unsafe_allow_html=True)

    # مقارنة بصرية
    fig_weight = go.Figure(go.Bar(
        x=["الوزن الحالي", "الوزن الأمثل"],
        y=[product.current_weight_g, weight_rec.optimal_weight_g],
        marker_color=["#1f77b4", "#2ca02c"],
        text=[f"{product.current_weight_g:.0f}g", f"{weight_rec.optimal_weight_g}g"],
        textposition="outside",
    ))
    fig_weight.update_layout(height=280, margin=dict(t=20, b=20), yaxis_title="الوزن (جرام)")
    st.plotly_chart(fig_weight, use_container_width=True)

    st.divider()

    # ── توصية B ──────────────────────────────────────────────────────
    st.markdown("## 🅑 تنبؤ نسبة المقاطعة")
    st.caption(f"لو رفعت السعر من {product.current_price:.0f} إلى {churn_pred.new_price:.0f} جنيه (+{churn_pred.price_increase_pct}%)")

    risk_colors = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}
    rc = risk_colors[churn_pred.risk_level]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("نسبة المقاطعة المتوقعة", f"{churn_pred.weighted_churn_pct}%")
    col2.metric("السكان المعرضون للخطر",  f"{churn_pred.at_risk_population_pct}%")
    col3.metric("احتمال المقاطعة (ML)",   f"{churn_pred.ml_churn_prob}%")
    col4.metric("مستوى الخطر",            churn_pred.risk_level)

    st.markdown(f"""
    <div class="metric-card {rc}">
        <strong>{churn_pred.recommendation}</strong>
    </div>
    """, unsafe_allow_html=True)

    # تفصيل الـ segments
    st.markdown("#### تفاصيل التأثير على كل فئة دخلية")
    seg_df = pd.DataFrame(churn_pred.segments_detail)

    fig_seg = go.Figure()

    colors = ["#d62728" if r else "#2ca02c" for r in seg_df["at_risk"]]
    fig_seg.add_trace(go.Bar(
        x=seg_df["bracket"],
        y=seg_df["price_burden_pct"],
        marker_color=colors,
        name="العبء السعري %",
        text=[f"{v:.1f}%" for v in seg_df["price_burden_pct"]],
        textposition="outside",
    ))
    fig_seg.add_trace(go.Scatter(
        x=seg_df["bracket"],
        y=seg_df["churn_threshold_pct"],
        mode="lines+markers",
        name="عتبة المقاطعة",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    ))
    fig_seg.add_trace(go.Bar(
        x=seg_df["bracket"],
        y=[v*100 for v in seg_df["ml_churn_prob"]],
        name="احتمال المقاطعة ML %",
        marker_color="rgba(148,103,189,0.6)",
        opacity=0.7,
    ))

    fig_seg.update_layout(
        barmode="overlay",
        height=380,
        xaxis_title="الفئة الدخلية",
        yaxis_title="%",
        legend=dict(x=0.7, y=0.95),
        margin=dict(t=20, b=80),
    )
    fig_seg.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_seg, use_container_width=True)

    # جدول مفصل
    seg_display = seg_df[[
        "bracket", "population_pct", "monthly_disposable",
        "price_burden_pct", "churn_threshold_pct", "at_risk", "ml_churn_prob"
    ]].copy()
    seg_display.columns = [
        "الفئة الدخلية", "% من السكان", "دخل متاح/شهر",
        "عبء سعري %", "عتبة المقاطعة %", "في خطر؟", "ML احتمال"
    ]
    seg_display["في خطر؟"] = seg_display["في خطر؟"].apply(lambda x: "🔴 نعم" if x else "🟢 لا")
    seg_display["ML احتمال"] = seg_display["ML احتمال"].apply(lambda x: f"{x:.0%}")
    st.dataframe(seg_display, use_container_width=True, hide_index=True)
