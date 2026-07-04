# ⚙️ محرك التسعير واقتراح الأوزان الذكي
### Smart Pricing & Weight Optimization Engine

> نظام ذكاء اصطناعي يساعد مصانع المواد الغذائية على اتخاذ قرارات التسعير بناءً على بيانات توزيع الدخل الحقيقية 2020 vs 2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 المشكلة اللي بيحلها المشروع

لما مصنع يفكر يرفع سعر منتجه أو يخفف وزنه بسبب التضخم، بيواجه سؤالين صعبين:

- **لو ثبتت السعر** وخففت الوزن — إيه الوزن الأمثل اللي يحافظ على هامش ربحي؟
- **لو رفعت السعر** وثبتت الوزن — كام زبون هيقاطعني؟

المشروع ده بيجاوب على السؤالين دول بناءً على بيانات توزيع الدخل الحقيقية ونموذج ML.

---

## 📊 الشاشات

### شاشة ١ — رادار انكماش الطبقات
مقارنة منحنى توزيع الدخل بين 2020 و2026 باستخدام **Log-Normal Distribution**.

- المنحنى الأزرق = 2020 (سنة الأساس)
- المنحنى الأحمر = 2026 (بعد تضخم تراكمي 2.47×)
- الإزاحة لليمين تعكس ارتفاع الأرقام الاسمية مع انخفاض القوة الشرائية الحقيقية

### شاشة ٢ — إدخال بيانات المصنع
المصنع بيدخل:
- السعر الحالي للمنتج
- الوزن الحالي بالجرام
- تكلفة الإنتاج لكل جرام
- السعر الجديد المقترح
- المنطقة (حضر / ريف)

### شاشة ٣ — التوصية الذكية
**توصية A:** الوزن الأمثل

```
optimal_weight = السعر × (1 - هامش الربح المستهدف) / تكلفة الجرام
```

**توصية B:** نسبة المقاطعة المتوقعة لكل فئة دخلية مع تصنيف الخطر (LOW / MEDIUM / HIGH)

---

## 🧠 الـ ML Model

### بيانات التدريب
مولّدة برمجياً من بيانات توزيع الدخل الحقيقية 2026:

```
9 فئات دخلية × 2 منطقة × 17 سعر × 80 سيناريو = 24,480 صف
```

### المنطق الاقتصادي
```
price_burden    = (السعر الجديد × التكرار) / الدخل المتاح الشهري
churn_threshold = 8% + 18% × income_percentile
churn_prob      = sigmoid(5 × excess + 2 × price_increase - 1.5)
```

### الموديل
- **الخوارزمية:** Gradient Boosting Classifier
- **الـ AUC:** 0.74 (5-fold cross validation)
- **أهم feature:** price_burden (76% من الـ importance)

---

## 🏗️ هيكل المشروع

```
pricing_engine/
│
├── streamlit_app.py          ← نقطة الدخول لـ Streamlit Cloud
│
├── data/
│   ├── ml_macro_metrics.csv              ← معدلات التضخم والقوة الشرائية
│   └── ml_ready_income_distribution.csv  ← توزيع الدخل 2020 و2026
│
├── core/
│   ├── data_loader.py    ← تحميل الداتا وحساب الـ Disposable Income
│   ├── curve_fitting.py  ← Log-Normal fitting للمنحنيين
│   ├── ml_model.py       ← توليد بيانات التدريب وتدريب الموديل
│   └── optimizer.py      ← محرك التوصيات (الوزن الأمثل + المقاطعة)
│
├── api/
│   └── main.py           ← FastAPI Backend (للاستخدام المستقل)
│
├── dashboard/
│   └── app.py            ← Streamlit Dashboard (3 شاشات)
│
├── models/               ← الموديل بيتولد تلقائياً هنا
│
├── requirements.txt
├── packages.txt
└── .gitignore
```

---

## 🚀 تشغيل محلي (VS Code)

### 1. استنسخ الـ Repository

```bash
git clone https://github.com/your-username/pricing-engine.git
cd pricing-engine
```

### 2. ثبت المكتبات

```bash
pip install -r requirements.txt
```

### 3. درّب الموديل (مرة واحدة بس)

```bash
python -c "from core.ml_model import train_churn_model; train_churn_model(save=True)"
```

### 4. شغل الـ Dashboard

```bash
streamlit run streamlit_app.py
```

افتح المتصفح على: `http://localhost:8501`

---

## ☁️ رفع على Streamlit Cloud

1. ارفع الـ repository على GitHub
2. روح على [share.streamlit.io](https://share.streamlit.io)
3. اختار الـ repository
4. في خانة **Main file path** اكتب: `streamlit_app.py`
5. اضغط Deploy

> ⚠️ الموديل بيتدرب تلقائياً أول مرة يشتغل التطبيق (دقيقتين تقريباً)

---

## 📐 المعادلات الأساسية

### حساب الدخل المتاح الشهري
```
real_income        = gross_annual × purchasing_power%
disposable_annual  = real_income × (1 - food_budget%)
disposable_monthly = disposable_annual ÷ 12
```

### الوزن الأمثل
```
optimal_weight = price × (1 - target_margin) ÷ cost_per_gram
```

### نسبة الانتقال للمنحنى
```
Log-Normal: PDF(x) = 1/(xσ√2π) × exp(-(ln x - μ)² / 2σ²)
μ و σ يتحسبوا بـ Weighted Average من الـ brackets
```

---

## 📁 مصادر الداتا

| الملف | المحتوى | المصدر |
|-------|---------|--------|
| `ml_macro_metrics.csv` | معدلات التضخم التراكمي، القوة الشرائية، ميزانية الأكل | محسوبة من شيتات التضخم والأجور |
| `ml_ready_income_distribution.csv` | توزيع الدخل 2020 و2026 (حضر/ريف) | بيانات أصلية + Re-bracketing بعامل التضخم |

---

## 🔧 تطوير مستقبلي

- [ ] إضافة بيانات استبيان حقيقية لإعادة تدريب الموديل
- [ ] دعم فئات منتجات مختلفة (غذاء / مستلزمات / إلخ)
- [ ] API مستقل للتكامل مع أنظمة ERP
- [ ] تقرير PDF تلقائي للتوصيات

---

## 👤 المطور

مبني بناءً على بيانات توزيع الدخل المصري 2020-2026

---

## 📄 الرخصة

MIT License — حر تستخدمه وتطوره
