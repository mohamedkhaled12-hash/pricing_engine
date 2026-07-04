"""
streamlit_app.py
================
نقطة الدخول لـ Streamlit Cloud.
بيستدعي dashboard/app.py مباشرة.
"""

import sys
from pathlib import Path

# عشان الـ imports تشتغل صح
sys.path.insert(0, str(Path(__file__).parent))

# استدعاء الـ dashboard
exec(open(Path(__file__).parent / "dashboard" / "app.py").read())
