import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from streamlit_folium import st_folium
import folium

# ====================================================
# 1. PAGE CONFIGURATION
# ====================================================
st.set_page_config(
    page_title="UMUL AIMAN 23146039 - Enterprise Mining Engine", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================
# 2. ENHANCED CUSTOM CSS - PREMIUM GLASSMORPHISM
# ====================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ===== GLOBAL RESET & TYPOGRAPHY ===== */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stApp {
        background: 
            radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 0.4) 0%, #080d1a 100%);
        background-color: #080d1a;
        color: #f1f5f9;
    }

    footer, #MainMenu { visibility: hidden; }

    [data-testid="stHeader"] {
        background-color: transparent !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    /* ===== PREMIUM GLASSMORPHISM CARDS ===== */
    .gemini-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.65) 0%, rgba(15, 23, 42, 0.55) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-radius: 24px;
        padding: 32px;
        margin-bottom: 28px;
        box-shadow: 
            0 4px 24px rgba(0, 0, 0, 0.4),
            0 0 0 1px rgba(255, 255, 255, 0.04) inset,
            0 100px 80px rgba(0, 0, 0, 0.2);
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    
    .gemini-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(168, 199, 250, 0.4), transparent);
        opacity: 0.5;
    }
    
    .gemini-card:hover {
        border-color: rgba(168, 199, 250, 0.25);
        box-shadow: 
            0 8px 40px rgba(0, 0, 0, 0.5),
            0 0 0 1px rgba(255, 255, 255, 0.06) inset,
            0 120px 100px rgba(0, 0, 0, 0.25);
        transform: translateY(-2px);
    }

    /* ===== BADGES & TITLES ===== */
    .gemini-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.08) 100%);
        border: 1px solid rgba(168, 199, 250, 0.35);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 8px 20px;
        border-radius: 999px;
        font-size: 0.78rem;
        color: #a8c7fa;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 16px;
        box-shadow: 
            0 0 20px rgba(59, 130, 246, 0.2),
            inset 0 0 12px rgba(168, 199, 250, 0.08);
        animation: glow 3s ease-in-out infinite;
    }

    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.2), inset 0 0 12px rgba(168, 199, 250, 0.08); }
        50% { box-shadow: 0 0 30px rgba(59, 130, 246, 0.3), inset 0 0 16px rgba(168, 199, 250, 0.12); }
    }

    .gemini-title {
        background: linear-gradient(135deg, #FFFFFF 0%, #E0E7FF 30%, #A8C7FA 60%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 2.6rem;
        letter-spacing: -1px;
        margin-bottom: 12px;
        line-height: 1.15;
        text-shadow: 0 0 40px rgba(168, 199, 250, 0.3);
    }

    .gemini-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 32px;
        font-weight: 400;
        line-height: 1.6;
        max-width: 700px;
    }

    /* ===== PREMIUM KPI CARDS ===== */
    .kpi-card {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }

    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.6), rgba(139, 92, 246, 0.6), transparent);
        animation: shimmer 2s infinite;
    }

    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(168, 199, 250, 0.2);
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.15);
    }

    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #A8C7FA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        margin-bottom: 6px;
    }

    .kpi-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1.2px;
    }

    /* ===== INPUT FIELDS & BUTTONS ===== */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.8) 0%, rgba(15, 23, 42, 0.6) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        backdrop-filter: blur(8px);
        transition: all 0.2s ease;
    }

    div[data-baseweb="input"] > div:hover,
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(168, 199, 250, 0.3) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 14px 28px !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        letter-spacing: 0.4px !important;
        font-size: 0.9rem !important;
        box-shadow: 
            0 4px 16px rgba(59, 130, 246, 0.35),
            0 0 0 1px rgba(255, 255, 255, 0.1) inset !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 
            0 8px 30px rgba(59, 130, 246, 0.5),
            0 0 0 1px rgba(255, 255, 255, 0.15) inset !important;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    /* ===== MODERN TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.7) 0%, rgba(15, 23, 42, 0.5) 100%);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(12px);
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 12px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.9rem;
        border: none !important;
        padding: 0 24px !important;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.25) 0%, rgba(139, 92, 246, 0.15) 100%) !important;
        color: #a8c7fa !important;
        border: 1px solid rgba(168, 199, 250, 0.35) !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.2);
    }

    /* ===== SIDEBAR CUSTOMIZATION ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 15, 29, 0.98) 0%, rgba(8, 13, 26, 0.95) 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(20px);
    }

    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 8px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* ===== STATUS CARDS ===== */
    .status-positive {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 
