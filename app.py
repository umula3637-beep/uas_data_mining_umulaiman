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
# 2. ENHANCED CUSTOM CSS
# ====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stApp {
    background: radial-gradient(circle at 50% -20%, #1e293b 0%, #0f172a 60%, #080d1a 100%);
    color: #f1f5f9;
}

footer, #MainMenu { visibility: hidden; }

[data-testid="stHeader"] {
    background-color: transparent !important;
}

.gemini-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 24px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
    transition: all 0.3s ease;
}

.gemini-card:hover {
    border-color: rgba(168, 199, 250, 0.25);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45);
    transform: translateY(-2px);
}

.gemini-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.08) 100%);
    border: 1px solid rgba(168, 199, 250, 0.35);
    backdrop-filter: blur(12px);
    padding: 8px 18px;
    border-radius: 999px;
    font-size: 0.75rem;
    color: #a8c7fa;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 16px;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);
}

.gemini-title {
    background: linear-gradient(135deg, #FFFFFF 0%, #E0E7FF 40%, #A8C7FA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    font-size: 2.5rem;
    letter-spacing: -0.8px;
    margin-bottom: 10px;
    line-height: 1.2;
}

.gemini-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 30px;
    font-weight: 400;
}

.kpi-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 22px;
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
    background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.6), transparent);
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
    letter-spacing: -0.5px;
    margin-bottom: 6px;
}

.kpi-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 1px;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: rgba(15, 23, 42, 0.7) !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
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
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.35) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(59, 130, 246, 0.5) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.6);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.stTabs [data-baseweb="tab"] {
    height: 46px;
    border-radius: 12px;
    color: #64748b;
    font-weight: 600;
    font-size: 0.9rem;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.25) 0%, rgba(139, 92, 246, 0.15) 100%) !important;
    color: #a8c7fa !important;
    border: 1px solid rgba(168, 199, 250, 0.35) !important;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.2);
}

[data-testid="stSidebar"] {
    background: rgba(10, 15, 29, 0.96) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

.status-positive {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(153, 27, 27, 0.12) 100%);
    border: 1px solid rgba(248, 113, 113, 0.35);
    color: #fca5a5;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(239, 68, 68, 0.15);
}

.status-negative {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.18) 0%, rgba(20, 83, 45, 0.12) 100%);
    border: 1px solid rgba(74, 222, 128, 0.35);
    color: #86efac;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(34, 197, 94, 0.15);
}
</style>
""", unsafe_allow_html=True)

# ====================================================
# 3. RESOURCE CACHING
# ====================================================
@st.cache_data
def load_dataset(filepath):
    return pd.read_csv(filepath)

@st.cache_resource
def load_ml_model(filepath):
    return joblib.load(filepath)

# ====================================================
# 4. SIDEBAR NAVIGATION
# ====================================================
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 14px; padding: 12px 4px; margin-bottom: 24px;">
            <div style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); width: 48px; height: 48px; border-radius: 14px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4);">
                <span style="font-size: 1.5rem;">✨</span>
            </div>
            <div>
                <div style="font-weight: 800; font-size: 1.1rem; color: #f8fafc; letter-spacing: -0.3px;">UMUL AIMAN</div>
                <div style="font-size: 0.78rem; color: #a8c7fa; font-weight: 600;">NIM: 23146039</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 0.72rem; color: #64748b; font-weight: 800; letter-spacing: 1.2px; margin-bottom: 14px; text-transform: uppercase;'>Modul Sistem</p>", unsafe_allow_html=True)
    
    page = st.radio(
        "Pilih Halaman:",
        ["1. Prediksi Diabetes (Klasifikasi)", "2. Clustering Gerai Kopi (Geospasial)"],
        label_visibility="collapsed"
    )
    
    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.06);'><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='font-size: 0.72rem; color: #64748b; text-align: center; line-height: 1.6;'>
            <b style='color: #94a3b8;'>Gemini Mining Engine v4.0</b><br>
            Scikit-Learn • Folium • Streamlit
        </div>
    """, unsafe_allow_html=True)

# ====================================================
# MODUL 1: KLASIFIKASI DIABETES
# ====================================================
if page == "1. Prediksi Diabetes (Klasifikasi)":
    
    st.markdown('<div class="gemini-badge">🩺 Healthcare Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-title">Prediksi Risiko Diabetes</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-subtitle">Sistem pendukung keputusan klinis berbasis AI dengan simulasi real-time dan analisis komprehensif.</div>', unsafe_allow_html=True)

    try:
        df_diab = load_dataset('diabetes.csv')
        knn = load_ml_model('model_knn.pkl')
        nb = load_ml_model('model_nb.pkl')
        dt = load_ml_model('model_dt.pkl')
    except Exception as e:
        st.error(f"⚠️ Gagal memuat dataset atau model ML. Pastikan file tersedia. Detail: {e}")
        st.stop()

    target_col = next((col for col in df_diab.columns if col.lower().strip() in ['outcome', 'target', 'class', 'diabetes']), df_diab.columns[-1])
    X = df_diab.drop(target_col, axis=1)
    y = df_diab[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {'KNN': knn, 'Naïve Bayes': nb, 'Decision Tree': dt}

    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df_diab):,}</div><div class="kpi-label">Total Sampel</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(X.columns)}</div><div class="kpi-label">Fitur Klinis</div></div>', unsafe_allow_html=True)
    with kpi3:
        best_acc = max([accuracy_score(y_test, m.predict(X_test)) for m in models.values()])
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{best_acc*100:.1f}%</div><div class="kpi-label">Akurasi Terbaik</div></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(models)}</div><div class="kpi-label">Model Aktif</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_sim, tab_batch, tab_eval = st.tabs(["📝 Simulasi", "📁 Batch Processing", "📊 Analytics"])

    # TAB 1: INPUT TUNGGAL
    with tab_sim:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Parameter Fisiologis Pasien")
        st.caption("Masukkan data klinis untuk prediksi risiko diabetes secara real-time.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        selected_model_name = st.selectbox("P
