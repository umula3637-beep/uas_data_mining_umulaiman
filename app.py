import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ====================================================
# CONFIG HALAMAN
# ====================================================
st.set_page_config(
    page_title="Gemini AI Analytics Workspace", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================
# CUSTOM CSS - ULTRA PREMIUM GEMINI GLASSMORPHISM
# ====================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Settings */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e2640 0%, #0d0e15 80%);
        color: #e3e2e6;
    }

    /* Sembunyikan Footer & Watermark Saja (Header tetap aktif untuk tombol sidebar) */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    /* Modifikasi Header Streamlit agar transparan & pas dengan tema gelap */
    [data-testid="stHeader"] {
        background-color: rgba(13, 14, 21, 0.0) !important;
    }
    
    /* Warna tombol toggle sidebar (hamburger/panah) */
    [data-testid="stHeader"] button {
        color: #a8c7fa !important;
    }

    /* Premium Gemini Badge */
    .gemini-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(168, 199, 250, 0.1) 0%, rgba(124, 172, 248, 0.05) 100%);
        border: 1px solid rgba(168, 199, 250, 0.25);
        backdrop-filter: blur(10px);
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.8rem;
        color: #a8c7fa;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    /* Gradient Title */
    .gemini-title {
        background: linear-gradient(90deg, #FFFFFF 0%, #A8C7FA 50%, #7CACF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.3rem;
        letter-spacing: -1px;
        margin-bottom: 8px;
    }

    .gemini-subtitle {
        color: #9aa0a6;
        font-size: 0.95rem;
        margin-bottom: 24px;
    }

    /* Glassmorphism Card Container */
    .gemini-card {
        background: rgba(23, 25, 35, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .gemini-card:hover {
        border-color: rgba(168, 199, 250, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: rgba(13, 14, 21, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Custom Radio Navigation di Sidebar */
    div[data-testid="stRadio"] > label {
        display: none;
    }
    
    div[data-testid="stRadio"] > div {
        gap: 8px;
    }

    /* Inputs Styling */
    div[data-baseweb="input"] {
        border-radius: 12px !important;
        background-color: rgba(30, 32, 48, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        transition: all 0.2s ease;
    }
    
    div[data-baseweb="input"]:focus-within {
        border-color: #a8c7fa !important;
        box-shadow: 0 0 12px rgba(168, 199, 250, 0.3) !important;
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        background-color: rgba(30, 32, 48, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* Glow Action Button */
    .stButton > button {
        background: linear-gradient(135deg, #a8c7fa 0%, #4285f4 100%);
        color: #040c1a;
        border: none;
        border-radius: 30px;
        padding: 12px 28px;
        font-weight: 700;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(66, 133, 244, 0.3);
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #d3e3fd 0%, #669df6 100%);
        box-shadow: 0 6px 28px rgba(66, 133, 244, 0.6);
        transform: translateY(-2px) scale(1.01);
        color: #000000;
    }

    /* Status Result Cards */
    .status-positive {
        background: linear-gradient(135deg, rgba(234, 67, 53, 0.15) 0%, rgba(154, 0, 0, 0.1) 100%);
        border: 1px solid rgba(242, 184, 181, 0.4);
        color: #f2b8b5;
        padding: 20px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(234, 67, 53, 0.15);
    }

    .status-negative {
        background: linear-gradient(135deg, rgba(52, 168, 83, 0.15) 0%, rgba(15, 81, 50, 0.1) 100%);
        border: 1px solid rgba(196, 238, 212, 0.4);
        color: #c4eed4;
        padding: 20px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(52, 168, 83, 0.15);
    }

    /* Table Customization */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Helper Function Styling Plot Dark Glass
def setup_plt_theme():
    plt.style.use("dark_background")
    bg_color = "#12141d"
    plt.rcParams['figure.facecolor'] = bg_color
    plt.rcParams['axes.facecolor'] = bg_color
    plt.rcParams['text.color'] = '#e3e2e6'
    plt.rcParams['axes.labelcolor'] = '#a8c7fa'
    plt.rcParams['xtick.color'] = '#9aa0a6'
    plt.rcParams['ytick.color'] = '#9aa0a6'
    plt.rcParams['grid.color'] = '#1f2233'

# ====================================================
# SIDEBAR NAVIGATION
# ====================================================
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <span style="font-size: 1.8rem;">✨</span>
            <span style="font-weight: 800; font-size: 1.2rem; color: #ffffff; letter-spacing: -0.5px;">GEMINI WORKSPACE</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 0.75rem; color: #a8c7fa; font-weight: 700; letter-spacing: 1px;'>PILIH MODUL ANALISIS</p>", unsafe_allow_html=True)
    
    page = st.radio(
        "Pilih Halaman:",
        ["1. Prediksi Diabetes (Klasifikasi)", "2. Clustering Gerai Kopi (K-Means)"],
        key="navigation_radio"
    )
    
    st.markdown("---")
    st.markdown("""
        <div style='font-size: 0.75rem; color: #6e727e; text-align: center; line-height: 1.5;'>
            <b>Gemini Mining Engine v2.5</b><br>
            Powered by Scikit-Learn & Streamlit
        </div>
    """, unsafe_allow_html=True)

# ====================================================
# HALAMAN 1: KLASIFIKASI DIABETES
# ====================================================
if page == "1. Prediksi Diabetes (Klasifikasi)":
    
    st.markdown('<div class="gemini-badge">🩺 Healthcare Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-title">Prediksi Risiko Diabetes Pasien</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-subtitle">Sistem pendukung keputusan klinis dengan pemodelan ensemble Machine Learning secara real-time.</div>', unsafe_allow_html=True)

    # Load Data & Model
    df_diab = pd.read_csv('diabetes.csv')
    
    target_col = None
    for col in df_diab.columns:
        if col.lower().strip() in ['outcome', 'target', 'class', 'diabetes']:
            target_col = col
            break
    if not target_col:
        target_col = df_diab.columns[-1]

    X = df_diab.drop(target_col, axis=1)
    y = df_diab[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn = joblib.load('model_knn.pkl')
    nb = joblib.load('model_nb.pkl')
    dt = joblib.load('model_dt.pkl')
    
    models = {'KNN': knn, 'Naïve Bayes': nb, 'Decision Tree': dt}
    
    # Visual Layout
    col_left, col_right = st.columns([1.1, 0.9])
    
    with col_left:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Benchmarking Model")
        st.caption("Perbandingan metrik performa algoritma pada data pengujian.")
        
        metrics_list = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            metrics_list.append({
                'Algoritma': name,
                'Akurasi': f"{accuracy_score(y_test, y_pred):.2f}",
                'Precision': f"{precision_score(y_test, y_pred):.2f}",
                'Recall': f"{recall_score(y_test, y_pred):.2f}",
                'F1-Score': f"{f1_score(y_test, y_pred):.2f}"
            })
        st.dataframe(pd.DataFrame(metrics_list), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 🧩 Confusion Matrix")
        selected_eval_model = st.selectbox("Pilih Model Evaluasi:", list(models.keys()), key="cm_select")
        
        setup_plt_theme()
        cm = confusion_matrix(y_test, models[selected_eval_model].predict(X_test))
        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='mako', ax=ax, cbar=False, annot_kws={"size": 11, "weight": "bold"})
        plt.xlabel('Prediksi Model', fontsize=8, color='#a8c7fa')
        plt.ylabel('Kondisi Real', fontsize=8, color='#a8c7fa')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # Input Form Card
    st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Form Parameter Klinis Pasien")
    st.caption("Isi nilai fitur di bawah untuk melakukan simulasi diagnosa prediktif.")
    
    selected_model_name = st.selectbox("Algoritma Eksekusi Prediksi:", list(models.keys()), key="pred_select")
    st.markdown("<br>", unsafe_allow_html=True)
    
    feature_names = X.columns.tolist()
    input_values = []
    
    cols = st.columns(3)
    for i, col_name in enumerate(feature_names):
        default_val = float(X[col_name].median())
        min_val = float(X[col_name].min())
        max_val = float(X[col_name].max())
        
        with cols[i % 3]:
            val = st.number_input(f"{col_name}", value=default_val, min_value=min_val, max_value=max_val)
            input_values.append(val)
            
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✨ Eksekusi Analisis Prediksi"):
        chosen_model = models[selected_model_name]
        prediction = chosen_model.predict([input_values])[0]
        
        if prediction == 1:
            st.markdown("""
                <div class="status-positive">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.5rem;">⚠️</span>
                        <div>
                            <b style="font-size: 1.05rem;">DIAGNOSA: RISIKO DIABETES TERDETEKSI (POSITIF)</b><br>
                            <span style="font-size: 0.85rem; opacity: 0.85;">Pola indikator indikatif terdeteksi tinggi. Disarankan pemeriksaan konfirmasi medis lebih lanjut.</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="status-negative">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.5rem;">✅</span>
                        <div>
                            <b style="font-size: 1.05rem;">DIAGNOSA: PASIEN DALAM KONDISI SEHAT (NEGATIF)</b><br>
                            <span style="font-size: 0.85rem; opacity: 0.85;">Profil indikator fisiologis pasien berada dalam kisaran ambang batas normal.</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

# ====================================================
# HALAMAN 2: CLUSTERING GERAI KOPI
# ====================================================
elif page == "2. Clustering Gerai Kopi (K-Means)":
    
    st.markdown('<div class="gemini-badge">☕ Spatial Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-title">Klaster Geospasial Gerai Kopi</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-subtitle">Pengelompokan titik lokasi gerai menggunakan Unsupervised Learning K-Means untuk identifikasi potensi ekspansi.</div>', unsafe_allow_html=True)
    
    df_kopi = pd.read_csv('lokasi_gerai_kopi_clean.csv')
    kmeans = joblib.load('model_kmeans.pkl')
    
    col_lat = [c for c in df_kopi.columns if any(k in c.lower() for k in ['lat', 'y', 'lintang'])][0]
    col_lon = [c for c in df_kopi.columns if any(k in c.lower() for k in ['lon', 'long', 'lng', 'x', 'bujur'])][0]
    
    X_kopi = df_kopi[[col_lat, col_lon]]
    df_kopi['Cluster'] = kmeans.labels_
    
    col_map, col_form = st.columns([1.2, 0.8])
    
    with col_map:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 📍 Visualisasi Klastering Spasial")
        
        setup_plt_theme()
        fig, ax = plt.subplots(figsize=(8, 5.2))
        sns.scatterplot(
            data=df_kopi, 
            x=col_lon, 
            y=col_lat, 
            hue='Cluster', 
            palette='viridis', 
            s=90, 
            ax=ax,
            edgecolor='#1e2233',
            alpha=0.9
        )
        plt.title("Persebaran Geospasial Gerai Kopi", fontsize=10, color='#a8c7fa', pad=12)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_form:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Evaluasi Titik Lokasi Baru")
        st.caption("Uji potensi koordinat lokasi untuk ekspansi outlet:")
        
        in_lat = st.number_input(f"Latitude ({col_lat})", value=float(X_kopi[col_lat].mean()))
        in_lon = st.number_input(f"Longitude ({col_lon})", value=float(X_kopi[col_lon].mean()))
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📌 Analisis Karakteristik Zona"):
            pred_cluster = kmeans.predict([[in_lat, in_lon]])[0]
            
            st.markdown(f"""
                <div style="background: rgba(168, 199, 250, 0.08); padding: 14px; border-radius: 12px; margin-bottom: 16px; border: 1px solid rgba(168, 199, 250, 0.2); font-weight: 600; color: #a8c7fa;">
                    📌 Hasil Pemetaan: Tergolong dalam Klaster {pred_cluster}
                </div>
            """, unsafe_allow_html=True)
            
            if pred_cluster == 0:
                st.markdown("""
                    <div class="status-positive">
                        <b style="font-size: 1rem;">⚠️ STATUS ZONA: POTENSI SEPI</b><br>
                        <span style="font-size: 0.85rem; opacity: 0.85;">Kepadatan outlet rendah. Kompetisi tergolong rendah namun membutuhkan strategi penetrasi pasar lebih kuat.</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="status-negative">
                        <b style="font-size: 1rem;">✅ STATUS ZONA: KAWASAN RAMAI</b><br>
                        <span style="font-size: 0.85rem; opacity: 0.85;">Tingkat konsentrasi tinggi. Indikasi demand dan foot-traffic yang sudah terbentuk.</span>
                    </div>
                """, unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)
