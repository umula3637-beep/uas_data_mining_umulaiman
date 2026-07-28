import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. KONFIGURASI HALAMAN & CSS PREMIUM (MINIMALIST & CLEAN)
# ==============================================================================
st.set_page_config(
    page_title="Data Mining Pro", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Clean, Minimalist, Premium
st.markdown("""
<style>
    /* Import Font: Inter (Standar industri untuk UI modern) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Reset & Background */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #FAFAFA !important;
        color: #111827 !important;
    }
    
    /* Hide Default Streamlit Clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDecoration {display: none;}
    
    /* Typography */
    h1, h2, h3 {
        color: #111827 !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
    }
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.25rem;
        letter-spacing: -0.03em;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6B7280;
        margin-bottom: 2rem;
        font-weight: 400;
        line-height: 1.6;
        max-width: 700px;
    }
    
    /* Clean Card Styling */
    .clean-card {
        background-color: #FFFFFF;
        border: 1px solid #F3F4F6;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02), 0 1px 2px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }
    .clean-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border-color: #E5E7EB;
    }
    
    /* Prediction Result Cards */
    .pred-card {
        padding: 24px;
        border-radius: 12px;
        margin-top: 16px;
        border: 1px solid;
        background-color: #FFFFFF;
    }
    .pred-positive {
        border-color: #FEE2E2;
        border-left: 4px solid #F43F5E;
    }
    .pred-negative {
        border-color: #D1FAE5;
        border-left: 4px solid #10B981;
    }
    .pred-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .pred-positive .pred-title { color: #9F1239; }
    .pred-negative .pred-title { color: #047857; }
    .pred-desc { 
        font-size: 0.95rem; 
        color: #4B5563; 
        line-height: 1.6; 
    }

    /* Metric Override */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: #6B7280 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sidebar Minimalist */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #F3F4F6 !important;
    }
    
    /* Sidebar Menu Button Polish (Menjamin Ikon Terlihat) */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 10px 16px !important;
        font-size: 0.9rem !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        margin-bottom: 4px !important;
        border: 1px solid transparent !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Inactive State */
    section[data-testid="stSidebar"] .stButton > button:not([data-testid="baseButton-primary"]) {
        background-color: transparent !important;
        color: #6B7280 !important;
    }
    section[data-testid="stSidebar"] .stButton > button:not([data-testid="baseButton-primary"]):hover {
        background-color: #F3F4F6 !important;
        color: #111827 !important;
    }
    
    /* Active State */
    section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border-color: #E5E7EB !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: #F9FAFB !important;
    }

    /* Main Content Button Premium */
    .stButton > button[kind="primary"] {
        background-color: #111827 !important;
        color: #FFFFFF !important;
        border: 1px solid #111827 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        transition: all 0.2s ease !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 8px !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #000000 !important;
        border-color: #000000 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Tabs Clean */
    button[data-baseweb="tab"] {
        font-weight: 500 !important;
        color: #6B7280 !important;
        font-size: 0.9rem !important;
        padding: 10px 20px !important;
        border: none !important;
        background-color: transparent !important;
        border-bottom: 2px solid transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #111827 !important;
        border-bottom: 2px solid #111827 !important;
        background-color: transparent !important;
    }
    
    /* Slider & Input Clean */
    .stSlider > div > div > div > div {
        background-color: #111827 !important;
    }
    .stTextInput > div > div > input, .stNumberInput > div > div > input {
        border-radius: 8px !important;
        border: 1px solid #E5E7EB !important;
        background-color: #FAFAFA !important;
        font-size: 0.9rem !important;
    }
    .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {
        border-color: #111827 !important;
        box-shadow: 0 0 0 1px #111827 !important;
    }

    /* Progress Bar Minimalist */
    .custom-progress-bg {
        background-color: #F3F4F6;
        border-radius: 99px;
        height: 6px;
        width: 100%;
        margin-top: 12px;
        overflow: hidden;
    }
    .custom-progress-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Section Divider */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. NAVIGASI SIDEBAR (DIPERBAIKI: MENGGUNAKAN SESSION STATE & BUTTON)
# ==============================================================================
with st.sidebar:
    # Header Sidebar dengan Ikon yang Dijamin Muncul
    st.markdown("""
        <div style="padding: 8px 0 24px 0; display: flex; align-items: center; gap: 12px;">
            <div style="background: #111827; color: white; width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem;">📊</div>
            <span style="font-size: 1.25rem; font-weight: 700; color: #111827; letter-spacing: -0.02em;">DataMining Pro</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='font-size: 0.7rem; font-weight: 600; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.05em; margin: 24px 0 12px 0;'>Modul Analisis</div>", unsafe_allow_html=True)
    
    # Inisialisasi session state untuk halaman aktif
    if "active_page" not in st.session_state:
        st.session_state.active_page = "🩺 Prediksi Diabetes"
        
    # Definisi item navigasi
    nav_items = [
        {"id": "🩺 Prediksi Diabetes", "icon": "🩺", "label": "Prediksi Diabetes"},
        {"id": "☕ Clustering Gerai Kopi", "icon": "☕", "label": "Clustering Gerai Kopi"}
    ]
    
    # Render tombol navigasi
    for item in nav_items:
        is_active = st.session_state.active_page == item["id"]
        btn_type = "primary" if is_active else "secondary"
        
        if st.button(f"{item['icon']}  {item['label']}", type=btn_type, use_container_width=True, key=f"nav_{item['id']}"):
            st.session_state.active_page = item["id"]
            st.rerun()
            
    st.divider()
    st.markdown("<p style='font-size: 0.75rem; color: #9CA3AF; line-height: 1.5; margin-top: 24px;'>v2.1 Premium Dashboard<br>Ditenagai oleh Scikit-Learn & Streamlit</p>", unsafe_allow_html=True)

# ==============================================================================
# 3. HALAMAN 1: PREDIKSI DIABETES
# ==============================================================================
if st.session_state.active_page == "🩺 Prediksi Diabetes":
    st.markdown('<h1 class="main-header">Prediksi Risiko Diabetes</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluasi risiko berdasarkan 8 indikator kesehatan utama menggunakan ensemble model machine learning.</p>', unsafe_allow_html=True)
    
    try:
        df_diab = pd.read_csv('diabetes.csv')
        knn = joblib.load('model_knn.pkl')
        nb = joblib.load('model_nb.pkl')
        dt = joblib.load('model_dt.pkl')
        
        target_col = next((col for col in df_diab.columns if col.lower() in ['outcome', 'target', 'class', 'diabetes']), df_diab.columns[-1])
        X = df_diab.drop(target_col, axis=1)
        y = df_diab[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {'KNN': knn, 'Naïve Bayes': nb, 'Decision Tree': dt}
        tab_eval, tab_pred, tab_info = st.tabs(["Evaluasi Model", "Prediksi Pasien Baru", "Info Dataset"])
        
        with tab_eval:
            st.markdown("<div class='section-label'>Performa Algoritma</div>", unsafe_allow_html=True)
            
            metrics_data = []
            cm_dict = {}
            for name, model in models.items():
                y_pred = model.predict(X_test)
                metrics_data.append({
                    'Model': name,
                    'Akurasi': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'F1-Score': f1_score(y_test, y_pred)
                })
                cm_dict[name] = confusion_matrix(y_test, y_pred)
            
            df_metrics = pd.DataFrame(metrics_data)
            
            cols = st.columns(3)
            for i, row in df_metrics.iterrows():
                with cols[i]:
                    st.metric(
                        label=f"Akurasi {row['Model']}", 
                        value=f"{row['Akurasi']*100:.1f}%", 
                        delta=f"F1: {row['F1-Score']:.2f}"
                    )
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_cm1, col_cm2 = st.columns([1, 3])
            with col_cm1:
                selected_cm = st.selectbox("Pilih Model", list(models.keys()), key="cm_select", label_visibility="collapsed")
            with col_cm2:
                fig, ax = plt.subplots(figsize=(6, 4.5))
                sns.heatmap(cm_dict[selected_cm], annot=True, fmt='d', cmap='Greys', 
                            xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'], 
                            ax=ax, cbar=False, linewidths=1, linecolor='#FFFFFF',
                            annot_kws={"size": 12, "weight": "600", "color": "#111827"})
                
                ax.set_title(f"Confusion Matrix: {selected_cm}", fontsize=13, fontweight='600', color='#111827', pad=15)
                ax.set_ylabel('Aktual', fontsize=10, color='#6B7280', fontweight='500')
                ax.set_xlabel('Prediksi', fontsize=10, color='#6B7280', fontweight='500')
                
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(axis='both', colors='#4B5563', labelsize=10)
                st.pyplot(fig)

        with tab_pred:
            st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Input Data Klinis</div>", unsafe_allow_html=True)
            
            selected_model = st.selectbox("Algoritma", list(models.keys()), index=2, label_visibility="collapsed")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                inp_pregnancies = st.slider("Kehamilan", 0, 20, int(X['Pregnancies'].median()))
                inp_age = st.slider("Usia (Tahun)", 10, 100, int(X['Age'].median()))
            with col2:
                inp_bmi = st.slider("BMI", 0.0, 70.0, float(X['BMI'].median()), 0.1)
                inp_glucose = st.slider("Glukosa (mg/dL)", 0, 200, int(X['Glucose'].median()))
            with col3:
                inp_bp = st.slider("Tekanan Darah", 0, 150, int(X['BloodPressure'].median()))
                inp_skin = st.slider("Ketebalan Kulit (mm)", 0, 100, int(X['SkinThickness'].median()))
            
            col4, col5 = st.columns(2)
            with col4:
                inp_insulin = st.slider("Insulin (mu U/ml)", 0, 900, int(X['Insulin'].median()))
            with col5:
                inp_dpf = st.slider("Faktor Genetik (DPF)", 0.0, 3.0, float(X['DiabetesPedigreeFunction'].median()), 0.01)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Analisis Risiko", type="primary", use_container_width=True):
                with st.spinner("Memproses..."):
                    input_array = np.array([[inp_pregnancies, inp_glucose, inp_bp, inp_skin, inp_insulin, inp_bmi, inp_dpf, inp_age]])
                    prediction = models[selected_model].predict(input_array)[0]
                    proba = models[selected_model].predict_proba(input_array)[0][1] if hasattr(models[selected_model], "predict_proba") else 0.0
                    
                    confidence_pct = proba * 100 if prediction == 1 else (1 - proba) * 100
                    bar_color = "#F43F5E" if prediction == 1 else "#10B981"
                    
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="pred-card pred-positive">
                            <div class="pred-title">⚠️ Risiko Tinggi Terdeteksi</div>
                            <div class="pred-desc">
                                Model <b>{selected_model}</b> memprediksi indikasi diabetes. <br>
                                <div style="margin-top: 12px; font-size: 0.85rem; color: #6B7280;">Tingkat Keyakinan Model</div>
                                <div class="custom-progress-bg">
                                    <div class="custom-progress-fill" style="width: {confidence_pct}%; background-color: {bar_color};"></div>
                                </div>
                                <div style="text-align: right; font-size: 0.8rem; color: #9CA3AF; margin-top: 4px;">{confidence_pct:.1f}%</div>
                                <br>
                                <b>Rekomendasi:</b> Segera konsultasikan dengan dokter untuk tes HbA1c lanjutan.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="pred-card pred-negative">
                            <div class="pred-title">✅ Risiko Rendah</div>
                            <div class="pred-desc">
                                Model <b>{selected_model}</b> memprediksi pasien <b>tidak</b> mengidap diabetes. <br>
                                <div style="margin-top: 12px; font-size: 0.85rem; color: #6B7280;">Tingkat Keyakinan Model</div>
                                <div class="custom-progress-bg">
                                    <div class="custom-progress-fill" style="width: {confidence_pct}%; background-color: {bar_color};"></div>
                                </div>
                                <div style="text-align: right; font-size: 0.8rem; color: #9CA3AF; margin-top: 4px;">{confidence_pct:.1f}%</div>
                                <br>
                                <b>Rekomendasi:</b> Pertahankan pola hidup sehat dan cek kesehatan rutin tahunan.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        with tab_info:
            st.markdown("<div class='section-label'>Statistik Deskriptif</div>", unsafe_allow_html=True)
            st.dataframe(df_diab.describe().round(2), use_container_width=True, height=250)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Matriks Korelasi</div>", unsafe_allow_html=True)
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            mask = np.triu(np.ones_like(df_diab.corr(), dtype=bool))
            sns.heatmap(df_diab.corr(), annot=True, cmap='Greys', fmt='.2f', ax=ax_corr, mask=mask, 
                        linewidths=0.5, linecolor='#FFFFFF', cbar_kws={"shrink": .8}, annot_kws={"size": 8, "color": "#4B5563"})
            ax_corr.set_title("Korelasi Fitur", fontsize=12, fontweight='600', pad=15)
            st.pyplot(fig_corr)

    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: `{e.filename}`")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# ==============================================================================
# 4. HALAMAN 2: CLUSTERING GERAI KOPI
# ==============================================================================
elif st.session_state.active_page == "☕ Clustering Gerai Kopi":
    st.markdown('<h1 class="main-header">Zonasi Gerai Kopi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Segmentasi lokasi berbasis koordinat geografis menggunakan K-Means untuk identifikasi zona bisnis potensial.</p>', unsafe_allow_html=True)
    
    try:
        df_kopi = pd.read_csv('lokasi_gerai_kopi_clean.csv')
        kmeans = joblib.load('model_kmeans.pkl')
        
        col_lat = next((c for c in df_kopi.columns if any(k in c.lower() for k in ['lat', 'y', 'lintang'])), df_kopi.columns[0])
        col_lon = next((c for c in df_kopi.columns if any(k in c.lower() for k in ['lon', 'long', 'lng', 'x', 'bujur'])), df_kopi.columns[1])
        
        X_kopi = df_kopi[[col_lat, col_lon]]
        df_kopi['Cluster'] = kmeans.labels_
        centroids = kmeans.cluster_centers_
        
        col_map, col_panel = st.columns([2.5, 1])
        
        with col_map:
            st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Peta Sebaran Klaster</div>", unsafe_allow_html=True)
            
            sns.set_theme(style="white", rc={"axes.facecolor": "#FAFAFA", "figure.facecolor": "#FAFAFA"})
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.scatterplot(
                data=df_kopi, x=col_lon, y=col_lat, hue='Cluster', 
                palette='Greys', s=40, alpha=0.6, edgecolor='#FFFFFF', linewidth=0.5, ax=ax, legend=False
            )
            
            ax.scatter(
                centroids[:, 1], centroids[:, 0], 
                s=120, c='#111827', marker='X', 
                edgecolors='#FFFFFF', linewidths=2, zorder=5, label='Centroid'
            )
            
            ax.set_title("Distribusi Geografis", fontsize=13, fontweight='600', color='#111827', pad=15)
            ax.set_xlabel("Longitude", fontsize=9, color='#6B7280')
            ax.set_ylabel("Latitude", fontsize=9, color='#6B7280')
            
            ax.grid(True, linestyle='--', alpha=0.3, color='#E5E7EB')
            for spine in ax.spines.values():
                spine.set_color('#F3F4F6')
                
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
            sns.reset_orig()

        with col_panel:
            st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Ringkasan Klaster</div>", unsafe_allow_html=True)
            
            n_clusters = len(np.unique(kmeans.labels_))
            st.metric("Total Klaster", n_clusters)
            
            cluster_counts = df_kopi['Cluster'].value_counts().sort_index()
            st.markdown("<div style='font-size: 0.85rem; color: #6B7280; margin-top: 16px; margin-bottom: 8px;'>Distribusi Anggota</div>", unsafe_allow_html=True)
            st.bar_chart(cluster_counts, color="#111827", height=150)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Simulasi Lokasi Baru</div>", unsafe_allow_html=True)
            
            in_lat = st.number_input("Latitude", value=float(X_kopi[col_lat].mean()), format="%.6f", label_visibility="collapsed")
            in_lon = st.number_input("Longitude", value=float(X_kopi[col_lon].mean()), format="%.6f", label_visibility="collapsed")
            
            if st.button("Analisis Zona", type="primary", use_container_width=True):
                pred_cluster = int(kmeans.predict([[in_lat, in_lon]])[0])
                distances = np.linalg.norm(centroids - [in_lat, in_lon], axis=1)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                min_cluster = cluster_counts.idxmin()
                
                st.markdown(f"<div style='text-align: center; padding: 12px; background: #F9FAFB; border-radius: 8px; margin-bottom: 12px;'><span style='font-size: 0.85rem; color: #6B7280;'>Hasil Zonasi:</span><br><b style='font-size: 1.1rem; color: #111827;'>Klaster {pred_cluster}</b></div>", unsafe_allow_html=True)
                
                if pred_cluster == min_cluster or min_dist > (max_dist * 0.7):
                    st.markdown("<div style='padding: 12px; background: #FFFBEB; border: 1px solid #FEF3C7; border-radius: 8px; font-size: 0.9rem; color: #92400E;'>⚠️ <b>Zona Sepi</b><br><span style='font-size: 0.85rem; color: #A16207;'>Kepadatan rendah. Potensi *blue ocean* jika ada target pasar spesifik.</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='padding: 12px; background: #F0FDF4; border: 1px solid #DCFCE7; border-radius: 8px; font-size: 0.9rem; color: #166534;'>✅ <b>Zona Potensial</b><br><span style='font-size: 0.85rem; color: #15803D;'>Dekat pusat klaster padat. Validasi pasar terbukti, *foot traffic* terjaga.</span></div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: `{e.filename}`")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. KONFIGURASI HALAMAN & CSS PREMIUM (MINIMALIST & CLEAN)
# ==============================================================================
st.set_page_config(
    page_title="Data Mining Pro", 
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Clean, Minimalist, Premium
st.markdown("""
<style>
    /* Import Font: Inter (Standar industri untuk UI modern) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Reset & Background */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #FAFAFA !important;
        color: #111827 !important;
    }
    
    /* Hide Default Streamlit Clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDecoration {display: none;}
    
    /* Typography */
    h1, h2, h3 {
        color: #111827 !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
    }
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.25rem;
        letter-spacing: -0.03em;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6B7280;
        margin-bottom: 2rem;
        font-weight: 400;
        line-height: 1.6;
        max-width: 700px;
    }
    
    /* Clean Card Styling */
    .clean-card {
        background-color: #FFFFFF;
        border: 1px solid #F3F4F6;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02), 0 1px 2px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }
    .clean-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border-color: #E5E7EB;
    }
    
    /* Prediction Result Cards */
    .pred-card {
        padding: 24px;
        border-radius: 12px;
        margin-top: 16px;
        border: 1px solid;
        background-color: #FFFFFF;
    }
    .pred-positive {
        border-color: #FEE2E2;
        border-left: 4px solid #F43F5E;
    }
    .pred-negative {
        border-color: #D1FAE5;
        border-left: 4px solid #10B981;
    }
    .pred-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .pred-positive .pred-title { color: #9F1239; }
    .pred-negative .pred-title { color: #047857; }
    .pred-desc { 
        font-size: 0.95rem; 
        color: #4B5563; 
        line-height: 1.6; 
    }

    /* Metric Override */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: #6B7280 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sidebar Minimalist */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #F3F4F6 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #111827;
    }
    
    /* Button Premium */
    .stButton>button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease !important;
        border: 1px solid transparent !important;
    }
    .stButton>button[kind="primary"] {
        background-color: #111827 !important;
        color: #FFFFFF !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #000000 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Tabs Clean */
    button[data-baseweb="tab"] {
        font-weight: 500 !important;
        color: #6B7280 !important;
        font-size: 0.9rem !important;
        padding: 10px 20px !important;
        border: none !important;
        background-color: transparent !important;
        border-bottom: 2px solid transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #111827 !important;
        border-bottom: 2px solid #111827 !important;
        background-color: transparent !important;
    }
    
    /* Slider & Input Clean */
    .stSlider > div > div > div > div {
        background-color: #111827 !important;
    }
    .stTextInput > div > div > input, .stNumberInput > div > div > input {
        border-radius: 8px !important;
        border: 1px solid #E5E7EB !important;
        background-color: #FAFAFA !important;
        font-size: 0.9rem !important;
    }
    .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {
        border-color: #111827 !important;
        box-shadow: 0 0 0 1px #111827 !important;
    }

    /* Progress Bar Minimalist */
    .custom-progress-bg {
        background-color: #F3F4F6;
        border-radius: 99px;
        height: 6px;
        width: 100%;
        margin-top: 12px;
        overflow: hidden;
    }
    .custom-progress-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Section Divider */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. NAVIGASI SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("<div style='padding: 8px 0 24px 0;'><span style='font-size: 1.5rem; font-weight: 700; color: #111827;'>◈ DataMining Pro</span></div>", unsafe_allow_html=True)
    
    page = st.radio(
        "Modul Analisis", 
        ["🩺 Prediksi Diabetes", "☕ Clustering Gerai Kopi"],
        label_visibility="collapsed",
        index=0
    )
    
    st.divider()
    st.markdown("<p style='font-size: 0.8rem; color: #9CA3AF; line-height: 1.5;'>v2.0 Premium Dashboard<br>Ditenagai oleh Scikit-Learn & Streamlit</p>", unsafe_allow_html=True)

# ==============================================================================
# 3. HALAMAN 1: PREDIKSI DIABETES
# ==============================================================================
if page == "🩺 Prediksi Diabetes":
    st.markdown('<h1 class="main-header">Prediksi Risiko Diabetes</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluasi risiko berdasarkan 8 indikator kesehatan utama menggunakan ensemble model machine learning.</p>', unsafe_allow_html=True)
    
    try:
        df_diab = pd.read_csv('diabetes.csv')
        knn = joblib.load('model_knn.pkl')
        nb = joblib.load('model_nb.pkl')
        dt = joblib.load('model_dt.pkl')
        
        target_col = next((col for col in df_diab.columns if col.lower() in ['outcome', 'target', 'class', 'diabetes']), df_diab.columns[-1])
        X = df_diab.drop(target_col, axis=1)
        y = df_diab[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {'KNN': knn, 'Naïve Bayes': nb, 'Decision Tree': dt}
        tab_eval, tab_pred, tab_info = st.tabs(["Evaluasi Model", "Prediksi Pasien Baru", "Info Dataset"])
        
        with tab_eval:
            st.markdown("<div class='section-label'>Performa Algoritma</div>", unsafe_allow_html=True)
            
            metrics_data = []
            cm_dict = {}
            for name, model in models.items():
                y_pred = model.predict(X_test)
                metrics_data.append({
                    'Model': name,
                    'Akurasi': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'F1-Score': f1_score(y_test, y_pred)
                })
                cm_dict[name] = confusion_matrix(y_test, y_pred)
            
            df_metrics = pd.DataFrame(metrics_data)
            
            cols = st.columns(3)
            for i, row in df_metrics.iterrows():
                with cols[i]:
                    st.metric(
                        label=f"Akurasi {row['Model']}", 
                        value=f"{row['Akurasi']*100:.1f}%", 
                        delta=f"F1: {row['F1-Score']:.2f}"
                    )
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_cm1, col_cm2 = st.columns([1, 3])
            with col_cm1:
                selected_cm = st.selectbox("Pilih Model", list(models.keys()), key="cm_select", label_visibility="collapsed")
            with col_cm2:
                fig, ax = plt.subplots(figsize=(6, 4.5))
                sns.heatmap(cm_dict[selected_cm], annot=True, fmt='d', cmap='Greys', 
                            xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'], 
                            ax=ax, cbar=False, linewidths=1, linecolor='#FFFFFF',
                            annot_kws={"size": 12, "weight": "600", "color": "#111827"})
                
                ax.set_title(f"Confusion Matrix: {selected_cm}", fontsize=13, fontweight='600', color='#111827', pad=15)
                ax.set_ylabel('Aktual', fontsize=10, color='#6B7280', fontweight='500')
                ax.set_xlabel('Prediksi', fontsize=10, color='#6B7280', fontweight='500')
                
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(axis='both', colors='#4B5563', labelsize=10)
                st.pyplot(fig)

        with tab_pred:
            st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Input Data Klinis</div>", unsafe_allow_html=True)
            
            selected_model = st.selectbox("Algoritma", list(models.keys()), index=2, label_visibility="collapsed")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                inp_pregnancies = st.slider("Kehamilan", 0, 20, int(X['Pregnancies'].median()))
                inp_age = st.slider("Usia (Tahun)", 10, 100, int(X['Age'].median()))
            with col2:
                inp_bmi = st.slider("BMI", 0.0, 70.0, float(X['BMI'].median()), 0.1)
                inp_glucose = st.slider("Glukosa (mg/dL)", 0, 200, int(X['Glucose'].median()))
            with col3:
                inp_bp = st.slider("Tekanan Darah", 0, 150, int(X['BloodPressure'].median()))
                inp_skin = st.slider("Ketebalan Kulit (mm)", 0, 100, int(X['SkinThickness'].median()))
            
            col4, col5 = st.columns(2)
            with col4:
                inp_insulin = st.slider("Insulin (mu U/ml)", 0, 900, int(X['Insulin'].median()))
            with col5:
                inp_dpf = st.slider("Faktor Genetik (DPF)", 0.0, 3.0, float(X['DiabetesPedigreeFunction'].median()), 0.01)
            
            st.markdown("</div>", unsafe_allow_html=True) # Close clean-card
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Analisis Risiko", type="primary", use_container_width=True):
                with st.spinner("Memproses..."):
                    input_array = np.array([[inp_pregnancies, inp_glucose, inp_bp, inp_skin, inp_insulin, inp_bmi, inp_dpf, inp_age]])
                    prediction = models[selected_model].predict(input_array)[0]
                    proba = models[selected_model].predict_proba(input_array)[0][1] if hasattr(models[selected_model], "predict_proba") else 0.0
                    
                    confidence_pct = proba * 100 if prediction == 1 else (1 - proba) * 100
                    bar_color = "#F43F5E" if prediction == 1 else "#10B981"
                    
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="pred-card pred-positive">
                            <div class="pred-title">⚠️ Risiko Tinggi Terdeteksi</div>
                            <div class="pred-desc">
                                Model <b>{selected_model}</b> memprediksi indikasi diabetes. <br>
                                <div style="margin-top: 12px; font-size: 0.85rem; color: #6B7280;">Tingkat Keyakinan Model</div>
                                <div class="custom-progress-bg">
                                    <div class="custom-progress-fill" style="width: {confidence_pct}%; background-color: {bar_color};"></div>
                                </div>
                                <div style="text-align: right; font-size: 0.8rem; color: #9CA3AF; margin-top: 4px;">{confidence_pct:.1f}%</div>
                                <br>
                                <b>Rekomendasi:</b> Segera konsultasikan dengan dokter untuk tes HbA1c lanjutan.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="pred-card pred-negative">
                            <div class="pred-title">✅ Risiko Rendah</div>
                            <div class="pred-desc">
                                Model <b>{selected_model}</b> memprediksi pasien <b>tidak</b> mengidap diabetes. <br>
                                <div style="margin-top: 12px; font-size: 0.85rem; color: #6B7280;">Tingkat Keyakinan Model</div>
                                <div class="custom-progress-bg">
                                    <div class="custom-progress-fill" style="width: {confidence_pct}%; background-color: {bar_color};"></div>
                                </div>
                                <div style="text-align: right; font-size: 0.8rem; color: #9CA3AF; margin-top: 4px;">{confidence_pct:.1f}%</div>
                                <br>
                                <b>Rekomendasi:</b> Pertahankan pola hidup sehat dan cek kesehatan rutin tahunan.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        with tab_info:
            st.markdown("<div class='section-label'>Statistik Deskriptif</div>", unsafe_allow_html=True)
            st.dataframe(df_diab.describe().round(2), use_container_width=True, height=250)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Matriks Korelasi</div>", unsafe_allow_html=True)
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            mask = np.triu(np.ones_like(df_diab.corr(), dtype=bool))
            sns.heatmap(df_diab.corr(), annot=True, cmap='Greys', fmt='.2f', ax=ax_corr, mask=mask, 
                        linewidths=0.5, linecolor='#FFFFFF', cbar_kws={"shrink": .8}, annot_kws={"size": 8, "color": "#4B5563"})
            ax_corr.set_title("Korelasi Fitur", fontsize=12, fontweight='600', pad=15)
            st.pyplot(fig_corr)

    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: `{e.filename}`")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# ==============================================================================
# 4. HALAMAN 2: CLUSTERING GERAI KOPI
# ==============================================================================
elif page == "☕ Clustering Gerai Kopi":
    st.markdown('<h1 class="main-header">Zonasi Gerai Kopi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Segmentasi lokasi berbasis koordinat geografis menggunakan K-Means untuk identifikasi zona bisnis potensial.</p>', unsafe_allow_html=True)
    
    try:
        df_kopi = pd.read_csv('lokasi_gerai_kopi_clean.csv')
        kmeans = joblib.load('model_kmeans.pkl')
        
        col_lat = next((c for c in df_kopi.columns if any(k in c.lower() for k in ['lat', 'y', 'lintang'])), df_kopi.columns[0])
        col_lon = next((c for c in df_kopi.columns if any(k in c.lower() for k in ['lon', 'long', 'lng', 'x', 'bujur'])), df_kopi.columns[1])
        
        X_kopi = df_kopi[[col_lat, col_lon]]
        df_kopi['Cluster'] = kmeans.labels_
        centroids = kmeans.cluster_centers_
        
        col_map, col_panel = st.columns([2.5, 1])
        
        with col_map:
            st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Peta Sebaran Klaster</div>", unsafe_allow_html=True)
            
            sns.set_theme(style="white", rc={"axes.facecolor": "#FAFAFA", "figure.facecolor": "#FAFAFA"})
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.scatterplot(
                data=df_kopi, x=col_lon, y=col_lat, hue='Cluster', 
                palette='Greys', s=40, alpha=0.6, edgecolor='#FFFFFF', linewidth=0.5, ax=ax, legend=False
            )
            
            ax.scatter(
                centroids[:, 1], centroids[:, 0], 
                s=120, c='#111827', marker='X', 
                edgecolors='#FFFFFF', linewidths=2, zorder=5, label='Centroid'
            )
            
            ax.set_title("Distribusi Geografis", fontsize=13, fontweight='600', color='#111827', pad=15)
            ax.set_xlabel("Longitude", fontsize=9, color='#6B7280')
            ax.set_ylabel("Latitude", fontsize=9, color='#6B7280')
            
            ax.grid(True, linestyle='--', alpha=0.3, color='#E5E7EB')
            for spine in ax.spines.values():
                spine.set_color('#F3F4F6')
                
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
            sns.reset_orig()

        with col_panel:
            st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Ringkasan Klaster</div>", unsafe_allow_html=True)
            
            n_clusters = len(np.unique(kmeans.labels_))
            st.metric("Total Klaster", n_clusters)
            
            cluster_counts = df_kopi['Cluster'].value_counts().sort_index()
            st.markdown("<div style='font-size: 0.85rem; color: #6B7280; margin-top: 16px; margin-bottom: 8px;'>Distribusi Anggota</div>", unsafe_allow_html=True)
            st.bar_chart(cluster_counts, color="#111827", height=150)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Simulasi Lokasi Baru</div>", unsafe_allow_html=True)
            
            in_lat = st.number_input("Latitude", value=float(X_kopi[col_lat].mean()), format="%.6f", label_visibility="collapsed")
            in_lon = st.number_input("Longitude", value=float(X_kopi[col_lon].mean()), format="%.6f", label_visibility="collapsed")
            
            if st.button("Analisis Zona", type="primary", use_container_width=True):
                pred_cluster = int(kmeans.predict([[in_lat, in_lon]])[0])
                distances = np.linalg.norm(centroids - [in_lat, in_lon], axis=1)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                min_cluster = cluster_counts.idxmin()
                
                st.markdown(f"<div style='text-align: center; padding: 12px; background: #F9FAFB; border-radius: 8px; margin-bottom: 12px;'><span style='font-size: 0.85rem; color: #6B7280;'>Hasil Zonasi:</span><br><b style='font-size: 1.1rem; color: #111827;'>Klaster {pred_cluster}</b></div>", unsafe_allow_html=True)
                
                if pred_cluster == min_cluster or min_dist > (max_dist * 0.7):
                    st.markdown("<div style='padding: 12px; background: #FFFBEB; border: 1px solid #FEF3C7; border-radius: 8px; font-size: 0.9rem; color: #92400E;'>⚠️ <b>Zona Sepi</b><br><span style='font-size: 0.85rem; color: #A16207;'>Kepadatan rendah. Potensi *blue ocean* jika ada target pasar spesifik.</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='padding: 12px; background: #F0FDF4; border: 1px solid #DCFCE7; border-radius: 8px; font-size: 0.9rem; color: #166534;'>✅ <b>Zona Potensial</b><br><span style='font-size: 0.85rem; color: #15803D;'>Dekat pusat klaster padat. Validasi pasar terbukti, *foot traffic* terjaga.</span></div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: `{e.filename}`")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
