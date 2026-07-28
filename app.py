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
# 1. CONFIG HALAMAN
# ====================================================
st.set_page_config(
    page_title="UMUL AIMAN 23146039 - Enterprise Mining Engine", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================
# 2. CUSTOM CSS - ULTRA PREMIUM GEMINI GLASSMORPHISM 2.0
# ====================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Reset & Font */
    * {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Animated Deep Space Background */
    .stApp {
        background: 
            radial-gradient(circle at 15% 50%, rgba(76, 29, 149, 0.15) 0%, transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(59, 130, 246, 0.1) 0%, transparent 25%),
            linear-gradient(180deg, #0a0b10 0%, #11131a 100%);
        background-attachment: fixed;
        color: #e3e2e6;
    }

    /* Hide Default Streamlit Elements */
    footer, #MainMenu, header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Custom Premium Scrollbar */
    ::-webkit-scrollbar {width: 6px; height: 6px;}
    ::-webkit-scrollbar-track {background: rgba(255, 255, 255, 0.02); border-radius: 10px;}
    ::-webkit-scrollbar-thumb {background: rgba(168, 199, 250, 0.2); border-radius: 10px;}
    ::-webkit-scrollbar-thumb:hover {background: rgba(168, 199, 250, 0.4);}

    /* Typography & Badges */
    .gemini-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(168, 199, 250, 0.12) 0%, rgba(124, 172, 248, 0.04) 100%);
        border: 1px solid rgba(168, 199, 250, 0.25);
        backdrop-filter: blur(12px);
        padding: 8px 18px;
        border-radius: 50px;
        font-size: 0.72rem;
        color: #a8c7fa;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(168, 199, 250, 0.08);
    }

    .gemini-title {
        background: linear-gradient(90deg, #FFFFFF 0%, #A8C7FA 40%, #7CACF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.6rem;
        letter-spacing: -1.5px;
        margin-bottom: 8px;
        line-height: 1.1;
    }

    .gemini-subtitle {
        color: #8b92a5;
        font-size: 1.05rem;
        margin-bottom: 36px;
        line-height: 1.6;
        max-width: 650px;
        font-weight: 400;
    }

    /* Glassmorphism Cards */
    .gemini-card {
        background: rgba(20, 22, 31, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-radius: 24px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .gemini-card:hover {
        border-color: rgba(168, 199, 250, 0.15);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        transform: translateY(-2px);
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 24px 16px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, #a8c7fa, transparent);
        opacity: 0.6;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(168, 199, 250, 0.25);
        box-shadow: 0 8px 24px rgba(168, 199, 250, 0.08);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF, #A8C7FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        line-height: 1;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #8b92a5;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.8px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 11, 16, 0.98) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(20px);
    }
    [data-testid="stSidebar"] .stRadio > div { gap: 10px; }
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
        color: #8b92a5 !important;
        font-weight: 500;
        font-size: 0.9rem;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(168, 199, 250, 0.08);
        border-color: rgba(168, 199, 250, 0.2);
        color: #e3e2e6 !important;
    }
    [data-testid="stSidebar"] .stRadio input:checked + div {
        background: linear-gradient(135deg, rgba(168, 199, 250, 0.15) 0%, rgba(124, 172, 248, 0.05) 100%) !important;
        border-color: rgba(168, 199, 250, 0.4) !important;
        color: #ffffff !important;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(168, 199, 250, 0.1);
    }

    /* Form Elements Override */
    .stSelectbox > div > div, .stNumberInput > div > div {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        color: #e3e2e6 !important;
        transition: all 0.2s ease;
    }
    .stSelectbox > div > div:hover, .stNumberInput > div > div:hover {
        border-color: rgba(168, 199, 250, 0.3) !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Premium Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #a8c7fa 0%, #7cacfa 100%) !important;
        color: #0a0b10 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 24px !important;
        width: 100% !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(168, 199, 250, 0.25) !important;
        font-size: 0.95rem !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(168, 199, 250, 0.35) !important;
        background: linear-gradient(135deg, #b8d4ff 0%, #8cbcff 100%) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(168, 199, 250, 0.25);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(168, 199, 250, 0.5);
        background: rgba(168, 199, 250, 0.04);
    }

    /* Semantic Status Boxes */
    .status-alert {
        background: linear-gradient(135deg, rgba(234, 67, 53, 0.12) 0%, rgba(154, 0, 0, 0.04) 100%);
        border: 1px solid rgba(242, 184, 181, 0.25);
        border-left: 4px solid #f2b8b5;
        color: #f2b8b5;
        padding: 20px 24px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    .status-success {
        background: linear-gradient(135deg, rgba(52, 168, 83, 0.12) 0%, rgba(15, 81, 50, 0.04) 100%);
        border: 1px solid rgba(196, 238, 212, 0.25);
        border-left: 4px solid #c4eed4;
        color: #c4eed4;
        padding: 20px 24px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 4px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: #8b92a5;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.2s ease;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(168, 199, 250, 0.12) !important;
        color: #a8c7fa !important;
        border: 1px solid rgba(168, 199, 250, 0.2) !important;
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.06);
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
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 32px; padding-bottom: 24px; border-bottom: 1px solid rgba(255,255,255,0.06);">
            <div style="font-size: 2rem; filter: drop-shadow(0 0 8px rgba(168, 199, 250, 0.4));">✨</div>
            <div>
                <div style="font-weight: 800; font-size: 1.15rem; color: #ffffff; letter-spacing: -0.5px;">UMUL AIMAN</div>
                <div style="font-size: 0.75rem; color: #a8c7fa; font-weight: 600; letter-spacing: 0.5px;">NIM: 23146039</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 0.7rem; color: #8b92a5; font-weight: 700; letter-spacing: 1.5px; margin-bottom: 12px;'>PILIH MODUL ANALISIS</p>", unsafe_allow_html=True)
    
    page = st.radio(
        "Pilih Halaman:",
        ["1. Prediksi Diabetes (Klasifikasi)", "2. Clustering Gerai Kopi (Geospasial)"],
        label_visibility="collapsed"
    )
    
    st.markdown("<div style='margin-top: auto; padding-top: 40px;'>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <div style='font-size: 0.7rem; color: #555a66; text-align: center; line-height: 1.6;'>
            <b style="color: #8b92a5;">Gemini Mining Engine v4.0</b><br>
            Powered by Scikit-Learn, Folium & Streamlit
        </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ====================================================
# HALAMAN 1: KLASIFIKASI DIABETES
# ====================================================
if page == "1. Prediksi Diabetes (Klasifikasi)":
    
    st.markdown('<div class="gemini-badge">🩺 Healthcare Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-title">Prediksi Risiko Diabetes Pasien</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-subtitle">Sistem pendukung keputusan klinis dengan simulasi tunggal, analisis batch upload, dan explainable AI berbasis ensemble learning.</div>', unsafe_allow_html=True)

    try:
        df_diab = load_dataset('diabetes.csv')
        knn = load_ml_model('model_knn.pkl')
        nb = load_ml_model('model_nb.pkl')
        dt = load_ml_model('model_dt.pkl')
    except Exception as e:
        st.error(f"⚠️ Gagal memuat dataset atau model. Pastikan file tersimpan di lokasi proyek. Detail: {e}")
        st.stop()

    target_col = next((col for col in df_diab.columns if col.lower().strip() in ['outcome', 'target', 'class', 'diabetes']), df_diab.columns[-1])
    X = df_diab.drop(target_col, axis=1)
    y = df_diab[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {'KNN': knn, 'Naïve Bayes': nb, 'Decision Tree': dt}
    feature_names = X.columns.tolist()

    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df_diab):,}</div><div class="kpi-label">Total Sampel Data</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(X.columns)}</div><div class="kpi-label">Fitur Indikator</div></div>', unsafe_allow_html=True)
    with kpi3:
        best_acc = max([accuracy_score(y_test, m.predict(X_test)) for m in models.values()])
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{best_acc*100:.1f}%</div><div class="kpi-label">Akurasi Tertinggi</div></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(models)}</div><div class="kpi-label">Model Terpasang</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_sim, tab_batch, tab_eval = st.tabs(["📝 Input Tunggal", "📁 Batch Prediction", "📊 Analytics & Evaluasi"])

    # TAB 1: INPUT TUNGGAL
    with tab_sim:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Input Parameter Pasien")
        
        selected_model_name = st.selectbox("Algoritma Eksekusi Prediksi:", list(models.keys()), key="pred_select")
        
        input_values = []
        cols = st.columns(3)
        
        for i, col_name in enumerate(feature_names):
            default_val = float(X[col_name].median())
            min_val = float(X[col_name].min())
            max_val = float(X[col_name].max())
            
            with cols[i % 3]:
                val = st.number_input(f"{col_name.replace('_', ' ').title()}", value=default_val, min_value=min_val, max_value=max_val, step=0.1)
                input_values.append(val)
                
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✨ Eksekusi Analisis Prediksi"):
            chosen_model = models[selected_model_name]
            prediction = int(chosen_model.predict([input_values])[0])
            
            if prediction == 1:
                st.markdown("""
                    <div class="status-alert">
                        <b style="font-size: 1.1rem;">⚠️ DIAGNOSA: RISIKO DIABETES TERDETEKSI (POSITIF)</b><br>
                        <span style="font-size: 0.9rem; opacity: 0.9; margin-top: 4px; display: block;">Pola indikator terdeteksi tinggi. Disarankan pemeriksaan medis lanjutan segera.</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="status-success">
                        <b style="font-size: 1.1rem;">✅ DIAGNOSA: PASIEN DALAM KONDISI SEHAT (NEGATIF)</b><br>
                        <span style="font-size: 0.9rem; opacity: 0.9; margin-top: 4px; display: block;">Profil fisiologis pasien berada dalam batas normal yang aman.</span>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2: BATCH PREDICTION
    with tab_batch:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 📁 Upload File CSV untuk Prediksi Massal")
        st.caption("Unggah dataset tanpa kolom target untuk memproses banyak pasien sekaligus.")
        
        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
        batch_model_name = st.selectbox("Model untuk Batch Processing:", list(models.keys()), key="batch_select")
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview Data Unggahan:")
            st.dataframe(batch_df.head(3), use_container_width=True)
            
            if st.button("🚀 Eksekusi Prediksi Massal"):
                try:
                    chosen_batch_model = models[batch_model_name]
                    preds = chosen_batch_model.predict(batch_df[feature_names])
                    
                    batch_df['Hasil_Prediksi'] = ["Positif Diabetes" if p == 1 else "Negatif (Sehat)" for p in preds]
                    
                    st.success("✅ Sukses memproses seluruh data!")
                    st.dataframe(batch_df, use_container_width=True)
                    
                    csv_data = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Hasil Prediksi (CSV)",
                        data=csv_data,
                        file_name="hasil_prediksi_diabetes.csv",
                        mime="text/csv"
                    )
                except Exception as err:
                    st.error(f"❌ Kolom dalam file tidak sesuai dengan fitur model. Error: {err}")
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 3: EVALUASI & EXPLAINABILITY
    with tab_eval:
        col_left, col_right = st.columns([1.1, 0.9])
        
        with col_left:
            st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Benchmarking Performa Model")
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
            
            if hasattr(dt, 'feature_importances_'):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 🔍 Feature Importance (Decision Tree)")
                imp_df = pd.DataFrame({'Fitur': feature_names, 'Tingkat Pengaruh': dt.feature_importances_}).sort_values('Tingkat Pengaruh', ascending=True)
                fig_imp = px.bar(imp_df, x='Tingkat Pengaruh', y='Fitur', orientation='h', color='Tingkat Pengaruh', color_continuous_scale='Viridis')
                fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#e3e2e6"), height=280, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_imp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
            st.markdown("### 🧩 Confusion Matrix")
            selected_eval_model = st.selectbox("Pilih Model Evaluasi:", list(models.keys()), key="cm_select")
            
            cm = confusion_matrix(y_test, models[selected_eval_model].predict(X_test))
            fig = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Prediksi Model", y="Kondisi Real", color="Jumlah"),
                x=['Negatif (0)', 'Positif (1)'],
                y=['Negatif (0)', 'Positif (1)'],
                color_continuous_scale='Blues'
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#e3e2e6"), height=320, margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ====================================================
# HALAMAN 2: CLUSTERING GERAI KOPI
# ====================================================
elif page == "2. Clustering Gerai Kopi (Geospasial)":
    
    st.markdown('<div class="gemini-badge">☕ Spatial Intelligence Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-title">Klaster Geospasial Gerai Kopi</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-subtitle">Pengelompokan geospasial interaktif berbasis Peta Bumi Nyata (OpenStreetMap Folium) untuk analisis penetrasi pasar.</div>', unsafe_allow_html=True)
    
    try:
        df_kopi = load_dataset('lokasi_gerai_kopi_clean.csv')
        kmeans = load_ml_model('model_kmeans.pkl')
    except Exception as e:
        st.error(f"⚠️ Gagal memuat dataset/model clustering. Detail: {e}")
        st.stop()

    col_lat = [c for c in df_kopi.columns if any(k in c.lower() for k in ['lat', 'y', 'lintang'])][0]
    col_lon = [c for c in df_kopi.columns if any(k in c.lower() for k in ['lon', 'long', 'lng', 'x', 'bujur'])][0]
    
    X_kopi = df_kopi[[col_lat, col_lon]]
    df_kopi['Cluster'] = kmeans.labels_
    
    col_map, col_form = st.columns([1.3, 0.7])
    
    with col_map:
        st.markdown('<div class="gemini-card" style="padding: 12px;">', unsafe_allow_html=True)
        st.markdown("### 🗺️ Peta Bumi Geospasial Interaktif", unsafe_allow_html=True)
        
        center_lat = float(df_kopi[col_lat].mean())
        center_lon = float(df_kopi[col_lon].mean())
        
        # Premium Dark Map Theme
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB dark_matter", attr="© CartoDB")
        
        colors = ['#FF4B4B', '#1E88E5', '#00E676', '#FFD600', '#AB47BC']
        
        for idx, row in df_kopi.iterrows():
            cluster_id = int(row['Cluster'])
            folium.CircleMarker(
                location=[row[col_lat], row[col_lon]],
                radius=7,
                color=colors[cluster_id % len(colors)],
                fill=True,
                fill_color=colors[cluster_id % len(colors)],
                fill_opacity=0.7,
                weight=1.5,
                popup=f"<b>Gerai #{idx}</b><br>Klaster: {cluster_id}"
            ).add_to(m)
            
        for c_idx, c_coord in enumerate(kmeans.cluster_centers_):
            folium.Marker(
                location=[c_coord[0], c_coord[1]],
                popup=f"<b>Pusat Centroid Klaster {c_idx}</b>",
                icon=folium.Icon(color='white', icon='star', prefix='fa')
            ).add_to(m)
            
        st_folium(m, width="100%", height=500, returned_objects=[])
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_form:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Evaluasi Titik Lokasi Baru")
        st.caption("Masukkan koordinat untuk menganalisis karakteristik zona pasar.")
        
        in_lat = st.number_input(f"Latitude ({col_lat})", value=center_lat, format="%.6f")
        in_lon = st.number_input(f"Longitude ({col_lon})", value=center_lon, format="%.6f")
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📌 Analisis Karakteristik Zona"):
            pred_cluster = int(kmeans.predict([[in_lat, in_lon]])[0])
            
            st.markdown(f"""
                <div style="background: rgba(168, 199, 250, 0.08); padding: 16px; border-radius: 14px; margin-bottom: 20px; border: 1px solid rgba(168, 199, 250, 0.2); font-weight: 600; color: #a8c7fa; text-align: center;">
                    📍 Tergolong dalam <b>Klaster {pred_cluster}</b>
                </div>
            """, unsafe_allow_html=True)
            
            if pred_cluster == 0:
                st.markdown("""
                    <div class="status-alert">
                        <b style="font-size: 1.05rem;">⚠️ STATUS ZONA: DENSITY RENDAH</b><br>
                        <span style="font-size: 0.85rem; opacity: 0.9; margin-top: 4px; display: block;">Tingkat kompetisi rendah. Area ini membutuhkan strategi penetrasi pasar agresif.</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="status-success">
                        <b style="font-size: 1.05rem;">✅ STATUS ZONA: DENSITY TINGGI</b><br>
                        <span style="font-size: 0.85rem; opacity: 0.9; margin-top: 4px; display: block;">Konsentrasi gerai tinggi. Menandakan foot-traffic yang sudah matang dan kompetitif.</span>
                    </div>
                """, unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)
