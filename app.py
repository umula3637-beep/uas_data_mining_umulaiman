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
from folium.plugins import HeatMap, MarkerCluster

# ====================================================
# 1. PAGE CONFIGURATION
# ====================================================
st.set_page_config(
    page_title="UMUL AIMAN 23146039 - Neural Mining Engine", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================
# 2. CUSTOM CSS - NEURAL EXPRESSIVE & GLASSMORPHISM
# ====================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #172033 0%, #0b1120 50%, #050811 100%);
        color: #f8fafc;
    }

    footer, #MainMenu { visibility: hidden; }

    [data-testid="stHeader"] {
        background-color: rgba(11, 17, 32, 0.0) !important;
    }

    /* Enterprise Glassmorphism Cards */
    .gemini-card {
        background: rgba(23, 32, 51, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-radius: 24px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .gemini-card:hover {
        border-color: rgba(96, 165, 250, 0.25);
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.5), 0 0 20px rgba(59, 130, 246, 0.1);
    }

    /* Badges & Header Styling */
    .gemini-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.15) 0%, rgba(37, 99, 235, 0.05) 100%);
        border: 1px solid rgba(96, 165, 250, 0.35);
        backdrop-filter: blur(12px);
        padding: 6px 18px;
        border-radius: 999px;
        font-size: 0.72rem;
        color: #93c5fd;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
        box-shadow: 0 0 16px rgba(59, 130, 246, 0.2);
    }

    .gemini-title {
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 50%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        letter-spacing: -1px;
        margin-bottom: 8px;
        line-height: 1.15;
    }

    .gemini-subtitle {
        color: #94a3b8;
        font-size: 1.02rem;
        margin-bottom: 32px;
        font-weight: 400;
        line-height: 1.5;
    }

    /* KPI Display Cards */
    .kpi-card {
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 22px 16px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-3px);
        background: rgba(255, 255, 255, 0.04);
    }

    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #60a5fa, #93c5fd);
    }

    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }

    .kpi-label {
        font-size: 0.7rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.8px;
    }

    /* Custom Form & Input Styling */
    div[data-baseweb="input"] > div {
        background-color: rgba(11, 17, 32, 0.7) !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        transition: all 0.2s ease;
    }
    div[data-baseweb="input"] > div:focus-within {
        border-color: #60a5fa !important;
        box-shadow: 0 0 12px rgba(96, 165, 250, 0.25) !important;
    }

    div[data-baseweb="select"] > div {
        background-color: rgba(11, 17, 32, 0.7) !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        padding: 14px 28px !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        box-shadow: 0 10px 28px rgba(59, 130, 246, 0.5) !important;
    }

    /* Tab Design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(11, 17, 32, 0.6);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 12px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.88rem;
        border: none !important;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(37, 99, 235, 0.25) !important;
        color: #93c5fd !important;
        border: 1px solid rgba(96, 165, 250, 0.4) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(5, 8, 17, 0.96) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Custom Status Cards */
    .status-positive {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(127, 29, 29, 0.1) 100%);
        border: 1px solid rgba(248, 113, 113, 0.4);
        color: #fca5a5;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.15);
    }

    .status-negative {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(20, 83, 45, 0.1) 100%);
        border: 1px solid rgba(74, 222, 128, 0.4);
        color: #86efac;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0 10px 25px rgba(34, 197, 94, 0.15);
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
        <div style="display: flex; align-items: center; gap: 14px; padding: 14px 6px; margin-bottom: 24px; background: rgba(255,255,255,0.03); border-radius: 16px; border: 1px solid rgba(255,255,255,0.06);">
            <div style="background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); width: 46px; height: 46px; border-radius: 14px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);">
                <span style="font-size: 1.5rem;">✨</span>
            </div>
            <div>
                <div style="font-weight: 800; font-size: 1.05rem; color: #f8fafc; letter-spacing: -0.3px;">UMUL AIMAN</div>
                <div style="font-size: 0.75rem; color: #93c5fd; font-weight: 700;">NIM: 23146039</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 0.72rem; color: #64748b; font-weight: 800; letter-spacing: 1px; margin-bottom: 12px;'>NAVIGASI MODUL ENGINE</p>", unsafe_allow_html=True)
    
    page = st.radio(
        "Pilih Halaman:",
        ["1. Prediksi Diabetes (Klasifikasi)", "2. Clustering Gerai Kopi (Geospasial)"],
        label_visibility="collapsed"
    )
    
    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08);'><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='font-size: 0.72rem; color: #64748b; text-align: center; line-height: 1.6;'>
            <b style='color: #94a3b8;'>Neural Mining Engine v4.5 Pro</b><br>
            Powered by Scikit-Learn & Folium
        </div>
    """, unsafe_allow_html=True)

# ====================================================
# MODUL 1: KLASIFIKASI DIABETES
# ====================================================
if page == "1. Prediksi Diabetes (Klasifikasi)":
    
    st.markdown('<div class="gemini-badge">🩺 Healthcare Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-title">Prediksi Risiko Diabetes Pasien</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-subtitle">Sistem pendukung keputusan klinis berbasis Machine Learning dengan opsi simulasi instan, batch processing, dan evaluasi performa model.</div>', unsafe_allow_html=True)

    try:
        df_diab = load_dataset('diabetes.csv')
        knn = load_ml_model('model_knn.pkl')
        nb = load_ml_model('model_nb.pkl')
        dt = load_ml_model('model_dt.pkl')
    except Exception as e:
        st.error(f"⚠️ Gagal memuat dataset atau model ML. Pastikan file tersedia di root folder. Detail: {e}")
        st.stop()

    target_col = next((col for col in df_diab.columns if col.lower().strip() in ['outcome', 'target', 'class', 'diabetes']), df_diab.columns[-1])
    X = df_diab.drop(target_col, axis=1)
    y = df_diab[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {'KNN': knn, 'Naïve Bayes': nb, 'Decision Tree': dt}

    # Dynamic Metric Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df_diab):,}</div><div class="kpi-label">Total Sampel Pasien</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(X.columns)}</div><div class="kpi-label">Fitur Fisiologis</div></div>', unsafe_allow_html=True)
    with kpi3:
        best_acc = max([accuracy_score(y_test, m.predict(X_test)) for m in models.values()])
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{best_acc*100:.1f}%</div><div class="kpi-label">Akurasi Best Model</div></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(models)}</div><div class="kpi-label">Model Terintegrasi</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_sim, tab_batch, tab_eval = st.tabs(["📝 Input Simulation", "📁 Batch Processing", "📊 Analytics & Benchmarks"])

    # TAB 1: INPUT SIMULATION
    with tab_sim:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Input Parameter Fisiologis")
        st.caption("Atur variabel klinis individual untuk memperoleh simulasi diagnosa risiko secara real-time.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        selected_model_name = st.selectbox("Algoritma Klasifikasi Presisi:", list(models.keys()), key="pred_select")
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
        if st.button("✨ Eksekusi Diagnosa AI"):
            chosen_model = models[selected_model_name]
            prediction = chosen_model.predict([input_values])[0]
            
            if prediction == 1:
                st.markdown("""
                    <div class="status-positive">
                        <b style="font-size: 1.1rem;">⚠️ DIAGNOSA: INDIKASI DIABETES TERDETEKSI (POSITIF)</b><br>
                        <span style="font-size: 0.88rem; opacity: 0.9;">Profil fisiologis berada pada rentang batas risiko tinggi. Direkomendasikan melakukan pemeriksaan medis laboratorium komprehensif.</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="status-negative">
                        <b style="font-size: 1.1rem;">✅ DIAGNOSA: INDIKASI SEHAT (NEGATIF DIABETES)</b><br>
                        <span style="font-size: 0.88rem; opacity: 0.9;">Seluruh parameter fisiologis berada dalam ambang batas standar normal.</span>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2: BATCH PROCESSING
    with tab_batch:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 📁 Batch Processing (Upload CSV)")
        st.caption("Proses dataset pasien berukuran besar secara otomatis hanya dalam hitungan detik.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Unggah berkas CSV data pasien", type=["csv"])
        batch_model_name = st.selectbox("Model untuk Batch Processing:", list(models.keys()), key="batch_select")
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.markdown("##### 👁️ Preview Data Terbaru")
            st.dataframe(batch_df.head(3), use_container_width=True)
            
            if st.button("🚀 Eksekusi Prediksi Massal"):
                try:
                    chosen_batch_model = models[batch_model_name]
                    preds = chosen_batch_model.predict(batch_df[feature_names])
                    
                    batch_df['Hasil_Prediksi'] = ["Positif Diabetes" if p == 1 else "Negatif (Sehat)" for p in preds]
                    
                    st.success("✅ Prediksi massal diselesaikan secara presisi!")
                    st.dataframe(batch_df, use_container_width=True)
                    
                    csv_data = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Unduh Berkas Hasil Prediksi (.CSV)",
                        data=csv_data,
                        file_name="hasil_prediksi_diabetes.csv",
                        mime="text/csv"
                    )
                except Exception as err:
                    st.error(f"Gagal memproses data. Pastikan format nama kolom sesuai. Detail: {err}")
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 3: ANALYTICS & BENCHMARKS
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
                    'Akurasi': f"{accuracy_score(y_test, y_pred):.3f}",
                    'Precision': f"{precision_score(y_test, y_pred):.3f}",
                    'Recall': f"{recall_score(y_test, y_pred):.3f}",
                    'F1-Score': f"{f1_score(y_test, y_pred):.3f}"
                })
            st.dataframe(pd.DataFrame(metrics_list), use_container_width=True, hide_index=True)
            
            if hasattr(dt, 'feature_importances_'):
                st.markdown("<br>#### 🔍 Feature Importance Analysis (Decision Tree)", unsafe_allow_html=True)
                imp_df = pd.DataFrame({'Fitur': feature_names, 'Bobot': dt.feature_importances_}).sort_values('Bobot', ascending=True)
                fig_imp = px.bar(imp_df, x='Bobot', y='Fitur', orientation='h', color='Bobot', color_continuous_scale='Viridis')
                fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#94a3b8"), height=260, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_imp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
            st.markdown("### 🧩 Confusion Matrix Visualization")
            selected_eval_model = st.selectbox("Pilih Model Evaluasi:", list(models.keys()), key="cm_select")
            
            cm = confusion_matrix(y_test, models[selected_eval_model].predict(X_test))
            fig = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Prediksi Model", y="Kondisi Real", color="Jumlah"),
                x=['Negatif (0)', 'Positif (1)'],
                y=['Negatif (0)', 'Positif (1)'],
                color_continuous_scale='Plasma'
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#94a3b8"), height=320, margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ====================================================
# MODUL 2: CLUSTERING GERAI KOPI (GEOSPASIAL ULTRA)
# ====================================================
elif page == "2. Clustering Gerai Kopi (Geospasial)":
    
    st.markdown('<div class="gemini-badge">☕ Spatial Intelligence Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-title">Klaster Geospasial Gerai Kopi</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-subtitle">Visualisasi distribusi spasial bisnis, estimasi kerapatan wilayah, dan zonasi ekspansi gerai berbasis K-Means.</div>', unsafe_allow_html=True)
    
    try:
        df_kopi = load_dataset('lokasi_gerai_kopi_clean.csv')
        kmeans = load_ml_model('model_kmeans.pkl')
    except Exception as e:
        st.error(f"⚠️ Gagal memuat dataset/model clustering. Detail: {e}")
        st.stop()

    # Cari nama kolom koordinat
    lat_candidates = [c for c in df_kopi.columns if any(k in c.lower() for k in ['lat', 'y', 'lintang', 'latitude'])]
    lon_candidates = [c for c in df_kopi.columns if any(k in c.lower() for k in ['lon', 'long', 'lng', 'x', 'bujur', 'longitude'])]
    
    if not lat_candidates or not lon_candidates:
        st.error(f"⚠️ Kolom koordinat Latitude/Longitude tidak ditemukan pada dataset. Kolom yang ada: {list(df_kopi.columns)}")
        st.stop()
        
    col_lat, col_lon = lat_candidates[0], lon_candidates[0]
    X_kopi = df_kopi[[col_lat, col_lon]]
    
    try:
        df_kopi['Cluster'] = kmeans.predict(X_kopi)
    except Exception:
        df_kopi['Cluster'] = kmeans.labels_
    
    name_candidates = [c for c in df_kopi.columns if any(k in c.lower() for k in ['nama', 'name', 'gerai', 'outlet', 'store', 'toko'])]
    col_name = name_candidates[0] if name_candidates else None
    
    col_map, col_form = st.columns([1.35, 0.65])
    
    with col_map:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        
        m_top_l, m_top_r = st.columns([1, 1])
        with m_top_l:
            st.markdown("### 🗺️ Peta Interaktif Spasial")
        with m_top_r:
            map_view = st.radio("Mode Layers:", ["Cluster Markers", "Heatmap Density", "Grouped Clusters"], horizontal=True, label_visibility="collapsed")

        center_lat = float(df_kopi[col_lat].mean())
        center_lon = float(df_kopi[col_lon].mean())
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB dark_matter")
        
        cluster_colors = ['#00E676', '#FF2A6D', '#05D5E7', '#FFD600', '#DDA0DD']
        
        if map_view == "Heatmap Density":
            heat_data = [[row[col_lat], row[col_lon]] for _, row in df_kopi.iterrows()]
            HeatMap(heat_data, radius=20, blur=15, min_opacity=0.45, gradient={0.2: '#05D5E7', 0.5: '#00E676', 0.8: '#FFD600', 1.0: '#FF2A6D'}).add_to(m)
        elif map_view == "Grouped Clusters":
            marker_cluster = MarkerCluster().add_to(m)
            for idx, row in df_kopi.iterrows():
                c_id = int(row['Cluster'])
                g_name = row[col_name] if col_name else f"Gerai #{idx+1}"
                
                folium.Marker(
                    location=[row[col_lat], row[col_lon]],
                    popup=f"<b>{g_name}</b><br>Klaster: {c_id}",
                    icon=folium.Icon(color='blue', icon='coffee', prefix='fa')
                ).add_to(marker_cluster)
        else:
            for idx, row in df_kopi.iterrows():
                c_id = int(row['Cluster'])
                c_color = cluster_colors[c_id % len(cluster_colors)]
                g_name = row[col_name] if col_name else f"Gerai #{idx+1}"
                
                popup_html = f"""
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; padding: 6px; width: 170px; color: #0f172a;">
                    <div style="font-size: 0.72rem; font-weight: 800; color: {c_color}; text-transform: uppercase; letter-spacing: 0.5px;">KLASTER {c_id}</div>
                    <div style="font-size: 0.95rem; font-weight: 700; margin: 2px 0 6px 0;">{g_name}</div>
                    <div style="font-size: 0.75rem; color: #475569;">
                        <b>Lat:</b> {row[col_lat]:.4f}<br>
                        <b>Lon:</b> {row[col_lon]:.4f}
                    </div>
                </div>
                """
                
                folium.CircleMarker(
                    location=[row[col_lat], row[col_lon]],
                    radius=7,
                    color=c_color,
                    fill=True,
                    fill_color=c_color,
                    fill_opacity=0.85,
                    popup=folium.Popup(popup_html, max_width=220)
                ).add_to(m)

        # Centroid Markers
        if hasattr(kmeans, 'cluster_centers_'):
            for c_idx, c_coord in enumerate(kmeans.cluster_centers_):
                c_color = cluster_colors[c_idx % len(cluster_colors)]
                folium.Marker(
                    location=[c_coord[0], c_coord[1]],
                    popup=f"<b>Pusat Centroid Klaster {c_idx}</b>",
                    icon=folium.Icon(color='black', icon_color=c_color, icon='coffee', prefix='fa')
                ).add_to(m)

        st_folium(m, width="100%", height=500)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_form:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Evaluasi Lokasi Baru")
        st.caption("Uji koordinat rencana gerai baru untuk mengetahui segmentasi zonanya.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        in_lat = st.number_input(f"Latitude ({col_lat})", value=float(center_lat), format="%.6f")
        in_lon = st.number_input(f"Longitude ({col_lon})", value=float(center_lon), format="%.6f")
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📌 Analisis Karakteristik Zona"):
            try:
                input_df = pd.DataFrame([[in_lat, in_lon]], columns=[col_lat, col_lon])
                pred_cluster = int(kmeans.predict(input_df)[0])
                assigned_color = cluster_colors[pred_cluster % len(cluster_colors)]
                
                st.markdown(f"""
                    <div style="background: rgba(11, 17, 32, 0.8); padding: 18px; border-radius: 16px; margin-bottom: 20px; border: 1px solid {assigned_color}; font-weight: 600; color: #f8fafc; display: flex; align-items: center; gap: 12px;">
                        <div style="width: 16px; height: 16px; border-radius: 50%; background: {assigned_color}; box-shadow: 0 0 12px {assigned_color};"></div>
                        <div>Hasil Zonasi: Masuk dalam <b style="color: {assigned_color};">Klaster {pred_cluster}</b></div>
                    </div>
                """, unsafe_allow_html=True)
                
                if pred_cluster == 0:
                    st.markdown("""
                        <div style="background: rgba(5, 213, 231, 0.08); border: 1px solid rgba(5, 213, 231, 0.35); color: #67e8f9; padding: 18px; border-radius: 16px;">
                            <b style="font-size: 1rem;">📍 ZONA POTENSIAL (BLUE OCEAN)</b><br>
                            <span style="font-size: 0.85rem; opacity: 0.9;">Tingkat kompetisi relatif rendah. Sangat strategis untuk membangun *brand awareness* wilayah baru.</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style="background: rgba(255, 42, 109, 0.08); border: 1px solid rgba(255, 42, 109, 0.35); color: #fca5a5; padding: 18px; border-radius: 16px;">
                            <b style="font-size: 1rem;">🔥 ZONA PADAT (HIGH DENSITY)</b><br>
                            <span style="font-size: 0.85rem; opacity: 0.9;">Konsentrasi gerai tinggi. Traksi pasar sangat kuat namun tingkat persaingan antar-gerai tinggi.</span>
                        </div>
                    """, unsafe_allow_html=True)
            except Exception as err:
                st.error(f"Gagal memprediksi zona. Detail: {err}")
                
        st.markdown('</div>', unsafe_allow_html=True)
