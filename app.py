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
# 2. CUSTOM CSS - NEXT-GEN GLASSMORPHISM & UI POLISH
# ====================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

    /* Global Reset & Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e293b 0%, #0f172a 60%, #080d1a 100%);
        color: #f1f5f9;
    }

    footer, #MainMenu { visibility: hidden; }

    [data-testid="stHeader"] {
        background-color: rgba(15, 23, 42, 0.0) !important;
    }

    /* Enterprise Glassmorphism Cards */
    .gemini-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .gemini-card:hover {
        border-color: rgba(168, 199, 250, 0.2);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45);
    }

    /* Badges & Titles */
    .gemini-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(168, 199, 250, 0.12) 0%, rgba(124, 172, 248, 0.04) 100%);
        border: 1px solid rgba(168, 199, 250, 0.3);
        backdrop-filter: blur(12px);
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 0.75rem;
        color: #a8c7fa;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 12px;
        box-shadow: 0 0 12px rgba(168, 199, 250, 0.15);
    }

    .gemini-title {
        background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 40%, #A8C7FA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.4rem;
        letter-spacing: -0.8px;
        margin-bottom: 8px;
        line-height: 1.2;
    }

    .gemini-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 28px;
        font-weight: 400;
    }

    /* Premium KPI Cards */
    .kpi-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(168, 199, 250, 0.4), transparent);
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }

    .kpi-label {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Custom Input Fields & Buttons Overrides */
    div[data-baseweb="input"] > div {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
    }

    div[data-baseweb="select"] > div {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5) !important;
    }

    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.88rem;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.2) !important;
        color: #a8c7fa !important;
        border: 1px solid rgba(168, 199, 250, 0.3) !important;
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 15, 29, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Medical Status Cards */
    .status-positive {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(153, 27, 27, 0.1) 100%);
        border: 1px solid rgba(248, 113, 113, 0.3);
        color: #fca5a5;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.1);
    }

    .status-negative {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(20, 83, 45, 0.1) 100%);
        border: 1px solid rgba(74, 222, 128, 0.3);
        color: #86efac;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(34, 197, 94, 0.1);
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
            <div style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);">
                <span style="font-size: 1.4rem;">✨</span>
            </div>
            <div>
                <div style="font-weight: 800; font-size: 1.05rem; color: #f8fafc; letter-spacing: -0.3px;">UMUL AIMAN</div>
                <div style="font-size: 0.75rem; color: #a8c7fa; font-weight: 600;">NIM: 23146039</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 0.72rem; color: #64748b; font-weight: 800; letter-spacing: 1px; margin-bottom: 12px;'>MODUL SISTEM</p>", unsafe_allow_html=True)
    
    page = st.radio(
        "Pilih Halaman:",
        ["1. Prediksi Diabetes (Klasifikasi)", "2. Clustering Gerai Kopi (Geospasial)"],
        label_visibility="collapsed"
    )
    
    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.06);'><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='font-size: 0.72rem; color: #64748b; text-align: center; line-height: 1.6;'>
            <b style='color: #94a3b8;'>Gemini Mining Engine v4.0 Ultra</b><br>
            Scikit-Learn • Folium • Streamlit Engine
        </div>
    """, unsafe_allow_html=True)

# ====================================================
# MODUL 1: KLASIFIKASI DIABETES
# ====================================================
if page == "1. Prediksi Diabetes (Klasifikasi)":
    
    st.markdown('<div class="gemini-badge">🩺 Healthcare Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-title">Prediksi Risiko Diabetes Pasien</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-subtitle">Sistem pendukung keputusan klinis berbasis AI dengan simulasi tunggal, analisis batch, dan transparansi model.</div>', unsafe_allow_html=True)

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

    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df_diab):,}</div><div class="kpi-label">Total Sampel Data</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(X.columns)}</div><div class="kpi-label">Fitur Fisiologis</div></div>', unsafe_allow_html=True)
    with kpi3:
        best_acc = max([accuracy_score(y_test, m.predict(X_test)) for m in models.values()])
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{best_acc*100:.1f}%</div><div class="kpi-label">Akurasi Best Model</div></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(models)}</div><div class="kpi-label">Algoritma Aktif</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_sim, tab_batch, tab_eval = st.tabs(["📝 Input Simulation", "📁 Batch Processing", "📊 Analytics & Benchmarks"])

    # TAB 1: INPUT TUNGGAL
    with tab_sim:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Parameter Fisiologis Pasien")
        st.caption("Masukkan data klinis individual untuk mensimulasikan hasil klasifikasi risiko secara real-time.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        selected_model_name = st.selectbox("Algoritma Model Presisi:", list(models.keys()), key="pred_select")
        
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
                        <b style="font-size: 1.1rem;">⚠️ RESULT: RISIKO DIABETES TERDETEKSI (POSITIF)</b><br>
                        <span style="font-size: 0.88rem; opacity: 0.9;">Pola parameter fisiologis berada pada rentang berisiko tinggi. Disarankan melakukan validasi laboratorium medis lanjutan.</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="status-negative">
                        <b style="font-size: 1.1rem;">✅ RESULT: PASIEN INDIKASI SEHAT (NEGATIF)</b><br>
                        <span style="font-size: 0.88rem; opacity: 0.9;">Seluruh profil indikator fisiologis berada di dalam ambang batas normal.</span>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2: BATCH PREDICTION
    with tab_batch:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 📁 Batch Processing (Upload CSV)")
        st.caption("Unggah dataset massal tanpa kolom label/target untuk klasifikasi otomatis.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Unggah berkas CSV data pasien", type=["csv"])
        batch_model_name = st.selectbox("Model untuk Batch Processing:", list(models.keys()), key="batch_select")
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.markdown("##### 👁️ Preview Data")
            st.dataframe(batch_df.head(3), use_container_width=True)
            
            if st.button("🚀 Eksekusi Prediksi Massal"):
                try:
                    chosen_batch_model = models[batch_model_name]
                    preds = chosen_batch_model.predict(batch_df[feature_names])
                    
                    batch_df['Hasil_Prediksi'] = ["Positif Diabetes" if p == 1 else "Negatif (Sehat)" for p in preds]
                    
                    st.success("✅ Prediksi massal berhasil diselesaikan!")
                    st.dataframe(batch_df, use_container_width=True)
                    
                    csv_data = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Unduh Hasil Prediksi (.CSV)",
                        data=csv_data,
                        file_name="hasil_prediksi_diabetes.csv",
                        mime="text/csv"
                    )
                except Exception as err:
                    st.error(f"Gagal memproses data. Pastikan nama kolom CSV sesuai. Detail: {err}")
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 3: ANALYTICS
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
                st.markdown("<br>#### 🔍 Relative Feature Importance (Decision Tree)", unsafe_allow_html=True)
                imp_df = pd.DataFrame({'Fitur': feature_names, 'Pengaruh': dt.feature_importances_}).sort_values('Pengaruh', ascending=True)
                fig_imp = px.bar(imp_df, x='Pengaruh', y='Fitur', orientation='h', color='Pengaruh', color_continuous_scale='Blues')
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
                labels=dict(x="Prediksi Model", y="Kondisi Aktual", color="Jumlah"),
                x=['Negatif (0)', 'Positif (1)'],
                y=['Negatif (0)', 'Positif (1)'],
                color_continuous_scale='Blues'
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#94a3b8"), height=320, margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ====================================================
# MODUL 2: CLUSTERING GERAI KOPI
# ====================================================
elif page == "2. Clustering Gerai Kopi (Geospasial)":
    
    st.markdown('<div class="gemini-badge">☕ Spatial Intelligence Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-title">Klaster Geospasial Gerai Kopi</div>', unsafe_allow_html=True)
    st.markdown('<div class="gemini-subtitle">Segmentasi wilayah bisnis berbasis geospasial interaktif (CartoDB Dark Matter).</div>', unsafe_allow_html=True)
    
    try:
        df_kopi = load_dataset('lokasi_gerai_kopi_clean.csv')
        kmeans = load_ml_model('model_kmeans.pkl')
    except Exception as e:
        st.error(f"⚠️ Gagal memuat dataset/model clustering. Detail: {e}")
        st.stop()

    lat_candidates = [c for c in df_kopi.columns if any(k in c.lower() for k in ['lat', 'y', 'lintang'])]
    lon_candidates = [c for c in df_kopi.columns if any(k in c.lower() for k in ['lon', 'long', 'lng', 'x', 'bujur'])]
    
    if not lat_candidates or not lon_candidates:
        st.error("⚠️ Kolom koordinat Latitude/Longitude tidak ditemukan pada dataset.")
        st.stop()
        
    col_lat, col_lon = lat_candidates[0], lon_candidates[0]
    
    X_kopi = df_kopi[[col_lat, col_lon]]
    df_kopi['Cluster'] = kmeans.labels_
    
    col_map, col_form = st.columns([1.3, 0.7])
    
    with col_map:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 🗺️ Peta Interaktif Sebaran Klaster")
        
        center_lat = df_kopi[col_lat].mean()
        center_lon = df_kopi[col_lon].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB dark_matter")
        
        colors = ['#FF4B4B', '#1E88E5', '#00E676', '#FFD600', '#AB47BC']
        
        for idx, row in df_kopi.iterrows():
            cluster_id = int(row['Cluster'])
            folium.CircleMarker(
                location=[row[col_lat], row[col_lon]],
                radius=6,
                color=colors[cluster_id % len(colors)],
                fill=True,
                fill_color=colors[cluster_id % len(colors)],
                fill_opacity=0.8,
                popup=f"Gerai #{idx} (Klaster {cluster_id})"
            ).add_to(m)
            
        for c_idx, c_coord in enumerate(kmeans.cluster_centers_):
            folium.Marker(
                location=[c_coord[0], c_coord[1]],
                popup=f"Pusat Centroid Klaster {c_idx}",
                icon=folium.Icon(color='white', icon='star', prefix='fa')
            ).add_to(m)
            
        st_folium(m, width="100%", height=480)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_form:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Evaluasi Lokasi Baru")
        st.caption("Uji koordinat lokasi rencana outlet baru untuk memprediksi karakteristik zona.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        in_lat = st.number_input(f"Latitude ({col_lat})", value=float(center_lat), format="%.6f")
        in_lon = st.number_input(f"Longitude ({col_lon})", value=float(center_lon), format="%.6f")
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📌 Analisis Karakteristik Zona"):
            pred_cluster = kmeans.predict([[in_lat, in_lon]])[0]
            
            st.markdown(f"""
                <div style="background: rgba(168, 199, 250, 0.1); padding: 14px 18px; border-radius: 12px; margin-bottom: 20px; border: 1px solid rgba(168, 199, 250, 0.3); font-weight: 600; color: #a8c7fa;">
                    📌 Hasil Pemetaan: Tergolong dalam <b>Klaster {pred_cluster}</b>
                </div>
            """, unsafe_allow_html=True)
            
            if pred_cluster == 0:
                st.markdown("""
                    <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); color: #fde68a; padding: 18px; border-radius: 14px;">
                        <b style="font-size: 1rem;">📍 ZONA KEPADATAN RENDAH (BLUE OCEAN)</b><br>
                        <span style="font-size: 0.85rem; opacity: 0.9;">Tingkat kompetisi antar-gerai rendah. Sangat potensial untuk ekspansi dan penguasaan pangsa pasar baru.</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background: rgba(59, 130, 246, 0.12); border: 1px solid rgba(59, 130, 246, 0.3); color: #bfdbfe; padding: 18px; border-radius: 14px;">
                        <b style="font-size: 1rem;">🏢 ZONA KEPADATAN TINGGI (RED OCEAN)</b><br>
                        <span style="font-size: 0.85rem; opacity: 0.9;">Konsentrasi outlet tinggi. Menandakan basis konsumen yang terbukti besar namun dengan kompetisi ketat.</span>
                    </div>
                """, unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)
