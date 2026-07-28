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
# 1. KONFIGURASI HALAMAN & CSS PREMIUM (ENHANCED)
# ==============================================================================
st.set_page_config(
    page_title="Data Mining Pro Dashboard", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan Premium & Modern
st.markdown("""
<style>
    /* Import Font Modern: Plus Jakarta Sans */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Font & Background */
    .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #f8fafc;
    }
    
    /* Hide default Streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header Styling dengan Gradient */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #64748b;
        margin-bottom: 2rem;
        font-weight: 400;
        line-height: 1.6;
        max-width: 800px;
    }
    
    /* Card Styling untuk Hasil (Glassmorphism) */
    .pred-card {
        padding: 28px;
        border-radius: 16px;
        margin-top: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease;
    }
    .pred-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08), 0 10px 10px -5px rgba(0, 0, 0, 0.03);
    }
    .pred-positive {
        background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%);
        border-left: 5px solid #ef4444;
    }
    .pred-negative {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border-left: 5px solid #10b981;
    }
    .pred-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .pred-positive .pred-title { color: #b91c1c; }
    .pred-negative .pred-title { color: #047857; }
    .pred-desc { 
        font-size: 1rem; 
        color: #475569; 
        line-height: 1.7; 
    }

    /* Metric Card Override */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sidebar Polish */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        box-shadow: 4px 0 24px rgba(0,0,0,0.02);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #0f172a;
    }
    
    /* Button Polish */
    .stButton>button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.2) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3) !important;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%) !important;
        color: white !important;
    }

    /* Slider & Input Polish */
    .stSlider > div > div > div > div {
        background-color: #3b82f6 !important;
    }
    
    /* Tabs Polish */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        color: #64748b !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 12px 24px !important;
        border: none !important;
        background-color: transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2563eb !important;
        background-color: #eff6ff !important;
        border-bottom: 3px solid #2563eb !important;
    }
    
    /* Custom Progress Bar */
    .custom-progress-bg {
        background-color: #e2e8f0;
        border-radius: 999px;
        height: 10px;
        width: 100%;
        margin-top: 12px;
        overflow: hidden;
    }
    .custom-progress-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. NAVIGASI SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("<div style='padding: 10px 0;'><span style='font-size: 1.8rem;'>📊</span></div>", unsafe_allow_html=True)
    st.markdown("### DataMining Pro")
    st.markdown("<p style='color: #64748b; font-size: 0.9rem; font-weight: 500; margin-top: -10px;'>Dashboard Analisis & Prediksi Cerdas</p>", unsafe_allow_html=True)
    st.divider()
    
    page = st.radio(
        "Pilih Modul Analisis:", 
        ["🩺 Prediksi Diabetes", "☕ Clustering Gerai Kopi"],
        label_visibility="collapsed",
        index=0
    )
    
    st.divider()
    st.markdown("#### ℹ️ Tentang Aplikasi")
    st.caption("Dibangun dengan Streamlit, Scikit-Learn, dan Seaborn. Memanfaatkan algoritma KNN, Naïve Bayes, Decision Tree, dan K-Means untuk analisis data yang akurat dan visualisasi yang modern.")

# ==============================================================================
# 3. HALAMAN 1: PREDIKSI DIABETES
# ==============================================================================
if page == "🩺 Prediksi Diabetes":
    st.markdown('<h1 class="main-header">🩺 Prediksi Risiko Diabetes</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluasi risiko diabetes berdasarkan 8 indikator kesehatan utama menggunakan ensemble model machine learning yang telah terlatih.</p>', unsafe_allow_html=True)
    
    try:
        # Load Data & Model
        df_diab = pd.read_csv('diabetes.csv')
        knn = joblib.load('model_knn.pkl')
        nb = joblib.load('model_nb.pkl')
        dt = joblib.load('model_dt.pkl')
        
        target_col = next((col for col in df_diab.columns if col.lower() in ['outcome', 'target', 'class', 'diabetes']), df_diab.columns[-1])
        X = df_diab.drop(target_col, axis=1)
        y = df_diab[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {'KNN': knn, 'Naïve Bayes': nb, 'Decision Tree': dt}
        
        # TAB LAYOUT
        tab_eval, tab_pred, tab_info = st.tabs(["📊 Evaluasi Model", "🔮 Prediksi Pasien Baru", "📖 Info Dataset"])
        
        with tab_eval:
            st.subheader("Performa Algoritma pada Data Uji")
            
            # Hitung Metrik
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
            
            # Tampilkan Metric Cards (Baris 1)
            cols = st.columns(3)
            for i, row in df_metrics.iterrows():
                with cols[i]:
                    st.metric(
                        label=f"Akurasi {row['Model']}", 
                        value=f"{row['Akurasi']*100:.1f}%", 
                        delta=f"F1: {row['F1-Score']:.2f}",
                        delta_color="normal"
                    )
            
            # Confusion Matrix Interaktif
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 🎯 Visualisasi Confusion Matrix")
            col_cm1, col_cm2 = st.columns([1, 3])
            with col_cm1:
                selected_cm = st.selectbox("Pilih Model:", list(models.keys()), key="cm_select", label_visibility="collapsed")
            with col_cm2:
                fig, ax = plt.subplots(figsize=(6, 4.5))
                # Modern heatmap styling
                sns.heatmap(cm_dict[selected_cm], annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Negatif (0)', 'Positif (1)'], 
                            yticklabels=['Negatif (0)', 'Positif (1)'], 
                            ax=ax, cbar=False, linewidths=1.5, linecolor='white',
                            annot_kws={"size": 14, "weight": "bold", "color": "#1e293b"})
                
                ax.set_title(f"Confusion Matrix: {selected_cm}", fontsize=14, fontweight='bold', color='#1e293b', pad=15)
                ax.set_ylabel('Kondisi Aktual', fontsize=11, color='#64748b', fontweight='600')
                ax.set_xlabel('Hasil Prediksi', fontsize=11, color='#64748b', fontweight='600')
                
                # Remove outer spines for cleaner look
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(axis='both', colors='#475569', labelsize=10)
                
                st.pyplot(fig)

        with tab_pred:
            st.subheader("Formulir Input Data Klinis")
            st.info("💡 **Tips:** Geser slider di bawah ini sesuai dengan hasil pemeriksaan medis pasien untuk mendapatkan prediksi real-time.")
            
            selected_model = st.selectbox("Algoritma yang Digunakan:", list(models.keys()), index=2) # Default DT
            
            # Pengelompokan Input agar lebih rapi
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📏 Data Fisik & Demografi")
            col1, col2, col3 = st.columns(3)
            with col1:
                inp_pregnancies = st.slider("Kehamilan (Pregnancies)", 0, 20, int(X['Pregnancies'].median()), help="Jumlah keer hamil")
                inp_age = st.slider("Usia (Age)", 10, 100, int(X['Age'].median()), help="Usia pasien dalam tahun")
            with col2:
                inp_bmi = st.slider("BMI (Body Mass Index)", 0.0, 70.0, float(X['BMI'].median()), 0.1, help="Indeks massa tubuh (kg/m²)")
            with col3:
                # Placeholder for layout balance
                st.markdown("<div style='margin-top: 24px; color: #64748b; font-size: 0.9rem;'>Pastikan data yang dimasukkan akurat untuk hasil terbaik.</div>", unsafe_allow_html=True)
            
            st.markdown("##### 🩸 Data Medis & Laboratorium")
            col4, col5, col6 = st.columns(3)
            with col4:
                inp_glucose = st.slider("Glukosa (Glucose)", 0, 200, int(X['Glucose'].median()), help="Kadar glukosa plasma (mg/dL)")
                inp_bp = st.slider("Tekanan Darah (BloodPressure)", 0, 150, int(X['BloodPressure'].median()), help="Tekanan darah diastolik (mm Hg)")
            with col5:
                inp_skin = st.slider("Ketebalan Kulit (SkinThickness)", 0, 100, int(X['SkinThickness'].median()), help="Ketebalan kulit triceps (mm)")
                inp_insulin = st.slider("Insulin", 0, 900, int(X['Insulin'].median()), help="Kadar insulin darah (mu U/ml)")
            with col6:
                inp_dpf = st.slider("Fungsi Silsilah Diabetes (DPF)", 0.0, 3.0, float(X['DiabetesPedigreeFunction'].median()), 0.01, help="Faktor risiko genetik")
            
            st.divider()
            
            # Tombol Prediksi
            if st.button("🚀 Analisis Risiko Sekarang", type="primary", use_container_width=True):
                with st.spinner("Sedang memproses data melalui model Machine Learning..."):
                    input_array = np.array([[inp_pregnancies, inp_glucose, inp_bp, inp_skin, inp_insulin, inp_bmi, inp_dpf, inp_age]])
                    prediction = models[selected_model].predict(input_array)[0]
                    proba = models[selected_model].predict_proba(input_array)[0][1] if hasattr(models[selected_model], "predict_proba") else 0.0
                    
                    confidence_pct = proba * 100 if prediction == 1 else (1 - proba) * 100
                    bar_color = "#ef4444" if prediction == 1 else "#10b981"
                    
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="pred-card pred-positive">
                            <div class="pred-title">⚠️ HASIL: POSITIF DIABETES (Risiko Tinggi)</div>
                            <div class="pred-desc">
                                Berdasarkan data yang dimasukkan, model <b>{selected_model}</b> memprediksi pasien mengidap diabetes.<br><br>
                                <b>Tingkat Keyakinan Model:</b>
                                <div class="custom-progress-bg">
                                    <div class="custom-progress-fill" style="width: {confidence_pct}%; background-color: {bar_color};"></div>
                                </div>
                                <div style="text-align: right; font-size: 0.85rem; color: #64748b; margin-top: 4px;">{confidence_pct:.1f}%</div>
                                <br>
                                <b>Rekomendasi:</b> Segera konsultasikan dengan dokter untuk tes HbA1c lanjutan dan evaluasi pola makan.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="pred-card pred-negative">
                            <div class="pred-title">✅ HASIL: NEGATIF DIABETES (Risiko Rendah)</div>
                            <div class="pred-desc">
                                Berdasarkan data yang dimasukkan, model <b>{selected_model}</b> memprediksi pasien <b>tidak</b> mengidap diabetes.<br><br>
                                <b>Tingkat Keyakinan Model:</b>
                                <div class="custom-progress-bg">
                                    <div class="custom-progress-fill" style="width: {confidence_pct}%; background-color: {bar_color};"></div>
                                </div>
                                <div style="text-align: right; font-size: 0.85rem; color: #64748b; margin-top: 4px;">{confidence_pct:.1f}%</div>
                                <br>
                                <b>Rekomendasi:</b> Pertahankan pola hidup sehat, olahraga teratur, dan cek kesehatan rutin tahunan.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        with tab_info:
            st.subheader("Statistik Deskriptif Dataset")
            st.dataframe(df_diab.describe().round(2), use_container_width=True, height=300)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 🔗 Korelasi Antar Fitur")
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(df_diab.corr(), dtype=bool))
            sns.heatmap(df_diab.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax_corr, mask=mask, 
                        linewidths=0.5, cbar_kws={"shrink": .8}, annot_kws={"size": 9})
            ax_corr.set_title("Matriks Korelasi Fitur", fontsize=14, fontweight='bold', pad=15)
            st.pyplot(fig_corr)

    except FileNotFoundError as e:
        st.error(f"❌ **File tidak ditemukan:** `{e.filename}`. Pastikan file dataset dan model `.pkl` berada di folder yang sama dengan script ini.")
    except Exception as e:
        st.error(f"Terjadi kesalahan tak terduga: {e}")

# ==============================================================================
# 4. HALAMAN 2: CLUSTERING GERAI KOPI
# ==============================================================================
elif page == "☕ Clustering Gerai Kopi":
    st.markdown('<h1 class="main-header">☕ Analisis Sebaran & Zonasi Gerai Kopi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Segmentasi lokasi berbasis koordinat geografis menggunakan K-Means untuk identifikasi zona bisnis potensial dan analisis kompetisi.</p>', unsafe_allow_html=True)
    
    try:
        df_kopi = pd.read_csv('lokasi_gerai_kopi_clean.csv')
        kmeans = joblib.load('model_kmeans.pkl')
        
        # Deteksi kolom koordinat secara cerdas
        col_lat = next((c for c in df_kopi.columns if any(k in c.lower() for k in ['lat', 'y', 'lintang'])), df_kopi.columns[0])
        col_lon = next((c for c in df_kopi.columns if any(k in c.lower() for k in ['lon', 'long', 'lng', 'x', 'bujur'])), df_kopi.columns[1])
        
        X_kopi = df_kopi[[col_lat, col_lon]]
        df_kopi['Cluster'] = kmeans.labels_
        centroids = kmeans.cluster_centers_
        
        # Layout Utama
        col_map, col_panel = st.columns([2.5, 1])
        
        with col_map:
            st.subheader("📍 Peta Sebaran Klaster")
            
            # Styling Plot agar terlihat seperti Dashboard Premium
            sns.set_theme(style="white", rc={"axes.facecolor": "#f8fafc", "figure.facecolor": "#f8fafc"})
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Scatter plot data points dengan palet modern
            scatter = sns.scatterplot(
                data=df_kopi, x=col_lon, y=col_lat, hue='Cluster', 
                palette='viridis', s=70, alpha=0.6, edgecolor='white', linewidth=0.5, ax=ax
            )
            
            # Plot Centroids dengan marker yang mencolok dan efek "glow" sederhana
            ax.scatter(
                centroids[:, 1], centroids[:, 0], 
                s=300, c='#f43f5e', marker='X', 
                label='Pusat Klaster (Centroid)', 
                edgecolors='white', linewidths=2.5, zorder=5
            )
            
            plt.title("Distribusi Geografis Gerai Kopi", fontsize=16, fontweight='800', color='#0f172a', pad=20)
            plt.xlabel(f"Longitude ({col_lon})", fontsize=11, color='#64748b', fontweight='600')
            plt.ylabel(f"Latitude ({col_lat})", fontsize=11, color='#64748b', fontweight='600')
            
            # Legend kustom yang bersih
            legend = ax.legend(title="Klaster", title_fontsize=11, fontsize=10, frameon=True, 
                             facecolor='white', edgecolor='#e2e8f0', loc='upper right', bbox_to_anchor=(1.15, 1))
            plt.setp(legend.get_texts(), color='#334155')
            plt.setp(legend.get_title(), color='#0f172a', fontweight='700')
            
            # Grid yang halus
            ax.grid(True, linestyle='--', alpha=0.3, color='#94a3b8')
            
            # Hapus spine untuk tampilan bersih
            for spine in ax.spines.values():
                spine.set_color('#e2e8f0')
                spine.set_linewidth(1)
                
            st.pyplot(fig)
            sns.reset_orig()

        with col_panel:
            st.subheader("📊 Ringkasan Klaster")
            
            # Metric jumlah klaster
            n_clusters = len(np.unique(kmeans.labels_))
            st.metric("Total Klaster Terbentuk", n_clusters, delta=f"{len(df_kopi)} Total Data", delta_color="off")
            
            # Bar chart distribusi
            cluster_counts = df_kopi['Cluster'].value_counts().sort_index()
            st.markdown("##### Distribusi Anggota Klaster")
            st.bar_chart(cluster_counts, color="#8b5cf6", height=200)
            
            st.divider()
            
            st.subheader("🔍 Simulasi Lokasi Baru")
            st.caption("Uji koordinat untuk melihat potensi zonanya.")
            
            in_lat = st.number_input("Latitude", value=float(X_kopi[col_lat].mean()), format="%.6f")
            in_lon = st.number_input("Longitude", value=float(X_kopi[col_lon].mean()), format="%.6f")
            
            if st.button("Analisis Zona", type="primary", use_container_width=True):
                pred_cluster = int(kmeans.predict([[in_lat, in_lon]])[0])
                
                # Hitung jarak ke centroid terdekat untuk "skor kepadatan"
                distances = np.linalg.norm(centroids - [in_lat, in_lon], axis=1)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                st.markdown(f"<div style='background: #f1f5f9; padding: 12px; border-radius: 8px; text-align: center; margin-bottom: 12px;'><b>Hasil:</b> Masuk ke <span style='color: #2563eb; font-size: 1.2rem;'>Klaster {pred_cluster}</span></div>", unsafe_allow_html=True)
                
                # Logika Bisnis Sederhana
                min_cluster = cluster_counts.idxmin()
                
                if pred_cluster == min_cluster or min_dist > (max_dist * 0.7):
                    st.warning("⚠️ **ZONA SEPI**\n\nKepadatan gerai di area ini rendah. Potensi *foot traffic* alami kecil, namun bisa menjadi peluang *blue ocean* jika ada target pasar spesifik (misal: area perumahan baru).")
                else:
                    st.success("✅ **ZONA RAMAI / POTENSIAL**\n\nArea ini berada di dekat pusat klaster yang padat. Persaingan tinggi, tetapi validasi pasar sudah terbukti dan *foot traffic* terjaga.")

    except FileNotFoundError as e:
        st.error(f"❌ **File tidak ditemukan:** `{e.filename}`. Pastikan file dataset dan model `.pkl` tersedia di direktori yang sama.")
    except Exception as e:
        st.error(f"Terjadi kesalahan pada modul Clustering: {e}")
