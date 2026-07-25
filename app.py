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
# 1. KONFIGURASI HALAMAN & CSS PREMIUM
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
    /* Import Font Modern */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Font */
    .stApp, .stMarkdown, .stTextInput, .stNumberInput, .stSelectbox {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    
    /* Card Styling untuk Hasil */
    .pred-card {
        padding: 24px;
        border-radius: 12px;
        margin-top: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 6px solid;
    }
    .pred-positive {
        background-color: #fef2f2;
        border-left-color: #ef4444;
    }
    .pred-negative {
        background-color: #f0fdf4;
        border-left-color: #22c55e;
    }
    .pred-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .pred-positive .pred-title { color: #991b1b; }
    .pred-negative .pred-title { color: #166534; }
    .pred-desc { font-size: 0.95rem; color: #475569; line-height: 1.5; }

    /* Metric Card Override */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar Polish */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Button Polish */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. NAVIGASI SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("### 📊 DataMining Pro")
    st.markdown("<p style='color: #64748b; font-size: 0.9rem;'>Dashboard Analisis & Prediksi Cerdas</p>", unsafe_allow_html=True)
    st.divider()
    
    page = st.radio(
        "Pilih Modul Analisis:", 
        ["🩺 Prediksi Diabetes", "☕ Clustering Gerai Kopi"],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.markdown("#### ℹ️ Tentang Aplikasi")
    st.caption("Dibangun dengan Streamlit, Scikit-Learn, dan Seaborn. Memanfaatkan algoritma KNN, Naïve Bayes, Decision Tree, dan K-Means.")

# ==============================================================================
# 3. HALAMAN 1: PREDIKSI DIABETES
# ==============================================================================
if page == "🩺 Prediksi Diabetes":
    st.markdown('<h1 class="main-header">🩺 Prediksi Risiko Diabetes</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluasi risiko diabetes berdasarkan 8 indikator kesehatan utama menggunakan ensemble model machine learning.</p>', unsafe_allow_html=True)
    
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
            st.markdown("##### Visualisasi Confusion Matrix")
            col_cm1, col_cm2 = st.columns([1, 2])
            with col_cm1:
                selected_cm = st.selectbox("Pilih Model:", list(models.keys()), key="cm_select")
            with col_cm2:
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm_dict[selected_cm], annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Negatif (0)', 'Positif (1)'], 
                            yticklabels=['Negatif (0)', 'Positif (1)'], 
                            ax=ax, cbar=False, linewidths=.5)
                plt.title(f"Confusion Matrix: {selected_cm}", fontsize=12, fontweight='bold')
                plt.ylabel('Kondisi Aktual', fontsize=10)
                plt.xlabel('Hasil Prediksi', fontsize=10)
                st.pyplot(fig)

        with tab_pred:
            st.subheader("Formulir Input Data Klinis")
            st.info("💡 Geser slider di bawah ini sesuai dengan hasil pemeriksaan medis pasien.")
            
            selected_model = st.selectbox("Algoritma yang Digunakan:", list(models.keys()), index=2) # Default DT
            
            # Pengelompokan Input agar lebih rapi
            st.markdown("##### 📏 Data Fisik & Demografi")
            col1, col2, col3 = st.columns(3)
            with col1:
                inp_pregnancies = st.slider("Kehamilan (Pregnancies)", 0, 20, int(X['Pregnancies'].median()), help="Jumlah keer hamil")
                inp_age = st.slider("Usia (Age)", 10, 100, int(X['Age'].median()), help="Usia pasien dalam tahun")
            with col2:
                inp_bmi = st.slider("BMI (Body Mass Index)", 0.0, 70.0, float(X['BMI'].median()), 0.1, help="Indeks massa tubuh (kg/m²)")
            
            st.markdown("##### 🩸 Data Medis & Laboratorium")
            col3, col4, col5 = st.columns(3)
            with col3:
                inp_glucose = st.slider("Glukosa (Glucose)", 0, 200, int(X['Glucose'].median()), help="Kadar glukosa plasma (mg/dL)")
                inp_bp = st.slider("Tekanan Darah (BloodPressure)", 0, 150, int(X['BloodPressure'].median()), help="Tekanan darah diastolik (mm Hg)")
            with col4:
                inp_skin = st.slider("Ketebalan Kulit (SkinThickness)", 0, 100, int(X['SkinThickness'].median()), help="Ketebalan kulit triceps (mm)")
                inp_insulin = st.slider("Insulin", 0, 900, int(X['Insulin'].median()), help="Kadar insulin darah (mu U/ml)")
            with col5:
                inp_dpf = st.slider("Fungsi Silsilah Diabetes (DPF)", 0.0, 3.0, float(X['DiabetesPedigreeFunction'].median()), 0.01, help="Faktor risiko genetik")
            
            st.divider()
            
            # Tombol Prediksi
            if st.button("🚀 Analisis Risiko Sekarang", type="primary", use_container_width=True):
                with st.spinner("Sedang memproses data melalui model Machine Learning..."):
                    input_array = np.array([[inp_pregnancies, inp_glucose, inp_bp, inp_skin, inp_insulin, inp_bmi, inp_dpf, inp_age]])
                    prediction = models[selected_model].predict(input_array)[0]
                    proba = models[selected_model].predict_proba(input_array)[0][1] if hasattr(models[selected_model], "predict_proba") else 0.0
                    
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="pred-card pred-positive">
                            <div class="pred-title">⚠️ HASIL: POSITIF DIABETES (Risiko Tinggi)</div>
                            <div class="pred-desc">
                                Berdasarkan data yang dimasukkan, model <b>{selected_model}</b> memprediksi pasien mengidap diabetes. 
                                {"(Tingkat keyakinan model: {:.1f}%)".format(proba*100) if proba > 0 else ""}<br><br>
                                <b>Rekomendasi:</b> Segera konsultasikan dengan dokter untuk tes HbA1c lanjutan dan evaluasi pola makan.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="pred-card pred-negative">
                            <div class="pred-title">✅ HASIL: NEGATIF DIABETES (Risiko Rendah)</div>
                            <div class="pred-desc">
                                Berdasarkan data yang dimasukkan, model <b>{selected_model}</b> memprediksi pasien <b>tidak</b> mengidap diabetes. 
                                {"(Tingkat keyakinan model: {:.1f}%)".format((1-proba)*100) if proba > 0 else ""}<br><br>
                                <b>Rekomendasi:</b> Pertahankan pola hidup sehat, olahraga teratur, dan cek kesehatan rutin tahunan.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        with tab_info:
            st.subheader("Statistik Deskriptif Dataset")
            st.dataframe(df_diab.describe().round(2), use_container_width=True)
            st.markdown("##### Korelasi Antar Fitur")
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            sns.heatmap(df_diab.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax_corr)
            st.pyplot(fig_corr)

    except FileNotFoundError as e:
        st.error(f"❌ File tidak ditemukan: `{e.filename}`. Pastikan file dataset dan model `.pkl` berada di folder yang sama dengan script ini.")
    except Exception as e:
        st.error(f"Terjadi kesalahan tak terduga: {e}")

# ==============================================================================
# 4. HALAMAN 2: CLUSTERING GERAI KOPI
# ==============================================================================
elif page == "☕ Clustering Gerai Kopi":
    st.markdown('<h1 class="main-header">☕ Analisis Sebaran & Zonasi Gerai Kopi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Segmentasi lokasi berbasis koordinat geografis menggunakan K-Means untuk identifikasi zona bisnis potensial.</p>', unsafe_allow_html=True)
    
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
        col_map, col_panel = st.columns([2, 1])
        
        with col_map:
            st.subheader("📍 Peta Sebaran Klaster")
            
            # Styling Plot agar terlihat seperti Dashboard Premium
            sns.set_theme(style="whitegrid", rc={"grid.linestyle": "--", "grid.alpha": 0.3})
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f8fafc')
            ax.set_facecolor('#ffffff')
            
            # Scatter plot data points
            scatter = sns.scatterplot(
                data=df_kopi, x=col_lon, y=col_lat, hue='Cluster', 
                palette='viridis', s=60, alpha=0.6, edgecolor=None, ax=ax
            )
            
            # Plot Centroids dengan marker yang mencolok
            ax.scatter(
                centroids[:, 1], centroids[:, 0], 
                s=250, c='#ef4444', marker='X', 
                label='Pusat Klaster (Centroid)', 
                edgecolors='white', linewidths=2, zorder=5
            )
            
            plt.title("Distribusi Geografis Gerai Kopi", fontsize=14, fontweight='bold', color='#1e293b', pad=15)
            plt.xlabel(f"Longitude ({col_lon})", fontsize=10, color='#64748b')
            plt.ylabel(f"Latitude ({col_lat})", fontsize=10, color='#64748b')
            
            # Legend kustom
            legend = ax.legend(title="Klaster", title_fontsize=11, fontsize=10, frameon=True, facecolor='white', edgecolor='#e2e8f0')
            plt.setp(legend.get_texts(), color='#334155')
            
            sns.despine(left=True, bottom=True) # Hapus spine kiri dan bawah untuk tampilan bersih
            st.pyplot(fig)
            
            # Reset theme
            sns.reset_orig()

        with col_panel:
            st.subheader("📊 Ringkasan Klaster")
            
            # Metric jumlah klaster
            n_clusters = len(np.unique(kmeans.labels_))
            st.metric("Total Klaster Terbentuk", n_clusters)
            
            # Bar chart distribusi
            cluster_counts = df_kopi['Cluster'].value_counts().sort_index()
            st.markdown("##### Distribusi Anggota Klaster")
            st.bar_chart(cluster_counts, color="#8b5cf6")
            
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
                
                st.markdown(f"**Hasil:** Masuk ke **Klaster {pred_cluster}**")
                
                # Logika Bisnis Sederhana (Sesuaikan dengan konteks data Anda)
                # Asumsi: Klaster dengan anggota paling sedikit = Zona Sepi, atau bisa disesuaikan
                min_cluster = cluster_counts.idxmin()
                
                if pred_cluster == min_cluster or min_dist > (np.max(distances) * 0.8):
                    st.warning("⚠️ **ZONA SEPI**\n\nKepadatan gerai di area ini rendah. Potensi *foot traffic* alami kecil, namun bisa menjadi peluang *blue ocean* jika ada target pasar spesifik.")
                else:
                    st.success("✅ **ZONA RAMAI / POTENSIAL**\n\nArea ini berada di dekat pusat klaster yang padat. Persaingan tinggi, tetapi validasi pasar sudah terbukti.")

    except FileNotFoundError as e:
        st.error(f"❌ File tidak ditemukan: `{e.filename}`. Pastikan file dataset dan model `.pkl` tersedia.")
    except Exception as e:
        st.error(f"Terjadi kesalahan pada modul Clustering: {e}")
