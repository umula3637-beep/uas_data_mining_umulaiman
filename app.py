import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

# --- KONFIGURASI HALAMAN & CSS CUSTOM ---
st.set_page_config(
    page_title="Proyek Data Mining Pro", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
<style>
    /* Font Global */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Card Style untuk Hasil Prediksi */
    .result-card-pos {
        padding: 20px;
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        border-radius: 5px;
        margin-top: 10px;
    }
    .result-card-neg {
        padding: 20px;
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
        border-radius: 5px;
        margin-top: 10px;
    }
    .result-text {
        font-size: 1.2rem;
        font-weight: bold;
        color: #155724;
    }
    .result-text-neg {
        font-size: 1.2rem;
        font-weight: bold;
        color: #721c24;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# NAVIGASI SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920323.png", width=100) # Placeholder Icon
    st.title("📌 Menu Navigasi")
    st.markdown("---")
    page = st.radio("Pilih Halaman Proyek:", [
        "1. Prediksi Diabetes (Klasifikasi)", 
        "2. Clustering Gerai Kopi (K-Means)"
    ], index=0)
    
    st.markdown("---")
    st.caption("© 2026 Proyek Data Mining")

# ====================================================
# HALAMAN 1: KLASIFIKASI DIABETES
# ====================================================
if page == "1. Prediksi Diabetes (Klasifikasi)":
    st.markdown('<h1 class="main-header">🩺 Prediksi Risiko Diabetes</h1>', unsafe_allow_html=True)
    st.write("Aplikasi ini menggunakan **Machine Learning** untuk memprediksi status diabetes pasien berdasarkan data medis.")
    
    try:
        # Load Data & Model
        df_diab = pd.read_csv('diabetes.csv')
        
        # Otomatis cari nama kolom target
        target_col = None
        for col in df_diab.columns:
            if col.lower().strip() in ['outcome', 'target', 'class', 'diabetes']:
                target_col = col
                break
        if not target_col:
            target_col = df_diab.columns[-1]

        X = df_diab.drop(target_col, axis=1)
        y = df_diab[target_col]
        
        # Split data hanya untuk evaluasi (tidak perlu di-load setiap kali jika model sudah jadi, tapi untuk demo kita biarkan)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Load 3 model pkl
        knn = joblib.load('model_knn.pkl')
        nb = joblib.load('model_nb.pkl')
        dt = joblib.load('model_dt.pkl')
        
        models = {'KNN': knn, 'Naïve Bayes': nb, 'Decision Tree': dt}
        
        # --- TAB LAYOUT ---
        tab1, tab2, tab3 = st.tabs(["📊 Evaluasi Model", "🔍 Prediksi Pasien Baru", "ℹ️ Info Dataset"])
        
        with tab1:
            st.subheader("Performa Algoritma")
            
            # Hitung Metrik
            metrics_data = []
            cm_dict = {}
            for name, model in models.items():
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                metrics_data.append({
                    'Algoritma': name,
                    'Akurasi': round(acc, 3),
                    'Precision': round(prec, 3),
                    'Recall': round(rec, 3),
                    'F1-Score': round(f1, 3)
                })
                cm_dict[name] = confusion_matrix(y_test, y_pred)
            
            df_metrics = pd.DataFrame(metrics_data)
            
            # Tampilkan Metric Cards untuk Akurasi
            cols = st.columns(len(models))
            for i, row in df_metrics.iterrows():
                with cols[i]:
                    st.metric(label=f"Akurasi {row['Algoritma']}", value=f"{row['Akurasi']*100:.1f}%")
            
            with st.expander("Lihat Detail Metrik Lengkap"):
                st.dataframe(df_metrics.style.highlight_max(axis=0, subset=['Akurasi', 'Precision', 'Recall', 'F1-Score']), use_container_width=True)
            
            # Confusion Matrix
            st.markdown("### Visualisasi Confusion Matrix")
            selected_cm_model = st.selectbox("Pilih Model:", list(models.keys()), key="cm_select")
            
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm_dict[selected_cm_model], annot=True, fmt='d', cmap='coolwarm', 
                        xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'], ax=ax)
            plt.title(f"Confusion Matrix - {selected_cm_model}")
            plt.ylabel('Aktual')
            plt.xlabel('Prediksi')
            st.pyplot(fig)

        with tab2:
            st.subheader("Formulir Input Data Pasien")
            st.info("Masukkan data medis pasien di bawah ini untuk mendapatkan prediksi.")
            
            selected_model_name = st.selectbox("Gunakan Algoritma:", list(models.keys()), key="pred_select")
            
            feature_names = X.columns.tolist()
            input_values = []
            
            # Buat input dalam grid 3 kolom agar tidak terlalu panjang ke bawah
            cols_input = st.columns(3)
            for i, col_name in enumerate(feature_names):
                default_val = float(X[col_name].median())
                min_val = float(X[col_name].min())
                max_val = float(X[col_name].max())
                
                with cols_input[i % 3]:
                    val = st.number_input(
                        f"{col_name.replace('_', ' ').title()}", 
                        value=default_val, 
                        min_value=min_val, 
                        max_value=max_val,
                        step=0.1,
                        key=f"input_{col_name}"
                    )
                    input_values.append(val)
            
            st.markdown("---")
            if st.button("🚀 Jalankan Analisis Prediksi", type="primary", use_container_width=True):
                with st.spinner("Sedang menganalisis data..."):
                    chosen_model = models[selected_model_name]
                    prediction = chosen_model.predict([input_values])[0]
                    
                    # Tampilan Hasil
                    if prediction == 1:
                        st.markdown("""
                        <div class="result-card-pos">
                            <p class="result-text">⚠️ HASIL: POSITIF DIABETES</p>
                            <p>Pasien terindikasi memiliki risiko diabetes tinggi. Segera konsultasikan dengan dokter.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="result-card-neg">
                            <p class="result-text-neg">✅ HASIL: NEGATIF DIABETES</p>
                            <p>Pasien terindikasi sehat dari risiko diabetes. Tetap jaga pola hidup sehat.</p>
                        </div>
                        """, unsafe_allow_html=True)

        with tab3:
            st.write("Statistik Dasar Dataset:")
            st.dataframe(df_diab.describe())
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data/model: {e}")

# ====================================================
# HALAMAN 2: CLUSTERING GERAI KOPI
# ====================================================
elif page == "2. Clustering Gerai Kopi (K-Means)":
    st.markdown('<h1 class="main-header">☕ Analisis Sebaran Gerai Kopi</h1>', unsafe_allow_html=True)
    st.write("Mengelompokkan lokasi gerai kopi menggunakan **K-Means Clustering** untuk identifikasi zona potensial.")
    
    try:
        df_kopi = pd.read_csv('lokasi_gerai_kopi_clean.csv')
        kmeans = joblib.load('model_kmeans.pkl')
        
        # Cari nama kolom koordinat secara dinamis
        col_lat = [c for c in df_kopi.columns if any(k in c.lower() for k in ['lat', 'y', 'lintang'])][0]
        col_lon = [c for c in df_kopi.columns if any(k in c.lower() for k in ['lon', 'long', 'lng', 'x', 'bujur'])][0]
        
        X_kopi = df_kopi[[col_lat, col_lon]]
        df_kopi['Cluster'] = kmeans.labels_
        
        # --- VISUALISASI UTAMA ---
        st.subheader("Peta Sebaran Klaster")
        
        # Hitung centroid untuk ditampilkan di plot
        centroids = kmeans.cluster_centers_
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = sns.scatterplot(
            data=df_kopi, 
            x=col_lon, 
            y=col_lat, 
            hue='Cluster', 
            palette='viridis', 
            s=60, 
            alpha=0.7,
            ax=ax
        )
        
        # Plot Centroids (Titik Pusat Klaster)
        plt.scatter(centroids[:, 1], centroids[:, 0], s=200, c='red', marker='X', label='Centroid Klaster', edgecolors='black', linewidths=2)
        
        plt.title("Sebaran Lokasi Gerai Kopi Berdasarkan Klaster", fontsize=14)
        plt.xlabel(f"Longitude ({col_lon})")
        plt.ylabel(f"Latitude ({col_lat})")
        plt.legend(title="Klaster")
        plt.grid(True, linestyle='--', alpha=0.3)
        
        st.pyplot(fig)
        
        st.markdown("---")
        
        # --- FITUR CEK LOKASI BARU ---
        col_check, col_info = st.columns([2, 1])
        
        with col_check:
            st.subheader("🔍 Cek Potensi Lokasi Baru")
            st.caption("Masukkan koordinat untuk mengetahui klaster dan estimasi keramaian.")
            
            c1, c2 = st.columns(2)
            with c1:
                in_lat = st.number_input(f"Latitude ({col_lat})", value=float(X_kopi[col_lat].mean()), format="%.6f")
            with c2:
                in_lon = st.number_input(f"Longitude ({col_lon})", value=float(X_kopi[col_lon].mean()), format="%.6f")
                
            if st.button("📌 Analisis Lokasi Ini", type="primary", use_container_width=True):
                pred_cluster = int(kmeans.predict([[in_lat, in_lon]])[0])
                
                # Logika Zona (Asumsi: Klaster 0 = Sepi, lainnya = Ramai. Sesuaikan dengan data Anda)
                # Untuk lebih akurat, hitung jarak ke centroid terdekat atau density
                is_sepi = (pred_cluster == 0) 
                
                if is_sepi:
                    st.warning(f"**Klaster {pred_cluster}**: Zona Sepi 📉")
                    st.markdown("Area ini memiliki kepadatan gerai rendah. Cocok untuk ekspansi jika target pasar spesifik, namun traffic mungkin rendah.")
                else:
                    st.success(f"**Klaster {pred_cluster}**: Zona Ramai 📈")
                    st.markdown("Area ini merupakan pusat keramaian. Persaingan tinggi, namun potensi pelanggan juga besar.")
                
                # Tampilkan titik baru di peta sementara (opsional, bisa ditambahkan logika plot ulang)
        
        with col_info:
            st.subheader("Info Klaster")
            n_clusters = len(np.unique(kmeans.labels_))
            st.metric("Total Klaster", n_clusters)
            
            cluster_counts = df_kopi['Cluster'].value_counts().sort_index()
            st.bar_chart(cluster_counts)
            
    except Exception as e:
        st.error(f"Terjadi kesalahan pada modul Clustering: {e}")
