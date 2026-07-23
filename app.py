import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Config Halaman
st.set_page_config(page_title="Proyek Data Mining", layout="wide")

# NAVIGASI SIDEBAR
st.sidebar.title("📌 Menu Navigasi")
page = st.sidebar.radio("Pilih Halaman Proyek:", [
    "1. Prediksi Diabetes (Klasifikasi)", 
    "2. Clustering Gerai Kopi (K-Means)"
])

# ====================================================
# HALAMAN 1: KLASIFIKASI DIABETES
# ====================================================
if page == "1. Prediksi Diabetes (Klasifikasi)":
    st.title("🩺 Prediksi Risiko Diabetes Berdasarkan Data Pasien")
    st.write("Aplikasi ini memprediksi status risiko diabetes pasien menggunakan 3 algoritma Machine Learning.")
    st.markdown("---")
    
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load 3 model pkl
    knn = joblib.load('model_knn.pkl')
    nb = joblib.load('model_nb.pkl')
    dt = joblib.load('model_dt.pkl')
    
    models = {'KNN': knn, 'Naïve Bayes': nb, 'Decision Tree': dt}
    
    # 1. Metrik Evaluasi
    st.subheader("📊 Metrik Evaluasi Model")
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
    st.table(pd.DataFrame(metrics_list))
    
    # 2. Confusion Matrix Visual
    st.subheader("🧩 Confusion Matrix Model")
    selected_eval_model = st.selectbox("Pilih Model untuk melihat Confusion Matrix:", list(models.keys()))
    cm = confusion_matrix(y_test, models[selected_eval_model].predict(X_test))
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    st.pyplot(fig)
    
    st.markdown("---")
    # 3. Form Input Pasien Baru
    st.subheader("📝 Input Fitur Pasien untuk Prediksi")
    selected_model_name = st.selectbox("Pilih Algoritma Prediksi:", list(models.keys()))
    
    col1, col2 = st.columns(2)
    feature_names = X.columns.tolist()
    input_values = []
    
    for i, col_name in enumerate(feature_names):
        default_val = float(X[col_name].median())
        min_val = float(X[col_name].min())
        max_val = float(X[col_name].max())
        
        with col1 if i % 2 == 0 else col2:
            val = st.number_input(f"Masukkan {col_name}", value=default_val, min_value=min_val, max_value=max_val)
            input_values.append(val)
            
    if st.button("🔴 Jalankan Prediksi"):
        chosen_model = models[selected_model_name]
        prediction = chosen_model.predict([input_values])[0]
        
        if prediction == 1:
            st.error("⚠️ **PASIEN DIPREDIKSI MENGIDAP DIABETES (POSITIF)**")
        else:
            st.success("✅ **PASIEN DIPREDIKSI TIDAK MENGIDAP DIABETES (NEGATIF)**")

# ====================================================
# HALAMAN 2: CLUSTERING GERAI KOPI
# ====================================================
elif page == "2. Clustering Gerai Kopi (K-Means)":
    st.title("☕ Analisis Klaster Lokasi Gerai Kopi & Deteksi Zona Sepi")
    st.write("Mengelompokkan sebaran lokasi gerai kopi menggunakan K-Means untuk mendeteksi potensi **Zona Sepi**.")
    st.markdown("---")
    
    df_kopi = pd.read_csv('lokasi_gerai_kopi_clean.csv')
    kmeans = joblib.load('model_kmeans.pkl')
    
    # Cari nama kolom koordinat
    col_lat = [c for c in df_kopi.columns if any(k in c.lower() for k in ['lat', 'y', 'lintang'])][0]
    col_lon = [c for c in df_kopi.columns if any(k in c.lower() for k in ['lon', 'long', 'lng', 'x', 'bujur'])][0]
    
    X_kopi = df_kopi[[col_lat, col_lon]]
    df_kopi['Cluster'] = kmeans.labels_
    
    # Visualisasi
    st.subheader("📍 Scatter Plot Sebaran Klaster Gerai Kopi")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_kopi, x=col_lon, y=col_lat, hue='Cluster', palette='Set1', s=70, ax=ax)
    plt.title("Visualisasi Koordinat Klaster Lokasi Gerai Kopi")
    st.pyplot(fig)
    
    st.markdown("---")
    # Cek Lokasi Baru
    st.subheader("🔍 Cek Potensi Lokasi Gerai Baru")
    
    col_a, col_b = st.columns(2)
    with col_a:
        in_lat = st.number_input(f"Masukkan {col_lat}", value=float(X_kopi[col_lat].mean()))
    with col_b:
        in_lon = st.number_input(f"Masukkan {col_lon}", value=float(X_kopi[col_lon].mean()))
        
    if st.button("📌 Cek Klaster & Status Zona"):
        pred_cluster = kmeans.predict([[in_lat, in_lon]])[0]
        st.info(f"Lokasi koordinat ini tergolong ke dalam: **Klaster {pred_cluster}**")
        
        # Logika penetapan zona
        if pred_cluster == 0:
            st.warning("⚠️ Status: **BERADA DI ZONA SEPI (Potensi Pelanggan Rendah)**")
        else:
            st.success("✅ Status: **BERADA DI ZONA RAMAI (Potensi Pelanggan Tinggi)**")