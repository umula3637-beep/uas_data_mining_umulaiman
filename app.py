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
    page_title="Data Mining Workspace", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================
# CUSTOM CSS - GEMINI STYLING THEME
# ====================================================
st.markdown("""
    <style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #0e0e11;
        color: #e3e2e6;
    }

    /* Custom Header Tag Gemini */
    .gemini-badge {
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, #1e1f29 0%, #171821 100%);
        border: 1px solid #2e303d;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        color: #a8c7fa;
        font-weight: 600;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    /* Card Container */
    .gemini-card {
        background: #171821;
        border: 1px solid #282a36;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    
    .gemini-card:hover {
        border-color: #44475a;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }

    /* Gradient Gradient Titles */
    .gemini-title {
        background: linear-gradient(90deg, #a8c7fa 0%, #7cacf8 50%, #d3e3fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #13131a !important;
        border-right: 1px solid #232430;
    }

    /* Streamlit Input Customization */
    div[data-baseweb="input"] {
        border-radius: 12px !important;
        background-color: #1e1f2b !important;
        border: 1px solid #2e303d !important;
        color: #e3e2e6 !important;
    }
    
    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        background-color: #1e1f2b !important;
        border: 1px solid #2e303d !important;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #a8c7fa 0%, #669df6 100%);
        color: #002d6c;
        border: none;
        border-radius: 24px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 157, 246, 0.2);
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #c2e7ff 0%, #7cacf8 100%);
        box-shadow: 0 6px 20px rgba(102, 157, 246, 0.4);
        transform: translateY(-2px);
        color: #001d4a;
    }

    /* Custom Alert Boxes */
    .status-positive {
        background-color: rgba(242, 184, 181, 0.12);
        border: 1px solid #f2b8b5;
        color: #f2b8b5;
        padding: 16px;
        border-radius: 12px;
        font-weight: 500;
    }

    .status-negative {
        background-color: rgba(196, 238, 212, 0.12);
        border: 1px solid #c4eed4;
        color: #c4eed4;
        padding: 16px;
        border-radius: 12px;
        font-weight: 500;
    }

    /* Matplotlib Figure Dark Background Override */
    </style>
""", unsafe_allow_html=True)

# Helper function untuk plotting ala Dark Theme
def apply_dark_theme_plt():
    plt.style.use("dark_background")
    fig_color = "#171821"
    plt.rcParams['figure.facecolor'] = fig_color
    plt.rcParams['axes.facecolor'] = fig_color
    plt.rcParams['text.color'] = '#e3e2e6'
    plt.rcParams['axes.labelcolor'] = '#a8c7fa'
    plt.rcParams['xtick.color'] = '#e3e2e6'
    plt.rcParams['ytick.color'] = '#e3e2e6'
    plt.rcParams['grid.color'] = '#282a36'

# ====================================================
# NAVIGASI SIDEBAR
# ====================================================
with st.sidebar:
    st.markdown("### ✨ Gemini Mining Workspace")
    st.caption("Pilih ruang kerja analisis data Anda:")
    
    page = st.radio(
        "",
        ["1. Prediksi Diabetes (Klasifikasi)", "2. Clustering Gerai Kopi (K-Means)"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
        <div style='font-size: 0.8rem; color: #8e919e; text-align: center;'>
            Powered by Streamlit & Scikit-Learn<br>© 2026 Analytics Dashboard
        </div>
    """, unsafe_allow_html=True)

# ====================================================
# HALAMAN 1: KLASIFIKASI DIABETES
# ====================================================
if page == "1. Prediksi Diabetes (Klasifikasi)":
    
    st.markdown('<div class="gemini-badge">🩺 Healthcare Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="gemini-title">Prediksi Risiko Diabetes Pasien</h1>', unsafe_allow_html=True)
    st.write("Analisis prediktif berbasis machine learning untuk membantu mendeteksi indikasi diabetes secara dini.")
    st.markdown("<br>", unsafe_allow_html=True)

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
    
    # Layout Split: Kiri Performance, Kanan Visualisasi
    col_left, col_right = st.columns([1.1, 0.9])
    
    with col_left:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Performa Model Klasifikasi")
        st.caption("Perbandingan metrik evaluasi dari 3 algoritma Machine Learning")
        
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
        selected_eval_model = st.selectbox("Pilih Model:", list(models.keys()), key="cm_select")
        
        apply_dark_theme_plt()
        cm = confusion_matrix(y_test, models[selected_eval_model].predict(X_test))
        fig, ax = plt.subplots(figsize=(4, 2.8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, annot_kws={"size": 12})
        plt.xlabel('Prediksi Model', fontsize=9)
        plt.ylabel('Kondisi Aktual', fontsize=9)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # Section Input Pasien Baru
    st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Simulasi Prediksi Pasien Baru")
    st.caption("Masukkan parameter klinis pasien di bawah ini:")
    
    selected_model_name = st.selectbox("Pilih Model Algoritma Eksekusi:", list(models.keys()), key="pred_select")
    st.markdown("<br>", unsafe_allow_html=True)
    
    feature_names = X.columns.tolist()
    input_values = []
    
    # Dynamic Form Input Grid 3 Kolom
    cols = st.columns(3)
    for i, col_name in enumerate(feature_names):
        default_val = float(X[col_name].median())
        min_val = float(X[col_name].min())
        max_val = float(X[col_name].max())
        
        with cols[i % 3]:
            val = st.number_input(f"{col_name}", value=default_val, min_value=min_val, max_value=max_val)
            input_values.append(val)
            
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✨ Jalankan Prediksi Risiko"):
        chosen_model = models[selected_model_name]
        prediction = chosen_model.predict([input_values])[0]
        
        if prediction == 1:
            st.markdown("""
                <div class="status-positive">
                    ⚠️ <b>HASIL PREDIKSI: POSITIF RISIKO DIABETES</b><br>
                    <small>Sistem mendeteksi pola indikasi risiko diabetes tinggi berdasarkan data yang dimasukkan.</small>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="status-negative">
                    ✅ <b>HASIL PREDIKSI: NEGATIF RISIKO DIABETES</b><br>
                    <small>Sistem mendeteksi parameter kesehatan pasien berada dalam batas aman.</small>
                </div>
            """, unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

# ====================================================
# HALAMAN 2: CLUSTERING GERAI KOPI
# ====================================================
elif page == "2. Clustering Gerai Kopi (K-Means)":
    
    st.markdown('<div class="gemini-badge">☕ Spatial Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="gemini-title">Analisis Sebaran Gerai Kopi & Deteksi Zona Sepi</h1>', unsafe_allow_html=True)
    st.write("Mengelompokkan titik lokasi gerai menggunakan Unsupervised Learning (K-Means) untuk rekomendasi ekspansi bisnis.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    df_kopi = pd.read_csv('lokasi_gerai_kopi_clean.csv')
    kmeans = joblib.load('model_kmeans.pkl')
    
    col_lat = [c for c in df_kopi.columns if any(k in c.lower() for k in ['lat', 'y', 'lintang'])][0]
    col_lon = [c for c in df_kopi.columns if any(k in c.lower() for k in ['lon', 'long', 'lng', 'x', 'bujur'])][0]
    
    X_kopi = df_kopi[[col_lat, col_lon]]
    df_kopi['Cluster'] = kmeans.labels_
    
    # Layout Map & Form Input Side-by-side
    col_map, col_form = st.columns([1.2, 0.8])
    
    with col_map:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 📍 Visualisasi Persebaran Klaster")
        
        apply_dark_theme_plt()
        fig, ax = plt.subplots(figsize=(8, 5.2))
        sns.scatterplot(
            data=df_kopi, 
            x=col_lon, 
            y=col_lat, 
            hue='Cluster', 
            palette='cool', 
            s=80, 
            ax=ax,
            edgecolor='#282a36'
        )
        plt.title("Peta Koordinat Klaster Lokasi Gerai Kopi", fontsize=11, color='#a8c7fa')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_form:
        st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Cek Potensi Lokasi Baru")
        st.caption("Masukkan koordinat calon lokasi gerai baru:")
        
        in_lat = st.number_input(f"Latitude ({col_lat})", value=float(X_kopi[col_lat].mean()))
        in_lon = st.number_input(f"Longitude ({col_lon})", value=float(X_kopi[col_lon].mean()))
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📍 Analisis Zonasi Lokasi"):
            pred_cluster = kmeans.predict([[in_lat, in_lon]])[0]
            
            st.markdown(f"""
                <div style="background-color: #1e1f2b; padding: 12px; border-radius: 8px; margin-bottom: 12px; border: 1px solid #2e303d;">
                    Teridentifikasi pada <b>Klaster {pred_cluster}</b>
                </div>
            """, unsafe_allow_html=True)
            
            if pred_cluster == 0:
                st.markdown("""
                    <div class="status-positive">
                        ⚠️ <b>STATUS: BERADA DI ZONA SEPI</b><br>
                        <small>Kepadatan gerai rendah. Potensi persaingan minim namun memerlukan riset pasar lebih mendalam.</small>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="status-negative">
                        ✅ <b>STATUS: BERADA DI ZONA RAMAI</b><br>
                        <small>Area dengan akumulasi gerai tinggi. Menandakan demand pasar tinggi namun kompetisi lebih ketat.</small>
                    </div>
                """, unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)
