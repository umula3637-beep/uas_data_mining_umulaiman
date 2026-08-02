# ☕🩺 Neural Mining Engine — Multi-Module Analytics Dashboard

Aplikasi web interaktif berbasis **Streamlit** yang menggabungkan dua solusi *Machine Learning* utama: **Klasifikasi Risiko Diabetes** dan **Clustering Geospasial Gerai Kopi**. Didesain dengan antarmuka futuristik (*Enterprise Glassmorphism UI*) untuk memberikan analisis data yang presisi, responsif, dan mudah dipahami.

---

## 👤 Informasi Pengembang

* **Nama** : UMUL AIMAN
* **NIM**  : 23146039

---

## 📌 Penjelasan Proyek

Proyek ini terdiri dari dua modul analisis data independen:

### 1. 🩺 Modul Prediksi Diabetes (Klasifikasi)
Solusi *healthcare intelligence* untuk mendukung keputusan medis awal dalam memprediksi risiko diabetes pada pasien berbasis parameter fisiologis.
* **Algoritma yang Digunakan**: K-Nearest Neighbors (KNN), Naïve Bayes, dan Decision Tree.
* **Fitur Utama**:
  * **Input Simulation**: Prediksi risiko secara real-time berdasarkan input manual parameter medis pasien.
  * **Batch Processing**: Pengolahan data pasien skala besar sekaligus melalui unggahan berkas `.csv`.
  * **Analytics & Benchmarks**: Perbandingan performa model (*Accuracy*, *Precision*, *Recall*, *F1-Score*), *Feature Importance*, dan visualisasi *Confusion Matrix*.

### 2. ☕ Modul Clustering Gerai Kopi (Geospasial)
Solusi *spatial intelligence* berbasis algoritma **K-Means Clustering** untuk memetakan distribusi lokasi gerai kopi, menganalisis kerapatan wilayah, dan mengevaluasi zonasi ekspansi bisnis.
* **Fitur Utama**:
  * **Peta Interaktif Spasial**: Visualisasi lokasi gerai berbasis peta digital (menggunakan *Folium*), mendukung *Heatmap Density*, *Cluster Markers*, dan *Grouped Clusters*.
  * **Evaluasi Lokasi Baru**: Simulasi titik koordinat (Latitude & Longitude) rencana gerai baru untuk menentukan kategori zona persaingan (*Blue Ocean* vs *High Density*).

---

## 🚀 Instruksi Menjalankan Aplikasi (Lokal)

Ikuti langkah-langkah berikut untuk menjalankan aplikasi di komputer lokal Anda:

### 1. Prasyarat (*Prerequisites*)
Pastikan Python (versi 3.8 ke atas) sudah terinstal di sistem Anda.

### 2. Kloning / Unduh Repositori
Masuk ke direktori proyek Anda melalui terminal / command prompt:
```bash
cd path/to/your/project
