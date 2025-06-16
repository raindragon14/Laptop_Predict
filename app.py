# app.py

import streamlit as st
import pandas as pd

# Konfigurasi halaman
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# Judul dan deskripsi
st.title("💻 Dashboard Prediksi Harga Laptop")
st.markdown("""
Aplikasi ini dirancang untuk menganalisis data laptop dan memprediksi harganya 
berdasarkan fitur-fitur yang ada.

**Gunakan menu di samping untuk navigasi:**
- **Exploratory Data Analysis**: Untuk melihat wawasan dari data.
- **Model Training**: Untuk melatih model machine learning.
- **Prediction**: Untuk memprediksi harga laptop baru.
""")

# Menampilkan data mentah
st.header("Cuplikan Dataset Harga Laptop")
try:
    df = pd.read_csv('laptop_prices.csv', encoding='latin-1')
    st.dataframe(df.head())
    st.info(f"Dataset ini memiliki **{df.shape[0]} baris** dan **{df.shape[1]} kolom**.")
except FileNotFoundError:
    st.error("File 'laptop_prices.csv' tidak ditemukan. Pastikan file tersebut berada di direktori yang sama dengan `app.py`.")

st.sidebar.success("Pilih halaman di atas.")
