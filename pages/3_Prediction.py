import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import re

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi Harga Laptop", page_icon="💻", layout="wide")

st.title("💻 Prediksi Harga Laptop")
st.markdown("Isi detail spesifikasi di bawah ini untuk mendapatkan estimasi harga laptop impian Anda.")

# --- Definisi Fungsi dan Konstanta ---
MODEL_PATH = 'final_model_pipeline.joblib'
DATA_PATH = 'laptop_prices.csv'

# Fungsi untuk memproses storage (sama seperti di skrip training)
def process_memory(memory_str):
    memory_str = str(memory_str).strip().lower()
    if '+' in memory_str:
        memory_str = memory_str.split('+')[0].strip()
    storage_type = 'Other'
    if 'ssd' in memory_str: storage_type = 'SSD'
    elif 'hdd' in memory_str: storage_type = 'HDD'
    elif 'flash' in memory_str: storage_type = 'Flash Storage'
    elif 'hybrid' in memory_str: storage_type = 'Hybrid'
    size_gb = 0
    num_match = re.search(r'(\d+\.?\d*)', memory_str)
    if num_match:
        size = float(num_match.group(1))
        size_gb = size * 1000 if 'tb' in memory_str else size
    return pd.Series([size_gb, storage_type])

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model '{MODEL_PATH}' tidak ditemukan. Harap latih model terlebih dahulu di halaman 'Model Training'.")
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Fungsi untuk memuat dan memproses data untuk opsi input
@st.cache_data
def get_data_options():
    if not os.path.exists(DATA_PATH):
        st.error(f"File data '{DATA_PATH}' tidak ditemukan.")
        return None
    try:
        df = pd.read_csv(DATA_PATH, encoding='latin-1')
        # Lakukan feature engineering yang sama persis seperti di training
        df['CPU_company'] = df['Cpu'].apply(lambda x: str(x).split()[0])
        df['CPU_freq'] = df['Cpu'].str.extract(r'(\d\.?\d*)GHz').astype(float).fillna(df['Cpu'].str.extract(r'(\d\.?\d*)GHz').astype(float).median())
        df['GPU_company'] = df['Gpu'].apply(lambda x: str(x).split()[0])
        storage_features = df['Memory'].apply(process_memory)
        storage_features.columns = ['PrimaryStorage', 'PrimaryStorageType']
        df = pd.concat([df, storage_features], axis=1)
        df.dropna(subset=['ScreenResolution'], inplace=True)
        screen_res = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)')
        df['ScreenW'] = screen_res[0].astype(int)
        df['ScreenH'] = screen_res[1].astype(int)
        if df['Ram'].dtype == 'object':
            df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('int32')
        return df
    except Exception as e:
        st.error(f"Gagal memproses data untuk opsi input: {e}")
        return None

# --- Logika Utama Aplikasi ---
model = load_model()
df_options = get_data_options()

if model and df_options is not None:
    with st.form("prediction_form"):
        st.header("Spesifikasi Utama")
        col1, col2, col3 = st.columns(3)
        with col1:
            company = st.selectbox("Brand", options=sorted(df_options['Company'].unique()))
            type_name = st.selectbox("Tipe Laptop", options=sorted(df_options['TypeName'].unique()))
            ram = st.selectbox("RAM (GB)", options=sorted(df_options['Ram'].unique()))
        with col2:
            opsys = st.selectbox("Sistem Operasi", options=sorted(df_options['OS'].unique()))
            weight = st.slider("Berat (kg)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
            inches = st.slider("Ukuran Layar (Inci)", min_value=10.0, max_value=18.0, value=15.6, step=0.1)
        with col3:
            cpu_company = st.selectbox("Brand CPU", options=sorted(df_options['CPU_company'].unique()))
            gpu_company = st.selectbox("Brand GPU", options=sorted(df_options['GPU_company'].unique()))
            cpu_freq = st.slider("Frekuensi CPU (GHz)", min_value=0.9, max_value=3.6, value=2.5, step=0.1)

        st.header("Penyimpanan & Layar")
        col4, col5, col6 = st.columns(3)
        with col4:
            storage_type = st.selectbox("Tipe Penyimpanan Utama", options=sorted(df_options['PrimaryStorageType'].unique()))
            primary_storage = st.select_slider("Kapasitas Penyimpanan (GB)", options=sorted(df_options[df_options['PrimaryStorage'] > 0]['PrimaryStorage'].unique()), value=256)
        with col5:
            screen_resolution = st.selectbox("Resolusi Layar", options=sorted(df_options['ScreenResolution'].unique()))
            # Ekstrak ScreenW dan ScreenH dari pilihan
            res_match = re.search(r'(\d+)x(\d+)', screen_resolution)
            screen_w = int(res_match.group(1)) if res_match else 1920
            screen_h = int(res_match.group(2)) if res_match else 1080
        with col6:
            touchscreen_str = st.radio("Layar Sentuh (Touchscreen)", options=['No', 'Yes'], horizontal=True)
            ips_str = st.radio("Panel IPS", options=['No', 'Yes'], horizontal=True)
            # Konversi input string ke numerik (0 atau 1)
            touchscreen = 1 if touchscreen_str == 'Yes' else 0
            ips = 1 if ips_str == 'Yes' else 0

        submit_button = st.form_submit_button(label="Prediksi Harga", type="primary", use_container_width=True)

    if submit_button:
        # Kumpulkan semua data ke dalam DataFrame dengan urutan kolom yang benar
        input_data = pd.DataFrame({
            'Company': [company], 'TypeName': [type_name], 'Ram': [ram],
            'Weight': [weight], 'OS': [opsys], 'Inches': [inches],
            'CPU_company': [cpu_company], 'CPU_freq': [cpu_freq],
            'GPU_company': [gpu_company], 'PrimaryStorage': [primary_storage],
            'PrimaryStorageType': [storage_type], 'ScreenW': [screen_w],
            'ScreenH': [screen_h], 'Touchscreen': [touchscreen], 'IPSpanel': [ips]
        })

        st.info("Data Input Anda:")
        st.dataframe(input_data)

        with st.spinner("Model sedang menganalisis spesifikasi..."):
            try:
                # Prediksi harga dalam skala logaritmik
                log_prediction = model.predict(input_data)
                # Transformasi balik ke skala harga asli (Juta Rupiah)
                predicted_price = np.expm1(log_prediction[0])

                st.balloons()
                st.success(f"## **Estimasi Harga: Rp {predicted_price:,.2f} Juta**")
            except Exception as e:
                st.error(f"Terjadi error saat melakukan prediksi: {e}")
