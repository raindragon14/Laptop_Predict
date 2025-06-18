import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Prediksi Harga Laptop (Fine Tuned)", page_icon="💻", layout="wide")

st.title("💻 Prediksi Harga Laptop")
st.markdown("Isi detail spesifikasi di bawah ini untuk mendapatkan estimasi harga laptop impian Anda.")

MODEL_PATH = 'optimized_weighted_ensemble_laptop_price_model.joblib'
DATA_PATH = 'laptop_prices.csv'

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

@st.cache_data
def get_data_options():
    if not os.path.exists(DATA_PATH):
        st.error(f"File data '{DATA_PATH}' tidak ditemukan.")
        return None
    try:
        df = pd.read_csv(DATA_PATH, encoding='latin-1')
    
        if 'Ram' in df.columns and df['Ram'].dtype == 'object':
            df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('int32')

        return df
    except Exception as e:
        st.error(f"Gagal memproses data untuk opsi input: {e}")
        return None

@st.cache_data
def load_model_performance():
    """Load model performance metrics if available"""
    try:
        # Coba load metrics dari file jika ada
        metrics_path = 'model_performance_metrics.joblib'
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
            return metrics
        else:
            # Jika tidak ada file metrics, hitung dari data
            return calculate_model_performance()
    except Exception as e:
        st.warning(f"Tidak dapat memuat performa model: {e}")
        return None

@st.cache_data
def calculate_model_performance():
    """Calculate model performance on test data"""
    try:
        if not os.path.exists(DATA_PATH):
            return None
            
        # Load data
        df = pd.read_csv(DATA_PATH, encoding='latin-1')
        
        # Preprocessing data (sesuaikan dengan preprocessing yang digunakan saat training)
        if 'Ram' in df.columns and df['Ram'].dtype == 'object':
            df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('int32')
        
        # Assumsi: ada kolom target 'Price_euros' atau 'Price'
        target_col = 'Price_euros' if 'Price_euros' in df.columns else 'Price'
        if target_col not in df.columns:
            return None
            
        # Split data untuk evaluasi (menggunakan 20% data terakhir sebagai test)
        test_size = int(len(df) * 0.2)
        test_data = df.tail(test_size)
        
        # Prepare features (sesuaikan dengan feature yang digunakan model)
        feature_cols = ['Company', 'TypeName', 'Ram', 'Weight', 'OS', 'Inches',
                       'CPU_company', 'CPU_freq', 'GPU_company', 'PrimaryStorage',
                       'PrimaryStorageType', 'ScreenW', 'ScreenH', 'Touchscreen', 'IPSpanel']
        
        available_cols = [col for col in feature_cols if col in test_data.columns]
        X_test = test_data[available_cols]
        y_test = test_data[target_col]
        
        # Load model
        model = load_model()
        if model is None:
            return None
            
        # Make predictions
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)  # Convert from log scale
        y_true = y_test
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'n_samples': len(y_test)
        }
        
        return metrics
        
    except Exception as e:
        st.warning(f"Error menghitung performa model: {e}")
        return None

def display_model_performance():
    """Display model performance metrics"""
    st.header("📊 Performa Model")
    
    metrics = load_model_performance()
    
    if metrics is None:
        st.warning("Performa model tidak tersedia. Pastikan model sudah dilatih dan data tersedia.")
        return
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="R² Score",
            value=f"{metrics['r2']:.4f}",
            help="Koefisien determinasi - semakin mendekati 1 semakin baik"
        )
    
    with col2:
        st.metric(
            label="MAE (Juta)",
            value=f"{metrics['mae']:.2f}",
            help="Mean Absolute Error - rata-rata kesalahan absolut"
        )
    
    with col3:
        st.metric(
            label="RMSE (Juta)",
            value=f"{metrics['rmse']:.2f}",
            help="Root Mean Square Error - akar rata-rata kuadrat kesalahan"
        )
    
    with col4:
        st.metric(
            label="MAPE (%)",
            value=f"{metrics['mape']:.2f}%",
            help="Mean Absolute Percentage Error - rata-rata persentase kesalahan"
        )
    
    # Interpretasi performa
    st.subheader("🎯 Interpretasi Performa")
    
    if metrics['r2'] >= 0.9:
        performance_level = "Sangat Baik"
        performance_color = "green"
    elif metrics['r2'] >= 0.8:
        performance_level = "Baik"
        performance_color = "blue"
    elif metrics['r2'] >= 0.7:
        performance_level = "Cukup"
        performance_color = "orange"
    else:
        performance_level = "Perlu Perbaikan"
        performance_color = "red"
    
    st.markdown(f"""
    <div style="padding: 1rem; border-left: 4px solid {performance_color}; background-color: rgba(0,0,0,0.1);">
        <h4>Tingkat Performa: <span style="color: {performance_color};">{performance_level}</span></h4>
        <ul>
            <li><strong>Akurasi Model (R²):</strong> {metrics['r2']:.1%} - Model dapat menjelaskan {metrics['r2']:.1%} variasi dalam harga laptop</li>
            <li><strong>Kesalahan Rata-rata:</strong> ±{metrics['mae']:.2f} juta rupiah</li>
            <li><strong>Kesalahan Persentase:</strong> ±{metrics['mape']:.1f}% dari harga sebenarnya</li>
            <li><strong>Jumlah Data Evaluasi:</strong> {metrics['n_samples']} laptop</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

model = load_model()
df_options = get_data_options()

# Display model performance at the top
if model is not None:
    display_model_performance()
    st.divider()

if model and df_options is not None:
    with st.form("prediction_form"):
        st.header("🔧 Spesifikasi Utama")
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
            cpu_freq = st.slider("Frekuensi CPU (GHz)", min_value=0.9, max_value=4.0, value=2.5, step=0.1)

        st.header("💾 Penyimpanan & Layar")
        col4, col5, col6 = st.columns(3)
        with col4:
            storage_type = st.selectbox("Tipe Penyimpanan Utama", options=sorted(df_options['PrimaryStorageType'].unique()))
            primary_storage = st.select_slider("Kapasitas Penyimpanan (GB)", options=sorted(df_options[df_options['PrimaryStorage'] > 0]['PrimaryStorage'].unique()), value=256)
        with col5:
            if 'ScreenResolution' in df_options.columns:
                 screen_resolution = st.selectbox("Resolusi Layar", options=sorted(df_options['ScreenResolution'].unique()))
                 res_match = re.search(r'(\d+)x(\d+)', screen_resolution)
                 screen_w = int(res_match.group(1)) if res_match else 1920
                 screen_h = int(res_match.group(2)) if res_match else 1080
            else:
                screen_w = st.number_input("Lebar Layar (px)", value=1920)
                screen_h = st.number_input("Tinggi Layar (px)", value=1080)
        with col6:
            touchscreen_str = st.radio("Layar Sentuh (Touchscreen)", options=['No', 'Yes'], horizontal=True)
            ips_str = st.radio("Panel IPS", options=['No', 'Yes'], horizontal=True)
            touchscreen = 1 if touchscreen_str == 'Yes' else 0
            ips = 1 if ips_str == 'Yes' else 0

        submit_button = st.form_submit_button(label="🚀 Prediksi Harga", type="primary", use_container_width=True)

    if submit_button:
        input_data = pd.DataFrame({
            'Company': [company], 'TypeName': [type_name], 'Ram': [ram],
            'Weight': [weight], 'OS': [opsys], 'Inches': [inches],
            'CPU_company': [cpu_company], 'CPU_freq': [cpu_freq],
            'GPU_company': [gpu_company], 'PrimaryStorage': [primary_storage],
            'PrimaryStorageType': [storage_type], 'ScreenW': [screen_w],
            'ScreenH': [screen_h], 'Touchscreen': [touchscreen], 'IPSpanel': [ips]
        })

        st.info("📋 Data Input Anda:")
        st.dataframe(input_data)

        with st.spinner("🤖 Model sedang menganalisis spesifikasi..."):
            try:
                log_prediction = model.predict(input_data)
                predicted_price = np.expm1(log_prediction[0])
                
                # Display prediction with confidence interval
                metrics = load_model_performance()
                if metrics:
                    confidence_interval = metrics['mae'] * 1.96  # 95% confidence interval
                    lower_bound = max(0, predicted_price - confidence_interval)
                    upper_bound = predicted_price + confidence_interval
                    
                    st.success(f"## 💰 **Estimasi Harga: Rp {predicted_price:,.2f} Juta**")
                    st.info(f"📊 **Rentang Kepercayaan (95%):** Rp {lower_bound:,.2f} - Rp {upper_bound:,.2f} Juta")
                    st.caption(f"*Berdasarkan akurasi model {metrics['r2']:.1%} dan kesalahan rata-rata ±{metrics['mae']:.2f} juta*")
                else:
                    st.success(f"## 💰 **Estimasi Harga: Rp {predicted_price:,.2f} Juta**")
                    
            except Exception as e:
                st.error(f"❌ Terjadi error saat melakukan prediksi: {e}")
else:
    st.warning("⚠️ Model atau data tidak tersedia. Pastikan file model dan data sudah tersedia.")
