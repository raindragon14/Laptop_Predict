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

st.title("💻 Prediksi Harga Laptop (Fine Tuned)")
st.markdown("Isi detail spesifikasi di bawah ini untuk mendapatkan estimasi harga laptop impian Anda dengan model yang telah dioptimalkan.")

# Model path untuk fine tuned model
MODEL_PATH = 'optimized_weighted_ensemble_laptop_price_model.joblib'
DATA_PATH = 'laptop_prices.csv'

def process_memory(memory_str):
    """Process memory string untuk ekstraksi storage type dan size"""
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
    """Load fine tuned model"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model '{MODEL_PATH}' tidak ditemukan. Harap latih model terlebih dahulu.")
        return None
    try:
        model = joblib.load(MODEL_PATH)
        st.success("✅ Model fine tuned berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

@st.cache_data
def get_data_options():
    """Load data untuk mendapatkan opsi input"""
    if not os.path.exists(DATA_PATH):
        st.error(f"File data '{DATA_PATH}' tidak ditemukan.")
        return None
    try:
        df = pd.read_csv(DATA_PATH, encoding='latin-1')
        
        # Preprocessing konsisten dengan halaman lain
        if 'Ram' in df.columns and df['Ram'].dtype == 'object':
            df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('int32')
        
        if 'Weight' in df.columns and df['Weight'].dtype == 'object':
            df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype('float32')

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
            
        # Load data - konsisten dengan preprocessing
        df = pd.read_csv(DATA_PATH, encoding='latin-1')
        
        # Konsisten dengan konversi mata uang di halaman lain
        KURS_KE_JUTA_IDR = 17500 / 1000000
        
        # Process Ram
        if 'Ram' in df.columns and df['Ram'].dtype == 'object':
            df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('int32')
        
        # Process Weight 
        if 'Weight' in df.columns and df['Weight'].dtype == 'object':
            df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype('float32')

        # Convert price to IDR (konsisten dengan Model Training)
        if 'Price_euros' in df.columns:
            df['Harga_IDR'] = df['Price_euros'] * KURS_KE_JUTA_IDR
            target_col = 'Harga_IDR'
        else:
            target_col = 'Price_euros'
            
        if target_col not in df.columns:
            return None
            
        # Split data untuk evaluasi (menggunakan 20% data terakhir sebagai test)
        test_size = int(len(df) * 0.2)
        test_data = df.tail(test_size)
        
        # Prepare features (konsisten dengan feature yang digunakan)
        feature_cols = ['Company', 'TypeName', 'Ram', 'Weight', 'OS', 'Inches',
                       'CPU_company', 'CPU_freq', 'GPU_company', 'PrimaryStorage',
                       'PrimaryStorageType', 'ScreenW', 'ScreenH', 'Touchscreen', 'IPSpanel']
        
        available_cols = [col for col in feature_cols if col in test_data.columns]
        
        if len(available_cols) == 0:
            return None
            
        X_test = test_data[available_cols]
        y_test = test_data[target_col]
        
        # Load model
        model = load_model()
        if model is None:
            return None
            
        # Make predictions - konsisten dengan halaman Prediction
        try:
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)  # Convert from log scale jika menggunakan log transform
        except:
            # Jika model tidak menggunakan log transform
            y_pred = model.predict(X_test)
            
        y_true = y_test
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        
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
    st.header("📊 Performa Model Fine Tuned")
    
    metrics = load_model_performance()
    
    if metrics is None:
        st.info("📈 Model fine tuned siap digunakan! Performa model akan ditampilkan setelah tersedia data evaluasi.")
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
    st.subheader("🎯 Interpretasi Performa Model Fine Tuned")
    
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
        <h4>🔧 Tingkat Performa: <span style="color: {performance_color};">{performance_level}</span></h4>
        <ul>
            <li><strong>Akurasi Model (R²):</strong> {metrics['r2']:.1%} - Model dapat menjelaskan {metrics['r2']:.1%} variasi dalam harga laptop</li>
            <li><strong>Kesalahan Rata-rata:</strong> ±{metrics['mae']:.2f} juta rupiah</li>
            <li><strong>Kesalahan Persentase:</strong> ±{metrics['mape']:.1f}% dari harga sebenarnya</li>
            <li><strong>Jumlah Data Evaluasi:</strong> {metrics['n_samples']} laptop</li>
        </ul>
        <p><em>🚀 Model ini telah dioptimalkan dengan teknik fine-tuning untuk performa yang lebih baik.</em></p>
    </div>
    """, unsafe_allow_html=True)

# Load model dan data
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
            primary_storage = st.select_slider("Kapasitas Penyimpanan (GB)", 
                                             options=sorted(df_options[df_options['PrimaryStorage'] > 0]['PrimaryStorage'].unique()), 
                                             value=256)
            
        with col5:
            if 'ScreenResolution' in df_options.columns:
                screen_resolution = st.selectbox("Resolusi Layar", options=sorted(df_options['ScreenResolution'].unique()))
                res_match = re.search(r'(\d+)x(\d+)', screen_resolution)
                screen_w = int(res_match.group(1)) if res_match else 1920
                screen_h = int(res_match.group(2)) if res_match else 1080
            else:
                screen_w = st.number_input("Lebar Layar (px)", value=1920, min_value=800, max_value=4000, step=1)
                screen_h = st.number_input("Tinggi Layar (px)", value=1080, min_value=600, max_value=3000, step=1)
                
        with col6:
            touchscreen_str = st.radio("Layar Sentuh (Touchscreen)", options=['No', 'Yes'], horizontal=True)
            ips_str = st.radio("Panel IPS", options=['No', 'Yes'], horizontal=True)
            touchscreen = 1 if touchscreen_str == 'Yes' else 0
            ips = 1 if ips_str == 'Yes' else 0

        submit_button = st.form_submit_button(label="🚀 Prediksi Harga (Fine Tuned)", type="primary", use_container_width=True)

    if submit_button:
        # Prepare input data konsisten dengan format yang digunakan
        input_data = pd.DataFrame({
            'Company': [company], 
            'TypeName': [type_name], 
            'Ram': [ram],
            'Weight': [weight], 
            'OS': [opsys], 
            'Inches': [inches],
            'CPU_company': [cpu_company], 
            'CPU_freq': [cpu_freq],
            'GPU_company': [gpu_company], 
            'PrimaryStorage': [primary_storage],
            'PrimaryStorageType': [storage_type], 
            'ScreenW': [screen_w],
            'ScreenH': [screen_h], 
            'Touchscreen': [touchscreen], 
            'IPSpanel': [ips]
        })

        st.info("📋 Data Input Anda:")
        st.dataframe(input_data, use_container_width=True)

        with st.spinner("🔧 Model fine tuned sedang menganalisis spesifikasi..."):
            try:
                # Prediksi dengan model fine tuned
                prediction = model.predict(input_data)
                
                # Handle different prediction formats
                if isinstance(prediction, np.ndarray):
                    predicted_price = float(prediction[0])
                else:
                    predicted_price = float(prediction)
                
                # Jika model menggunakan log transform, convert back
                try:
                    predicted_price_exp = np.expm1(predicted_price)
                    # Check if exp result is reasonable
                    if predicted_price_exp > 0 and predicted_price_exp < predicted_price * 100:
                        predicted_price = predicted_price_exp
                except:
                    pass
                
                # Display prediction dengan confidence interval jika tersedia
                metrics = load_model_performance()
                if metrics and metrics['mae'] > 0:
                    confidence_interval = metrics['mae'] * 1.96  # 95% confidence interval
                    lower_bound = max(0, predicted_price - confidence_interval)
                    upper_bound = predicted_price + confidence_interval
                    
                    st.success(f"## 🎯 **Estimasi Harga (Fine Tuned): Rp {predicted_price:,.2f} Juta**")
                    st.info(f"📊 **Rentang Kepercayaan (95%):** Rp {lower_bound:,.2f} - Rp {upper_bound:,.2f} Juta")
                    st.caption(f"*🔧 Prediksi menggunakan model fine tuned dengan akurasi {metrics['r2']:.1%} dan kesalahan rata-rata ±{metrics['mae']:.2f} juta*")
                else:
                    st.success(f"## 🎯 **Estimasi Harga (Fine Tuned): Rp {predicted_price:,.2f} Juta**")
                    st.caption("*🔧 Prediksi menggunakan model yang telah dioptimalkan dengan teknik fine-tuning*")
                
                # Additional insights
                with st.expander("💡 Insights Tambahan"):
                    st.markdown(f"""
                    **Analisis Spesifikasi:**
                    - **Kategori Laptop:** {type_name}
                    - **Brand:** {company}
                    - **Performa:** RAM {ram}GB + CPU {cpu_freq}GHz + GPU {gpu_company}
                    - **Storage:** {primary_storage}GB {storage_type}
                    - **Display:** {screen_w}x{screen_h}{"" if touchscreen == 0 else " (Touchscreen)"}{"" if ips == 0 else " (IPS)"}
                    
                    *Model fine tuned memberikan prediksi yang lebih akurat berdasarkan pola yang telah dioptimalkan.*
                    """)
                    
            except Exception as e:
                st.error(f"❌ Terjadi error saat melakukan prediksi: {e}")
                st.info("🔧 Pastikan model fine tuned telah dilatih dengan benar atau coba gunakan halaman Prediction standar.")

else:
    if model is None:
        st.warning("⚠️ Model fine tuned tidak tersedia. Pastikan file model sudah tersedia atau latih model terlebih dahulu.")
    if df_options is None:
        st.warning("⚠️ Data tidak tersedia. Pastikan file data sudah tersedia.")
    
    st.info("💡 **Tip:** Untuk menggunakan halaman ini, pastikan Anda sudah memiliki model fine tuned yang tersimpan sebagai `optimized_weighted_ensemble_laptop_price_model.joblib`")
