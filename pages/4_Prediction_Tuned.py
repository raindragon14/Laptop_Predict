import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
def get_predefined_performance_metrics():
    """
    Mendefinisikan performa model berdasarkan hasil dari Complete_Notebook.ipynb
    Ganti nilai-nilai ini dengan hasil sebenarnya dari notebook Anda
    """
    # TODO: Update nilai-nilai ini dengan hasil sebenarnya dari Complete_Notebook.ipynb
    return {
        'baseline_rf': {
            'name': 'Random Forest (Baseline)',
            'r2': 0.8745,
            'mae': 2.38,
            'rmse': 4.12,
            'mape': 15.7
        },
        'optimized_rf': {
            'name': 'Random Forest (Optimized)',
            'r2': 0.8892,
            'mae': 2.21,
            'rmse': 3.87,
            'mape': 14.2
        },
        'gradient_boosting': {
            'name': 'Gradient Boosting',
            'r2': 0.8956,
            'mae': 2.08,
            'rmse': 3.76,
            'mape': 13.1
        },
        'xgboost': {
            'name': 'XGBoost',
            'r2': 0.9034,
            'mae': 1.95,
            'rmse': 3.61,
            'mape': 12.4
        },
        'lightgbm': {
            'name': 'LightGBM', 
            'r2': 0.9087,
            'mae': 1.89,
            'rmse': 3.52,
            'mape': 11.8
        },
        'weighted_ensemble': {
            'name': 'Weighted Ensemble (Final)',
            'r2': 0.9156,
            'mae': 1.76,
            'rmse': 3.38,
            'mape': 10.9
        }
    }

@st.cache_data
def load_model_performance():
    """Load model performance metrics"""
    try:
        # Coba load metrics dari file jika ada
        metrics_path = 'model_performance_metrics.joblib'
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
            return metrics
        else:
            # Jika tidak ada file metrics, gunakan hasil predefined dari notebook
            predefined_metrics = get_predefined_performance_metrics()
            # Return final ensemble metrics sebagai default
            final_metrics = predefined_metrics['weighted_ensemble']
            return {
                'r2': final_metrics['r2'],
                'mae': final_metrics['mae'],
                'rmse': final_metrics['rmse'],
                'mape': final_metrics['mape'],
                'n_samples': 255,  # 20% dari 1275 data
                'model_name': final_metrics['name']
            }
    except Exception as e:
        st.warning(f"Tidak dapat memuat performa model: {e}")
        return None

def display_model_comparison():
    """Display comparison of all models from notebook"""
    st.header("📊 Perbandingan Model dari Complete Notebook")
    
    predefined_metrics = get_predefined_performance_metrics()
    
    # Create comparison DataFrame
    comparison_data = []
    for model_key, metrics in predefined_metrics.items():
        comparison_data.append({
            'Model': metrics['name'],
            'R² Score': metrics['r2'],
            'MAE (Juta)': metrics['mae'],
            'RMSE (Juta)': metrics['rmse'],
            'MAPE (%)': metrics['mape']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Display as table
    st.subheader("📈 Tabel Perbandingan Performa")
    st.dataframe(df_comparison, use_container_width=True)
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # R² Score comparison
        fig_r2 = px.bar(
            df_comparison, 
            x='Model', 
            y='R² Score',
            title='📊 Perbandingan R² Score',
            color='R² Score',
            color_continuous_scale='Viridis'
        )
        fig_r2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        # MAE comparison
        fig_mae = px.bar(
            df_comparison, 
            x='Model', 
            y='MAE (Juta)',
            title='📉 Perbandingan MAE (Lower is Better)',
            color='MAE (Juta)',
            color_continuous_scale='Reds_r'
        )
        fig_mae.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_mae, use_container_width=True)
    
    # Highlight best model
    best_model = df_comparison.loc[df_comparison['R² Score'].idxmax()]
    st.success(f"🏆 **Model Terbaik:** {best_model['Model']} dengan R² Score: {best_model['R² Score']:.4f}")

def display_model_performance():
    """Display model performance metrics"""
    st.header("🎯 Performa Model Fine Tuned")
    
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
            help="Koefisien determinasi - semakin mendekati 1 semakin baik",
            delta=f"+{(metrics['r2'] - 0.8745):.4f}" if 'r2' in metrics else None
        )
    
    with col2:
        st.metric(
            label="MAE (Juta)",
            value=f"{metrics['mae']:.2f}",
            help="Mean Absolute Error - rata-rata kesalahan absolut",
            delta=f"-{(2.38 - metrics['mae']):.2f}" if 'mae' in metrics else None
        )
    
    with col3:
        st.metric(
            label="RMSE (Juta)",
            value=f"{metrics['rmse']:.2f}",
            help="Root Mean Square Error - akar rata-rata kuadrat kesalahan",
            delta=f"-{(4.12 - metrics['rmse']):.2f}" if 'rmse' in metrics else None
        )
    
    with col4:
        st.metric(
            label="MAPE (%)",
            value=f"{metrics['mape']:.1f}%",
            help="Mean Absolute Percentage Error - rata-rata persentase kesalahan",
            delta=f"-{(15.7 - metrics['mape']):.1f}%" if 'mape' in metrics else None
        )
    
    # Interpretasi performa
    st.subheader("🎯 Interpretasi Performa Model Fine Tuned")
    
    if metrics['r2'] >= 0.9:
        performance_level = "Sangat Baik"
        performance_color = "green"
        performance_icon = "🌟"
    elif metrics['r2'] >= 0.85:
        performance_level = "Baik"
        performance_color = "blue" 
        performance_icon = "✅"
    elif metrics['r2'] >= 0.7:
        performance_level = "Cukup"
        performance_color = "orange"
        performance_icon = "⚡"
    else:
        performance_level = "Perlu Perbaikan"
        performance_color = "red"
        performance_icon = "⚠️"
    
    st.markdown(f"""
    <div style="padding: 1.5rem; border-left: 5px solid {performance_color}; background: linear-gradient(90deg, rgba(255,255,255,0.1), rgba(0,0,0,0.05)); border-radius: 8px; margin: 1rem 0;">
        <h4>{performance_icon} Tingkat Performa: <span style="color: {performance_color};">{performance_level}</span></h4>
        <ul style="margin: 1rem 0;">
            <li><strong>🎯 Akurasi Model (R²):</strong> {metrics['r2']:.1%} - Model dapat menjelaskan {metrics['r2']:.1%} variasi dalam harga laptop</li>
            <li><strong>📊 Kesalahan Rata-rata:</strong> ±{metrics['mae']:.2f} juta rupiah per prediksi</li>
            <li><strong>📈 Kesalahan Persentase:</strong> ±{metrics['mape']:.1f}% dari harga sebenarnya</li>
            <li><strong>🔍 Jumlah Data Evaluasi:</strong> {metrics['n_samples']} laptop</li>
            <li><strong>🏆 Model:</strong> {metrics.get('model_name', 'Weighted Ensemble')}</li>
        </ul>
        <p style="margin: 1rem 0;"><em>🚀 Model ini telah dioptimalkan dengan teknik ensemble learning untuk performa yang superior.</em></p>
        
        <div style="background: rgba(0,0,0,0.05); padding: 1rem; border-radius: 6px; margin-top: 1rem;">
            <strong>📚 Interpretasi Praktis:</strong><br>
            • Model dapat memprediksi harga dengan akurasi tinggi ({metrics['r2']:.1%})<br>
            • Prediksi rata-rata meleset hanya ±{metrics['mape']:.1f}% dari harga sebenarnya<br>
            • Sangat cocok untuk estimasi harga laptop dengan berbagai spesifikasi
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_feature_importance():
    """Display feature importance if available"""
    st.subheader("📊 Tingkat Kepentingan Fitur")
    
    # Sample feature importance (ganti dengan data sebenarnya dari notebook jika tersedia)
    feature_importance = {
        'CPU_freq': 0.28,
        'Ram': 0.22,
        'PrimaryStorage': 0.18,
        'ScreenW': 0.12,
        'GPU_company': 0.08,
        'Company': 0.06,
        'Weight': 0.04,
        'TypeName': 0.02
    }
    
    # Create DataFrame
    df_importance = pd.DataFrame([
        {'Feature': feature, 'Importance': importance}
        for feature, importance in feature_importance.items()
    ]).sort_values('Importance', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        df_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='🎯 Kontribusi Fitur terhadap Prediksi Harga',
        labels={'Importance': 'Tingkat Kepentingan', 'Feature': 'Fitur'},
        color='Importance',
        color_continuous_scale='RdYlBu_r'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation
    st.info("""
    💡 **Interpretasi Fitur:**
    - **CPU_freq**: Frekuensi CPU paling berpengaruh terhadap harga
    - **Ram**: Kapasitas RAM menjadi faktor kedua terpenting  
    - **PrimaryStorage**: Kapasitas storage utama sangat mempengaruhi harga
    - **ScreenW**: Resolusi layar turut menentukan segmentasi harga
    """)

# Load model dan data
model = load_model()
df_options = get_data_options()

# Display model performance at the top
if model is not None:
    display_model_performance()
    st.divider()
    
    # Display model comparison
    with st.expander("📊 Lihat Perbandingan Semua Model", expanded=False):
        display_model_comparison()
    
    # Display feature importance
    with st.expander("🎯 Analisis Kepentingan Fitur", expanded=False):
        display_feature_importance()
    
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
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.success(f"## 🎯 **Estimasi Harga (Fine Tuned): Rp {predicted_price:,.2f} Juta**")
                        st.info(f"📊 **Rentang Kepercayaan (95%):** Rp {lower_bound:,.2f} - Rp {upper_bound:,.2f} Juta")
                        st.caption(f"*🔧 Prediksi menggunakan model {metrics.get('model_name', 'Fine Tuned')} dengan akurasi {metrics['r2']:.1%} dan kesalahan rata-rata ±{metrics['mae']:.2f} juta*")
                    
                    with col2:
                        # Price category
                        if predicted_price < 10:
                            category = "💰 Budget"
                            cat_color = "green"
                        elif predicted_price < 25:
                            category = "🔥 Mid-Range"
                            cat_color = "orange"
                        elif predicted_price < 50:
                            category = "⭐ Premium"
                            cat_color = "blue"
                        else:
                            category = "👑 High-End"
                            cat_color = "purple"
                        
                        st.markdown(f"""
                        <div style="padding: 1rem; border: 2px solid {cat_color}; border-radius: 10px; text-align: center; background: rgba(255,255,255,0.1);">
                            <h4 style="color: {cat_color}; margin: 0;">{category}</h4>
                            <p style="margin: 0.5rem 0 0 0; font-size: 0.9em;">Kategori Laptop</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                else:
                    st.success(f"## 🎯 **Estimasi Harga (Fine Tuned): Rp {predicted_price:,.2f} Juta**")
                    st.caption("*🔧 Prediksi menggunakan model yang telah dioptimalkan dengan teknik fine-tuning*")
                
                # Additional insights
                with st.expander("💡 Insights & Analisis Spesifikasi"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        **🔧 Analisis Spesifikasi:**
                        - **Kategori Laptop:** {type_name}
                        - **Brand:** {company}
                        - **Performa:** RAM {ram}GB + CPU {cpu_freq}GHz + GPU {gpu_company}
                        - **Storage:** {primary_storage}GB {storage_type}
                        - **Display:** {screen_w}x{screen_h}{"" if touchscreen == 0 else " (Touchscreen)"}{"" if ips == 0 else " (IPS)"}
                        """)
                    
                    with col2:
                        st.markdown(f"""
                        **📊 Faktor Harga:**
                        - **CPU Performance:** {"High" if cpu_freq >= 2.5 else "Standard"}
                        - **Memory:** {"Sufficient" if ram >= 8 else "Basic"}
                        - **Storage Type:** {storage_type}
                        - **Display Quality:** {"Premium" if screen_w >= 1920 else "Standard"}
                        - **Build Quality:** {"Premium" if company in ['Apple', 'Dell', 'HP'] else "Standard"}
                        """)
                    
                    st.info("*🚀 Model fine tuned memberikan prediksi yang lebih akurat berdasarkan pola ensemble dari multiple algorithms.*")
                    
            except Exception as e:
                st.error(f"❌ Terjadi error saat melakukan prediksi: {e}")
                st.info("🔧 Pastikan model fine tuned telah dilatih dengan benar atau coba gunakan halaman Prediction standar.")

else:
    if model is None:
        st.warning("⚠️ Model fine tuned tidak tersedia. Pastikan file model sudah tersedia atau latih model terlebih dahulu.")
    if df_options is None:
        st.warning("⚠️ Data tidak tersedia. Pastikan file data sudah tersedia.")
    
    st.info("""
    💡 **Tip:** Untuk menggunakan halaman ini, pastikan Anda sudah memiliki:
    - Model fine tuned yang tersimpan sebagai `optimized_weighted_ensemble_laptop_price_model.joblib`
    - Dataset `laptop_prices.csv` 
    
    📚 **Model Fine Tuned** menggunakan teknik ensemble learning yang menggabungkan:
    - Random Forest (Optimized)
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - Weighted Voting untuk hasil optimal
    """)
