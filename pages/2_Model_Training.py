import streamlit as st
import pandas as pd
import numpy as np
import re # Diperlukan untuk ekstraksi fitur dengan regular expression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

st.set_page_config(page_title="Model Training (Advanced)", page_icon="🚀", layout="wide")

st.title("🚀 Pelatihan Model Machine Learning (Versi Lanjutan)")
st.markdown("Model ini telah disempurnakan dengan menambahkan fitur-fitur baru dari CPU, GPU, dan Storage untuk prediksi harga yang lebih akurat.")

# Konstanta untuk konversi ke Juta Rupiah
KURS_KE_JUTA_IDR = 17500 / 1000000 # Kurs Euro ke Rupiah, lalu ke Juta

def process_memory(memory_str):
    """Fungsi untuk mengekstrak ukuran dan tipe storage utama."""
    memory_str = str(memory_str).strip().lower()
    
    # Ambil komponen pertama jika ada kombinasi storage (e.g., '256GB SSD + 1TB HDD')
    if '+' in memory_str:
        memory_str = memory_str.split('+')[0].strip()

    # Tentukan tipe storage
    storage_type = 'Other'
    if 'ssd' in memory_str:
        storage_type = 'SSD'
    elif 'hdd' in memory_str:
        storage_type = 'HDD'
    elif 'flash' in memory_str:
        storage_type = 'Flash Storage'
    elif 'hybrid' in memory_str:
        storage_type = 'Hybrid'
    
    # Ekstrak ukuran storage dalam GB
    size_gb = 0
    num_match = re.search(r'(\d+\.?\d*)', memory_str)
    if num_match:
        size = float(num_match.group(1))
        if 'tb' in memory_str:
            size_gb = size * 1000  # Konversi TB ke GB (1TB ~ 1000GB)
        else:  # Asumsi dalam GB
            size_gb = size
            
    return pd.Series([size_gb, storage_type])

@st.cache_data
def load_data():
    """Memuat, membersihkan, dan melakukan rekayasa fitur pada data."""
    try:
        df = pd.read_csv('laptop_prices.csv', encoding='latin-1')
        
        # --- Pembersihan Data Awal ---
        if 'Ram' in df.columns and df['Ram'].dtype == 'object':
            df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('int32')
        if 'Weight' in df.columns and df['Weight'].dtype == 'object':
            df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype('float32')
        
        # --- Rekayasa Fitur (Feature Engineering) ---
        # 1. Fitur CPU
        df['CPU_company'] = df['Cpu'].apply(lambda x: str(x).split()[0])
        # Ekstrak frekuensi CPU (GHz), tangani jika tidak ditemukan
        df['CPU_freq'] = df['Cpu'].str.extract(r'(\d\.?\d*)GHz').astype(float)
        df['CPU_freq'].fillna(df['CPU_freq'].median(), inplace=True) # Isi nilai hilang dengan median
        
        # 2. Fitur GPU
        df['GPU_brand'] = df['Gpu'].apply(lambda x: str(x).split()[0])

        # 3. Fitur Storage
        storage_features = df['Memory'].apply(process_memory)
        storage_features.columns = ['PrimaryStorage', 'PrimaryStorageType']
        df = pd.concat([df, storage_features], axis=1)

        # Hapus kolom yang tidak terpakai setelah ekstraksi
        df = df.drop(['Cpu', 'Gpu', 'Memory'], axis=1)
        
        # --- Konversi Harga & Finalisasi ---
        df['Harga_IDR'] = df['Price_euros'] * KURS_KE_JUTA_IDR
        df = df.drop('Price_euros', axis=1)
        
        # Menghapus baris dengan data yang hilang pada fitur utama
        df.dropna(subset=['Inches', 'Weight', 'Ram', 'PrimaryStorage'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Gagal memuat atau memproses data: {e}")
        return None

df = load_data()

if df is not None:
    st.header("Persiapan Data untuk Pelatihan")

    # Fitur yang digunakan sekarang lebih banyak
    features = ['Company', 'TypeName', 'Ram', 'Weight', 'OS', 'Inches', 
                'CPU_company', 'GPU_brand', 'PrimaryStorage', 'PrimaryStorageType', 'CPU_freq']
    target = 'Harga_IDR'
    
    X = df[features]
    y = df[target]

    st.write("Fitur yang akan digunakan:", features)
    st.write("Target prediksi:", target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.markdown(f"Data telah dibagi menjadi:  \n- **Data Latih**: {X_train.shape[0]} baris  \n- **Data Uji**: {X_test.shape[0]} baris")

    # --- Pipeline Pra-pemrosesan Data (Diperbarui) ---
    numeric_features = ['Ram', 'Weight', 'Inches', 'CPU_freq', 'PrimaryStorage']
    categorical_features = ['Company', 'TypeName', 'OS', 'CPU_company', 'GPU_brand', 'PrimaryStorageType']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ], remainder='passthrough')

    # --- Pipeline Model ---
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    st.header("Mulai Pelatihan Model")
    if st.button("Latih Model dengan Fitur Baru", type="primary"):
        with st.spinner("Model sedang dilatih dengan data yang lebih kaya, mohon tunggu..."):
            
            # Transformasi logaritmik pada target untuk menstabilkan varians
            y_train_log = np.log1p(y_train)
            
            # Melatih model dengan target yang sudah ditransformasi
            model.fit(X_train, y_train_log)
            
            # Menyimpan model yang sudah dilatih
            joblib.dump(model, 'advanced_model_pipeline.joblib')
            
            # Prediksi dan transformasi balik ke skala asli
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

        st.success("🎉 Model lanjutan berhasil dilatih dan disimpan!")
        st.info("Model ini sekarang menggunakan detail CPU, GPU, dan Storage untuk meningkatkan performa.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="**R-squared (R²)**", value=f"{r2:.3f}", help="Metrik ini menunjukkan seberapa baik fitur-fitur baru dapat menjelaskan variasi harga laptop. Semakin mendekati 1, semakin baik.")
        with col2:
            st.metric(label="**Mean Absolute Error (MAE)**", value=f"Rp {mae:.2f} Juta", help="Rata-rata kesalahan absolut prediksi harga. Semakin rendah, semakin akurat modelnya.")
else:
    st.error("Gagal memuat data. Pelatihan tidak bisa dilanjutkan.")

