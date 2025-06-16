import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import lightgbm as lgb # Menggunakan LightGBM sebagai model yang lebih kuat
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Model Training (No Tuning)", page_icon="⚡", layout="wide")
st.title("⚡ Pelatihan Model Cepat (Tanpa Tuning)")
st.markdown("Melatih model **LightGBM** dengan **Rekayasa Fitur** menggunakan pengaturan default.")

# Konstanta
KURS_KE_JUTA_IDR = 17500 / 1000000
DATA_PATH = 'laptop_prices.csv'
MODEL_PATH = 'lgbm_model_default.joblib' # Nama model baru

# --- Fungsi-fungsi ---
@st.cache_data
def load_and_engineer_data(file_path):
    """Memuat data dan melakukan rekayasa fitur tingkat lanjut."""
    if not os.path.exists(file_path):
        st.error(f"File data '{file_path}' tidak ditemukan.")
        return None
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        df.rename(columns={'Price_euros': 'Price'}, inplace=True, errors='ignore')
        
        # --- Rekayasa Fitur ---
        if 'Cpu' in df.columns and 'CPU_company' not in df.columns:
            df['CPU_company'] = df['Cpu'].apply(lambda x: str(x).split()[0])
            df['CPU_freq'] = df['Cpu'].str.extract(r'(\d\.?\d*)GHz').astype(float)
            df['CPU_freq'].fillna(df['CPU_freq'].median(), inplace=True)
        if 'Gpu' in df.columns and 'GPU_company' not in df.columns:
            df['GPU_company'] = df['Gpu'].apply(lambda x: str(x).split()[0])
        if 'Memory' in df.columns and 'PrimaryStorage' not in df.columns:
            df['Memory_lower'] = df['Memory'].str.lower()
            df['PrimaryStorage'] = df['Memory_lower'].str.extract(r'(\d+\.?\d*)').astype(float)
            df.loc[df['Memory_lower'].str.contains('tb', na=False), 'PrimaryStorage'] *= 1024
            df['PrimaryStorageType'] = 'Other'
            df.loc[df['Memory_lower'].str.contains('ssd', na=False), 'PrimaryStorageType'] = 'SSD'
            df.loc[df['Memory_lower'].str.contains('hdd', na=False), 'PrimaryStorageType'] = 'HDD'
            df.loc[df['Memory_lower'].str.contains('flash', na=False), 'PrimaryStorageType'] = 'Flash Storage'
            df.drop(columns=['Memory_lower'], inplace=True)
        if 'ScreenResolution' in df.columns and 'ScreenW' not in df.columns:
            res = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)')
            df['ScreenW'], df['ScreenH'] = pd.to_numeric(res[0]), pd.to_numeric(res[1])
        
        df.dropna(subset=['ScreenW', 'ScreenH', 'Inches'], inplace=True)
        st.info("Menciptakan fitur Kepadatan Piksel (PPI)...")
        df['PPI'] = (((df['ScreenW']**2) + (df['ScreenH']**2))**0.5 / df['Inches']).astype('float')

        # --- Pembersihan dan Konversi ---
        for col in ['Touchscreen', 'IPSpanel']:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = LabelEncoder().fit_transform(df[col])
        if 'Ram' in df.columns and df['Ram'].dtype == 'object':
            df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('int32')
        if 'Weight' in df.columns and df['Weight'].dtype == 'object':
            df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype('float32')
        if 'Price' in df.columns:
            df['Harga_IDR'] = df['Price'] * KURS_KE_JUTA_IDR
            df.drop('Price', axis=1, inplace=True)
        
        required_cols = ['Inches', 'Ram', 'Weight', 'Harga_IDR', 'CPU_freq', 'PrimaryStorage', 'ScreenW', 'ScreenH', 'PPI']
        df.dropna(subset=[col for col in required_cols if col in df.columns], inplace=True)
        
        st.success("Data berhasil dimuat dan fitur baru 'PPI' berhasil dibuat!")
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")
        return None

# --- Logika Utama ---
df = load_and_engineer_data(DATA_PATH)

if df is not None:
    st.header("Persiapan Data untuk Pelatihan")
    features = [
        'Company', 'TypeName', 'Ram', 'Weight', 'OS', 'Inches', 
        'CPU_company', 'CPU_freq', 'GPU_company', 'PrimaryStorage', 'PrimaryStorageType',
        'ScreenW', 'ScreenH', 'Touchscreen', 'IPSpanel', 'PPI'
    ]
    available_features = [f for f in features if f in df.columns]
    target = 'Harga_IDR'

    X = df[available_features]
    y = df[target]

    st.write("**Fitur yang akan digunakan:**", available_features)
    st.dataframe(X.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = [f for f in ['Ram', 'Weight', 'Inches', 'CPU_freq', 'PrimaryStorage', 'ScreenW', 'ScreenH', 'Touchscreen', 'IPSpanel', 'PPI'] if f in X.columns]
    categorical_features = [f for f in ['Company', 'TypeName', 'OS', 'CPU_company', 'GPU_company', 'PrimaryStorageType'] if f in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ], remainder='passthrough')

    # --- Pipeline Langsung Tanpa Tuning ---
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(random_state=42)) # Gunakan LightGBM dengan parameter default
    ])

    st.header("Mulai Pelatihan Model")
    if st.button("Latih Model Cepat (Tanpa Tuning)", type="primary"):
        with st.spinner("Model sedang dilatih, ini akan berjalan sangat cepat..."):
            # Langsung melatih pipeline
            y_train_log = np.log1p(y_train)
            pipeline.fit(X_train, y_train_log)
            
            st.success("🎉 Model berhasil dilatih!")
            
            joblib.dump(pipeline, MODEL_PATH)
            st.info(f"Model disimpan ke '{MODEL_PATH}'")
            
            # Evaluasi model
            y_pred_log = pipeline.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            st.write("---")
            st.header("Hasil Evaluasi Model (Default)")
            col1, col2 = st.columns(2)
            col1.metric("R-squared (R²)", f"{r2:.4f}")
            col2.metric("Mean Absolute Error (MAE)", f"Rp {mae:.2f} Juta")

