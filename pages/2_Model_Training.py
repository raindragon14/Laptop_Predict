import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

st.set_page_config(page_title="Model Training (Final)", page_icon="🏆", layout="wide")

st.title("🏆 Pelatihan Model Machine Learning (Versi Final)")
st.markdown("Model ini disesuaikan untuk bekerja dengan dataset yang sudah diproses dan menggunakan fitur-fitur yang lebih lengkap seperti detail layar dan komponen.")

KURS_KE_JUTA_IDR = 17500 / 1000000

@st.cache_data
def load_data(file_path='laptop_prices.csv'):
    """Memuat data yang sudah bersih dan diproses."""
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        df.rename(columns={
            'Price': 'Price_euros',
            'ScreenResolution': 'Screen',
            'Cpu': 'CPU_model',
            'Gpu': 'GPU_model',
            'Memory': 'Storage'
        }, inplace=True, errors='ignore')
        
        if 'Gpu' in df.columns and 'GPU_company' not in df.columns:
            st.info("Melakukan rekayasa fitur untuk GPU...")
            df['GPU_company'] = df['Gpu'].apply(lambda x: str(x).split()[0])

        for col in ['Touchscreen', 'IPSpanel']:
            if col in df.columns and df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
 
        if 'Ram' in df.columns and df['Ram'].dtype == 'object':
            df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('int32')
        if 'Weight' in df.columns and df['Weight'].dtype == 'object':
            df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype('float32')

        if 'Price_euros' in df.columns:
            df['Harga_IDR'] = df['Price_euros'] * KURS_KE_JUTA_IDR
            df = df.drop('Price_euros', axis=1)
        
        required_cols = ['Inches', 'Ram', 'Weight', 'Harga_IDR', 'CPU_freq', 'PrimaryStorage', 'ScreenW', 'ScreenH']
        df.dropna(subset=[col for col in required_cols if col in df.columns], inplace=True)
        
        st.success("Data berhasil dimuat dan disiapkan!")
        return df
        
    except FileNotFoundError:
        st.error(f"File '{file_path}' tidak ditemukan. Pastikan file berada di direktori yang benar.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat atau memproses data: {e}")
        return None

df = load_data('laptop_prices.csv')

if df is not None:
    st.header("Persiapan Data untuk Pelatihan")

    features = [
        'Company', 'TypeName', 'Ram', 'OS', 'Inches','Weight','PrimaryStorage',
        'CPU_company', 'CPU_freq', 'GPU_company', 'PrimaryStorageType',
        'ScreenW', 'ScreenH', 'Touchscreen', 'IPSpanel'
    ]
    
    available_features = [f for f in features if f in df.columns]
    target = 'Harga_IDR'

    if target not in df.columns:
        st.error(f"Kolom target '{target}' tidak ditemukan dalam data.")
    else:
        X = df[available_features]
        y = df[target]

        st.write("**Fitur yang akan digunakan:**", available_features)
        st.write("**Target prediksi:**", target)
        st.dataframe(X.head())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.markdown(f"Data telah dibagi menjadi:  \n- **Data Latih**: {X_train.shape[0]} baris  \n- **Data Uji**: {X_test.shape[0]} baris")

        numeric_features = [f for f in ['Ram', 'CPU_freq', 'ScreenW', 'ScreenH', 'Touchscreen', 'IPSpanel','Inches','Weight', 'PrimaryStorage' ] if f in X.columns]
        categorical_features = [f for f in ['Company', 'TypeName', 'OS', 'CPU_company', 'GPU_company', 'PrimaryStorageType'] if f in X.columns]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
            ], remainder='passthrough')

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True))
        ])

        st.header("Mulai Pelatihan Model")
        if st.button("Latih Model dengan Dataset Final", type="primary"):
            with st.spinner("Model sedang dilatih, mohon tunggu..."):
                
                y_train_log = np.log1p(y_train)
                model.fit(X_train, y_train_log)
                
                joblib.dump(model, 'final_model_pipeline.joblib')
                
                y_pred_log = model.predict(X_test)
                y_pred = np.expm1(y_pred_log)
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                oob = model.named_steps['regressor'].oob_score_

            st.success("🎉 Model final berhasil dilatih dan disimpan!")
            st.info("Model ini dilatih menggunakan fitur-fitur yang lebih spesifik.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="**R-squared (R²)**", value=f"{r2:.3f}", help="Menunjukkan seberapa baik fitur-fitur menjelaskan variasi harga. Semakin mendekati 1, semakin baik.")
            with col2:
                st.metric(label="**Mean Absolute Error (MAE)**", value=f"Rp {mae:.2f} Juta", help="Rata-rata kesalahan prediksi harga. Semakin rendah, semakin akurat.")
            with col3:
                st.metric(label="**Out-of-Bag (OOB) Score**", value=f"{oob:.3f}", help="Skor validasi internal dari RandomForest, mirip dengan R² pada data yang tidak terlihat saat training.")

else:
    st.error("Gagal memuat data. Pelatihan tidak bisa dilanjutkan.")
