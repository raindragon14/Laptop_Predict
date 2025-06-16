import streamlit as st
import pandas as pd
import numpy as np  # Diperlukan untuk transformasi logaritmik
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

st.title("🤖 Pelatihan Model Machine Learning")
st.markdown("Model ini ditingkatkan berdasarkan insight dari EDA untuk menangani distribusi harga yang tidak normal.")

# Konstanta untuk konversi ke Juta Rupiah
KURS_KE_JUTA_IDR = 0.0175

@st.cache_data
def load_data():
    """Memuat, membersihkan, dan melakukan pra-pemrosesan data."""
    try:
        df = pd.read_csv('laptop_prices.csv', encoding='latin-1')
        
        # Membersihkan kolom 'Ram' dan 'Weight'
        if 'Ram' in df.columns and df['Ram'].dtype == 'object':
            df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('int32')
        if 'Weight' in df.columns and df['Weight'].dtype == 'object':
            df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype('float32')
        
        # Konversi harga dan hapus kolom asli
        df['Harga_IDR'] = df['Price_euros'] * KURS_KE_JUTA_IDR
        df = df.drop('Price_euros', axis=1)
        
        # Menghapus baris dengan data yang hilang pada fitur utama
        df.dropna(subset=['Inches', 'Weight', 'Ram'], inplace=True)
        return df
    except (FileNotFoundError, AttributeError, KeyError) as e:
        st.error(f"Gagal memuat atau memproses data: {e}")
        return None

df = load_data()

if df is not None:
    st.header("Persiapan Data untuk Pelatihan")

    features = ['Company', 'TypeName', 'Ram', 'Weight', 'OS', 'Inches']
    target = 'Harga_IDR'
    
    X = df[features]
    y = df[target]

    st.write("Fitur yang akan digunakan:", features)
    st.write("Target prediksi:", target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.markdown(f"Data telah dibagi menjadi:  \n- **Data Latih**: {X_train.shape[0]} baris  \n- **Data Uji**: {X_test.shape[0]} baris")

    # --- Pipeline Pra-pemrosesan Data ---
    numeric_features = ['Ram', 'Weight', 'Inches']
    categorical_features = ['Company', 'TypeName', 'OS']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # --- Pipeline Model ---
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    st.header("Mulai Pelatihan Model")
    if st.button("Latih Model Sekarang", type="primary"):
        with st.spinner("Model sedang dilatih, mohon tunggu..."):
            
            y_train_log = np.log1p(y_train)
            
            # Melatih model dengan target yang sudah ditransformasi
            model.fit(X_train, y_train_log)
            
            # Menyimpan model yang sudah dilatih
            joblib.dump(model, 'model_pipeline.joblib')
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

        st.success("🎉 Model berhasil dilatih dan disimpan!")
        st.info("Model ini dilatih menggunakan target harga dalam skala logaritmik untuk meningkatkan akurasi.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="**R-squared (R²)**", value=f"{r2:.3f}", help="Semakin mendekati 1, semakin baik model menjelaskan variasi data.")
        with col2:
            st.metric(label="**Mean Absolute Error (MAE)**", value=f"Rp {mae:.2f} Juta", help="Rata-rata selisih harga prediksi dengan harga asli.")
else:
    st.error("Gagal memuat data. Pelatihan tidak bisa dilanjutkan.")
