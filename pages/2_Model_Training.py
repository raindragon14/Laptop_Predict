# pages/2_🤖_Model_Training.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

st.title("🤖 Pelatihan Model Machine Learning")
st.markdown("Di halaman ini, kita akan melatih model untuk memprediksi harga laptop.")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('laptop_prices.csv', encoding='latin-1')
        
        # PERBAIKAN: Hanya proses jika kolomnya adalah string (object)
        if df['Ram'].dtype == 'object':
            df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')
        
        if df['Weight'].dtype == 'object':
            df['Weight'] = df['Weight'].str.replace('kg', '').astype('float32')

        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is not None:
    st.header("Persiapan Data untuk Pelatihan")

    # Memilih fitur dan target
    features = ['Company', 'TypeName', 'Ram', 'Weight', 'OpSys', 'Inches']
    target = 'Price_euros'
    X = df[features]
    y = df[target]

    st.write("Fitur yang digunakan:", features)
    st.write("Target prediksi:", target)

    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.markdown(f"""
    Data dibagi menjadi:
    - **Data Latih (Training)**: {X_train.shape[0]} baris
    - **Data Uji (Testing)**: {X_test.shape[0]} baris
    """)

    # Membuat pipeline preprocessing
    numeric_features = ['Ram', 'Weight', 'Inches']
    categorical_features = ['Company', 'TypeName', 'OpSys']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Membuat pipeline lengkap dengan model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    st.header("Mulai Pelatihan Model")
    if st.button("Latih Model Sekarang", type="primary"):
        with st.spinner("Model sedang dilatih, mohon tunggu..."):
            # Melatih model
            model.fit(X_train, y_train)

            # Menyimpan pipeline (model + preprocessor)
            joblib.dump(model, 'model_pipeline.joblib')

            # Evaluasi model
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            st.success("🎉 Model berhasil dilatih dan disimpan!")
            st.metric(label="**R-squared (R²)**", value=f"{r2:.2f}")
            st.metric(label="**Mean Absolute Error (MAE)**", value=f"€{mae:.2f}")

            st.info("""
            **Penjelasan Metrik:**
            - **R-squared (R²)**: Menunjukkan seberapa baik fitur yang kita gunakan dapat menjelaskan variasi harga. Nilai 0.82 berarti sekitar 82% variasi harga dapat dijelaskan oleh model. Semakin mendekati 1, semakin baik.
            - **Mean Absolute Error (MAE)**: Rata-rata kesalahan prediksi dalam Euro. Nilai €208 berarti prediksi model kita rata-rata meleset sekitar 208 Euro dari harga sebenarnya.
            """)
    
    st.warning("Klik tombol di atas untuk memulai proses pelatihan. File `model_pipeline.joblib` akan dibuat/diperbarui.")

else:
    st.error("Gagal memuat data. Tidak dapat melanjutkan pelatihan.")
