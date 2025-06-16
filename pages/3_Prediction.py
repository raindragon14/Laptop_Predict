import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Prediksi Harga Laptop")
st.markdown("Masukkan spesifikasi laptop untuk mendapatkan estimasi harga.")

# Path ke model
MODEL_PATH = 'model_pipeline.joblib'

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        return model
    return None

model = load_model()

if model is None:
    st.error("Model belum dilatih! Silakan pergi ke halaman 'Model Training' untuk melatih model terlebih dahulu.")
else:
    # Memuat data untuk mendapatkan opsi input
    @st.cache_data
def get_data_options():
    df_options = pd.read_csv('laptop_prices.csv', encoding='latin-1')

    if df_options['Ram'].dtype == 'object':
        df_options['Ram'] = df_options['Ram'].str.replace('GB', '').astype('int32')
    
    return df_options

    df_options = get_data_options()

    # Membuat form input
    with st.form("prediction_form"):
        st.header("Masukkan Detail Laptop")

        col1, col2, col3 = st.columns(3)

        with col1:
            company = st.selectbox("Brand", options=df_options['Company'].unique())
            ram = st.selectbox("RAM (GB)", options=sorted(df_options['Ram'].unique()))
        
        with col2:
            type_name = st.selectbox("Tipe", options=df_options['TypeName'].unique())
            weight = st.number_input("Berat (kg)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)

        with col3:
            opsys = st.selectbox("Sistem Operasi", options=df_options['OS'].unique())
            inches = st.number_input("Ukuran Layar (Inci)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)

        submit_button = st.form_submit_button(label="Prediksi Harga", type="primary")

    if submit_button:
        # Membuat DataFrame dari input
        input_data = pd.DataFrame({
            'Company': [company],
            'TypeName': [type_name],
            'Ram': [ram],
            'Weight': [weight],
            'OS': [os],
            'Inches': [inches]
        })

        st.subheader("Data Input:")
        st.write(input_data)

        # Melakukan prediksi
        with st.spinner("Menghitung prediksi..."):
            prediction = model.predict(input_data)
            predicted_price = prediction[0]

        st.subheader("🎉 Hasil Prediksi Harga")
        st.success(f"**Estimasi Harga Laptop: €{predicted_price:,.2f}**")
