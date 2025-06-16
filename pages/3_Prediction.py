import streamlit as st
import pandas as pd
import joblib
import os

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Prediksi Harga Laptop")
st.markdown("Masukkan spesifikasi laptop untuk mendapatkan estimasi harga.")

# --- Definisi Fungsi di Awal Skrip ---

# Path ke model yang sudah disimpan
MODEL_PATH = 'model_pipeline.joblib'

# Fungsi untuk memuat model (menggunakan cache resource agar model hanya dimuat sekali)
@st.cache_resource
def load_model():
    """Memuat model pipeline dari file joblib."""
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None
    return None

# Fungsi untuk memuat dan membersihkan data (menggunakan cache data agar data hanya dibaca sekali)
@st.cache_data
def get_data_options():
    """Membaca file CSV dan menyiapkan data untuk opsi input."""
    try:
        df_options = pd.read_csv('laptop_prices.csv', encoding='latin-1')
        # Membersihkan kolom 'Ram' dengan menghapus 'GB' dan mengubah tipe data
        if df_options['Ram'].dtype == 'object':
            df_options['Ram'] = df_options['Ram'].str.replace('GB', '', regex=False).astype('int32')
        return df_options
    except FileNotFoundError:
        st.error("File 'laptop_prices.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return None
    except Exception as e:
        st.error(f"Gagal memproses data: {e}")
        return None

# --- Logika Utama Aplikasi ---

model = load_model()
df_options = get_data_options()

# Tampilkan error jika model atau data tidak berhasil dimuat
if model is None:
    st.error("Model 'model_pipeline.joblib' tidak ditemukan! Silakan pergi ke halaman 'Model Training' untuk melatih model terlebih dahulu.")
elif df_options is None:
    st.warning("Data untuk opsi input tidak dapat dimuat. Fungsionalitas prediksi tidak akan berjalan.")
else:
    # Membuat form input agar prediksi hanya berjalan saat tombol ditekan
    with st.form("prediction_form"):
        st.header("Masukkan Detail Laptop")

        col1, col2, col3 = st.columns(3)

        with col1:
            company = st.selectbox("Brand", options=sorted(df_options['Company'].unique()))
            ram = st.selectbox("RAM (GB)", options=sorted(df_options['Ram'].unique()))

        with col2:
            type_name = st.selectbox("Tipe", options=sorted(df_options['TypeName'].unique()))
            weight = st.number_input("Berat (kg)", min_value=0.1, max_value=5.0, value=1.5, step=0.1)

        with col3:
            opsys = st.selectbox("Sistem Operasi", options=sorted(df_options['OS'].unique()))
            inches = st.number_input("Ukuran Layar (Inci)", min_value=10.0, max_value=24.0, value=15.6, step=0.1)
        
        # Tombol submit untuk form
        submit_button = st.form_submit_button(label="Prediksi Harga", type="primary", use_container_width=True)

    # Blok ini akan dieksekusi hanya jika tombol submit di dalam form ditekan
    if submit_button:
        # Membuat DataFrame dari input pengguna untuk diproses oleh model
        input_data = pd.DataFrame({
            'Company': [company],
            'TypeName': [type_name],
            'Ram': [ram],
            'Weight': [weight],
            # PERBAIKAN: Menggunakan variabel 'opsys' bukan 'os' untuk menghindari konflik
            'OS': [opsys],
            'Inches': [inches]
        })

        st.subheader("Data Input Anda:")
        st.dataframe(input_data)

        # Melakukan prediksi dengan menampilkan spinner
        with st.spinner("Menghitung estimasi harga..."):
            try:
                prediction = model.predict(input_data)
                # Mengambil nilai pertama dari hasil prediksi
                predicted_price = prediction[0]

                st.subheader("🎉 Hasil Prediksi Harga")
                # Menampilkan hasil dengan format mata uang Euro
                st.success(f"**Estimasi Harga Laptop: €{predicted_price:,.2f}**")
            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")
