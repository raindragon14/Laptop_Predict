import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Data", page_icon="📊", layout="wide")

# Judul utama aplikasi
st.title("📊 Analisis Data Eksploratif (EDA) Harga Laptop")
st.markdown("Mari kita jelajahi karakteristik dari dataset harga laptop dalam Rupiah (IDR).")

# Asumsi kurs untuk konversi
KURS_EUR_TO_IDR = 17500

# Fungsi untuk memuat dan memproses data dengan cache
@st.cache_data
def load_and_process_data():
    """
    Memuat dataset, membersihkan data, dan melakukan konversi mata uang.
    """
    try:
        df = pd.read_csv('laptop_prices.csv', encoding='latin-1')
        
        # Membersihkan kolom 'Ram'
        if df['Ram'].dtype == 'object':
            df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('int32')
        
        # Membersihkan kolom 'Weight'
        if df['Weight'].dtype == 'object':
            df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype('float32')
        
        # Konversi harga dari Euro ke IDR dan membulatkannya
        df['Harga_IDR'] = (df['Price_euros'] * KURS_EUR_TO_IDR).astype(np.int64)
        
        return df
        
    except FileNotFoundError:
        st.error("File 'laptop_prices.csv' tidak ditemukan. Pastikan file berada di direktori yang sama dengan skrip Anda.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat atau memproses data: {e}")
        return None

# Memuat data
df = load_and_process_data()

# Melanjutkan hanya jika data berhasil dimuat
if df is not None:
    st.header("Informasi Dasar Dataset")
    st.info(f"Harga dikonversi dari Euro ke Rupiah dengan asumsi kurs **€1 = Rp{KURS_EUR_TO_IDR:,}**.")
    st.dataframe(df.head())

    # Menampilkan statistik deskriptif untuk kolom numerik
    st.subheader("Statistik Deskriptif")
    st.write(df.describe())

    st.header("Visualisasi Data")

    # 1. Distribusi Harga
    st.subheader("1. Distribusi Harga Laptop (IDR)")
    fig_harga = px.histogram(
        df, 
        x='Harga_IDR', 
        nbins=50, 
        title='Distribusi Harga Laptop dalam Rupiah',
        labels={'Harga_IDR': 'Harga (Juta Rupiah)'}
    )
    st.plotly_chart(fig_harga, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # 2. Jumlah Laptop per Brand
        st.subheader("2. Jumlah Laptop per Brand")
        company_counts = df['Company'].value_counts().reset_index()
        company_counts.columns = ['Brand', 'Jumlah']
        fig_company = px.bar(
            company_counts, 
            x='Brand', 
            y='Jumlah', 
            title='Jumlah Laptop Berdasarkan Brand', 
            labels={'Brand': 'Brand Laptop', 'Jumlah': 'Jumlah Unit'}
        )
        st.plotly_chart(fig_company, use_container_width=True)

    with col2:
        # 3. Proporsi Tipe Laptop
        st.subheader("3. Proporsi Tipe Laptop")
        fig_type = px.pie(
            df, 
            names='TypeName', 
            title='Proporsi Tipe Laptop', 
            hole=0.3
        )
        st.plotly_chart(fig_type, use_container_width=True)

    # 4. Hubungan RAM dengan Harga
    st.subheader("4. Hubungan antara Ukuran RAM dan Harga")
    fig_ram_harga = px.box(
        df.sort_values('Ram'), 
        x='Ram', 
        y='Harga_IDR', 
        title='Box Plot Harga berdasarkan Ukuran RAM',
        labels={'Ram': 'RAM (GB)', 'Harga_IDR': 'Harga (Juta Rupiah)'}
    )
    st.plotly_chart(fig_ram_harga, use_container_width=True)

    # 5. Heatmap Korelasi
    st.subheader("5. Heatmap Korelasi Fitur Numerik")
    # Memilih hanya kolom numerik untuk perhitungan korelasi
    numeric_df = df.select_dtypes(include=np.number)
    # Menghapus kolom Euro asli agar tidak redundan
    if 'Price_euros' in numeric_df.columns:
        numeric_df = numeric_df.drop('Price_euros', axis=1)
        
    corr = numeric_df.corr()
    
    fig_heatmap, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Korelasi antara Fitur-fitur Numerik')
    st.pyplot(fig_heatmap)
else:
    st.warning("Analisis tidak dapat ditampilkan karena data gagal dimuat.")
