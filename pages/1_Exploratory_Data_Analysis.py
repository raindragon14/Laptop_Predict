import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis (EDA)")
st.markdown("Mari kita jelajahi karakteristik dari dataset harga laptop.")
KURS_EUR_TO_IDR = 0.0175

# Fungsi untuk memuat data (dengan caching dan perbaikan)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('laptop_prices.csv', encoding='latin-1')
        
        if df['Ram'].dtype == 'object':
            df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')
        
        if df['Weight'].dtype == 'object':
            df['Weight'] = df['Weight'].str.replace('kg', '').astype('float32')

        df['Harga_IDR'] = (df['Price_euros'] * KURS_EUR_TO_IDR).astype(np.int64)
            
        return df
    except FileNotFoundError:
        st.error("Dataset tidak ditemukan. Pastikan 'laptop_prices.csv' ada.")
        return None

df = load_data()

if df is not None:
    st.header("Informasi Dasar Dataset")
    st.dataframe(df.head())

    st.subheader("Statistik Deskriptif")
    st.write(df.describe())

    # --- Visualisasi ---
    st.header("Visualisasi Data")

    # 1. Distribusi Harga
    st.subheader("1. Distribusi Harga Laptop")
    fig_price = px.histogram(df, x='Harga_IDR', nbins=50, title='Distribusi Harga Laptop (Satuan Juta)')
    st.plotly_chart(fig_price, use_container_width=True)
    st.markdown("""
    **Insight**: Distribusi harga cenderung *right-skewed*, yang berarti sebagian besar laptop berada di rentang harga yang lebih rendah, dengan beberapa model premium yang sangat mahal.
    """)

    col1, col2 = st.columns(2)

    with col1:
        # 2. Jumlah Laptop per Perusahaan (Dengan Perbaikan)
        st.subheader("2. Jumlah Laptop per Brand")

        # Buat DataFrame baru dengan nama kolom yang jelas
        company_counts = df['Company'].value_counts().reset_index()
        company_counts.columns = ['Brand', 'Jumlah']  # Ganti nama kolom

        # Gunakan nama kolom baru di dalam plot
        fig_company = px.bar(company_counts,
                             x='Brand',
                             y='Jumlah',
                             title='Jumlah Laptop Berdasarkan Brand',
                             labels={'Brand': 'Brand Laptop', 'Jumlah': 'Jumlah Unit'})

        st.plotly_chart(fig_company, use_container_width=True)
        st.markdown("**Insight**: Dell, Lenovo, dan HP mendominasi pasar.")

    with col2:
        # 3. Tipe Laptop
        st.subheader("3. Proporsi Tipe Laptop")
        fig_type = px.pie(df, names='TypeName', title='Proporsi Tipe Laptop', hole=0.3)
        st.plotly_chart(fig_type, use_container_width=True)
        st.markdown("**Insight**: Notebook adalah tipe yang paling umum, diikuti oleh Gaming dan Ultrabook.")

    # 4. Hubungan antara RAM dan Harga
    st.subheader("4. Hubungan RAM dengan Harga")
    fig_ram_price = px.box(df, x='Ram', y='Harga_IDR', title='Box Plot Harga berdasarkan Ukuran RAM',
                           labels={'Ram': 'RAM (GB)', 'Harga_IDR': 'Harga'})
    st.plotly_chart(fig_ram_price, use_container_width=True)
    st.markdown("""
    **Insight**: Terlihat jelas korelasi positif. Semakin besar RAM, median harga cenderung semakin tinggi. Laptop dengan RAM 32GB memiliki variasi harga yang sangat luas.
    """)

    # 5. Korelasi antar fitur numerik
    st.subheader("5. Heatmap Korelasi Fitur Numerik")
    numeric_df = df.select_dtypes(include=['int32', 'float32', 'float64'])
    corr = numeric_df.corr()
    fig_heatmap, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig_heatmap)
    st.markdown("""
    **Insight**: **RAM** memiliki korelasi positif yang cukup kuat dengan harga. Ini adalah kandidat fitur yang baik untuk model kita.
    """)
else:
    st.warning("Data tidak dapat dimuat.")
