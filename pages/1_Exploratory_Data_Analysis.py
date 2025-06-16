import streamlit as st

import pandas as pd

import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt



st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")



st.title("📊 Exploratory Data Analysis (EDA)")

st.markdown("Mari kita jelajahi karakteristik dari dataset harga laptop.")



@st.cache_data

def load_data():

    try:

        df = pd.read_csv('laptop_prices.csv', encoding='latin-1')

        

        if df['Ram'].dtype == 'object':

            df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')

        if df['Weight'].dtype == 'object':

            df['Weight'] = df['Weight'].str.replace('kg', '').astype('float32')

            

        return df

    except (FileNotFoundError, AttributeError) as e:

        st.error(f"Gagal memuat atau memproses data: {e}")

        return None



df = load_data()



if df is not None and 'Inches' in df.columns:

    st.header("Informasi Dasar Dataset")

    st.dataframe(df.head())



    st.subheader("Statistik Deskriptif")

    st.write(df.describe())



    st.header("Visualisasi Data")



    st.subheader("1. Distribusi Harga Laptop")

    fig_price = px.histogram(df, x='Price_euros', nbins=50, title='Distribusi Harga Laptop (dalam Euro)')

    st.plotly_chart(fig_price, use_container_width=True)



    col1, col2 = st.columns(2)



    with col1:

        st.subheader("2. Jumlah Laptop per Brand")

        company_counts = df['Company'].value_counts().reset_index()

        company_counts.columns = ['Brand', 'Jumlah']

        fig_company = px.bar(company_counts, x='Brand', y='Jumlah', title='Jumlah Laptop Berdasarkan Brand', labels={'Brand': 'Brand Laptop', 'Jumlah': 'Jumlah Unit'})

        st.plotly_chart(fig_company, use_container_width=True)



    with col2:

        st.subheader("3. Proporsi Tipe Laptop")

        fig_type = px.pie(df, names='TypeName', title='Proporsi Tipe Laptop', hole=0.3)

        st.plotly_chart(fig_type, use_container_width=True)



    st.subheader("4. Hubungan RAM dengan Harga")

    fig_ram_price = px.box(df, x='Ram', y='Price_euros', title='Box Plot Harga berdasarkan Ukuran RAM', labels={'Ram': 'RAM (GB)', 'Price_euros': 'Harga (Euro)'})

    st.plotly_chart(fig_ram_price, use_container_width=True)



    st.subheader("5. Heatmap Korelasi Fitur Numerik")

    numeric_df = df.select_dtypes(include=['int32', 'float32', 'float64'])

    corr = numeric_df.corr()

    fig_heatmap, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)

    st.pyplot(fig_heatmap)

else:

    st.warning("Data tidak dapat dimuat atau kolom 'Inches' tidak dapat dibuat.")
