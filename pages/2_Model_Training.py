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

KURS_EUR_TO_IDR = 0.0175

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('laptop_prices.csv', encoding='latin-1')
        
        if 'Ram' in df.columns and df['Ram'].dtype == 'object':
            df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')
        if 'Weight' in df.columns and df['Weight'].dtype == 'object':
            df['Weight'] = df['Weight'].str.replace('kg', '').astype('float32')
        df['Harga_IDR'] = (df['Price_euros'] * KURS_EUR_TO_IDR).astype(np.int64)
        
        df.dropna(subset=['Inches', 'Weight', 'Ram'], inplace=True)
        return df
    except (FileNotFoundError, AttributeError, KeyError) as e:
        st.error(f"Gagal memuat atau memproses data: {e}")
        return None

df = load_data()

if df is not None and 'Inches' in df.columns:
    st.header("Persiapan Data untuk Pelatihan")

    features = ['Company', 'TypeName', 'Ram', 'Weight', 'OS', 'Inches']
    target = 'Harga_IDR'
    
    X = df[features]
    y = df[target]

    st.write("Fitur yang akan digunakan:", features)
    st.write("Target prediksi:", target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.markdown(f"Data telah dibagi menjadi:  \n- **Data Latih**: {X_train.shape[0]} baris  \n- **Data Uji**: {X_test.shape[0]} baris")

    numeric_features = ['Ram', 'Weight', 'Inches']
    categorical_features = ['Company', 'TypeName', 'OS']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    st.header("Mulai Pelatihan Model")
    if st.button("Latih Model Sekarang", type="primary"):
        with st.spinner("Model sedang dilatih, mohon tunggu..."):
            model.fit(X_train, y_train)
            joblib.dump(model, 'model_pipeline.joblib')
            
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            st.success("🎉 Model berhasil dilatih dan disimpan!")
            st.metric(label="**R-squared (R²)**", value=f"{r2:.2f}")
            st.metric(label="**Mean Absolute Error (MAE)**", value=f"€{mae:.2f}")
else:
    st.error("Gagal memuat data atau membuat fitur 'Inches'. Pastikan file 'laptop_prices.csv' benar dan memiliki kolom 'ScreenResolution'. Pelatihan tidak dapat dilanjutkan.")
