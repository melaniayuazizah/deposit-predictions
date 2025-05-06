import streamlit as st
import pandas as pd
import pickle

# Load model dan kolom yang dipakai saat training
with open("model_logregg.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    trained_columns = pickle.load(f)

st.set_page_config(layout='centered', initial_sidebar_state='expanded', page_title='Prediksi Deposito', menu_items= {'Get Help': 'https://docs.streamlit.io/en/stable/'})
st.title("Prediksi Deposito")
st.write("Akan diprediksi apakah seseorang akan berlangganan deposito berjangka")

# Input user
umur = st.number_input("Umur", min_value=18, max_value=100, value=25)
education = st.selectbox("Education", ["illiterate", "university.degree", "professional.course","basic.9y","basic.6y","basic.4y","high.school","unknown"])
job = st.selectbox("Job", ["admin", "enterpreneur", "student","retired","technician","management","unemployed","services","self-employed","blue-collar","housemaid","unknown"])
marital = st.selectbox("Marital", ["single", "married", "divorce", "unknown"])

if st.button("Prediksi"):
    # Buat dataframe dari input
    input_data = pd.DataFrame({
        "age": [umur],
        "education": [education],
        "job": [job],
        "marital": [marital]
    })

    # Konversi ke dummy dan sesuaikan kolom
    input_dummy = pd.get_dummies(input_data)
    input_dummy = input_dummy.reindex(columns=trained_columns, fill_value=0)
    # Prediksi
    pred = model.predict(input_dummy)[0]

    # Tampilkan hasil
    if pred == 1:
        st.markdown(
        "<div style='background-color:#d4edda;padding:10px;border-radius:5px;color:#155724;'>"
        "<b>YES </b> - Nasabah akan belangganan deposito berjangka ✅"
        "</div>", unsafe_allow_html=True)
    else:
        st.markdown(
        "<div style='background-color:#f8d7da;padding:10px;border-radius:5px;color:#721c24;'>"
        "<b>NO </b> - Nasabah tidak akan berlangganan deposito berjangka ❌"
        "</div>", unsafe_allow_html=True)




