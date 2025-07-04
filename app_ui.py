import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")



# Load dataset
@st.cache_data
def load_data():
    response = requests.get(f'{BASE_URL}/customers')
    return pd.DataFrame(response.json())

@st.cache_data
def load_reviews():
    response = requests.get(f'{BASE_URL}/reviews')
    return pd.DataFrame(response.json())

reviews = load_reviews()
data = load_data()



# Fungsi untuk menampilkan halaman
def dataSciencePage():
    st.title("Data Sains")
    st.subheader("Analisa Sentimen, Segmentasi Pelanggan, dan Rekomendasi Produk menggunakan Machine Learning.")

    # **📌 Statistik Umum**
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Rata-rata Usia", f"{data['Age'].mean():.1f}")
    col2.metric("Pendapatan Rata-rata", f"${data['Income'].mean():,.0f}")
    col3.metric("Rata-rata Spending Score", f"{data['SpendingScore'].mean():.1f}")
    col4.metric("Rata-rata Loyalty Score", f"{data['LoyaltyScore'].mean():.2f}")
    col5.metric("Rata-rata CLV", f"${data['CLV'].mean():,.2f}")

    # **📈 Visualisasi Cluster**
    st.subheader("📊 Segmentasi Pelanggan (Clustering)")
    fig = px.scatter(data, x="Income", y="SpendingScore", color=data["Cluster"].astype(str),
                    hover_data=["Name", "Age"], title="Segmentasi Pelanggan Berdasarkan Pendapatan & Skor Belanja")
    st.plotly_chart(fig)


    # **📈 Visualisasi Loyalty Score & CLV**
    st.subheader("📈 Loyalty Score vs CLV")
    fig = px.scatter(data, x="LoyaltyScore", y="CLV", color=data["Churn"].astype(str),
                    hover_data=["Name", "Age"], title="Loyalty Score vs CLV")
    st.plotly_chart(fig)

    # **📩 Download Laporan PDF**
    st.subheader("📜 Laporan Analisis Pelanggan")
    if st.button("Download Laporan PDF"):
        pdf_url = f'{BASE_URL}/report'
        st.markdown(f"[Klik di sini untuk mendownload laporan]({pdf_url})")

    # **🔮 Prediksi Cluster untuk Pelanggan Baru**
    st.subheader("🔮 Prediksi Segmentasi Pelanggan Baru")

    age = st.number_input("Masukkan Umur", min_value=18, max_value=80, step=1, key="1")
    income = st.number_input("Masukkan Pendapatan Tahunan", min_value=10000, max_value=200000, step=1000, key="2")
    spending = st.number_input("Masukkan Spending Score", min_value=0, max_value=100, step=1, key="3")

    if st.button("Prediksi Cluster", key="btn_cluster"):
        response = requests.post(f'{BASE_URL}/predict', json={"Age": age, "Income": income, "SpendingScore": spending})
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Pelanggan termasuk dalam Cluster: {result['Cluster']}")
        else:
            st.error("Gagal mendapatkan prediksi.")

    # **🔮 Prediksi Churn untuk Pelanggan Baru**
    st.subheader("🔮 Prediksi Churn")

    age_churn = st.number_input("Masukkan Umur", min_value=18, max_value=80, step=1, key="4")
    income_churn = st.number_input("Masukkan Pendapatan Tahunan", min_value=10000, max_value=200000, step=1000, key="5")
    spending_churn = st.number_input("Masukkan Spending Score", min_value=0, max_value=100, step=1, key="6")

    if st.button("Prediksi Churn", key="btn_churn"):
        response = requests.post(f'{BASE_URL}/predict_churn', json={"Age": age_churn, "Income": income_churn, "SpendingScore": spending_churn})
        
        if response.status_code == 200:
            result = response.json()
            print(result)
            churn_prob = result["Churn_Probability"]
            if churn_prob > 0.5:
                st.error(f"Pelanggan memiliki risiko churn tinggi: {churn_prob:.2%}")
            else:
                st.success(f"Pelanggan memiliki risiko churn rendah: {churn_prob:.2%}")
        else:
            st.error("Gagal mendapatkan prediksi.")

    # **📌 Tampilkan Ulasan Pelanggan**
    st.title("Ulasan Pelanggan & Analisis Sentimen")
    st.dataframe(reviews)
    # **🔍 Analisis Sentimen untuk Review Baru**
    st.subheader("🔍 Cek Sentimen Ulasan")

    review_text = st.text_area("Masukkan Ulasan Pelanggan", key="7")

    if st.button("Analisis Sentimen", key="8"):
        response = requests.post(f'{BASE_URL}/analyze_review', json={"Review": review_text})
        
        if response.status_code == 200:
            result = response.json()
            sentiment = result["Sentiment"]
            
            if sentiment == "Positif":
                st.success(f"Sentimen: {sentiment} 😊")
            elif sentiment == "Negatif":
                st.error(f"Sentimen: {sentiment} 😞")
            else:
                st.warning(f"Sentimen: {sentiment} 😐")
        else:
            st.error("Gagal menganalisis ulasan.")

    # **📌 Rekomendasi Produk**
    st.subheader("🛍 Rekomendasi Produk")

    customer_name = st.text_input("Masukkan Nama Pelanggan", key="9")

    if st.button("Dapatkan Rekomendasi", key="10"):
        response = requests.post(f'{BASE_URL}/recommend', json={"Name": customer_name})
        
        if response.status_code == 200:
            result = response.json()
            recommendations = result["Recommendations"]
            
            if recommendations:
                st.success(f"Produk yang direkomendasikan untuk {customer_name}: {', '.join(recommendations)}")
            else:
                st.warning(f"Tidak ada rekomendasi untuk {customer_name}.")
        else:
            st.error("Gagal mendapatkan rekomendasi.")

def chatbotPage():
    st.title("Chatbot")
    st.write("Chatbot adalah program komputer yang dirancang untuk meniru percakapan manusia dan memberikan respon yang mirip dengan manusia. Biasanya, chatbot digunakan untuk membantu pengguna dalam melakukan tugas-tugas tertentu atau memberikan informasi yang dibutuhkan. Chatbot menggunakan algoritma dan kecerdasan buatan untuk memahami bahasa manusia dan merespons dengan tepat. Seiring dengan perkembangan teknologi, chatbot semakin canggih dan mampu menangani percakapan yang lebih kompleks.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Anda:")

    if st.button("Kirim"):
        if user_input:
            response = requests.post(f'{BASE_URL}/chat', json={"message": user_input})
            bot_response = response.json().get("response", "Error")
            
            # Simpan percakapan ke dalam session state
            st.session_state.chat_history.append(("Anda", user_input))
            st.session_state.chat_history.append(("Chatbot", bot_response))

    # Tampilkan chat history dengan multi-scroll UI
    st.subheader("Percakapan")
    chat_container = st.container()

    with chat_container:
        for sender, message in st.session_state.chat_history:
            st.markdown(f"**{sender}:** {message}")

def informationExtractorPage():
    st.title("PDF Information Extractor")
    st.write("Information Extractor digunakan untuk mengekstrak teks dan entitas dari file PDF. Anda dapat mengunggah file PDF dan melihat teks yang diekstrak beserta entitas yang teridentifikasi.")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        files = {"file": uploaded_file.getvalue()}
        
        with st.spinner("Processing..."):
            response = requests.post(f'{BASE_URL}/extract', files=files)
            
            if response.status_code == 200:
                data = response.json()
                st.subheader("Extracted Text:")
                st.text_area("Text", data["text"], height=300)

                st.subheader("Extracted Entities:")
                for entity in data["entities"]:
                    st.write(f'{entity} : {data["entities"][entity]}')
            else:
                st.error("Failed to process the file.")


def faceRecognitionPage():
    st.title("Face Recognition")
    st.subheader('Register New User')
    name = st.text_input('Name')
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if st.button('Register'):
        if name and uploaded_file:
            files = {'image': uploaded_file}
            data = {'name': name}
            response = requests.post(f'{BASE_URL}/register', files=files, data=data)

            if response.status_code == 200:
                st.success(response.json().get('message'))
            else:
                st.error(response.json().get('message'))
        else:
            st.error('Please provide both name and image.')
     
    st.header('Recognize User')
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if st.button('Recognize'):
        if uploaded_file:
            files = {'image': uploaded_file}
            response = requests.post(f'{BASE_URL}/recognize', files=files)

            if response.status_code == 200:
                name = response.json().get('name')
                st.success(f'Recognized as: {name}')
            else:
                st.error(response.json().get('message'))
        else:
            st.error('Please upload an image.')


# Sidebar Navigasi
st.sidebar.title("CKA Artificial Intelligence")
menu = st.sidebar.radio("List Application", ["Data Science", "Chatbot", "Information Extractor", "Face Recognition"])

# Routing halaman berdasarkan menu yang dipilih
if menu == "Data Science":
    dataSciencePage()
elif menu == "Chatbot":
    chatbotPage()
elif menu == "Information Extractor":
    informationExtractorPage()
elif menu == "Face Recognition":
    faceRecognitionPage()