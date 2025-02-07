from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity  
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Load dataset
data = pd.read_csv("customer_data.csv")

products = ["Laptop", "Smartphone", "Tablet", "Headphone", "Smartwatch", "Kamera", "Speaker", "Mouse", "Keyboard"]


# Preprocessing: Standarisasi Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[["Age", "Income", "SpendingScore"]])

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data["Cluster"] = kmeans.fit_predict(data_scaled)

# **Loyalty Score** → Skor berdasarkan jumlah transaksi & nilai total pembelian
data["LoyaltyScore"] = (data["SpendingScore"] + (data["Income"] / 1000)) / 2

# **Customer Lifetime Value (CLV)** → Menghitung CLV sederhana
data["CLV"] = data["SpendingScore"] * data["Income"] / 1000

# **Prediksi Churn**
# Simulasi: Anggap pelanggan dengan Spending Score < 30 memiliki risiko churn tinggi
data["Churn"] = (data["SpendingScore"] < 30).astype(int)

# Model Prediksi Churn
X = data[["Age", "Income", "SpendingScore", "LoyaltyScore", "CLV"]]
y = data["Churn"]
churn_model = xgb.XGBClassifier()
churn_model.fit(X, y)

@app.route('/customers', methods=['GET'])
def get_customers():
    return jsonify(data.to_dict(orient="records"))

@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    try:
        request_data = request.get_json()
        age_churn = request_data["Age"]
        income_churn = request_data["Income"]
        spending_churn = request_data["SpendingScore"]
        
        loyalty = (spending_churn + (income_churn / 1000)) / 2
        clv = spending_churn * income_churn / 1000

        input_data = np.array([[age_churn, income_churn, spending_churn, loyalty, clv]])
        churn_prob = churn_model.predict_proba(input_data)[0][1]  
        return jsonify({"Churn_Probability": float(churn_prob)})
    except Exception as e:
        print('Res Error')
        return jsonify({"error": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.get_json()
        age = request_data["Age"]
        income = request_data["Income"]
        spending = request_data["SpendingScore"]

        # Standarisasi input
        input_scaled = scaler.transform([[age, income, spending]])
        cluster = kmeans.predict(input_scaled)

        return jsonify({"Cluster": int(cluster[0])})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/report', methods=['GET'])
def generate_report():
    pdf_filename = "customer_report.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.drawString(100, 750, "Customer Analytics Report")
    
    y_position = 730
    for index, row in data.iterrows():
        text = f"{row['Name']} - Age: {row['Age']}, Income: {row['Income']}, Spending Score: {row['SpendingScore']}, Cluster: {row['Cluster']}"
        c.drawString(100, y_position, text)
        y_position -= 20
        if y_position < 100:
            c.showPage()
            y_position = 750

    c.save()
    return send_file(pdf_filename, as_attachment=True)

@app.route('/report_churn', methods=['GET'])
def generate_report_churn():
    pdf_filename = "customer_report_churn.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.drawString(100, 750, "Customer Analytics Report")
    
    y_position = 730
    for index, row in data.iterrows():
        text = f"{row['Name']} - Loyalty: {row['LoyaltyScore']:.2f}, CLV: {row['CLV']:.2f}, Churn Risk: {row['Churn']}"
        c.drawString(100, y_position, text)
        y_position -= 20
        if y_position < 100:
            c.showPage()
            y_position = 750

    c.save()
    return send_file(pdf_filename, as_attachment=True)

# **Fungsi Analisis Sentimen**
def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    return "Positif" if sentiment > 0 else "Negatif" if sentiment < 0 else "Netral"

data["Sentiment"] = data["Review"].apply(analyze_sentiment)

@app.route('/reviews', methods=['GET'])
def get_reviews():
    return jsonify(data[["Name", "Review", "Sentiment"]].to_dict(orient="records"))

@app.route('/analyze_review', methods=['POST'])
def analyze_review():
    try:
        request_data = request.get_json()
        review_text = request_data["Review"]
        sentiment = analyze_sentiment(review_text)
        return jsonify({"Review": review_text, "Sentiment": sentiment})
    except Exception as e:
        return jsonify({"error": str(e)})


# **Fungsi Rekomendasi Produk**
def recommend_products(customer_name):
    if customer_name not in data["Name"].values:
        return []
    
    # Ambil data pelanggan
    customer_index = data[data["Name"] == customer_name].index[0]
    # Hitung kemiripan dengan pelanggan lain
    product_matrix = data.iloc[:, 1:].values
    similarities = cosine_similarity(product_matrix)
    
    
    # Rekomendasi berdasarkan pelanggan paling mirip
    similar_customer_index = np.argsort(similarities[customer_index])[-2]  # Ambil pelanggan paling mirip selain dirinya sendiri
    recommended_products = []
    
    for product in products:
        if data.loc[customer_index, product] == 0 and data.loc[similar_customer_index, product] == 1:
            recommended_products.append(product)
    
    return recommended_products

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    try:
        request_data = request.get_json()
        customer_name = request_data["Name"]
        recommendations = recommend_products(customer_name)
        
        return jsonify({"Name": customer_name, "Recommendations": str(recommendations)})
    except Exception as e:
        return jsonify({"error": str(e)})

# CHATBOT

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Inisialisasi Chatbot dengan LangChain
llm = OpenAI()
conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    
    response = conversation.predict(input=user_input)
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
