# File: fake_news_detector.py

import joblib
import re

model = joblib.load('C:\\Users\\bhanu\\Downloads\\fake_news_detection\\models\\ensemble_model.pkl')
vectorizer = joblib.load('C:\\Users\\bhanu\\Downloads\\fake_news_detection\\models\\tfidf_vectorizer.pkl')

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Prediction function
def predict_news(text):
    clean = preprocess(text)
    vectorized = vectorizer.transform([clean])
    prediction = model.predict(vectorized)
    return "Fake News" if prediction[0] == 1 else "Real News"
