import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import gdown
import os

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)


# Load Models
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        model = joblib.load("sentiment_model.pkl")
        return vectorizer, model
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first.")
        return None, None


vectorizer, model = load_models()

# Streamlit App
st.set_page_config(
    page_title="Sentiment Analysis App", page_icon="😊", layout="centered"
)

st.title("Product Review Sentiment Analysis")
st.markdown("Enter a product review below to analyze its sentiment.")

st.sidebar.header("About")
st.sidebar.info(
    "This application uses a trained Linear SVC model to predict whether a review is Positive, Negative, or Neutral based on its content."
)

user_input = st.text_area("Review text", height=150)

if st.button("Predict Sentiment"):
    if user_input:
        if vectorizer and model:
            cleaned_input = clean_text(user_input)
            transformed_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(transformed_input)[0]

            st.markdown("### Prediction:")
            if prediction == "Positive":
                st.success(f"**{prediction}** ")
            elif prediction == "Negative":
                st.error(f"**{prediction}** ")
            else:
                st.warning(f"**{prediction}** ")
    else:
        st.warning("Please enter a review to analyze.")
