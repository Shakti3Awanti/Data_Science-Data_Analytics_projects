# Product Review Sentiment Analysis

A Streamlit-based NLP project that classifies customer product reviews as **Positive**, **Neutral**, or **Negative**.  
Built for portfolio and job-search visibility, this project demonstrates the full workflow from text preprocessing and model training to interactive deployment.

## Overview

This project takes customer review text, cleans and vectorizes it with TF-IDF, and uses a trained **LinearSVC** model to predict sentiment. The app is deployed with **Streamlit** for a simple, interactive user experience.

## Business Objective

Extract sentiment from customer reviews on a product and present the result in a deployed web application.

## Key Features

- Text preprocessing with lowercasing, punctuation removal, and stopword filtering
- TF-IDF feature extraction
- Sentiment classification into **Positive**, **Neutral**, and **Negative**
- Streamlit web interface for live prediction
- Cached model loading for faster app startup

## Tech Stack

- **Python**
- **Pandas**
- **scikit-learn**
- **NLTK**
- **Streamlit**
- **Joblib**

## Project Structure

```text
.
├── app.py
├── train_save_model.py
├── model_code.py
├── Group Details.xlsx
├── tfidf_vectorizer.pkl
├── sentiment_model.pkl
├── Instructions.md
├── NLP[EDA+Model EVAL]shakti.ipynb
├── Untitled-1.ipynb
└── P652-Group Details - Sheet1.pdf
```

## How It Works

1. Reviews are loaded from `Group Details.xlsx`.
2. Ratings are mapped to sentiment labels:
   - `1–2` → Negative
   - `3` → Neutral
   - `4–5` → Positive
3. Review text is cleaned using NLTK stopwords and regex preprocessing.
4. TF-IDF converts the cleaned text into numerical features.
5. A **LinearSVC** model is trained and saved as `sentiment_model.pkl`.
6. The Streamlit app loads the model and predicts sentiment for new input text.

## Setup and Installation

### 1) Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2) Create a virtual environment (recommended)
```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install pandas scikit-learn nltk streamlit openpyxl joblib
```

## Train the Model

Run the training script to generate the vectorizer and classifier:

```bash
python train_save_model.py
```

This will create:

- `tfidf_vectorizer.pkl`
- `sentiment_model.pkl`

## Run the App

Start the Streamlit application with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```bash
http://localhost:8501
```

## Usage

1. Paste a product review into the text box.
2. Click **Predict Sentiment**.
3. View the predicted label on the screen.

## Example

**Input:**
> The product quality is excellent and delivery was fast.

**Output:**
> Positive

## Model Notes

- The final deployed model uses **TF-IDF + LinearSVC**
- The application caches loaded models with `st.cache_resource`
- The preprocessing pipeline is shared between training and inference to keep predictions consistent



