import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def map_sentiment(rating):
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'

def clean_text(text):
    text = str(text).lower() 
    text = re.sub(r'[^a-z\s]', '', text) 
    words = text.split() 
    cleaned_words = [word for word in words if word not in stop_words] 
    return ' '.join(cleaned_words)

def main():
    print("Loading data...")
    try:
        df = pd.read_excel('Group Details.xlsx', engine='openpyxl')
    except Exception as e:
        df = pd.read_excel('Group Details.xlsx')
        
    print("Preprocessing data...")
    df['Sentiment'] = df['rating'].apply(map_sentiment)
    df['cleaned_review'] = df['body'].apply(clean_text)
    
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['Sentiment']
    
    print("Training model...")
    model = LinearSVC()
    model.fit(X, y)
    
    print("Saving model and vectorizer...")
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    joblib.dump(model, 'sentiment_model.pkl')
    print("Done!")

if __name__ == '__main__':
    main()
