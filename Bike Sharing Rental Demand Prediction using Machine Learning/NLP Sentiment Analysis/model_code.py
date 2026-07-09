import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

try:
    import openpyxl
except ImportError:
    print("openpyxl not installed, install if reading .xlsx files")
df = pd.read_excel('Group Details.xlsx', engine='openpyxl')
df.head()
df = pd.read_excel('Group Details.xlsx')
df
df.head()
df.tail()
df.isnull().sum()
df[['rating','body']].head()
def map_sentiment(rating):
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'
df['Sentiment'] = df['rating'].apply(map_sentiment)

df['Sentiment'].value_counts()
plt.figure(figsize=(8,5))
sns.countplot(x='Sentiment',order=['Negative','Neutral','Positive'],
              data=df,palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')

import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower() 
    text = re.sub(r'[^a-z\s]', '', text) 
    words = text.split() 
    cleaned_words = [word for word in words if word not in stop_words] 
    
    return ' '.join(cleaned_words)
df['cleaned_review'] = df['body'].apply(clean_text)
print(df[['body', 'cleaned_review']].head())
from sklearn.feature_extraction.text import TfidfVectorizer
vecotorizer=TfidfVectorizer(max_features=5000)
x=vecotorizer.fit_transform(df['cleaned_review'])
y=df['Sentiment']
TfidfVectorizer(ngram_range=(1,2))
df
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)
cm
print(classification_report(y_test,y_pred))
df['Sentiment'].value_counts()
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression(max_iter=1000)
lr_model.fit(x_train,y_train)
y_pred_lr=lr_model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred_lr)
accuracy
cm=confusion_matrix(y_test,y_pred_lr)
cm
print(classification_report(y_test,y_pred_lr))
from sklearn.svm import LinearSVC
svm_model=LinearSVC()
svm_model.fit(x_train,y_train)
y_pred_svm=svm_model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred_svm)
accuracy
cm=confusion_matrix(y_test,y_pred_svm)
cm
print(classification_report(y_test,y_pred_svm))
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative','Neutral','Positive'],
            yticklabels=['Negative','Neutral','Positive'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM Model")

plt.show()
