import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load your labeled dataset from 'cleaned_messages.csv'
data = pd.read_csv('cleaned_messages.csv')

# Tokenization, removing punctuation, and stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def process_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['subject'] = data['subject'].apply(process_text)
data['message'] = data['message'].apply(process_text)

# Splitting the data
X = data['subject'] + ' ' + data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction (using CountVectorizer and TF-IDF)
count_vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vectorizer.fit_transform(X_train)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Train a Multinomial Naive Bayes model
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Transform and predict on the test data
X_test_counts = count_vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
