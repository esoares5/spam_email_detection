import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

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

# Feature extraction (using TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_tfidf, y_train)

# Function to classify a new email
def classify_new_email(new_email_text):
    new_email_text = process_text(new_email_text)
    new_email_tfidf = tfidf_vectorizer.transform([new_email_text])
    prediction = clf.predict(new_email_tfidf)[0]
    if prediction == 0:
        return "Not Spam"
    else:
        return "Spam"

# Prompt the user to input a new email
new_email = input("Enter a new email message: ")

# Classify the new email
classification_result = classify_new_email(new_email)
print("Classification result: This email is", classification_result)

# ... (remaining code for user feedback, evaluation, etc.)

# Re-evaluate the model after updating it
y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Updated Model Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the percentage of spam and hams identified correctly versus incorrectly
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap='viridis')  # Change the cmap here
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Ham", "Spam"])
plt.yticks(tick_marks, ["Ham", "Spam"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix[i][j]), horizontalalignment='center', verticalalignment='center', color='white')

plt.show()
