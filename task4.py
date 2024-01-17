import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# Print predicted and actual values
print("\nPredicted Labels:\n", y_pred)
print("\nActual Labels:\n", y_test.values)  # Convert to NumPy array

# Calculate the actual percentage of each class in the test data
actual_ham_percentage = (np.sum(y_test == 0) / len(y_test)) * 100
actual_spam_percentage = (np.sum(y_test == 1) / len(y_test)) * 100

# Calculate the predicted percentage for each class
predicted_ham_percentage = (np.sum(y_pred == 0) / len(y_pred)) * 100
predicted_spam_percentage = (np.sum(y_pred == 1) / len(y_pred)) * 100

# Print accuracy percentages for Ham and Spam
print(f"Actual Percentage for Ham: {actual_ham_percentage:.2f}%")
print(f"Actual Percentage for Spam: {actual_spam_percentage:.2f}%")
print(f"Predicted Percentage for Ham: {predicted_ham_percentage:.2f}%")
print(f"Predicted Percentage for Spam: {predicted_spam_percentage:.2f}%")

# Calculate the percentage of correct predictions for each class
conf_matrix = confusion_matrix(y_test, y_pred)
total_ham = np.sum(conf_matrix[0, :])  # Total actual "Ham" instances
total_spam = np.sum(conf_matrix[1, :])  # Total actual "Spam" instances
correct_ham_percentage = (conf_matrix[0, 0] / total_ham) * 100
correct_spam_percentage = (conf_matrix[1, 1] / total_spam) * 100

# Calculate the actual percentage of each class in the test data
actual_ham_percentage = (np.sum(y_test == 0) / len(y_test)) * 100
actual_spam_percentage = (np.sum(y_test == 1) / len(y_test)) * 100

# Plot all four percentages
labels = ['Correctly Predicted Ham', 'Correctly Predicted Spam', 'Actual Ham Percentage', 'Actual Spam Percentage']
percentages = [correct_ham_percentage, correct_spam_percentage, actual_ham_percentage, actual_spam_percentage]

plt.figure(figsize=(10, 6))
plt.bar(labels, percentages, color=['blue', 'green', 'orange', 'red'])
plt.ylabel('Percentage')
plt.title('Percentages of Predictions and Actuals')
plt.ylim(0, 100)

plt.show()

# Create a confusion matrix plot
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
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
