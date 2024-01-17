import pandas as pd
import numpy as np
import nltk
import os
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
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

# Feature extraction (using CountVectorizer and TF-IDF)
count_vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vectorizer.fit_transform(X_train)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Train a Multinomial Naive Bayes model
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Define X_test_tfidf as a global variable
X_test_tfidf = None

# Function to classify a new email and update the model
def classify_and_update_model(new_email):
    global data  # Declare 'data' as a global variable
    global X_test_tfidf  # Declare 'X_test_tfidf' as a global variable
    # Preprocess the new email text
    new_email = process_text(new_email)

    # Classify the new email using the trained model
    new_email_counts = count_vectorizer.transform([new_email])
    new_email_tfidf = tfidf_transformer.transform(new_email_counts)
    predicted_label = clf.predict(new_email_tfidf)

    # Provide the classification result to the user
    if predicted_label[0] == 0:
        print("The given email is NOT spam.")
    else:
        print("The given email is SPAM.")

    # Ask the user to confirm the classification
    user_confirmation = input("Is this classification correct? (yes/no): ").lower()

    if user_confirmation == "no":
        # If the classification is incorrect, ask the user to provide the correct label
        corrected_label = int(input("Please enter the correct label (0 for NOT spam, 1 for SPAM): "))
        
        # Add the new email and its corrected label to the training dataset
        new_data = pd.DataFrame({'subject': [new_email], 'message': [''], 'label': [corrected_label]})
        data = pd.concat([data, new_data], ignore_index=True)

        # Retrain the model with the updated training dataset
        X = data['subject'] + ' ' + data['message']
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_counts = count_vectorizer.fit_transform(X_train)
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        clf.fit(X_train_tfidf, y_train)

        print("Model has been updated with the corrected label.")

# Evaluate model accuracy before updating with new data
X_test_counts = count_vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
y_pred_before = clf.predict(X_test_tfidf)
accuracy_before_update = accuracy_score(y_test, y_pred_before)
print("Model Accuracy before update:", accuracy_before_update)

# Read a new email from the command prompt
new_email_text = input("Enter the text of the new email: ")
classify_and_update_model(new_email_text)

# Evaluate model accuracy after updating with new data
X_test_counts = count_vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
y_pred_after = clf.predict(X_test_tfidf)
accuracy_after_update = accuracy_score(y_test, y_pred_after)
print("Model Accuracy after update:", accuracy_after_update)

# Compare the results
if accuracy_after_update > accuracy_before_update:
    print("The model has improved with new data.")
elif accuracy_after_update < accuracy_before_update:
    print("The model's accuracy has decreased with new data.")
else:
    print("The model's accuracy remains the same with new data.")

# Create ROC curve for before and after update
fpr_before, tpr_before, _ = roc_curve(y_test, y_pred_before)
roc_auc_before = auc(fpr_before, tpr_before)

fpr_after, tpr_after, _ = roc_curve(y_test, y_pred_after)
roc_auc_after = auc(fpr_after, tpr_after)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_before, tpr_before, color='blue', lw=2, label=f'ROC curve (before update, area = {roc_auc_before:.2f})')
plt.plot(fpr_after, tpr_after, color='green', lw=2, label=f'ROC curve (after update, area = {roc_auc_after:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid('true')

# Create a bar chart to show the differences in accuracy before and after the update
accuracy_values = [accuracy_before_update, accuracy_after_update]
labels = ["Before Update", "After Update"]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels, accuracy_values, color=['blue', 'green'])
plt.title("Model Accuracy Comparison")
plt.xlabel("Model State")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid('true')

# Add values on top of the bars
for bar, value in zip(bars, accuracy_values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, value + 0.02, f'{value:.2f}', fontsize=12, color='black')

plt.tight_layout()
plt.show()
