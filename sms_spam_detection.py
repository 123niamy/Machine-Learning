# Step 1: Load and preprocess SMS dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Load dataset
data_path = '02-machine-learning/sms_spam_data/SMSSpamCollection'
df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'text'])

# Basic preprocessing: lowercase, strip whitespace
df['text'] = df['text'].str.lower().str.strip()

# Encode labels: ham=0, spam=1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Split dataset
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
	df['text'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
)

# Step 2: TF-IDF vectorization
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Step 3: Train Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

# Step 4: Evaluate model performance
y_pred = nb.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Model Evaluation Report:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# Step 5: Prediction script
def predict_sms(message: str) -> str:
	# Preprocess input
	msg_clean = message.lower().strip()
	msg_tfidf = tfidf.transform([msg_clean])
	pred = nb.predict(msg_tfidf)[0]
	return 'spam' if pred == 1 else 'ham'

# Example usage:
if __name__ == "__main__":
	test_sms = "Congratulations! You've won a free ticket. Reply now!"
	result = predict_sms(test_sms)
	print(f"Prediction for test SMS: '{test_sms}' => {result}")
