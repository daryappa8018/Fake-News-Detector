import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Load datasets
fake_news = pd.read_csv('Fake.csv')
real_news = pd.read_csv('True.csv')

# Label the data
fake_news['class'] = 0
real_news['class'] = 1

# Keep last 10 rows separately for manual testing
manual_fake = fake_news.tail(10)
manual_real = real_news.tail(10)

# Remove those last 10 rows from the main dataset
fake_news = fake_news[:-10]
real_news = real_news[:-10]

# Combine the datasets
combined_data = pd.concat([fake_news, real_news], ignore_index=True)

# Drop unnecessary columns
combined_data = combined_data.drop(['title', 'subject', 'date'], axis=1)

# Shuffle the dataset
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

# Clean the text column
combined_data['text'] = combined_data['text'].apply(clean_text)

X = combined_data['text']
y = combined_data['class']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Vectorize text using TF-IDF
tfidf = TfidfVectorizer()
X_train_vectors = tfidf.fit_transform(X_train)
X_test_vectors = tfidf.transform(X_test)

# Initialize and train models
log_reg = LogisticRegression()
log_reg.fit(X_train_vectors, y_train)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_vectors, y_train)

grad_boost = GradientBoostingClassifier(random_state=42)
grad_boost.fit(X_train_vectors, y_train)

rand_forest = RandomForestClassifier(random_state=42)
rand_forest.fit(X_train_vectors, y_train)

# Predict and display reports for each model
models = {
    "Logistic Regression": log_reg,
    "Decision Tree": decision_tree,
    "Gradient Boosting": grad_boost,
    "Random Forest": rand_forest
}

for name, model in models.items():
    predictions = model.predict(X_test_vectors)
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, predictions))


def predict_news_label(label):
    return "Fake News" if label == 0 else "Not Fake News"

def test_custom_news(text):
    cleaned_text = clean_text(text)
    vector = tfidf.transform([cleaned_text])

    predictions = {
        "Logistic Regression": log_reg.predict(vector)[0],
        "Decision Tree": decision_tree.predict(vector)[0],
        "Gradient Boosting": grad_boost.predict(vector)[0],
        "Random Forest": rand_forest.predict(vector)[0]
    }

    for model_name, pred in predictions.items():
        print(f"{model_name} Prediction: {predict_news_label(pred)}")


# Input from user
user_input = input("Enter news text: ")
test_custom_news(user_input)

