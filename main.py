import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("news.csv")

# Features and labels
X = data['text']
y = data['label']

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test custom input
news = input("Enter news: ")
news_vector = vectorizer.transform([news])
prediction = model.predict(news_vector)

print("Prediction:", prediction[0])
