# ==========================================================
# SOCIAL MEDIA SENTIMENT ANALYSIS USING MACHINE LEARNING
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from wordcloud import WordCloud

# ==========================================================
# DOWNLOAD NLTK DATA
# ==========================================================

nltk.download('stopwords')

# ==========================================================
# LOAD DATASET
# ==========================================================

data = {
    'text': [
        "I love this phone",
        "Excellent product",
        "Amazing service",
        "Very happy with purchase",
        "Worst experience ever",
        "Terrible customer support",
        "Not satisfied",
        "Very disappointing",
        "Average quality",
        "It's okay",
        "Nothing special",
        "Can be improved",
        "Fantastic performance",
        "Highly recommended",
        "Poor battery life",
        "Waste of money"
    ],
    'sentiment': [
        'Positive',
        'Positive',
        'Positive',
        'Positive',
        'Negative',
        'Negative',
        'Negative',
        'Negative',
        'Neutral',
        'Neutral',
        'Neutral',
        'Neutral',
        'Positive',
        'Positive',
        'Negative',
        'Negative'
    ]
}

df = pd.DataFrame(data)

print("\nDataset Shape:", df.shape)
print(df.head())

# ==========================================================
# DATA CLEANING
# ==========================================================

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# ==========================================================
# TEXT PREPROCESSING
# ==========================================================

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["clean_text"] = df["text"].apply(preprocess_text)

# ==========================================================
# EXPLORATORY DATA ANALYSIS
# ==========================================================

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='sentiment')
plt.title("Sentiment Distribution")
plt.show()

# ==========================================================
# WORD CLOUD
# ==========================================================

all_words = " ".join(df["clean_text"])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(all_words)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud")
plt.show()

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["clean_text"])

y = df["sentiment"]

# ==========================================================
# SPLIT DATA
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# MODEL TRAINING
# ==========================================================

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Naive Bayes":
        MultinomialNB(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}

best_model = None
best_accuracy = 0

print("\nMODEL COMPARISON")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"\n{name}")
    print("Accuracy:", accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# ==========================================================
# EVALUATION
# ==========================================================

final_predictions = best_model.predict(X_test)

print("\nBEST MODEL ACCURACY")
print(best_accuracy)

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        final_predictions
    )
)

cm = confusion_matrix(
    y_test,
    final_predictions
)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    best_model,
    "sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)

print("\nModel Saved Successfully")

# ==========================================================
# LOAD MODEL
# ==========================================================

loaded_model = joblib.load(
    "sentiment_model.pkl"
)

loaded_vectorizer = joblib.load(
    "vectorizer.pkl"
)

# ==========================================================
# REAL TIME PREDICTION
# ==========================================================

print("\nSOCIAL MEDIA SENTIMENT PREDICTION")

while True:

    user_text = input(
        "\nEnter a post (type exit to quit): "
    )

    if user_text.lower() == "exit":
        break

    cleaned = preprocess_text(
        user_text
    )

    transformed = loaded_vectorizer.transform(
        [cleaned]
    )

    prediction = loaded_model.predict(
        transformed
    )[0]

    print(
        "\nPredicted Sentiment:",
        prediction
    )

print("\nProgram Ended Successfully")