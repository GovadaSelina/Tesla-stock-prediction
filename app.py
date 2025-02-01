import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import re

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load datasets
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Add labels
df_fake["class"] = 0
df_true["class"] = 1

# Combine datasets
df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)
df["text"] = df["text"].apply(preprocess_text)

# Split data
X = df["text"]
y = df["class"]

# Vectorize text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title("Fake News Detection")

# User input
user_input = st.text_area("Enter news content:", height=200)

if st.button("Predict"):
    if user_input:
        preprocessed_text = preprocess_text(user_input)
        input_vector = vectorizer.transform([preprocessed_text])
        prediction = model.predict(input_vector)
        result = "FAKE NEWS" if prediction[0] == 0 else "TRUE NEWS"
        st.write(f"The given news is classified as: **{result}**")
    else:
        st.write("Please enter news content to predict.")
