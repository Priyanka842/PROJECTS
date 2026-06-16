import wave
from collections import Counter

import pandas as pd
import nltk
import speech_recognition as sr
import spacy

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download VADER lexicon (first run only)
nltk.download("vader_lexicon")

# =====================================================
# TASK 1: Speech Recognition + Audio Statistics
# =====================================================

with wave.open("sample_customer_call.wav", "rb") as audio:
    frame_rate = audio.getframerate()
    number_channels = audio.getnchannels()

recognizer = sr.Recognizer()

with sr.AudioFile("sample_customer_call.wav") as source:
    audio_data = recognizer.record(source)

transcribed_text = recognizer.recognize_google(audio_data)

# =====================================================
# TASK 2: Sentiment Analysis
# =====================================================

df = pd.read_csv("customer_call_transcriptions.csv")

sia = SentimentIntensityAnalyzer()

def predict_sentiment(text):
    score = sia.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["predicted_sentiment"] = df["text"].apply(predict_sentiment)

# True positives = predicted positive AND actual positive
true_positive = int(
    (
        (df["predicted_sentiment"] == "positive")
        & (df["sentiment_label"] == "positive")
    ).sum()
)

# =====================================================
# TASK 3: Named Entity Recognition
# =====================================================

nlp = spacy.load("en_core_web_sm")

entities = []

for text in df["text"]:
    doc = nlp(str(text))
    entities.extend([ent.text for ent in doc.ents])

most_freq_ent = Counter(entities).most_common(1)[0][0]

# =====================================================
# TASK 4: Similarity Search
# =====================================================

query = "wrong package delivery"

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(
    [query] + df["text"].astype(str).tolist()
)

similarities = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:]
).flatten()

most_similar_text = df.iloc[similarities.argmax()]["text"]

# =====================================================
# OUTPUT
# =====================================================

print("\n" + "=" * 50)
print("CUSTOMER CALL ANALYTICS RESULTS")
print("=" * 50)

print(f"\nFrame Rate: {frame_rate}")
print(f"Number of Channels: {number_channels}")

print("\nTranscribed Text:")
print(transcribed_text)

print(f"\nTrue Positive Predictions: {true_positive}")

print(f"\nMost Frequent Entity: {most_freq_ent}")

print("\nMost Similar Call:")
print(most_similar_text)