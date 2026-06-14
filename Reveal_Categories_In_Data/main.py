import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (only needed first time)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load dataset
reviews = pd.read_csv("reviews.csv")

# Keep only negative reviews (scores 1 and 2)
negative_reviews = reviews[reviews['score'].isin([1, 2])].copy()

# English stop words
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())

    cleaned_tokens = [
        word
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(cleaned_tokens)

# Create preprocessed reviews dataframe
preprocessed_reviews = pd.DataFrame()

preprocessed_reviews["content"] = (
    negative_reviews["content"]
    .apply(preprocess_text)
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(
    preprocessed_reviews["content"]
)

# K-Means Clustering
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(tfidf_matrix)

predicted_labels = kmeans.predict(tfidf_matrix)

categories = predicted_labels.tolist()

# Extract most frequent term from each cluster
feature_names = vectorizer.get_feature_names_out()

topic_data = []

for cluster in range(5):

    cluster_indices = np.where(
        np.array(categories) == cluster
    )[0]

    if len(cluster_indices) == 0:
        continue

    cluster_matrix = tfidf_matrix[cluster_indices]

    term_scores = np.asarray(
        cluster_matrix.sum(axis=0)
    ).flatten()

    top_term_index = term_scores.argmax()

    identified_term = feature_names[top_term_index]

    frequency = term_scores[top_term_index]

    topic_data.append(
        [cluster, identified_term, frequency]
    )

topic_terms = pd.DataFrame(
    topic_data,
    columns=[
        "label",
        "identified_term",
        "frequency"
    ]
)

# Display results
print("\n=== TOPIC TERMS ===\n")
print(topic_terms)

# Save outputs
preprocessed_reviews.to_csv(
    "preprocessed_reviews.csv",
    index=False
)

topic_terms.to_csv(
    "topic_terms.csv",
    index=False
)

print("\nFiles created successfully:")
print("- preprocessed_reviews.csv")
print("- topic_terms.csv")