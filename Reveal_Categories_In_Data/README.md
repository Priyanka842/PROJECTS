# Google Play Store Review Topic Discovery

A Natural Language Processing (NLP) project that analyzes negative Google Play Store reviews using TF-IDF vectorization and K-Means clustering to identify common user concerns.

## Features

- Filters negative reviews (ratings 1 and 2)
- Cleans and preprocesses text using NLTK
- Converts reviews into TF-IDF vectors
- Groups similar reviews using K-Means clustering
- Extracts representative keywords from each cluster

## Technologies

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn

## Run

```bash
pip install -r requirements.txt
python main.py

```

## Output

The project generates:

- `preprocessed_reviews.csv` – cleaned review text
- `topic_terms.csv` – representative terms for each cluster

This project demonstrates text preprocessing, feature extraction, and unsupervised machine learning for topic discovery.