# Customer Call Analytics

A Natural Language Processing (NLP) project that analyzes customer support interactions from both audio recordings and text transcripts. The pipeline converts customer calls into text, evaluates sentiment, extracts key entities, and identifies similar customer issues using text similarity techniques.

## Features

- Converts customer support audio into text using speech recognition
- Extracts audio metadata such as sample rate and channel count
- Performs sentiment analysis on customer conversations using VADER
- Identifies named entities (people, dates, organizations, etc.) with spaCy
- Uses TF-IDF vectorization and cosine similarity to find related customer complaints
- Processes unstructured customer interaction data into actionable insights

## Tech Stack

**Python, Pandas, NLTK, spaCy, SpeechRecognition, Scikit-learn**

## Run Locally

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python main.py

```

## Skills Demonstrated

- Natural Language Processing (NLP)
- Speech Recognition
- Sentiment Analysis
- Named Entity Recognition (NER)
- Text Similarity Search
- Data Processing & Analysis
- Machine Learning Workflows

