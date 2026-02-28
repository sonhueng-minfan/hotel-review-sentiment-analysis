import pandas as pd
from transformers import pipeline
from nltk.corpus import stopwords
import re

# Load BERT sentiment pipeline (downloads model first time, ~500MB)
print("Loading BERT model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512
)

def load_data(filepath="hotel_reviews_sample.csv", sample_size=1000):
    df = pd.read_csv(filepath)
    df = df[['Hotel_Name', 'Reviewer_Nationality', 'Reviewer_Score',
             'Positive_Review', 'Negative_Review']].dropna()
    df['review_text'] = df['Positive_Review'] + " " + df['Negative_Review']
    df['review_length'] = df['review_text'].apply(len)
    return df.sample(n=sample_size, random_state=42)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def analyze_sentiment(df):
    print("Running BERT sentiment analysis (this may take a few minutes)...")
    df['clean_review'] = df['review_text'].apply(clean_text)

    results = sentiment_pipeline(df['clean_review'].tolist(), batch_size=16)

    df['sentiment'] = [r['label'].capitalize() for r in results]
    df['confidence'] = [round(r['score'], 3) for r in results]

    return df

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    df = analyze_sentiment(df)
    df.to_csv("analyzed_reviews.csv", index=False)
    print(df['sentiment'].value_counts())