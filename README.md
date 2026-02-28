# 🏨 Hotel Review Sentiment Analysis

An end-to-end NLP project that performs sentiment analysis, aspect-based analysis, and topic modeling on Kaggle's 500K+ real European hotel reviews using state-of-the-art transformer models.


## 🚀 Features

- **BERT Sentiment Analysis** — Classifies each review as Positive or Negative using a pretrained DistilBERT transformer model
- **Aspect-Based Sentiment** — Scores guest sentiment specifically for Rooms, Staff, Food, Cleanliness, and Location
- **Topic Modeling (LDA)** — Automatically discovers hidden themes across thousands of reviews
- **Interactive Dashboard** — Filter by hotel, nationality, and score; compare hotels side by side
- **Live Prediction** — Type any review and get an instant BERT sentiment prediction with a confidence gauge
- **Export** — Download filtered results as a CSV

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| HuggingFace Transformers | BERT sentiment model |
| Gensim | LDA topic modeling |
| Streamlit | Interactive web dashboard |
| Plotly | Charts and visualizations |
| Pandas | Data manipulation |
| NLTK | Text preprocessing |
| WordCloud | Word cloud generation |

---

## 📁 Project Structure

```
hotel-review-sentiment-analysis/
├── app.py                  # Main Streamlit dashboard
├── sentiment_analysis.py   # BERT sentiment pipeline
├── aspect_sentiment.py     # Aspect-based sentiment analysis
├── topic_modeling.py       # LDA topic modeling
├── setup_nltk.py           # One-time NLTK resource downloader
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files (venv, CSV, cache)
└── README.md               # This file
```

---

## 📂 File Explanations

### `app.py`
The main application file that runs the Streamlit dashboard. It imports from all other modules and organizes everything into 4 tabs:
- **📊 Sentiment Dashboard** — KPI metrics, sentiment distribution, reviewer score analysis, review length vs sentiment, nationality insights, hotel leaderboard (top/bottom 10), hotel comparison tool, word clouds, and a raw data table. Includes sidebar filters for sentiment, nationality, hotel name, and minimum score, plus a CSV export button.
- **🏷️ Aspect Analysis** — Runs BERT on review sentences to score sentiment for each of 5 hotel aspects (Rooms, Staff, Food, Cleanliness, Location). Displays results as bar charts and a radar chart.
- **🔍 Topic Modeling** — Runs LDA to discover hidden themes across reviews. Users can choose the number of topics and sample size before computing.
- **🤖 Try It Yourself** — A live text input where users can type any review and receive an instant BERT sentiment prediction with a confidence score gauge.

---

### `sentiment_analysis.py`
Handles all core data loading and BERT sentiment classification.

**How it works:**
1. Loads the raw `Hotel_Reviews.csv` dataset using Pandas
2. Combines the `Positive_Review` and `Negative_Review` columns into a single `review_text` field
3. Cleans the text using NLTK — converts to lowercase, removes punctuation, and strips common stopwords
4. Loads the `distilbert-base-uncased-finetuned-sst-2-english` model from HuggingFace — a lightweight version of BERT fine-tuned specifically for sentiment classification
5. Runs the cleaned reviews through BERT in batches of 16 for efficiency
6. Returns a `sentiment` label (Positive/Negative) and a `confidence` score (0–1) for each review

---

### `aspect_sentiment.py`
Performs aspect-based sentiment analysis — instead of one overall sentiment score, it scores how guests feel about specific aspects of their stay.

**How it works:**
1. Defines keyword lists for 5 aspects: Rooms, Staff, Food, Cleanliness, and Location
2. For each review, splits the text into individual sentences
3. Filters sentences that contain keywords relevant to each aspect
4. Passes those sentences through the BERT pipeline to get a sentiment score
5. Returns a score between -1 (very negative) and +1 (very positive) for each aspect, or `None` if the aspect wasn't mentioned in the review

---

### `topic_modeling.py`
Uses Latent Dirichlet Allocation (LDA) to automatically discover recurring themes across reviews without any manual labeling.

**How it works:**
1. Preprocesses the review text — removes stopwords, short words, and hotel-specific common words that wouldn't form meaningful topics (e.g. "hotel", "stay")
2. Builds a Gensim dictionary mapping each unique word to an ID
3. Filters out very rare words (appearing in fewer than 5 reviews) and very common words (appearing in more than 50% of reviews)
4. Converts each review into a bag-of-words representation
5. Trains an LDA model which learns to group words into topics based on co-occurrence patterns
6. Assigns each review a dominant topic
7. Returns the top 8 words per topic for display

---

### `setup_nltk.py`
A simple one-time setup script that downloads the required NLTK language resources:
- `vader_lexicon` — sentiment word dictionary (used in early versions)
- `stopwords` — common words to filter out (the, and, is, etc.)
- `punkt` — sentence tokenizer

Only needs to be run once before launching the app for the first time.

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/hotel-review-sentiment-analysis.git
cd hotel-review-sentiment-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK resources (one time only)
```bash
python setup_nltk.py
```

### 4. Get the dataset
- Go to [Kaggle — 515K Hotel Reviews](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe)
- Download `Hotel_Reviews.csv`
- Place it in the root of the project folder

### 5. Launch the app
```bash
python -m streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

> **Note:** The first launch downloads the BERT model (~500MB) and may take a few minutes. Subsequent launches are much faster.

---

## 📦 requirements.txt

```
pandas
nltk
streamlit
plotly
scikit-learn
wordcloud
matplotlib
transformers
torch
gensim
pyLDAvis
```

---

## 📊 Dataset

[515K Hotel Reviews Data in Europe](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe) — Kaggle

Contains 515,000+ reviews for 1,493 hotels across Europe, including reviewer nationality, scores, and free-text positive/negative comments.

---

## 💡 Resume Bullet Points

- Built an end-to-end NLP pipeline processing 500+ hotel reviews using Python, HuggingFace Transformers (DistilBERT), and Pandas
- Implemented aspect-based sentiment analysis to score guest experience across 5 hotel dimensions (rooms, staff, food, cleanliness, location)
- Applied Latent Dirichlet Allocation (LDA) topic modeling to automatically discover hidden themes across review corpora
- Developed a 4-tab interactive dashboard with Streamlit and Plotly featuring real-time BERT predictions, hotel comparison tools, and dynamic filtering