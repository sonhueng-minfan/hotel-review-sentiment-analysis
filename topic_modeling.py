import pandas as pd
import re
from nltk.corpus import stopwords
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

def preprocess_for_lda(texts):
    stop_words = set(stopwords.words('english'))
    # Add hotel-specific stopwords that aren't useful as topics
    custom_stops = {'hotel', 'room', 'stayed', 'stay', 'would', 'also', 
                    'one', 'get', 'got', 'us', 'na', 'wa', 'nothing'}
    stop_words.update(custom_stops)
    
    processed = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        words = [w for w in words if w not in stop_words and len(w) > 3]
        processed.append(words)
    return processed

def run_lda(texts, num_topics=6, passes=10):
    processed = preprocess_for_lda(texts)
    dictionary = corpora.Dictionary(processed)
    
    # Filter out very rare and very common words
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in processed]
    
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=42
    )
    return lda_model, corpus, dictionary

def get_topic_labels(lda_model, num_topics=6):
    """Extract top words per topic for display"""
    topics = []
    for i in range(num_topics):
        top_words = lda_model.show_topic(i, topn=8)
        words = [w for w, _ in top_words]
        topics.append({
            'topic_id': i,
            'top_words': ', '.join(words),
            'label': f"Topic {i+1}"
        })
    return topics

def assign_topics_to_reviews(df, lda_model, corpus):
    """Assign dominant topic to each review"""
    dominant_topics = []
    for doc_bow in corpus:
        topic_dist = lda_model.get_document_topics(doc_bow)
        if topic_dist:
            dominant = max(topic_dist, key=lambda x: x[1])
            dominant_topics.append(dominant[0])
        else:
            dominant_topics.append(0)
    df = df.copy()
    df['dominant_topic'] = dominant_topics
    return df