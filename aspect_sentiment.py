import re

# Keywords associated with each aspect
ASPECT_KEYWORDS = {
    'Rooms': ['room', 'bed', 'bedroom', 'suite', 'accommodation', 'pillow', 'mattress',
              'shower', 'bathroom', 'toilet', 'bath', 'towel', 'view', 'balcony',
              'window', 'curtain', 'air conditioning', 'ac', 'heating', 'noise', 'quiet'],
    'Staff': ['staff', 'service', 'receptionist', 'concierge', 'manager', 'employee',
              'helpful', 'friendly', 'rude', 'professional', 'attentive', 'welcoming',
              'check in', 'check out', 'front desk', 'team', 'personnel', 'crew'],
    'Cleanliness': ['clean', 'dirty', 'hygiene', 'hygienic', 'tidy', 'spotless',
                    'dust', 'smell', 'odor', 'stain', 'mold', 'bug', 'insect',
                    'cockroach', 'maintenance', 'worn', 'shabby', 'fresh'],
    'Food': ['food', 'breakfast', 'dinner', 'lunch', 'restaurant', 'meal', 'eat',
             'cuisine', 'menu', 'buffet', 'taste', 'delicious', 'bland', 'portion',
             'chef', 'cooking', 'dining', 'bar', 'drink', 'coffee', 'tea'],
    'Location': ['location', 'located', 'area', 'neighborhood', 'centre', 'center',
                 'transport', 'metro', 'subway', 'bus', 'walk', 'walking', 'distance',
                 'nearby', 'close', 'far', 'convenient', 'accessible', 'airport',
                 'station', 'attraction', 'beach', 'city']
}

def extract_aspect_sentences(text, aspect_keywords):
    """Extract sentences related to a specific aspect."""
    sentences = re.split(r'[.!?]', text.lower())
    relevant = [s.strip() for s in sentences
                if any(kw in s for kw in aspect_keywords) and len(s.strip()) > 10]
    return relevant

def analyze_aspects(df, sentiment_pipeline):
    """Add aspect sentiment columns to the dataframe."""
    print("Running aspect-based sentiment analysis...")

    for aspect, keywords in ASPECT_KEYWORDS.items():
        scores = []
        for text in df['review_text']:
            sentences = extract_aspect_sentences(text, keywords)
            if sentences:
                combined = ' '.join(sentences)[:512]  # truncate for BERT
                try:
                    result = sentiment_pipeline(combined, truncation=True, max_length=512)[0]
                    label = result['label'].capitalize()
                    score = result['score'] if label == 'Positive' else -result['score']
                except Exception:
                    score = 0.0
            else:
                score = None  # No mention of this aspect
            scores.append(score)
        df[f'aspect_{aspect.lower()}'] = scores

    return df