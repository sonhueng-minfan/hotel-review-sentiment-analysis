import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sentiment_analysis import load_data, analyze_sentiment, sentiment_pipeline
from topic_modeling import run_lda, get_topic_labels, assign_topics_to_reviews
from aspect_sentiment import analyze_aspects, ASPECT_KEYWORDS

st.set_page_config(page_title="Hotel Review Sentiment Analysis", layout="wide")

st.title("🏨 Hotel Review Sentiment Analysis")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Sentiment Dashboard",
    "🏷️ Aspect Analysis",
    "🔍 Topic Modeling",
    "🤖 Try It Yourself"
])

# ─────────────────────────────────────────────
# Load & cache data
# ─────────────────────────────────────────────
@st.cache_data
def get_data():
    df = load_data()
    return analyze_sentiment(df)

with st.spinner("Loading and analyzing reviews with BERT..."):
    df = get_data()

# ─────────────────────────────────────────────
# SIDEBAR FILTERS (affect Tab 1)
# ─────────────────────────────────────────────
st.sidebar.header("🔧 Filters")

sentiment_filter = st.sidebar.multiselect(
    "Sentiment", options=["Positive", "Negative"],
    default=["Positive", "Negative"]
)

nationality_options = ["All"] + sorted(df['Reviewer_Nationality'].dropna().unique().tolist())
nationality_filter = st.sidebar.selectbox("Reviewer Nationality", nationality_options)

hotel_options = ["All"] + sorted(df['Hotel_Name'].dropna().unique().tolist())
hotel_filter = st.sidebar.selectbox("Hotel Name", hotel_options)

min_score = st.sidebar.slider("Minimum Reviewer Score", 0.0, 10.0, 0.0, 0.5)

# Apply filters
filtered_df = df[df['sentiment'].isin(sentiment_filter)]
filtered_df = filtered_df[filtered_df['Reviewer_Score'] >= min_score]
if nationality_filter != "All":
    filtered_df = filtered_df[filtered_df['Reviewer_Nationality'] == nationality_filter]
if hotel_filter != "All":
    filtered_df = filtered_df[filtered_df['Hotel_Name'] == hotel_filter]

# ─────────────────────────────────────────────
# TAB 1 — Sentiment Dashboard
# ─────────────────────────────────────────────
with tab1:
    st.markdown("Analyzing real European hotel reviews using BERT NLP")

    if filtered_df.empty:
        st.warning("No reviews match your current filters. Try adjusting the sidebar.")
    else:
        # --- KPI Row ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", len(filtered_df))
        col2.metric("Avg Reviewer Score", f"{filtered_df['Reviewer_Score'].mean():.2f}")
        col3.metric("% Positive", f"{(filtered_df['sentiment']=='Positive').mean()*100:.1f}%")
        col4.metric("Avg BERT Confidence", f"{filtered_df['confidence'].mean():.2%}")

        st.markdown("---")

        # --- Export Button ---
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Export Filtered Reviews as CSV",
            data=csv,
            file_name="filtered_hotel_reviews.csv",
            mime="text/csv"
        )

        st.markdown("---")

        # --- Charts Row 1 ---
        col5, col6 = st.columns(2)

        with col5:
            st.subheader("Sentiment Distribution")
            fig = px.pie(filtered_df, names='sentiment',
                         color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)

        with col6:
            st.subheader("Sentiment vs. Reviewer Score")
            fig2 = px.box(filtered_df, x='sentiment', y='Reviewer_Score',
                          color='sentiment',
                          color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'})
            st.plotly_chart(fig2, use_container_width=True)

        # --- Review Length vs Sentiment ---
        st.subheader("Review Length vs. Sentiment")
        st.markdown("Do longer reviews tend to be more negative?")
        fig_len = px.box(filtered_df, x='sentiment', y='review_length',
                         color='sentiment',
                         color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'},
                         labels={'review_length': 'Review Length (characters)'})
        st.plotly_chart(fig_len, use_container_width=True)

        # --- Nationality Insights ---
        st.subheader("🌍 Nationality Insights")
        st.markdown("Which nationalities leave the most positive or negative reviews?")
        nat_counts = filtered_df.groupby(['Reviewer_Nationality', 'sentiment']).size().reset_index(name='count')
        top_nats = (filtered_df['Reviewer_Nationality']
                    .value_counts()
                    .head(10)
                    .index.tolist())
        nat_top = nat_counts[nat_counts['Reviewer_Nationality'].isin(top_nats)]
        fig_nat = px.bar(nat_top, x='Reviewer_Nationality', y='count',
                         color='sentiment', barmode='group',
                         color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'})
        fig_nat.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_nat, use_container_width=True)

        # --- Top / Bottom Hotels Leaderboard ---
        st.subheader("🏆 Hotel Leaderboard")
        col_top, col_bot = st.columns(2)

        hotel_sentiment = df.groupby('Hotel_Name').apply(
            lambda x: round((x['sentiment'] == 'Positive').mean() * 100, 1)
        ).reset_index(name='positive_pct')
        hotel_counts = df['Hotel_Name'].value_counts().reset_index()
        hotel_counts.columns = ['Hotel_Name', 'review_count']
        hotel_sentiment = hotel_sentiment.merge(hotel_counts, on='Hotel_Name')
        hotel_sentiment = hotel_sentiment[hotel_sentiment['review_count'] >= 5]

        with col_top:
            st.markdown("**🥇 Top 10 Hotels by Positive Sentiment**")
            top10 = hotel_sentiment.nlargest(10, 'positive_pct')[['Hotel_Name', 'positive_pct', 'review_count']]
            top10.columns = ['Hotel', '% Positive', 'Reviews']
            st.dataframe(top10, hide_index=True)

        with col_bot:
            st.markdown("**⚠️ Bottom 10 Hotels by Positive Sentiment**")
            bot10 = hotel_sentiment.nsmallest(10, 'positive_pct')[['Hotel_Name', 'positive_pct', 'review_count']]
            bot10.columns = ['Hotel', '% Positive', 'Reviews']
            st.dataframe(bot10, hide_index=True)

        # --- Hotel Comparison Tool ---
        st.subheader("🆚 Hotel Comparison Tool")
        hotels_list = sorted(df['Hotel_Name'].dropna().unique().tolist())
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            hotel_a = st.selectbox("Select Hotel A", hotels_list, index=0, key="hotel_a")
        with col_h2:
            hotel_b = st.selectbox("Select Hotel B", hotels_list, index=1, key="hotel_b")

        if hotel_a and hotel_b:
            compare_df = df[df['Hotel_Name'].isin([hotel_a, hotel_b])]
            fig_comp = px.histogram(compare_df, x='Reviewer_Score', color='Hotel_Name',
                                    barmode='overlay', opacity=0.7,
                                    title=f"Score Distribution: {hotel_a} vs {hotel_b}")
            st.plotly_chart(fig_comp, use_container_width=True)

            comp_sent = compare_df.groupby(['Hotel_Name', 'sentiment']).size().reset_index(name='count')
            fig_comp2 = px.bar(comp_sent, x='Hotel_Name', y='count', color='sentiment',
                               barmode='group',
                               color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'},
                               title="Sentiment Comparison")
            st.plotly_chart(fig_comp2, use_container_width=True)

        # --- Word Cloud ---
        st.subheader("☁️ Most Common Words in Reviews")
        wc_sentiment = st.radio("Select sentiment:", ["Positive", "Negative"], horizontal=True)
        text = " ".join(filtered_df[filtered_df['sentiment'] == wc_sentiment]['clean_review'].tolist())
        if text:
            wc = WordCloud(width=800, height=300, background_color='white').generate(text)
            fig_wc, ax = plt.subplots(figsize=(10, 3))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)

        # --- Raw Data ---
        st.subheader("📋 Sample Reviews")
        st.dataframe(
            filtered_df[['Hotel_Name', 'Reviewer_Nationality', 'review_text',
                          'sentiment', 'confidence', 'Reviewer_Score']].head(20)
        )

# ─────────────────────────────────────────────
# TAB 2 — Aspect-Based Sentiment
# ─────────────────────────────────────────────
with tab2:
    st.header("🏷️ Aspect-Based Sentiment Analysis")
    st.markdown("How do guests feel about **specific aspects** of their stay — rooms, staff, food, cleanliness, and location?")

    aspect_sample = st.slider("Reviews to analyze for aspects", 50, 300, 100, step=50)
    run_aspect = st.button("🚀 Run Aspect Analysis")

    if run_aspect:
        with st.spinner("Analyzing aspects with BERT... this may take a few minutes"):
            aspect_df = df.sample(n=aspect_sample, random_state=42).copy()
            aspect_df = analyze_aspects(aspect_df, sentiment_pipeline)

        st.success("Done!")
        st.markdown("---")

        aspects = list(ASPECT_KEYWORDS.keys())
        aspect_cols = [f'aspect_{a.lower()}' for a in aspects]

        # Average sentiment score per aspect
        avg_scores = []
        for aspect, col in zip(aspects, aspect_cols):
            scores = aspect_df[col].dropna()
            if len(scores) > 0:
                avg_scores.append({
                    'Aspect': aspect,
                    'Avg Sentiment Score': round(scores.mean(), 3),
                    'Reviews Mentioning': len(scores)
                })

        avg_df = pd.DataFrame(avg_scores)

        st.subheader("Average Sentiment Score by Aspect")
        st.markdown("Scores range from -1 (very negative) to +1 (very positive)")
        fig_asp = px.bar(avg_df, x='Aspect', y='Avg Sentiment Score',
                         color='Avg Sentiment Score',
                         color_continuous_scale=['#e74c3c', '#f39c12', '#2ecc71'],
                         text='Avg Sentiment Score')
        fig_asp.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_asp, use_container_width=True)

        st.subheader("How Many Reviews Mention Each Aspect?")
        fig_count = px.bar(avg_df, x='Aspect', y='Reviews Mentioning', color='Aspect')
        st.plotly_chart(fig_count, use_container_width=True)

        # Radar chart
        st.subheader("Aspect Sentiment Radar Chart")
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=avg_df['Avg Sentiment Score'].tolist(),
            theta=avg_df['Aspect'].tolist(),
            fill='toself',
            line_color='#3498db'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-1, 1])))
        st.plotly_chart(fig_radar, use_container_width=True)

        st.subheader("Raw Aspect Scores per Review")
        display_cols = ['Hotel_Name', 'sentiment'] + aspect_cols
        st.dataframe(aspect_df[display_cols].head(20))

# ─────────────────────────────────────────────
# TAB 3 — Topic Modeling
# ─────────────────────────────────────────────
with tab3:
    st.header("🔍 Topic Modeling with LDA")
    st.markdown("Discover hidden themes across hotel reviews — what do guests talk about most?")

    num_topics = st.slider("Number of topics to discover", min_value=3, max_value=10, value=6)
    lda_sample_size = st.slider("Reviews to analyze", min_value=100, max_value=1000, value=300, step=100)

    run_button = st.button("🚀 Run Topic Modeling")

    if run_button:
        with st.spinner("Running LDA... this takes 1-2 minutes"):
            sample_df = df.sample(n=lda_sample_size, random_state=42)
            lda_model, corpus, dictionary = run_lda(
                sample_df['review_text'].tolist(),
                num_topics=num_topics
            )
            topics = get_topic_labels(lda_model, num_topics)
            sample_df = assign_topics_to_reviews(sample_df, lda_model, corpus)

        st.success("Done! Here are the discovered topics:")
        st.markdown("---")

        cols = st.columns(2)
        for i, topic in enumerate(topics):
            with cols[i % 2]:
                st.markdown(f"### 🏷️ {topic['label']}")
                st.markdown(f"**Top words:** {topic['top_words']}")
                count = (sample_df['dominant_topic'] == topic['topic_id']).sum()
                st.markdown(f"**Reviews in this topic:** {count}")
                st.markdown("---")

        st.subheader("Topic Distribution Across Reviews")
        topic_counts = sample_df['dominant_topic'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']
        topic_counts['Topic'] = topic_counts['Topic'].apply(lambda x: f"Topic {x+1}")
        fig_topics = px.bar(topic_counts, x='Topic', y='Count', color='Topic')
        st.plotly_chart(fig_topics, use_container_width=True)

        st.subheader("Sentiment Breakdown per Topic")
        sample_df['topic_label'] = sample_df['dominant_topic'].apply(lambda x: f"Topic {x+1}")
        fig_sent = px.histogram(sample_df, x='topic_label', color='sentiment', barmode='group',
                                color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'})
        st.plotly_chart(fig_sent, use_container_width=True)

        st.subheader("Sample Reviews per Topic")
        selected_topic = st.selectbox("Pick a topic to explore", [f"Topic {i+1}" for i in range(num_topics)])
        topic_num = int(selected_topic.split()[1]) - 1
        samples = sample_df[sample_df['dominant_topic'] == topic_num][
            ['review_text', 'sentiment', 'Hotel_Name']
        ].head(5)
        st.dataframe(samples)

# ─────────────────────────────────────────────
# TAB 4 — Try It Yourself (Live BERT Prediction)
# ─────────────────────────────────────────────
with tab4:
    st.header("🤖 Try It Yourself")
    st.markdown("Type any hotel review below and BERT will predict its sentiment in real time.")

    user_review = st.text_area(
        "Enter a hotel review:",
        placeholder="e.g. The room was spotless and the staff were incredibly welcoming. Breakfast was a bit disappointing though.",
        height=150
    )

    if st.button("🔍 Analyze My Review"):
        if user_review.strip() == "":
            st.warning("Please enter a review first.")
        else:
            with st.spinner("Running BERT..."):
                result = sentiment_pipeline(user_review[:512], truncation=True, max_length=512)[0]
                label = result['label'].capitalize()
                confidence = result['score']

            st.markdown("---")
            col_r1, col_r2 = st.columns(2)

            with col_r1:
                if label == "Positive":
                    st.success(f"### ✅ Sentiment: {label}")
                else:
                    st.error(f"### ❌ Sentiment: {label}")

            with col_r2:
                st.metric("BERT Confidence", f"{confidence:.2%}")

            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={'text': f"Confidence ({label})"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#2ecc71' if label == 'Positive' else '#e74c3c'},
                    'steps': [
                        {'range': [0, 50], 'color': '#f8f9fa'},
                        {'range': [50, 75], 'color': '#e9ecef'},
                        {'range': [75, 100], 'color': '#dee2e6'}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("**Your review:**")
            st.info(user_review)