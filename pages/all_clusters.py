import pandas as pd
import streamlit as st
import json
import toml
from clustering import compute_tfidf, AgglomerativeClustering, Helper
import numpy as np
import os
from datetime import datetime

# PAGE FORMAT
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Load configuration from TOML file
config = toml.load('config.toml')

# Sidebar logo
st.sidebar.image('app/logo.png', use_column_width=True)

# Load the JSON file with article data
file_path = 'article_cache.json'

@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    else:
        return {}

# Define custom CSS for the Streamlit app
st.markdown(f"""
    <style>
        body {{
            color: {config['theme']['textColor']};
            background-color: {config['theme']['backgroundColor']};
            font-family: {config['theme']['font']};
            font-size: {config['theme']['fontSize']}px;
        }}
        .primary {{
            color: {config['theme']['primaryColor']};
        }}
        .secondary-bg {{
            background-color: {config['theme']['secondaryBackgroundColor']};
        }}
        .article-image {{
            width: 100%;
            height: 200px;
            object-fit: cover;
        }}
    </style>
    """, unsafe_allow_html=True)

# Load data from the JSON file
original_data = load_data(file_path)

# Extract necessary data from the loaded JSON
articles = list(original_data.values())

if articles:
    # Convert date strings to datetime objects
    for article in articles:
        article['date'] = datetime.strptime(article['date'], '%Y-%m-%d') if 'date' in article else None

    # Date filter
    min_date = min(article['date'] for article in articles if article['date'] is not None)
    max_date = max(article['date'] for article in articles if article['date'] is not None)
    start_date, end_date = st.sidebar.slider(
        'Select date range:',
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    # Filter articles by selected date range
    filtered_articles = [article for article in articles if article['date'] and start_date <= article['date'] <= end_date]

    # Re-cluster the articles
    filtered_titles = [article['title'] for article in filtered_articles]

    # Prepare data for clustering
    helper = Helper()
    news_df = pd.DataFrame(filtered_articles)
    news_df['clean_body'] = news_df['body']  # Assuming that body is already cleaned

    tfidf_array = compute_tfidf(news_df)

    # Cluster articles using AgglomerativeClustering with distance threshold
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
    news_df['cluster_id'] = clustering_model.fit_predict(tfidf_array)

    clusters = {str(cluster_id): news_df[news_df.cluster_id == cluster_id].to_dict(orient='records')
                for cluster_id in np.unique(news_df.cluster_id)}

    # Initialize session state for displaying articles
    if 'article_counts' not in st.session_state:
        st.session_state.article_counts = {cluster_id: 3 for cluster_id in clusters.keys()}

    # Function to truncate body text to 100 words
    def truncate_body(text, max_words=100):
        words = text.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words]) + '...'
        return text

    # Display clusters and articles
    for cluster_id, cluster_articles in clusters.items():
        st.title(f"Cluster {int(cluster_id) + 1}")

        cols = st.columns(3)
        col_index = 0

        num_articles_to_show = st.session_state.article_counts[cluster_id]

        for article in cluster_articles[:num_articles_to_show]:  # Show only the specified number of articles per cluster
            with cols[col_index]:
                image_url = article.get("image_url", "default_image.png")
                st.markdown(f'<img src="{image_url}" class="article-image">', unsafe_allow_html=True)
                st.markdown(f"## {article['title']}")
                st.markdown(f"**Source:** {article.get('source', 'N/A')}")
                st.markdown(f"**Published on:** {article['date'].strftime('%Y-%m-%d') if article['date'] else 'N/A'}")
                st.markdown(truncate_body(article['body']))
                st.markdown(f"**Frequent Words:** {', '.join(article.get('keywords', []))}")
                st.markdown(f"**Sentiment:** {article.get('sentiment_category', 'N/A')}")
                st.markdown("---")
            
            col_index = (col_index + 1) % 3

        if num_articles_to_show < len(cluster_articles):
            if st.button(f"Read more from Cluster {int(cluster_id) + 1}", key=f"read_more_{cluster_id}"):
                st.session_state.article_counts[cluster_id] += 3
                st.experimental_rerun()

else:
    st.write("No articles data available.")
