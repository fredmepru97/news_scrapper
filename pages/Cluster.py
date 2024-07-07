import pandas as pd
import streamlit as st
import json
import toml
from clustering import compute_tfidf, AgglomerativeClustering, Helper
import numpy as np

# PAGE FORMAT
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Load configuration from TOML file
config = toml.load('config.toml')

# Sidebar logo
st.sidebar.image('app/logo.png', use_column_width=True)

# Load the JSON file with article data
file_path = 'article_cache.json'

@st.cache_data
def load_data():
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load configuration from TOML file
config = toml.load('config.toml')

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

# Check if cluster ID is provided
cluster_id = int(st.query_params.get('cluster_id', [0])[0])

# Load data from the JSON file
original_data = load_data()

# Extract necessary data from the loaded JSON
articles = list(original_data.values())

# Re-cluster the articles
filtered_titles = [article['title'] for article in articles]

# Prepare data for clustering
helper = Helper()
news_df = pd.DataFrame(articles)
news_df['clean_body'] = news_df['body']  # Assuming that body is already cleaned

tfidf_array = compute_tfidf(news_df)

# Cluster articles using AgglomerativeClustering with distance threshold
clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
news_df['cluster_id'] = clustering_model.fit_predict(tfidf_array)

clusters = {str(cluster_id): news_df[news_df.cluster_id == cluster_id].to_dict(orient='records')
            for cluster_id in np.unique(news_df.cluster_id)}

# Function to truncate body text to 100 words
def truncate_body(text, max_words=100):
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '...'
    return text

# Get articles in the selected cluster
if str(cluster_id) in clusters:
    cluster_articles = clusters[str(cluster_id)]

    # Display articles in the cluster
    st.title(f"Articles in Cluster {int(cluster_id) + 1}")

    cols = st.columns(3)
    col_index = 0

    for article in cluster_articles:
        with cols[col_index]:
            st.markdown(f'<img src="{article.get("image_url", "")}" class="article-image">', unsafe_allow_html=True)
            st.markdown(f"## {article['title']}")
            st.markdown(f"**Source:** {article.get('source', 'N/A')}")
            st.markdown(f"**Published on:** {article.get('date', 'N/A')}")
            st.markdown(truncate_body(article['body']))
            st.markdown(f"**Frequent Words:** {', '.join(article.get('keywords', []))}")
            st.markdown(f"**Sentiment:** {article.get('sentiment_category', 'N/A')}")
            st.markdown("---")
        
        col_index = (col_index + 1) % 3

else:
    st.write(f"No articles found for cluster {int(cluster_id) + 1}.")
