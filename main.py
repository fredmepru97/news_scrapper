import pandas as pd
import streamlit as st
import altair as alt
import json
import toml
import os
from clustering import compute_tfidf, AgglomerativeClustering, Helper
import numpy as np
from collections import Counter

# PAGE FORMAT
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

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
    </style>
    """, unsafe_allow_html=True)

# Sidebar logo
st.sidebar.image('app/Cat.png', use_column_width=True)

# Header for the Streamlit app
st.markdown(f'<h1 class="primary">Giki News for People in a Hurry!</h1>', unsafe_allow_html=True)


file_path = 'article_cache.json'
LOGO_PATH = 'app/Cat.png'
KEYWORD_LOGO_PATH = 'app/logo.png'

@st.cache_data
def load_data():
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load the data from the JSON file
original_data = load_data()

# Extract necessary data from the loaded JSON
articles = list(original_data.values())

# Sidebar filters
st.sidebar.header('Filters')
search_topic = st.sidebar.text_input("Search for a topic")
selected_sentiment = st.sidebar.multiselect(
    "Select Sentiment Category",
    options=list(set(article['sentiment_category'] for article in articles)),
    default=list(set(article['sentiment_category'] for article in articles))
)
all_sources = list(set(article['source'] for article in articles))
selected_sources = st.sidebar.multiselect(
    "Select Sources",
    options=all_sources,
    default=[]
)

# If no sources are selected, use all sources
if not selected_sources:
    selected_sources = all_sources

# Filter articles based on the search topic, selected sentiment category, and selected sources
filtered_articles = [
    article for article in articles 
    if (search_topic.lower() in article['title'].lower() or search_topic.lower() in article['body'].lower())
    and article['sentiment_category'] in selected_sentiment
    and article['source'] in selected_sources
]

# Extract necessary data from the filtered articles
filtered_titles = [article['title'] for article in filtered_articles]
filtered_sources = [article['source'] for article in filtered_articles]
filtered_keywords = [keyword for article in filtered_articles for keyword in article.get('keywords', [])]
filtered_sentiments = [article['sentiment_category'] for article in filtered_articles]

# Recalculate metrics for filtered articles
filtered_total_articles = len(filtered_articles)
filtered_unique_sources = len(set(filtered_sources))

# Create a dataframe for the filtered donut chart showing sources
df_filtered = pd.DataFrame({'Source': filtered_sources})

# Calculate counts for each source in the filtered data
filtered_source_counts = df_filtered['Source'].value_counts().reset_index()
filtered_source_counts.columns = ['Source', 'Count']

# Create a donut chart using Altair for the filtered data
donut_chart = alt.Chart(filtered_source_counts).mark_arc(innerRadius=50).encode(
    theta=alt.Theta(field='Count', type='quantitative'),
    color=alt.Color(field='Source', type='nominal', legend=alt.Legend(title="Sources")),
    tooltip=['Source', 'Count']
).properties(
    width=300,
    height=300
).configure_legend(
    titleFontSize=14,
    labelFontSize=12
).configure_view(
    strokeWidth=0
)

# Prepare data for the sentiment stacked bar chart
sentiment_counts = pd.DataFrame({'Sentiment': filtered_sentiments, 'Source': filtered_sources})
sentiment_counts = sentiment_counts.groupby(['Source', 'Sentiment']).size().reset_index(name='Count')

# Create a stacked bar chart for sentiment categories
stacked_bar_chart = alt.Chart(sentiment_counts).mark_bar().encode(
    x=alt.X('Source:N', title='Source'),
    y=alt.Y('Count:Q', title='Count'),
    color=alt.Color('Sentiment:N', title='Sentiment'),
    tooltip=['Source', 'Sentiment', 'Count']
).properties(
    width=300,
    height=300
).configure_legend(
    titleFontSize=14,
    labelFontSize=12
).configure_view(
    strokeWidth=0
)

# Prepare data for bubble chart
df_keywords = pd.DataFrame({'Keyword': filtered_keywords})

# Calculate counts for each keyword
keyword_counts = df_keywords['Keyword'].value_counts().reset_index()
keyword_counts.columns = ['Keyword', 'Count']

# Filter keywords that appear at least 8 times
filtered_keyword_counts = keyword_counts[keyword_counts['Count'] >= 3]

# Create a sorting function to sort numbers first, then words
def sort_keywords(keyword):
    if keyword.isdigit():
        return (0, int(keyword))
    return (1, keyword.lower())

filtered_keyword_counts['Keyword'] = pd.Categorical(
    filtered_keyword_counts['Keyword'],
    categories=sorted(filtered_keyword_counts['Keyword'], key=sort_keywords),
    ordered=True
)

# Create a bubble chart using Altair with sorted keywords
bubble_chart = alt.Chart(filtered_keyword_counts).mark_circle().encode(
    x=alt.X('Keyword:N', sort='ascending'),
    y=alt.Y('Count:Q', title='Frequency'),
    size=alt.Size('Count:Q', legend=None),
    color=alt.Color('Keyword:N', legend=None),
    tooltip=['Keyword', 'Count']
).properties(
    width=600,
    height=400
)

# Determine the number of clusters dynamically
if filtered_total_articles == 0:
    st.write("No articles found")
else:
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

    # Store filtered_articles and clusters in session state for access on another page
    st.session_state.filtered_articles = filtered_articles
    st.session_state.clusters = clusters

    # Display metrics and charts in Streamlit
    st.title("Article Metrics")

    # Row 1
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.metric(label="Total Articles", value=filtered_total_articles)
    with row1_col2:
        st.metric(label="Total Sources", value=filtered_unique_sources)

    # Row 2
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.altair_chart(stacked_bar_chart, use_container_width=True)
    with row2_col2:
        st.altair_chart(donut_chart, use_container_width=True)

    # Row 3
    st.altair_chart(bubble_chart, use_container_width=True)

    # Display clusters in a markdown format in two columns
    st.title("Article Clusters")

    cols = st.columns(2)
    col_index = 0

    # Sort clusters by cluster number
    sorted_clusters = sorted(clusters.items(), key=lambda x: int(x[0]))

    for cluster_id, titles in sorted_clusters:
        # Define cluster name and sample keywords
        cluster_name = f"<a href='/cluster?cluster_id={cluster_id}' target=_top>Cluster {int(cluster_id) + 1}</a>"
        
        # Calculate top 5 most frequent keywords in the cluster
        all_keywords = [keyword for article in titles for keyword in article.get('keywords', [])]
        keyword_counts = Counter(all_keywords).most_common(10)
        top_keywords = [keyword for keyword, count in keyword_counts]
        cluster_keywords_str = f"Keywords: {', '.join(top_keywords)}"
        
        num_articles = len(titles)
        
        # Representative articles (sample titles)
        representative_articles = "Representative Articles:\n" + "\n".join([f"- \"{article['title']}\"" for article in titles[:2]])
        
        # Create markdown content
        cluster_markdown = f"""
        - **{cluster_name}** ({num_articles} articles)
            - {cluster_keywords_str}
            - {representative_articles}
        """
        
        # Display in columns
        cols[col_index].markdown(cluster_markdown, unsafe_allow_html=True)
        col_index = (col_index + 1) % 2
