import pandas as pd
import streamlit as st
import altair as alt
import json
import toml
import os
from clustering import cluster_articles

# Other necessary imports...

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

# Header for the Streamlit app
st.markdown(f'<h1 class="primary">Giki News for People in a Hurry!</h1>', unsafe_allow_html=True)

# Load the JSON file with article data
file_path = 'article_cache.json'

@st.cache_data
def load_data():
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load the data from the JSON file
original_data = load_data()

# Extract necessary data from the loaded JSON
articles = list(original_data.values())

# Create a search bar
search_topic = st.text_input("Search for a topic")

# Filter articles based on the search topic
if search_topic:
    filtered_articles = [article for article in articles if search_topic.lower() in article['title'].lower() or search_topic.lower() in article['body'].lower()]
else:
    filtered_articles = articles

# Extract necessary data from the filtered articles
filtered_titles = [article['title'] for article in filtered_articles]
filtered_sources = [article['source'] for article in filtered_articles]
filtered_keywords = [keyword for article in filtered_articles for keyword in article.get('keywords', [])]

# Recalculate metrics for filtered articles
filtered_total_articles = len(filtered_articles)
filtered_unique_sources = len(set(filtered_sources))

# Create a dataframe for the filtered donut chart
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
    width=400,
    height=400
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
filtered_keyword_counts = keyword_counts[keyword_counts['Count'] >= 8]

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

# Display metrics and charts in Streamlit
st.title("Article Metrics")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Articles", value=filtered_total_articles)
with col2:
    st.metric(label="Total Sources", value=filtered_unique_sources)
with col3:
    st.altair_chart(donut_chart, use_container_width=True)

st.title("Keyword Frequency")
st.altair_chart(bubble_chart, use_container_width=True)

# Optionally, display articles or additional information here
st.subheader("Article Titles")
for title in filtered_titles:
    st.write(f"- {title}")
