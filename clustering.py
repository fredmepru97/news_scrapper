import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import json
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

CACHE_FILE = 'article_cache.json'

class Helper:
    @staticmethod
    def print_scrape_status(count):
        logging.info(f'Scraped {count} articles so far...')

    @staticmethod
    def clean_dataframe(news_df):
        logging.info("Cleaning dataframe")
        news_df = news_df[news_df.title != '']
        news_df = news_df[news_df.body != '']
        news_df = news_df[news_df.image_url != '']
        news_df = news_df[news_df.title.str.count(r'\s+').ge(3)]
        news_df = news_df[news_df.body.str.count(r'\s+').ge(20)]
        return news_df

def compute_tfidf(news_df):
    logging.info("Computing TF-IDF values")
    tfidf_matrix = TfidfVectorizer().fit_transform(news_df['clean_body'])
    tfidf_array = np.asarray(tfidf_matrix.todense())
    return tfidf_array

def find_featured_clusters(clusters):
    logging.info("Finding clusters with articles from multiple sources")
    featured_clusters = {}
    for i in clusters.keys():
        if len(set([j["source"] for j in clusters[i]])) > 1:
            featured_clusters[i] = clusters[i]
    return featured_clusters

def main():
    logging.info("Loading articles from cache")
    with open(CACHE_FILE, 'r') as f:
        articles = json.load(f)
    
    news_df = pd.DataFrame(articles.values())
    helper = Helper()
    news_df = helper.clean_dataframe(news_df)
    tfidf_array = compute_tfidf(news_df)
    
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
    news_df['cluster_id'] = clustering_model.fit_predict(tfidf_array)
    
    clusters = {str(cluster_id): news_df[news_df.cluster_id == cluster_id].to_dict(orient='records')
                for cluster_id in np.unique(news_df.cluster_id)}
    
    featured_clusters = find_featured_clusters(clusters)
    
    logging.info("Saving clusters to cache")
    with open(CACHE_FILE, 'w') as f:
        json.dump(articles, f, indent=4)

if __name__ == "__main__":
    main()

def cluster_articles(titles, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(titles)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    clusters = {}
    for idx, label in enumerate(kmeans.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(titles[idx])

    return clusters

def clusters_2():
    logging.info("Loading articles from cache")
    with open(CACHE_FILE, 'r') as f:
        articles = json.load(f)
    
    news_df = pd.DataFrame(articles.values())
    helper = Helper()
    news_df = helper.clean_dataframe(news_df)
    tfidf_array = compute_tfidf(news_df)
    
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
    news_df['cluster_id'] = clustering_model.fit_predict(tfidf_array)
    
    clusters = {str(cluster_id): news_df[news_df.cluster_id == cluster_id].to_dict(orient='records')
                for cluster_id in np.unique(news_df.cluster_id)}
    
    featured_clusters = find_featured_clusters(clusters)
    
    logging.info("Saving clusters to cache")
    with open(CACHE_FILE, 'w') as f:
        json.dump(articles, f, indent=4)
