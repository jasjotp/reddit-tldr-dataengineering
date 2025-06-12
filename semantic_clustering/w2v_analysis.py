import os 
os.environ['OMP_NUM_THREADS'] = '3' # set the OP NUM THREADS env variable to 3 to avoid the potential memory leak caused my using KMeans on a Windows Intel machine
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import re 
from gensim.models import Word2Vec
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline # for huggingface 
from nltk.corpus import stopwords 
import nltk
import sys 
from adjustText import adjust_text
from scipy.spatial import ConvexHull
import plotly.express as px
from kneed import KneeLocator
from datetime import datetime, timedelta, timezone
from bertopic import BERTopic
import plotly.io as pio 

nltk.download('stopwords')

# set the current and output dirs for Airflow and make the output directory if it does not exist already 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../data/output")
GRAPHS_DIR = os.path.join(CURRENT_DIR, "../graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

# add root directory of project to Python's import path so we can import modules from older folders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.s3_helpers import upload_to_s3

'''
This program uses Word2Vec to get similar clusters of words, and measures each cluster's average score.
'''

# preprocess: change all words to lowercase, remove punctuation, and stopwords amd tokenize (split bodies into words/tokens)
stop_words = set(stopwords.words('english'))

# function to split all of the text of the body into words/tokens
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text) # replace all non characters and white spaces with a blank 
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2] # filter out words less than 2 characters and that are in the stop words
    return tokens
    
def kmeans_elbow_method(max_k, data, output_file_path):
    inertia = []
    k_values = range(2, max_k + 1)

    # find inertia values for each cluster
    for k in k_values:
        kmeans = KMeans(n_clusters = k, random_state = 42, max_iter = 1000)
        kmeans.fit_predict(data)

        inertia.append(kmeans.inertia_) # saves the sum of squared distance between the cluster centroid and the data poits in each cluster
    
    # use the KneeLocator library to find the optimal k 
    kn = KneeLocator(k_values, inertia, curve = 'convex', direction = 'decreasing')
    optimal_k = kn.knee 

    # plot the intertia values to find the optimal k for K Means using the K Means method 
    plt.figure(figsize = (14, 6))
    plt.plot(k_values, inertia, marker = 'o', label = 'Inertia')
    plt.axvline(x = optimal_k, color = 'red', linestyle = '--', label = f'Optimal k = {optimal_k}')
    plt.title('KMeans Inertia for Different Values of k')
    plt.xlabel('Numer of Clusters (k)')
    plt.ylabel('Total WCSS/Error')
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file_path)

    return optimal_k

# add root directory of project to Python's import path so we can import modules from older folders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import aws_access_key, aws_secret_access_key, aws_bucket_name, aws_region
from utils.s3_helpers import load_latest_data_from_s3

def run_nlp_clustering_pipeline():

    # read the updated combined data using the load_combined_data heper function from utils to get the updated (daily) data
    combined_df = load_latest_data_from_s3(
        aws_access_key, 
        aws_secret_access_key, 
        aws_region, 
        aws_bucket_name, 
        prefix = 'processed',
        keyword = 'reddit_combined_data'
    )

    # initialize the huggingface sentiment pipeline to get a 
    sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

    # truncate the bodies of the posts to fit the models limits 
    combined_df['cleaned_body'] = combined_df['body'].fillna("").apply(lambda x: x[:512]) # 512 token limit for BERT

    # run the sentiment analysis in batches, with a batch size of 32 for each batch - returns a dictionary of 'label': and 'score':
    sentiment_results = sentiment_pipeline(combined_df['cleaned_body'].tolist(), truncation = True, batch_size = 32)

    # extract the sentiment label and sentiment score 
    combined_df['sentiment_label'] = [res['label'] for res in sentiment_results]
    combined_df['sentiment_score'] = [res['score'] if res['label'] == 'POSITIVE' else -res['score'] for res in sentiment_results]

    # get all preprocessed tokens for each post that are not null 
    combined_df['tokens'] = combined_df['body'].fillna("").apply(preprocess)

    # for TF-IDF vectorization, rejoin the tokens into strings 
    combined_df['processed_body'] = combined_df['tokens'].apply(lambda tokens: " ".join(tokens))

    # fit the TF-IDF vectorizer to the post bodies to extract the most important words based on their frequency across documents, while down-weighting common words
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_df['processed_body'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # switch the above df to a long formatted df so each row is a word with its tdidf score 
    tfidf_df_long = tfidf_df.stack().reset_index()
    tfidf_df_long.columns = ['post_num', 'word', 'tfidf']

    # find each word's avg TF-IDF 
    avg_tfidf_scores = tfidf_df_long.groupby('word')['tfidf'].mean().reset_index()
    avg_tfidf_scores.columns = ['word', 'avg_tfidf']

    # train the word2vec model 
    w2v_model = Word2Vec(
        sentences = combined_df['tokens'],
        vector_size = 100,
        window = 5, 
        min_count = 5,
        sg = 1, # skip gram model 
        workers = 4,
        epochs=50
    )

    # extract word vectors 
    word_vectors = w2v_model.wv
    words = list(word_vectors.index_to_key)
    vectors = word_vectors[words]

    # reduce the vectors to 2d for visualization purposes 
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

   # use the elbow method to check how many clusters to pick 
    kmeans_elbow_path = os.path.join(GRAPHS_DIR, "k_means_interia.png")
    optimal_k = kmeans_elbow_method(20, reduced, kmeans_elbow_path)
    upload_to_s3(kmeans_elbow_path, 'graphs')

    # use k-means clustering to get clusters of word groups 
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(reduced)

    # create a df for each cluster
    word_cluster_df = pd.DataFrame({
        "word": words, 
        "x": reduced[:, 0], 
        "y": reduced[:, 1], 
        "cluster": labels
    })

    # merge the above avg_tfidf_scores and the word cluster df on word to get each word's avg tfidf score for clustering 
    word_cluster_df = word_cluster_df.merge(avg_tfidf_scores, on='word', how='left')

    # filter for the top 10 words for each cluster
    top_words = (word_cluster_df
                .sort_values(by='avg_tfidf',ascending=False)
                .groupby('cluster')
                .head(10)
                .reset_index(drop=True))

    # visualize the vectors 
    plt.figure(figsize=(14, 8))
    sns.scatterplot(data=top_words, x='x', y='y', hue='cluster', palette='tab10', legend=True)

    top10words_clusters_path = os.path.join(GRAPHS_DIR, 'top_10_words_by_cluster.png')
    plt.savefig(top10words_clusters_path)
    upload_to_s3(top10words_clusters_path, 'graphs')

    texts = []

    # visualize the actual words as text in their cluster
    for _, row in top_words.iterrows():
        texts.append(plt.text(row["x"], row["y"], row["word"], fontsize=9))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # plot hull blocks around each cluster so they are easily distinguishable
    unique_clusters = top_words['cluster'].unique()
    colors = sns.color_palette('tab10', n_colors=len(unique_clusters))  # match colors to clusters

    for cluster_id, color in zip(unique_clusters, colors):
        cluster_points = word_cluster_df[word_cluster_df['cluster'] == cluster_id][['x', 'y']].values
        if len(cluster_points) >= 3:  # only draw a convex hull block if the cluster has at least 3 points/words
            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]
            plt.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.2, color=color, label=f"Cluster {cluster_id}")

    plt.title("Word2Vec Clusters of Reddit Words")
    
    word2vec_clusters_path = os.path.join(GRAPHS_DIR, 'word2vec_clusters.png')
    plt.savefig(word2vec_clusters_path)
    upload_to_s3(word2vec_clusters_path, 'graphs')

    ### - Based on the above clusters, compute the avg score and num of comments for each cluster based on if the cluster's words match a posts
    # store the cluster's scores in a list 
    cluster_scores = []

    # calculate the thresholds to see which cluster is in which threshold (top 30% or bottom 30%)
    top_30_threshold = combined_df['score'].quantile(0.70)
    bottom_30_threshold = combined_df['score'].quantile(0.30)

    # label each engagement level so we can add a new feature to the result set to measure a post by high, mid or low engagement 
    def label_engagement(score):
        if score > top_30_threshold:
            return 'High'
        elif score < bottom_30_threshold:
            return 'Low'
        else:
            return 'Mid'

    # get the engagement level of each post 
    combined_df['engagement_level'] = combined_df['score'].apply(label_engagement)

    for cluster_id in word_cluster_df['cluster'].unique():
        words_in_cluster = word_cluster_df[word_cluster_df['cluster'] == cluster_id]['word'].tolist() # grab all words in each cluster

        # build regex pattern to serach for any word from the cluster in each post body 
        pattern = r'\b(?:' + '|'.join(map(re.escape, words_in_cluster)) + r')\b'

        # see if the above pattern/any words from the cluster match words in a post and return the matching posts
        matched_posts = combined_df[combined_df['body'].str.contains(pattern, case = False, na = False)]

        # if matched posts are not empty, calculate the avg statistics for those posts that contain words that are in the cluster
        if not matched_posts.empty:
            avg_score = matched_posts['score'].mean()
            avg_comments = matched_posts['num_comments'].mean()
            avg_word_count = matched_posts['word_count'].mean()
            avg_sentiment = matched_posts['sentiment_score'].mean()
            avg_char_count = matched_posts['char_count'].mean()
            avg_readability = matched_posts['flesch_reading_ease'].mean()
            avg_sentence_count = matched_posts['sentence_count'].mean()
            avg_syllable_count = matched_posts['syllable_count'].mean()
            avg_smog_index = matched_posts['smog_index'].mean()
            pct_has_url = matched_posts['has_url'].mean()  # percentage of posts with URLs
            pct_has_code = matched_posts['has_code'].mean()  # percentage of posts with code

            # divide the post distributions into the top 30%, middle 40%, and bottom 30% of posts, and count how many posts fall into each category (boolean evaluates to 1 if True, 0 if False, so sum gives us the total posts per category)
            top_engagement_count = (matched_posts['score'] > top_30_threshold).sum()
            mid_engagement_count = ((matched_posts['score'] <= top_30_threshold) & 
                                    (matched_posts['score'] >= bottom_30_threshold)).sum()
            bottom_engagement_count = (matched_posts['score'] < bottom_30_threshold).sum()

            # create the dataframe that has the avg statistics for each cluster
            cluster_scores.append({
                "cluster": cluster_id,
                "keywords": ", ".join(words_in_cluster),  # show all words in cluster
                "avg_score": round(avg_score, 1),
                "avg_comments": round(avg_comments, 1),
                "avg_word_count": round(avg_word_count, 1),
                "avg_sentiment": round(avg_sentiment, 3),
                "avg_char_count": round(avg_char_count, 1),
                "avg_readability": round(avg_readability, 1),
                "avg_sentence_count": round(avg_sentence_count, 1),
                "avg_syllable_count": round(avg_syllable_count, 1),
                "avg_smog_index": round(avg_smog_index, 1),
                "pct_has_url": round(pct_has_url * 100, 1),
                "pct_has_code": round(pct_has_code * 100, 1),
                "top_30%_post_count": int(top_engagement_count),
                "middle_40%_post_count": int(mid_engagement_count),
                "bottom_30%_post_count": int(bottom_engagement_count),
                "example_post": matched_posts.iloc[0]['body']
            })

    cluster_insights_df = pd.DataFrame(cluster_scores)

    # rank by avg_score and avg_comments
    cluster_insights_df["score_rank"] = cluster_insights_df["avg_score"].rank(ascending=False).astype(int)
    cluster_insights_df["comment_rank"] = cluster_insights_df["avg_comments"].rank(ascending=False).astype(int)

    # sort the df by score_rank and comment_rank so highest engagement word clusters are at the top 
    cluster_insights_df = cluster_insights_df.sort_values(by=['score_rank', 'comment_rank'])

    cluster_insights_path = os.path.join(OUTPUT_DIR, 'cluster_engagement_insights.csv')
    cluster_insights_df.to_csv(cluster_insights_path, index = False)
    upload_to_s3(cluster_insights_path, 'processed')

    # draw a heatmap to get a comparison of each cluster's avg statistics 
    # filter the columns that we want to compare (cluster id, avg score, avg comments, numer of posts in each quantile)
    heatmap_df = cluster_insights_df[['cluster', 'avg_score', 
                                    'avg_comments', 'avg_word_count', 'avg_sentiment',
                                    'avg_char_count', 'avg_readability', 'avg_sentence_count', 'avg_syllable_count', 
                                    'avg_smog_index', 'pct_has_url', 'pct_has_code', 
                                    'top_30%_post_count', 'middle_40%_post_count', 'bottom_30%_post_count'
                                    ]].set_index('cluster')

    plt.figure(figsize = (14, 10))
    sns.heatmap(data = heatmap_df, annot = True, fmt = '.2f', cmap = 'YlGnBu')
    plt.title('Engagement Metrics by Word Cluster')
    plt.xticks(rotation = 45) # rotate x acix labels so thet titles are not cut off 
    plt.tight_layout() # precents cutting of labels and title
    
    cluster_engagement_metrics_path = os.path.join(GRAPHS_DIR, 'cluster_engagement_metrics.png')
    plt.savefig(cluster_engagement_metrics_path)
    upload_to_s3(cluster_engagement_metrics_path, 'graphs')

    # get each post's avg statistics like the cluster-wise avg statistics above 
    keyword_to_cluster = {} # initialize an empty hashmap that lists all keywords and their cluster they belong to - stores keyword_to_cluster[word] = cluster_id 

    # map each word of each post to a cluster
    for _,row in cluster_insights_df.iterrows():
        for word in str(row['keywords']).split(', '):
            keyword_to_cluster[word.strip()] = row['cluster']

    # function to assign the dominant cluster based on word presense in the body 
    def assign_dominant_cluster(text, keyword_map): 
        cluster_counts = {}
        words = str(text).lower().split()

        for word in words:
            if word in keyword_map:
                cluster = keyword_map[word] # get the cluster id of the word 
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1 # get a count of how many times this cluster's words appear in a post 

        if not cluster_counts: # if no matching cluster words are found, return -1 
            return -1 
        
        return max(cluster_counts, key = cluster_counts.get) # return the cluster ID with the max number of keyword matches in the post 

    # assign a domiannt cluster to each post 
    combined_df['dominant_cluster'] = combined_df['body'].apply(lambda text: assign_dominant_cluster(text, keyword_to_cluster))

    # see the original row count of combined_df before filtering to see if there were any rows removed after the below filter 
    original_row_count = combined_df.shape[0]
    print(f'Original Row Count without Removing Unmatched Posts: {original_row_count}')

    # remove any unmatched posts (where dominant_cluster returned - 1)
    combined_df = combined_df[
                            combined_df['dominant_cluster'] != -1] # remove any unmatched posts

    new_row_count = combined_df.shape[0]
    print(f'New Row Count without Removing Unmatched Posts: {new_row_count}')

    # recalculate the engagement thresholds only if filtering changed the dataset 
    if new_row_count < original_row_count: # recakculate the row counts if the new row count is less that the orignial row count 
        top_30_threshold = combined_df["score"].quantile(0.70)
        bottom_30_threshold = combined_df["score"].quantile(0.30)

        # reassign the engagement level 
        def label_engagement(score):
            if score > top_30_threshold:
                return 'High'
            elif score < bottom_30_threshold:
                return 'Low'
            else:
                return 'Mid'

        combined_df['engagement_level'] = combined_df['score'].apply(label_engagement)

    # merge all the cluster's statistics with each post's statistics using the dominant cluster for each post 
    combined_df = combined_df.merge(
            cluster_insights_df, 
            left_on = 'dominant_cluster',
            right_on = 'cluster',
            suffixes = ('', '_cluster')
    )

    # check the new combined df and all of its columns to see if it was merged correctly 
    print(combined_df.head())
    print(combined_df.columns)
    print(combined_df.info())

    # now after combined df is loaded, use BERTopic to cluster together topics
    # first, ensure that the 'Created_utc' column is in a datetime format
    combined_df['created_utc'] = pd.to_datetime(combined_df['created_utc'])

    # filter today's and this week's posts 
    today = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    start_of_week = today - timedelta(days = today.weekday())

    # since the DAG runs daily at 12:00 AM UTC, we subtract one day to analyze posts created on the previous day (i.e., "yesterday" in UTC time). Since the reddit has a posfix of today's date it means that the file storing a date's posts actually contains that previous day's posts
    today_posts = combined_df[combined_df['created_utc'].dt.date == today]
    week_posts = combined_df[combined_df['created_utc'].dt.date >= start_of_week]

    # extract the text from all the reddit posts (all historical text up until today, today's posts, and the week's post (since the start of the week on Monday))
    docs_all = combined_df['body'].fillna("").tolist()
    docs_today = today_posts['body'].fillna("").tolist()
    docs_week = week_posts['body'].fillna("").tolist()

    # Clustering all historical topics: fit BERTopic model on all the posts to date to get all topics/themes to date 
    topic_model = BERTopic(embedding_model = "all-MiniLM-L6-v2")
    topics_all, probs_all = topic_model.fit_transform(docs_all)

    # create the dataframe for all topics with the body, topic, and created_utc timestamp 
    all_topics_df = pd.DataFrame({
        'post': docs_all,
        'topic': topics_all,
        'created_utc': combined_df['created_utc']
    })
    all_topics_df = all_topics_df.sort_values(by='topic')

    topic_info_df = topic_model.get_topic_info()
    topic_info_path = os.path.join(OUTPUT_DIR, 'bertopic_overall_topic_info.csv')
    topic_info_df.to_csv(topic_info_path, index = False)
    upload_to_s3(topic_info_path, 'processed')

    # visualize the topics with a chart of clusters
    overall_topic_vis_path = os.path.join(GRAPHS_DIR, 'bertopic_overall_topics.html')
    topic_model.visualize_topics().write_html(overall_topic_vis_path)
    upload_to_s3(overall_topic_vis_path, 'graphs')

    # visualize the top 10 topics in a barchart format 
    bar_chart_path = os.path.join(GRAPHS_DIR, "bertopic_overall_barchart.html")
    topic_model.visualize_barchart(top_n_topics = 10).write_html(bar_chart_path)
    upload_to_s3(bar_chart_path, 'graphs')

    # Clustering the week's topics (starting from Monday UTC Time)
    topics_week, probs_week = topic_model.transform(docs_week)

    # create the dataframe for all topics with the body, topic, and created_utc timestamp 
    week_topics_df = pd.DataFrame({
        'post': docs_week,
        'topic': topics_week,
        'created_utc': week_posts['created_utc']
    })
    week_topics_df = week_topics_df.sort_values(by='topic')

    # visualize the weeks topics with a chart of clusters
    weekly_topic_vis_path = os.path.join(GRAPHS_DIR, 'bertopic_weekly_topics.html')
    topic_model.visualize_topics().write_html(weekly_topic_vis_path)
    upload_to_s3(weekly_topic_vis_path, 'graphs')

    weekly_barchart_path = os.path.join(GRAPHS_DIR, "bertopic_weekly_barchart.html")
    topic_model.visualize_barchart(top_n_topics = 10, topics = list(set(topics_week))).write_html(weekly_barchart_path)
    upload_to_s3(weekly_barchart_path, 'graphs')

    # Clustering the day's topics (UTC Time)
    topics_today, probs_today = topic_model.transform(docs_today)

    # create the dataframe for all topics with the body, topic, and created_utc timestamp 
    today_topics_df = pd.DataFrame({
        'post': docs_today,
        'topic': topics_today,
        'created_utc': today_posts['created_utc']
    })
    today_topics_df = today_topics_df.sort_values(by = 'topic')

    # visualize the weeks topics with a chart of clusters
    today_topic_vis_path = os.path.join(GRAPHS_DIR, 'bertopic_today_topics.html')
    topic_model.visualize_topics().write_html(today_topic_vis_path)
    upload_to_s3(today_topic_vis_path, 'graphs')

    today_barchart_path = os.path.join(GRAPHS_DIR, "bertopic_today_barchart.html")
    topic_model.visualize_barchart(top_n_topics = 10, topics = list(set(topics_today))).write_html(today_barchart_path)
    upload_to_s3(today_barchart_path, 'graphs')
    
    # save all, today's and weekly topic breakdown
    today_path = os.path.join(OUTPUT_DIR, 'bertopic_today_topics.csv')
    week_path = os.path.join(OUTPUT_DIR, 'bertopic_weekly_topics.csv')
    all_to_date_path = os.path.join(OUTPUT_DIR, 'bertopic_all_topics.csv')

    today_topics_df.to_csv(today_path, index = False)
    week_topics_df.to_csv(week_path, index = False)
    all_topics_df.to_csv(all_to_date_path, index = False)

    upload_to_s3(today_path, 'processed')
    upload_to_s3(week_path, 'processed')
    upload_to_s3(all_to_date_path, 'processed')

    # drop all of the columns that are not needed to train the model, like id, url, author, cleaned_body (only used for transformer based NLP), example posts
    combined_df = combined_df.drop(columns = ['id', 'url', 'author', 'cleaned_body'])

    post_engagement_insights_path = os.path.join(OUTPUT_DIR, 'post_engagement_insights.csv')
    combined_df.to_csv(post_engagement_insights_path, index = False)
    upload_to_s3(post_engagement_insights_path, 'processed')

    return {
    "rows_processed": combined_df.shape[0],
    "clusters_found": len(cluster_insights_df),
    "output_csv": os.path.join(OUTPUT_DIR, 'post_engagement_insights.csv')
    }

if __name__ == "__main__":
    run_nlp_clustering_pipeline()