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
from nltk.corpus import stopwords 
import nltk
import sys 
from adjustText import adjust_text
from scipy.spatial import ConvexHull
import plotly.express as px
from kneed import KneeLocator

nltk.download('stopwords')
'''
This program uses Word2Vec to get similar clusters of words, and measures each cluster's average score.
'''

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

from eda.eda import get_combined_data

combined_df = get_combined_data()

print(combined_df.head())

# preprocess: change all words to lowercase, remove punctuation, and stopwords amd tokenize (split bodies into words/tokens)
stop_words = set(stopwords.words('english'))

# get all preprocessed tokens for each post that are not null 
combined_df['tokens'] = combined_df['body'].dropna().apply(preprocess)

# for TF-IDF vectorization, rejoin the tokens into strings 
combined_df['processed_body'] = combined_df['tokens'].apply(lambda tokens: " ".join(tokens))

# fit the TF-IDF vectorizer to the post bodies to extract the most important words based on their frequency across documents, while down-weighting common words
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_df['processed_body'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# swtich the above df to a long formatted df so each row is a word with its tdidf score 
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
optimal_k = kmeans_elbow_method(20, reduced, 'k_means_interia.png')

# use k-means clustering to get clusters of wrod groups 
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
plt.savefig("word2vec_clusters.png", bbox_inches='tight')

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

combined_df['engagement_level'] = combined_df['score'].apply(label_engagement)

for cluster_id in word_cluster_df['cluster'].unique():
    words_in_cluster = word_cluster_df[word_cluster_df['cluster'] == cluster_id]['word'].tolist() # grab all words in each cluster

    # build regex pattern to serach for any word from the cluster in each post body 
    pattern = r'\b(?:' + '|'.join(map(re.escape, words_in_cluster)) + r')\b'

    # see if the above pattern/any words from the cluster match words in a post and return the matching posts
    matched_posts = combined_df[combined_df['body'].str.contains(pattern, case = False, na = False)]

    # if matched posts are not empty, calculate the avg score and avg num of comments for those posts that contain words that are in the cluster
    if not matched_posts.empty:
        avg_score = matched_posts['score'].mean()
        avg_comments = matched_posts['num_comments'].mean()
        avg_word_count = matched_posts['word_count'].mean()
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
            "keywords": ", ".join(words_in_cluster[:10]),  # show only the top 10 keywords for readability
            "avg_score": round(avg_score, 1),
            "avg_comments": round(avg_comments, 1),
            "avg_word_count": round(avg_word_count, 1),
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

# save the df to a csv file 
cluster_insights_df.to_csv('../data/output/cluster_engagement_insights.csv')

# draw a heatmap to get a comparison of each cluster's avg statistics 
# filter the columns that we want to compare (cluster id, avg score, avg comments, numer of posts in each quantile)
heatmap_df = cluster_insights_df[['cluster', 'avg_score', 
                                  'avg_comments', 'avg_word_count', 'avg_char_count', 
                                  'avg_readability', 'avg_sentence_count', 'avg_syllable_count', 
                                  'avg_smog_index', 'pct_has_url', 'pct_has_code', 
                                  'top_30%_post_count', 'middle_40%_post_count', 'bottom_30%_post_count'
                                  ]].set_index('cluster')

plt.figure(figsize = (14, 10))
sns.heatmap(data = heatmap_df, annot = True, fmt = '.2f', cmap = 'YlGnBu')
plt.title('Engagement Metrics by Word Cluster')
plt.savefig('cluster_engagement_metrics.png')