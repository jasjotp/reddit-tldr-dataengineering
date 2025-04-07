import os 
os.environ['OMP_NUM_THREADS'] = '3' # set the OP NUM THREADS env variable to 3 to avoid the potential memory leak caused my using KMeans on a Windows Intel machine
import pandas as pd 
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

'''
This program uses Word2Vec to get similar clusters of words, and measures each cluster's average score.
'''

nltk.download('stopwords')

# add root directory of project to Python's import path so we can import modules from older folders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eda.eda import get_combined_data

combined_df = get_combined_data()

print(combined_df.head())

# preprocess: change all words to lowercase, remove punctuation, and stopwords amd tokenize (split bodies into words/tokens)
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text) # replace all non characters and white spaces with a blank 
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2] # filter out words less than 2 characters and that are in the stop words
    return tokens

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

# use k-means clustering to get clusters of wrod groups 
kmeans = KMeans(n_clusters=10, random_state=42)
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
plt.savefig("semantic_clustering/word2vec_clusters.png", bbox_inches='tight')

### - Based on the above clusters, compute the avg score and num of comments for each cluster based on if the cluster's words match a posts
# store the cluster's scores in a list 
cluster_scores = []

for cluster_id in word_cluster_df['cluster'].unique():
    words_in_cluster = word_cluster_df[word_cluster_df['cluster'] == cluster_id]['word'].tolist() # grab all words in each cluster

    # build regex pattern to serach for any word from the cluster in each post body 
    

