import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import re 
from gensim.models import Word2Vec
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from nltk.corpus import stopwords 
import nltk
import os 
import sys 
from adjustText import adjust_text

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

# filter for the top 10 words for each cluster
top_words = word_cluster_df.groupby("cluster").apply(lambda df: df.sample(n=min(10, len(df)), random_state=42)).reset_index(drop=True)

# visualize the vectors 
plt.figure(figsize=(14, 8))
sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette='tab10', legend=False)

texts = []

for _, row in top_words.iterrows():
    texts.append(plt.text(row["x"], row["y"], row["word"], fontsize=9))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

plt.title("Word2Vec Clusters of Reddit Words")
os.makedirs("semantic_clustering", exist_ok=True)
plt.savefig("semantic_clustering/word2vec_clusters.png", bbox_inches='tight')