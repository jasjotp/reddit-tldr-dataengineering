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


# add root directory of project to Python's import path so we can import modules from older folders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eda.eda import get_combined_data

combined_df = get_combined_data()

print(combined_df.head())