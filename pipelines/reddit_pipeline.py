import sys
import os 
import pandas as pd

# add root directory of project to Python's import path so we can import modules from older folders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import client_id, secret, output_path
from etls.reddit_etl import connect_to_reddit, extract_posts, transform_data, load_data_to_csv

def reddit_pipeline(file_name: str, subreddit:str, time_filter='day', limit=None):
    # connect to reddit instance 
    instance = connect_to_reddit(client_id, secret, 'Jasjot Paarmars Agent')
    
    # extract the data 
    posts = extract_posts(instance, subreddit, time_filter, limit)
    
    posts_df = pd.DataFrame(posts)

    # transform the data 
    posts_df = transform_data(posts_df)

    # load the data to a csv
    filepath = f'{output_path}/{file_name}.csv'

    load_data_to_csv(posts_df, filepath)

    return filepath