import praw
from praw import Reddit
import sys
import os
import pandas as pd
import numpy as np

# add root directory of project to Python's import path so we can import modules from older folders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import post_fields

# takes the client id, secret, and the agent and returns the reddit instance 
def connect_to_reddit(client_id, client_secret, user_agent) -> Reddit:
    '''
    returns the reddit instnance 
    '''
    try:
        reddit = praw.Reddit(client_id = client_id,
                              client_secret = client_secret,
                              user_agent = user_agent)
        print("Successfully connected to reddit")
        return reddit
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) # exit if the program fails to connect to reddit

# define a function to extract reddit posts 
def extract_posts(reddit_instance, subreddit, time_filter, limit):
    subreddit = reddit_instance.subreddit(subreddit)
    posts = subreddit.top(time_filter=time_filter, limit=limit)

    post_lists = []

    for post in posts:
        post_dict = vars(post) # convert the post object to a dict 
       
        post = {key: post_dict[key] for key in post_fields}

        post_lists.append(post)
    
    return post_lists

# define a function to transform the data 
def transform_data(post_df: pd.DataFrame):
    # change date to a dt time format in pandas 
    post_df['created_utc'] = pd.to_datetime(post_df['created_utc'], unit='s')
    post_df['over_18'] = np.where((post_df['over_18'] == True), True, False)
    post_df['author'] = post_df['author'].astype(str)

    # change edited to a datetime format, and if it is false, keep it the same
    post_df['edited'] = post_df['edited'].apply(
        lambda x: pd.to_datetime(x, unit = 's') if x not in [False, None] else x
    )

    post_df['num_comments'] = post_df['num_comments'].astype(int)
    post_df['score'] = post_df['score'].astype(int)
    post_df['upvote_ratio'] = post_df['upvote_ratio'].astype(int)
    post_df['selftext'] = post_df['selftext'].astype(str)
    post_df['title'] = post_df['title'].astype(str)
    
    return post_df

# define a function to load all the data to a csv
def load_data_to_csv(data: pd.DataFrame, path: str):
    data.to_csv(path, index=False)