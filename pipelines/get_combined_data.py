import pandas as pd 
import numpy as np 
import boto3
import io 
import sys
import os
import textstat 
import re
from datetime import datetime

# add root directory of project to Python's import path so we can import modules from older folders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import aws_access_key, aws_secret_access_key, aws_bucket_name, aws_region
from utils.s3_helpers import upload_to_s3

# define current and output directory 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../data/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_combined_data():
    '''
    connects to Amazon S3, grabs all of the files in the S3 bucket and combines them, before cleaning the data and creating nccessessary columns for EDA
    '''
    # create the session 
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )

    # initialize the s3 client
    s3 = session.client('s3')

    # prefix for S3 folder to pull from 
    prefix = 'raw/'

    response = s3.list_objects_v2(Bucket=aws_bucket_name, Prefix=prefix)

    files = [f['Key'] for f in response.get('Contents', []) if f['Key'].endswith('.csv')]

    # Load and combine
    dfs = []

    for file_key in files:
        obj = s3.get_object(Bucket=aws_bucket_name, Key=file_key)

        df = pd.read_csv(io.BytesIO(obj['Body'].read()))

        dfs.append(df)

    # Combine into one DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(dfs)} files. Combined shape: {combined_df.shape}")

    # check all posts that are null or blank (have spaces or just newlines), and clean and prepare all columns needed for EDA
    nulls = combined_df['selftext'].isnull()
    blanks = combined_df['selftext'].fillna('').str.strip() == ''
    removed_or_deleted = combined_df['selftext'].isin(['[removed]', '[deleted]']) # filter for posts that contain the reddit removed and deleted tags
    empty_posts = nulls | blanks | removed_or_deleted
    combined_df = combined_df.rename(columns = {'selftext': 'body'})
    combined_df['created_dt'] = pd.to_datetime(combined_df['created_utc'])
    combined_df['edited'] = combined_df['edited'].astype(str)
    combined_df['hour'] = combined_df['created_dt'].dt.hour
    combined_df['weekday'] = combined_df['created_dt'].dt.day_name()
    combined_df['word_count'] = combined_df['body'].str.split().str.len()
    combined_df['body'] = combined_df['body'].fillna("").astype(str) # fill all blank bodies with empty strings so the vectorizer can successfully analyze them
    combined_df['char_count'] = combined_df['body'].apply(len) # character count of a post 
    combined_df['flesch_reading_ease'] = combined_df['body'].apply(textstat.flesch_reading_ease) # readability score
    combined_df['sentence_count'] = combined_df['body'].apply(textstat.sentence_count)
    combined_df['syllable_count'] = combined_df['body'].apply(textstat.syllable_count)
    combined_df['smog_index'] = combined_df['body'].apply(textstat.smog_index) # score that ranks the level of education one needs to understand the most (higher score = need more education)
    combined_df['has_url'] = combined_df['body'].str.contains('http|www', case = False).astype(int)  # boolean that checks if a post contains a URL (starts with www or http)

    # check for posts that contain code (check for triple backticks or tabs or common variables in languages)
    code_pattern = r'(?i)(```|^ {4,}|\t|def |class |import |SELECT |<div>|function|console\.log|var |let |const )'
    combined_df['has_code'] = combined_df['body'].str.contains(code_pattern, regex=True, flags=re.MULTILINE).astype(int)

    # save locally 
    local_path = os.path.join(OUTPUT_DIR, 'reddit_combined_data.csv')
    combined_df.to_csv(local_path, index = False)

    # upload the combined df csv to S3
    upload_to_s3(local_path, 'processed')
    
    print(f"Uploaded cleaned combined data to S3 and saved to: {local_path}")