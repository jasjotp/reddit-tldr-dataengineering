import pandas as pd 
import numpy as np 
import sklearn 
import io 
import os 
import sys 

# add root directory of project to Python's import path so we can import modules from older folders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import aws_access_key, aws_secret_access_key, aws_bucket_name, aws_region
from utils.s3_helpers import load_latest_data_from_s3

# read the updated combined data using the load_combined_data heper function from utils to get the updated (daily) data
combined_df = load_latest_data_from_s3(
    aws_access_key, 
    aws_secret_access_key, 
    aws_region, 
    aws_bucket_name, 
    prefix = 'processed',
    keyword = 'post_engagement_insights'
)

print(combined_df.head())

