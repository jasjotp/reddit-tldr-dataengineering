import pandas as pd 
import numpy as np 
import sklearn 
import io 
import os 

# read in the csv to get each cluster's features 
cluster_insights_df = pd.read_csv(r'C:\Users\Jasjot Parmar\Airflow-Docker\reddit_pipeline_project\data\output\cluster_engagement_insights.csv')

print(cluster_insights_df.head())

