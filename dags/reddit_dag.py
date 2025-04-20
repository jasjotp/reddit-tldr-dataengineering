from airflow import DAG 
from airflow.operators.python import PythonOperator
from datetime import datetime 
import os 
import sys

# add root directory of project to Python's import path so we can import modules from older folders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.reddit_pipeline import reddit_pipeline
from pipelines.aws_s3_pipeline import upload_to_s3_pipeline
from pipelines.get_combined_data import get_combined_data
from pipelines.eda_pipeline import run_reddit_eda
from semantic_clustering.w2v_analysis import run_nlp_clustering_pipeline

# set default arguments 
default_args = {
    'owner': 'Jasjot Parmar',
    'start_date': datetime(year=2025, month=3, day=27)
}

# the format we are appending our files with 
file_postfix = datetime.now().strftime("%Y%m%d")

dag = DAG(
    dag_id = 'etl_reddit_pipeline',
    default_args = default_args,
    schedule_interval = '@daily',
    catchup=False, # don't run previously missed runs, only run from the current date forward
    tags = ['reddit', 'etl', 'pipeline']

)

# extract from Reddit
extract = PythonOperator(
    task_id = 'reddit_extraction',
    python_callable = reddit_pipeline,
    op_kwargs = {
        'file_name': f'reddit_{file_postfix}',
        'subreddit': 'dataengineering',
        'time_filter': 'day', 
        'limit': 100
    }, 
    dag = dag
)

# upload to s3 bucket 
upload_to_s3 = PythonOperator(
    task_id = 'S3_Upload',
    python_callable = upload_to_s3_pipeline,
    dag = dag
)

# grab the updated data from the s3 bucket daily and conduct cleaning and feature engineering through rubbing the get_combined_data.py file 
combine_and_clean_data = PythonOperator(
    task_id = 'combine_and_clean_data',
    python_callable = get_combined_data,
    dag = dag
)

# run EDA on the updated data and generate graphs 
generate_eda_and_graphs = PythonOperator(
    task_id = 'generate_eda_and_graphs',
    python_callable = run_reddit_eda, 
    dag = dag
)

# run the w2v analysis and NLP pipeline on the updated data before generating updated post level statistics and keywords for each cluster
run_w2v_nlp_analysis = PythonOperator(
    task_id = 'run_w2v_and_nlp_analysis',
    python_callable = run_nlp_clustering_pipeline,
    dag = dag
)

extract >> upload_to_s3 >> combine_and_clean_data >> generate_eda_and_graphs >> run_w2v_nlp_analysis