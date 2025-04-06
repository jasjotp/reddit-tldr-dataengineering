from airflow import DAG 
from airflow.operators.python import PythonOperator
from datetime import datetime 
import os 
import sys

# add root directory of project to Python's import path so we can import modules from older folders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.reddit_pipeline import reddit_pipeline
from pipelines.aws_s3_pipeline import upload_to_s3_pipeline

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
    python_callable=upload_to_s3_pipeline,
    dag = dag
)

extract >> upload_to_s3

