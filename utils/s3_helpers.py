import boto3
import os 
from datetime import datetime 
import pandas as pd
import io
from utils.constants import aws_access_key, aws_secret_access_key, aws_bucket_name, aws_region

# helper function to upload the graphs to s3
def upload_graph_to_s3(local_path, s3_folder_name = 'graphs'):
    """
    connects to AWS S3 and cretes a folder called graphs in the exisiting bucket name.
    Uploads each graph with its name and timestamp to the /graphs directory.
    """
    file_name = os.path.basename(local_path)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    s3_key = f'{s3_folder_name}/{os.path.splitext(file_name)[0]}_{timestamp}{os.path.splitext(file_name)[1]}'

    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )

    s3 = session.client('s3')

    s3.upload_file(local_path, aws_bucket_name, s3_key)
    print(f'Uploaded {local_path} to S3 at: {aws_bucket_name}/{s3_key}')

def load_combined_data_from_s3(aws_access_key, aws_secret_access_key, aws_region, aws_bucket_name, key='processed/reddit_combined_data.csv'):
    """
    connects to AWS S3 and loads the processed combined Reddit data from a specified key.
    Returns the combined DataFrame.
    """
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )

    s3 = session.client('s3')

    # read the processed combined data file from the aws bucket and read it into a pandas df
    obj = s3.get_object(Bucket=aws_bucket_name, Key=key)

    combined_df = pd.read_csv(io.BytesIO(obj['Body'].read()))

    return combined_df