import boto3
import os 
from datetime import datetime 
import pandas as pd
import io
from utils.constants import aws_access_key, aws_secret_access_key, aws_bucket_name, aws_region

# helper function to upload the graphs to s3
def upload_to_s3(local_path, s3_folder_name):
    """
    connects to AWS S3 and uploads a file into the specified folder name
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

def load_latest_data_from_s3(aws_access_key, aws_secret_access_key, aws_region, aws_bucket_name, prefix, keyword):
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

    # list all files under the prefix 
    response = s3.list_objects_v2(
        Bucket = aws_bucket_name,
        Prefix = prefix
    )
    if 'Contents' not in response:
        raise FileNotFoundError(f'No files found under: {prefix}')
    
    # filter for files by keyword and .csv extension
    matching_files = [
        obj['Key']
        for obj in response.get('Contents', [])
            if keyword in obj['Key'] and obj['Key'].endswith('.csv')
    ]
    if not matching_files:
        raise FileNotFoundError(f'No files found for the keyword: {keyword}')

    # sort the matching files and select the latest file 
    latest_key = sorted(matching_files, reverse = True) # sort in descending order

    obj = s3.get_object(Bucket = aws_bucket_name, Key = latest_key[0])

    # read the processed combined data file from the aws bucket an
    combined_df = pd.read_csv(io.BytesIO(obj['Body'].read()))

    print(f"Loaded latest file from S3: {latest_key[0]}")
    return combined_df