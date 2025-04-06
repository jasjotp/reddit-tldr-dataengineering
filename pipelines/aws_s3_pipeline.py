from etls.aws_etl import connect_to_s3, create_bucket_if_not_exists, upload_to_s3
from utils.constants import aws_bucket_name
from airflow.utils.log.logging_mixin import LoggingMixin

logger = LoggingMixin().log

def upload_to_s3_pipeline(**kwargs):
    task_instance = kwargs['ti']  # Airflow task instance

    file_path = task_instance.xcom_pull(task_ids = 'reddit_extraction', key = 'return_value') # gets the .csv's file path from Airflow
    logger.info(f"Pulled file path from XCom: {file_path}")

    s3 = connect_to_s3()
    logger.info("Connected to S3")

    create_bucket_if_not_exists(s3, aws_bucket_name)
    logger.info(f"Checked/created bucket: {aws_bucket_name}")

    upload_to_s3(s3, file_path, aws_bucket_name, file_path.split('/')[-1])
    logger.info(f"Uploaded file to S3: {file_path}")
