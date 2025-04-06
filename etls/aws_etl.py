from utils.constants import aws_access_key, aws_secret_access_key
import s3fs

def connect_to_s3():
    try:
        s3 = s3fs.S3FileSystem(
            anon = False,
            key = aws_access_key, 
            secret = aws_secret_access_key
        )

        return s3
    except Exception as e:
        print(e)

def create_bucket_if_not_exists(s3: s3fs.S3FileSystem, bucket_name: str):
    try:
        if not s3.exists(bucket_name):
            s3.mkdir(bucket_name)
            print('New bucket created')
        else:
            print('Bucket already exists')

    except Exception as e:
        print(e)

def upload_to_s3(s3: s3fs.S3FileSystem, file_path: str, bucket_name: str, s3_file_name: str):
    try:
        s3.put(file_path, bucket_name + '/raw/' + s3_file_name)
        print('File uploaded to S3')

    except FileNotFoundError:
        print('The file wsa not found')