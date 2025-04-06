import configparser
import os 

parser = configparser.ConfigParser()

# specify the source of the config file 
parser.read(os.path.join(os.path.dirname(__file__), '../config/config.conf'))

# get the reddit api key and client id
secret = parser.get('api_keys', 'reddit_secret_key')
client_id = parser.get('api_keys', 'reddit_client_id')

# get the database details
database_host = parser.get('database', 'database_host')
database_name = parser.get('database', 'database_name')
database_port = parser.get('database', 'database_port')
database_user = parser.get('database', 'database_username')
database_pw = parser.get('database', 'database_password')

# get aws details 
aws_secret_access_key = parser.get('aws', 'aws_secret_access_key')
aws_access_key = parser.get('aws', 'aws_access_key')
aws_region = parser.get('aws', 'aws_region')
aws_bucket_name = parser.get('aws', 'aws_bucket_name')

# get the input and output path 
input_path = parser.get('file_paths', 'input_path')
output_path = parser.get('file_paths', 'output_path')

post_fields = [
    'id',
    'title',
    'selftext',
    'score',
    'num_comments',
    'author',
    'created_utc',
    'url',
    'upvote_ratio',
    'over_18',
    'edited',
    'spoiler',
    'stickied'
]

