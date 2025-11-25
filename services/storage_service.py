import boto3
from botocore.exceptions import ClientError
from core.config import settings

# Initialize S3 client using settings
s3_client = boto3.client(
    "s3",
    region_name=settings.SPACES_REGION,
    endpoint_url=settings.SPACES_ENDPOINT,
    aws_access_key_id=settings.ACCESS_KEY,
    aws_secret_access_key=settings.SECRET_KEY,
)

def generate_presigned_upload_url(s3_key: str, content_type: str) -> str:
    """
    Generates a pre-signed URL for the client to upload a file directly to S3.
    
    :param s3_key: The full path/key in the S3 bucket (e.g., 'uploads/job_id/filename.png').
    :param content_type: The MIME type of the file.
    :return: The generated pre-signed URL string.
    :raises ClientError: If there's an issue with S3.
    """
    try:
        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': settings.SPACES_NAME,
                'Key': s3_key,
                'ContentType': content_type
            },
            ExpiresIn=3600  # URL is valid for 1 hour
        )
        return presigned_url
    except ClientError as e:
        raise ClientError(f"Error generating pre-signed URL: {e}", operation_name='generate_presigned_url')


def download_file(s3_key: str, local_path: str):
    """
    Downloads a file from S3 to a local path.
    
    :param s3_key: The path/key in the S3 bucket.
    :param local_path: The local path to save the file.
    :raises ClientError: If the file is not found or other S3 error occurs.
    """
    try:
        s3_client.download_file(
            settings.SPACES_NAME,  # Bucket name
            s3_key,                # The path to the file in the bucket
            local_path             # The local path to save it to
        )
    except ClientError as e:
        raise ClientError(f"Error downloading file {s3_key}: {e}", operation_name='download_file')


def list_files_in_prefix(s3_prefix: str) -> list[str]:
    """
    Lists all object keys under a given prefix in the S3 bucket.
    
    :param s3_prefix: The prefix to search under (e.g., 'uploads/job_id/').
    :return: A list of object keys (full S3 paths).
    :raises ClientError: If there's an issue with S3.
    """
    try:
        list_response = s3_client.list_objects_v2(
            Bucket=settings.SPACES_NAME, 
            Prefix=s3_prefix
        )
        
        if 'Contents' not in list_response:
            return [] # No files found
            
        # Return only the keys that represent actual files
        return [obj['Key'] for obj in list_response['Contents'] if obj['Size'] > 0]
        
    except Exception as e:
            print(f"S3 LIST ERROR: {e}")  # This will print the real error to your logs
            raise e  # Raise the original error, don't try to wrap it