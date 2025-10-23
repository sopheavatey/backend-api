from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import os
from datetime import datetime
from helper.ocr import run_prediction


#Load environment variables from .env
load_dotenv()

app = FastAPI(title="Image Upload & OCR Backend")

#DigitalOcean Spaces Configuration
SPACES_REGION = os.getenv("SPACES_REGION")
SPACES_NAME = os.getenv("SPACES_NAME")
SPACES_ENDPOINT = os.getenv("SPACES_ENDPOINT")
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

YOLO_MODEL_PATH = "./model/YOLO.pt"
CRNN_CHECKPOINT_PATH = "./model/CRNN.pth"

#Create a boto3 client for Spaces 
s3_client = boto3.client(
    "s3",
    region_name=SPACES_REGION,
    endpoint_url=SPACES_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

#Image URL for Digital Ocean Spaces Object Storage 
@app.post("/upload")
async def generate_upload_url(filename: str):
    try:
        # Give file a unique key (path) inside the bucket
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        object_key = f"uploads/{timestamp}_{filename}"

        # Generate pre-signed URL valid for 1 hour (3600 seconds)
        presigned_url = s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": SPACES_NAME, "Key": object_key, "ContentType": "image/jpeg"},
            ExpiresIn=3600,
        )

        return JSONResponse(
            content={
                "upload_url": presigned_url,
                "file_key": object_key,
                "message": "Use this URL to upload directly to DigitalOcean Spaces."
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# OCR Processing Endpoint
@app.get("/process/{image_id}")
async def process_image(image_id: str):
    
    object_key = f"uploads/{image_id}" 
    
    local_download_dir = "./temp_downloads"
    os.makedirs(local_download_dir, exist_ok=True)
    local_image_path = os.path.join(local_download_dir, image_id)

    try:
        print(f"Attempting to download: {object_key} from bucket: {SPACES_NAME}")
        s3_client.download_file(
            SPACES_NAME,       # Bucket name
            object_key,        # The path to the file in the bucket
            local_image_path   # The local path to save it to
        )
        print(f"Successfully downloaded to: {local_image_path}")

        text = run_prediction(
            YOLO_MODEL_PATH, 
            CRNN_CHECKPOINT_PATH, 
            local_image_path
        )
        
        return {"image_id": image_id, "extracted_text": text}
    
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            raise HTTPException(status_code=404, detail=f"File not found in Spaces: {object_key}")
        else:
            raise HTTPException(status_code=500, detail=f"S3 Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing Error: {str(e)}")
    finally:
        if os.path.exists(local_image_path):
            os.remove(local_image_path)
            print(f"Cleaned up temporary file: {local_image_path}")

