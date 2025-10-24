# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from botocore.exceptions import ClientError
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os
import uuid
from datetime import datetime
from helper.ocr import run_prediction


# Load environment variables from .env
load_dotenv()

app = FastAPI(title="Image Upload & OCR Backend")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DigitalOcean Spaces Configuration
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


class UploadRequest(BaseModel):
    filename: str
    content_type: str


@app.post("/upload")
async def get_upload_url(request: UploadRequest):
    """
    Generate a presigned URL for uploading a file to DigitalOcean Spaces.
    Returns both the upload URL and the public URL.
    """
    try:
        print(f"\n[UPLOAD REQUEST] Filename: {request.filename}, Content-Type: {request.content_type}")
        
        # Generate unique filename to avoid collisions
        file_extension = request.filename.split(".")[-1] if "." in request.filename else ""
        unique_filename = f"{uuid.uuid4()}.{file_extension}" if file_extension else str(uuid.uuid4())
        
        # Organize files in folders with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        s3_key = f"ocr-uploads/{timestamp}/{unique_filename}"
        
        print(f"[S3 KEY] {s3_key}")
        
        # Generate presigned URL for PUT operation
        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': SPACES_NAME,
                'Key': s3_key,
                'ContentType': request.content_type,
                'ACL': 'public-read'  # Make uploaded files publicly readable
            },
            ExpiresIn=3600  # URL expires in 1 hour
        )
        
        print(f"[PRESIGNED URL GENERATED] Success")
        
        # Construct public URL for DigitalOcean Spaces
        public_url = f"https://{SPACES_NAME}.{SPACES_REGION}.digitaloceanspaces.com/{s3_key}"
        
        print(f"[PUBLIC URL] {public_url}")
        
        return {
            "upload_url": presigned_url,
            "public_url": public_url,
            "s3_key": s3_key,
            "bucket": SPACES_NAME,
            "timestamp": timestamp
        }
    
    except ClientError as e:
        print(f"[ERROR] ClientError: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Spaces Error: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


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

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "OCR Upload API is running",
        "status": "healthy",
        "spaces_configured": bool(SPACES_NAME and ACCESS_KEY)
    }


@app.get("/test-connection")
async def test_spaces_connection():
    """Test DigitalOcean Spaces connection"""
    try:
        print("\n[TEST CONNECTION] Attempting to connect to Spaces...")
        print(f"[TEST CONNECTION] Endpoint: {SPACES_ENDPOINT}")
        print(f"[TEST CONNECTION] Region: {SPACES_REGION}")
        print(f"[TEST CONNECTION] Bucket: {SPACES_NAME}")
        
        # Try a simpler operation first - check if bucket exists
        try:
            s3_client.head_bucket(Bucket=SPACES_NAME)
            print(f"[TEST CONNECTION] ✓ Bucket '{SPACES_NAME}' exists and is accessible")
            
            return {
                "status": "success",
                "message": f"Successfully connected to Space: {SPACES_NAME}",
                "bucket_accessible": True,
                "endpoint": SPACES_ENDPOINT,
                "region": SPACES_REGION
            }
        except ClientError as e:
            error_code = e.response['Error']['Code']
            print(f"[TEST CONNECTION] ✗ Bucket check failed with error: {error_code}")
            
            if error_code == '404':
                raise HTTPException(
                    status_code=404,
                    detail=f"Bucket '{SPACES_NAME}' not found. Please check your SPACES_NAME configuration."
                )
            elif error_code == '403':
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied to bucket '{SPACES_NAME}'. Check your ACCESS_KEY and SECRET_KEY."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Bucket access error ({error_code}): {str(e)}"
                )
        
    except ClientError as e:
        print(f"[TEST CONNECTION] ClientError: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Connection failed: {str(e)}"
        )
    except Exception as e:
        print(f"[TEST CONNECTION] Exception: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error: {str(e)}"
        )


@app.get("/debug-config")
async def debug_config():
    """Debug endpoint to check configuration (NEVER use in production!)"""
    return {
        "spaces_region": SPACES_REGION,
        "spaces_name": SPACES_NAME,
        "spaces_endpoint": SPACES_ENDPOINT,
        "access_key_set": bool(ACCESS_KEY),
        "secret_key_set": bool(SECRET_KEY),
        "access_key_prefix": ACCESS_KEY[:10] if ACCESS_KEY else None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)