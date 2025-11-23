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
from typing import List

# Load environment variables from .env
from dotenv import load_dotenv
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

YOLO_MODEL_PATH = "model/yolo_best_inference.pt"
CRNN_CHECKPOINT_PATH = "model/crnn_best.onnx"

#Create a boto3 client for Spaces 
s3_client = boto3.client(
    "s3",
    region_name="sgp1",
    endpoint_url="https://sgp1.digitaloceanspaces.com",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

# Define class for type
class UploadRequest(BaseModel):
    filename: str
    content_type: str


class FileMetadata(BaseModel):
    filename: str
    content_type: str

class InitiateUploadRequest(BaseModel):
    files: List[FileMetadata]

class PresignedUpload(BaseModel):
    filename: str
    key: str
    upload_url: str

class InitiateUploadResponse(BaseModel):
    job_id: str
    uploads: List[PresignedUpload]

class StartJobRequest(BaseModel):
    job_id: str

class OCRResult(BaseModel):
    filename: str
    text: str

# --- Endpoint 1: Initiate Uploads ---

@app.post("/api/initiate-uploads", response_model=InitiateUploadResponse)
async def initiate_uploads(request: InitiateUploadRequest):
    """
    Called by React. Generates a unique Job ID and pre-signed URLs
    for the frontend to upload files directly to S3.
    """
    job_id = str(uuid.uuid4()) # Your unique session ID
    uploads = []

    for file in request.files:
        # Create a unique key for S3
        # e.g., "uploads/job-123-abc/image.png"
        s3_key = f"uploads/{job_id}/{file.filename}"

        try:
            # Generate the pre-signed URL
            presigned_url = s3_client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': SPACES_NAME,
                    'Key': s3_key,
                    'ContentType': file.content_type
                },
                ExpiresIn=3600  # URL is valid for 1 hour
            )
            
            uploads.append(
                PresignedUpload(
                    filename=file.filename,
                    key=s3_key,
                    upload_url=presigned_url
                )
            )

        except Exception as e:
            print(f"Error generating pre-signed URL: {e}")
            # Handle error appropriately
            pass

    return InitiateUploadResponse(job_id=job_id, uploads=uploads)




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

@app.get("/api/health-check")
async def health_check():
    """Health check endpoint for React frontend"""
    response = s3_client.list_objects_v2(Bucket=SPACES_NAME)
    for obj in response.get("Contents", []):
        return obj["Key"]

@app.post("/api/get-ocr-results", response_model=List[OCRResult])
async def get_ocr_results(request: StartJobRequest):
    job_id = request.job_id
    
    # --- FIX #2: THE S3 PREFIX ---
    # This prefix MUST match the key from Endpoint 1
    s3_prefix = f"uploads/{job_id}/"
    # ---
    
    local_download_dir = "./temp_downloads"
    os.makedirs(local_download_dir, exist_ok=True)
    
    ocr_results = []

    try:
        # This will now list objects in:
        # Bucket: "fyp-ocr-25"
        # Prefix: "fyp-ocr-25/uploads/{job_id}/"
        print(f"Listing objects in bucket '{SPACES_NAME}' with prefix '{s3_prefix}'")
        list_response = s3_client.list_objects_v2(Bucket=SPACES_NAME, Prefix=s3_prefix)
        
        if 'Contents' not in list_response:
            print(f"No contents found for prefix: {s3_prefix}")
            raise HTTPException(status_code=404, detail=f"No files found for job ID: {job_id}")

        # ... (rest of your function remains the same) ...
        for obj in list_response['Contents']:
            object_key = obj['Key']
            filename = os.path.basename(object_key)
            if not filename:
                continue
                
            local_image_path = os.path.join(local_download_dir, filename)

            try:
                print(f"Downloading: {object_key}")
                s3_client.download_file(SPACES_NAME, object_key, local_image_path)

                print(f"Running OCR on: {filename}")
                text = run_prediction(
                    YOLO_MODEL_PATH, 
                    CRNN_CHECKPOINT_PATH, 
                    local_image_path
                )
                
                ocr_results.append(
                    OCRResult(filename=filename, text=text)
                )

            except ClientError as e:
                print(f"S3 Error downloading {object_key}: {e}")
                ocr_results.append(
                    OCRResult(filename=filename, text=f"Error: S3 download error.")
                )
            except Exception as e:
                print(f"Processing Error on {filename}: {e}")
                ocr_results.append(
                    OCRResult(filename=filename, text=f"Error: Processing failed.")
                )
            finally:
                if os.path.exists(local_image_path):
                    os.remove(local_image_path)

    except ClientError as e:
        print(f"S3 Error listing objects: {e}")
        raise HTTPException(status_code=500, detail="Error listing files in S3.")

    if not ocr_results:
        raise HTTPException(status_code=404, detail="No files were processed.")

    return ocr_results

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