from fastapi import APIRouter, HTTPException, status, Depends
from botocore.exceptions import ClientError
import uuid
import os
from typing import List, Annotated

# Import dependencies, schemas, services, and core logic
from middleware.auth.auth_deps import user_dependency
from schemas.ocr_schema import (
    InitiateUploadRequest, InitiateUploadResponse, 
    PresignedUpload, StartJobRequest, OCRResult
)
from services import storage_service
from core.config import settings
from helper.ocr import run_prediction # Unchanged external dependency

router = APIRouter(
    prefix="/ocr",
    tags=["ocr", "upload"],
)

# --- Endpoint 1: Initiate Uploads ---

@router.post("/initiate-uploads", response_model=InitiateUploadResponse)
async def initiate_uploads(request: InitiateUploadRequest, user: user_dependency):
    """
    Generates a unique Job ID and pre-signed URLs 
    for the authenticated frontend to upload files directly to S3.
    Requires authentication via user_dependency.
    """
    job_id = str(uuid.uuid4()) # Unique session ID
    uploads = []

    for file in request.files:
        # Create a unique key for S3: uploads/{job_id}/{filename}
        s3_key = f"uploads/{job_id}/{file.filename}"

        try:
            # Use the storage service to get the URL
            presigned_url = storage_service.generate_presigned_upload_url(
                s3_key=s3_key,
                content_type=file.content_type
            )
            
            uploads.append(
                PresignedUpload(
                    filename=file.filename,
                    key=s3_key,
                    upload_url=presigned_url
                )
            )

        except ClientError as e:
            print(f"Error generating pre-signed URL for {file.filename}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate upload URL for {file.filename}."
            )

    return InitiateUploadResponse(job_id=job_id, uploads=uploads)


# --- Endpoint 2: Process a Single Image (Legacy/Direct) ---

@router.get("/process/{image_id}")
async def process_image(image_id: str):
    """
    Legacy/Direct endpoint to process a single image using its S3 key.
    NOTE: This key should be the full path, e.g., 'uploads/job_id/filename.jpg'.
    """
    # NOTE: The original logic in main.py had an issue where it assumed 
    # image_id was the full S3 key, but then prepended 'uploads/'.
    # For robustness, we assume image_id is the full key path.
    object_key = image_id
    
    local_download_dir = "./temp_downloads"
    os.makedirs(local_download_dir, exist_ok=True)
    # Use the base filename for the local path
    local_image_path = os.path.join(local_download_dir, os.path.basename(object_key))

    try:
        print(f"Attempting to download: {object_key}")
        storage_service.download_file(object_key, local_image_path)
        print(f"Successfully downloaded to: {local_image_path}")

        # Run OCR prediction
        text = run_prediction(
            settings.YOLO_MODEL_PATH, 
            settings.CRNN_MODEL_PATH, 
            local_image_path,
            settings.OCR_MODE
        )
        
        return {"image_id": image_id, "extracted_text": text}
    
    except ClientError as e:
        if '404' in str(e):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File not found in Spaces: {object_key}")
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"S3 Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Processing Error: {str(e)}")
    finally:
        if os.path.exists(local_image_path):
            os.remove(local_image_path)
            print(f"Cleaned up temporary file: {local_image_path}")


# --- Endpoint 3: Process All Images in a Job ---

@router.post("/get-ocr-results", response_model=List[OCRResult])
async def get_ocr_results(request: StartJobRequest, user: user_dependency):
    """
    Processes all files associated with a given job_id.
    Requires authentication via user_dependency.
    """
    job_id = request.job_id
    s3_prefix = f"uploads/{job_id}/"
    local_download_dir = "./temp_downloads"
    os.makedirs(local_download_dir, exist_ok=True)
    
    ocr_results = []
    
    try:
        # Get list of all file keys for this job
        object_keys = storage_service.list_files_in_prefix(s3_prefix)
        
        if not object_keys:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail=f"No files found for job ID: {job_id}"
            )

        for object_key in object_keys:
            filename = os.path.basename(object_key)
            local_image_path = os.path.join(local_download_dir, filename)

            try:
                # Download the file
                print(f"Downloading: {object_key}")
                storage_service.download_file(object_key, local_image_path)

                # Run OCR prediction
                print(f"Running OCR on: {filename}")
                text = run_prediction(
                    settings.YOLO_MODEL_PATH, 
                    settings.CRNN_MODEL_PATH, 
                    local_image_path,
                    settings.OCR_MODE
                )
                
                ocr_results.append(
                    OCRResult(filename=filename, text=text)
                )

            except ClientError as e:
                print(f"S3 Error downloading {object_key}: {e}")
                ocr_results.append(OCRResult(filename=filename, text=f"Error: S3 download failed."))
            except Exception as e:
                print(f"Processing Error on {filename}: {e}")
                ocr_results.append(OCRResult(filename=filename, text=f"Error: Processing failed."))
            finally:
                if os.path.exists(local_image_path):
                    os.remove(local_image_path)

    except ClientError as e:
        print(f"S3 Error during list operation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Error listing files in S3."
        )

    if not ocr_results:
        # Should be caught by the list_files_in_prefix check, but good to have
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No files were processed.")

    return ocr_results