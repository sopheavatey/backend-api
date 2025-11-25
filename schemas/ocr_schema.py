from pydantic import BaseModel
from typing import List

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