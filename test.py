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


response = s3_client.list_objects_v2(Bucket=SPACES_NAME)

print("response:", response)