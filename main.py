from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Image Upload & OCR Backend")

# --- Image URL for Digital Ocean Spaces Object Storage ---
@app.get("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Get a unique url from digital ocean spaces object storage and send it back to frontend to upload the images to digital ocean spaces 
        return JSONResponse(content={"message": "Image URL generated", "filename": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- OCR Processing Endpoint ---
@app.get("/process/{image_id}")
async def process_image(image_id: str):
    try:
        # OCR 
        text = "OCR result placeholder"
        return {"image_id": image_id, "extracted_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
