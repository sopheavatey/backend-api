from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Image Upload & OCR Backend")

# --- Image Upload Endpoint ---
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # upload logic
        return JSONResponse(content={"message": "Image received", "filename": file.filename})
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
