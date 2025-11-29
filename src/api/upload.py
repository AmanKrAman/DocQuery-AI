from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
from pathlib import Path
import secrets
import string

router = APIRouter()

def generate_unique_id(length: int = 10) -> str:
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_extension = Path(file.filename).suffix.lower()
        print(f"Received file: {file.filename} with extension: {file_extension}")
        if file_extension not in [".txt", ".pdf"]:
            raise HTTPException(
                status_code=400, 
                detail="Only .txt and .pdf files allowed"
            )
        
        unique_id = generate_unique_id(10)
        
        upload_folder = os.path.join("workfiles", unique_id)
        os.makedirs(upload_folder, exist_ok=True)
        
        file_path = os.path.join(upload_folder, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        from src.ingestion.ingestion_pipeline import DocumentIngestionPipeline
        
        pipeline = DocumentIngestionPipeline()
        pipeline.ingest_file(file_path, unique_id=unique_id)
        
        return JSONResponse(content={
            "message": f"File '{file.filename}' uploaded successfully",
            "unique_id": unique_id,
            "file_path": file_path,
            "filename": file.filename,
            "status": "uploaded"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
