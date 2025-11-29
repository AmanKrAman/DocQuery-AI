from fastapi import FastAPI
import os

from src.api.upload import router as upload_router
from src.api.query import router as query_router
# from src.api.files import router as files_router

app = FastAPI(title="RAG Document Search API")

os.makedirs("workfiles", exist_ok=True)
os.makedirs("data/vector_db", exist_ok=True)

app.include_router(upload_router, tags=["Upload"])
app.include_router(query_router, tags=["Query"])
# app.include_router(files_router, tags=["Files"])


@app.get("/")
async def root():
    return {
        "message": "RAG API is running!", 
        "status": "healthy",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
