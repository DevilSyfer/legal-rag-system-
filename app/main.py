from fastapi import APIRouter, UploadFile, File, HTTPException, FastAPI
import os
import shutil
from app.api.routes import router

app = FastAPI(title="Legal RAG System")

app.include_router(router)
