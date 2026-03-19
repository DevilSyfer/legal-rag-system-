from fastapi import APIRouter, UploadFile, File, HTTPException, FastAPI
import os
import shutil
from app.api.routes import router
from app.core.logging_config import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Legal RAG System")
app.include_router(router)

logger.info("Legal RAG System started successfully")