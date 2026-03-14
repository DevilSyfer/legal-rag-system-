from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.pdf_service import extract_text_from_pdf
from app.services.chunking_service import chunk_text
from app.services.vector_store_service import create_collection, store_chunks,search_results
from app.services.embedding_service import get_embeddings
import shutil
import os
from app.services.llm_service import get_answer

import logging
logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = "uploads"


@router.get("/")
def health_check():
    return {"status":"running"}

@router.post("/upload")
async def upload_pdf(file:UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF Files allowed")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    logger.info(f"File Path:- {file_path}")
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
        
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except IOError as e:
        logger.error(f"\nFIle Not Saved:- {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File saving failed: {str(e)}")

    
    try:
        text = extract_text_from_pdf(file_path)
    except Exception as e:
        logger.error(f"\nPDF parsing failed:- {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {str(e)}")
    
    collection_name = file.filename.replace(".pdf", "").replace(" ", "_")
    
    try:
        create_collection(collection_name)
    except Exception :
        print("Collection already exists, Continue")
    
    chunks = chunk_text(text)
    chunk_texts = [chunk['text'] for chunk in chunks]
    try:
        # risky operation 
        embeddings = get_embeddings(chunk_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
   
    try:
        # risky operation 
        store_chunks(collection_name, chunks,embeddings)
    except Exception as e:
       
       raise HTTPException(status_code=500, detail=f"Storing in Qdrant failed: {str(e)}")

    logger.info(f"Successfully processed {file.filename} — {len(chunks)} chunks stored")
    return{
        "filename": file.filename,
        "status"  : "uploaded",
        "path"    : file_path,
        "extracted_character" : len(text),
        "total_chunks" : len(chunks),
        "chunk_Preview" : chunks[:2]
    }

@router.post("/user_query") 
def user_query(collection_name:str,userquery:str,limit:int):
    try:
        query_embedding = get_embeddings([userquery])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User Query Embedding failed: {str(e)}")
    try:
        result = search_results(collection_name,query_embedding,limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search result not found: {str(e)}")

    try:
        llm_answer = get_answer(userquery,result)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"LLM answer generation failed: {str(e)}")
       
    
    return llm_answer

