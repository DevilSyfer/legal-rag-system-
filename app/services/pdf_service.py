import fitz
import os

def extract_text_from_pdf(file_path:str) ->str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FIle not found: {file_path}")
    
    doc = fitz.open(file_path)
    full_text = ""
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n---Page {page_num + 1} ---\n{text}"
    
    doc.close()
    
    return full_text

