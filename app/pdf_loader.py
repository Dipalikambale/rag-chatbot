import logging
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger("uvicorn.error")

def load_pdf(pdf_path: str):
    try:
        loader = PyPDFLoader(pdf_path)
        # Sequential data extraction loop
        return loader.load()
    except Exception as e:
        logger.error(f"[CRITICAL PARSE ERROR] Failed to parse PDF structure at: {pdf_path}. Details: {str(e)}")
        raise RuntimeError("The uploaded PDF is corrupted, password-protected, or layout-locked.")