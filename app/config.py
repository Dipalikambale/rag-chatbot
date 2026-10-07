
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # Application Identification
    APP_NAME: str = "Secure Local-RAG Engine"

    # Highly optimized open-source weights running locally at zero cost
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "google/flan-t5-base"

    # On-Premise Data Directory Settings
    VECTORSTORE_DIR: str = "vectorstore"
    UPLOAD_DIR: str = "uploaded_documents"


settings = Settings()

# Instantly auto-create secure workspace directories on server startup
os.makedirs(
    settings.UPLOAD_DIR,
    exist_ok=True
)
