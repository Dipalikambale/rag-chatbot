from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings

def get_embeddings():
    # Mounts the high-efficiency embedding layer straight into system memory
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL
    )