import os
from langchain_community.vectorstores import FAISS
from app.embeddings import get_embeddings
from app.config import settings

def load_vectorstore():
    # Verify index configuration existence to prevent application boot errors
    if not os.path.exists(os.path.join(settings.VECTORSTORE_DIR, "index.faiss")):
        return None

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        settings.VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True  # Required for local file processing
    )
    return vectorstore