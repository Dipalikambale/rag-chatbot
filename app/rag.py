import logging
from app.llm import generate_answer

logger = logging.getLogger("uvicorn.error")

# Semantic boundary for smaller open-source models (lower distance = closer match)
THRESHOLD = 1.2  

def get_answer(question: str, vectorstore):
    if vectorstore is None:
        return "ERROR: No documents have been indexed yet. Please upload a PDF file via the sidebar interface first."

    # Execute a vector similarity search pulling the top 3 closest text blocks
    results = vectorstore.similarity_search_with_score(question, k=5)
    docs = []

    for doc, score in results:
        if score <= THRESHOLD:
            docs.append(doc)

    if not docs:
        logger.warning(f"Query match failed threshold boundary check for: '{question}'")
        return "DATA NOT FOUND IN DOCUMENT."

    # Merge match segments into a clean context block
    context = "\n\n".join(doc.page_content for doc in docs)

    # Fire local language execution model pipeline
    answer = generate_answer(context=context, question=question)
    return answer