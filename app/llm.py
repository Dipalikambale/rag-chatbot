from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from app.config import settings

print("[SYSTEM] Loading local Language Model weights into memory...")
tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(settings.LLM_MODEL)

local_generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512
)

def generate_answer(context: str, question: str):
    # A highly detailed corporate prompt to force structural text extraction
    prompt = f"""You are a precise corporate recruitment AI assistant.
Your task is to extract exact information from the text context provided.

Instructions:
- Provide a direct, descriptive answer using the factual context.
- If the question asks for "education", extract the degree names, colleges, or years mentioned.
- If the question asks for "experience", extract company names, job titles, or dates.
- Do not repeat generic resume headers or skills if they do not answer the prompt.
- If the text context contains absolutely no information to answer the question, state exactly: DATA NOT FOUND IN DOCUMENT.

CONTEXT:
{context}

QUESTION:
{question}

DETAILED ANSWER:"""

    response = local_generator(
        prompt, 
        max_new_tokens=150,  # Slightly increased to allow descriptive extraction
        do_sample=False,
        repetition_penalty=3.0  # Increased to strongly suppress repeating words
    )
    return response[0]['generated_text'].strip()