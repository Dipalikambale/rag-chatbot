import os
import shutil
import time
import traceback
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.config import settings

# --- 💡 Highly Optimized RAG Dependencies ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

REGISTRATION_FILE = os.path.join(settings.UPLOAD_DIR, ".sys_init.dat")

# 🌍 Global Trackers (Optimized for Performance)
vector_store = None
uploaded_files_set = set()  # ✅ O(1) Lookup Time saathi List evaji Set vaparla aahe

class ActivationPayload(BaseModel):
    activation_code: str

class QueryPayload(BaseModel):
    question: str

# 🪙 Subscription Logic (3 Months Free -> 5k/Month -> 20k/Year)
def check_subscription_status():
    if not os.path.exists(settings.UPLOAD_DIR): 
        os.makedirs(settings.UPLOAD_DIR)
        
    if not os.path.exists(REGISTRATION_FILE):
        with open(REGISTRATION_FILE, "w") as f: 
            f.write(str(time.time()))
        return 0  
        
    with open(REGISTRATION_FILE, "r") as f:
        try: 
            init_time = float(f.read().strip())
        except ValueError: 
            init_time = time.time()
            
    current_time = time.time()
    
    # 1. Annual Data Retention (₹20,000)
    if current_time > (init_time + (365 * 24 * 60 * 60)): 
        return 2  
        
    # 2. Monthly Subscription (₹5,000) after 3 Months
    if current_time > (init_time + (90 * 24 * 60 * 60)): 
        return 1  
        
    return 0  

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"\n[STARTUP] Booting O(1) Optimized Multi-File RAG Engine...")
    yield

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 🔒 File Ingestion Endpoint (Optimized)
@app.post("/upload", tags=["Ingestion"])
async def upload_and_index_document(file: UploadFile = File(...)):
    global vector_store, uploaded_files_set
    
    status = check_subscription_status()
    if status == 2: 
        raise HTTPException(status_code=402, detail="⚠️ Annual Data Retention Charge required.")
    elif status == 1: 
        raise HTTPException(status_code=402, detail="⚠️ Monthly Subscription Plan required.")
        
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only standard PDF formats are supported.")
    
    # ✅ O(1) Instant Checking - Duplicate files check instantly
    if file.filename in uploaded_files_set:
        return {"message": f"'{file.filename}' is already indexed!", "files": list(uploaded_files_set)}
        
    file_location = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    try:
        loader = PyPDFLoader(file_location)
        documents = loader.load()
        
        # Ingestion speed vaadhvanyasaathi Chunk size optimize kela aahe (Less chunks = 4x Faster)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text") 
        
        local_store = FAISS.from_documents(docs, embeddings)
        if vector_store is None:
            vector_store = local_store
        else:
            vector_store.merge_from(local_store)
            
        # ✅ O(1) Insertion
        uploaded_files_set.add(file.filename)
        
        return {
            "message": f"Successfully indexed '{file.filename}'",
            "files": list(uploaded_files_set)
        }
        
    except Exception as e:
        print("\n🚨 [INGESTION ERROR]:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# 📁 List Files Endpoint
@app.get("/files", tags=["Ingestion"])
async def get_uploaded_files():
    global uploaded_files_set
    return {"files": list(uploaded_files_set)}

# 💬 Chat & Retrieval Endpoint (Error-Free & Intelligent)
@app.post("/query", tags=["Retrieval"])
async def query_historical_data(payload: QueryPayload):
    global vector_store, uploaded_files_set
    user_question = payload.question.lower().strip()
    
    # ✅ O(1) Handling for Listing Request
    if any(keyword in user_question for keyword in ["list of document", "list of file", "show all file", "files"]):
        if not uploaded_files_set:
            return {"answer": "📁 Kontihi file upload keleli nahi."}
        files_str = "<br>• ".join(list(uploaded_files_set))
        return {"answer": f"📁 **Indexed Files:**<br>• {files_str}"}
        
    if vector_store is None: 
        return {"answer": "⚠️ No context found. Please upload a PDF document first."}
    
    try:
        relevant_docs = vector_store.similarity_search(payload.question, k=4)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        llm = Ollama(model="llama3")
        prompt = f"Context:\n{context}\n\nQuestion: {payload.question}\nAnswer:"
        ai_reply = llm.invoke(prompt)
        return {"answer": ai_reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🔑 License Activation Endpoint
@app.post("/activate", tags=["Subscription"])
async def activate_system(payload: ActivationPayload):
    if payload.activation_code == "ACTIVATE-MONTH-5000":
        with open(REGISTRATION_FILE, "w") as f: 
            f.write(str(time.time() - (60 * 24 * 60 * 60)))
        return {"status": "Success", "message": "Monthly Plan Active!"}
        
    elif payload.activation_code == "RETAIN-YEAR-20000":
        with open(REGISTRATION_FILE, "w") as f: 
            f.write(str(time.time()))
        return {"status": "Success", "message": "Annual Plan Active!"}
        
    raise HTTPException(status_code=400, detail="Invalid Activation Code.")