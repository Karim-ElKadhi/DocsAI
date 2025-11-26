import os
import asyncio
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import os
import google.generativeai as genai
from utils.utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    chunk_text,
    embed_text,
)   
#Docs storage folder
UPLOAD_FOLDER = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Gemini setup
genai.configure(api_key="AIzaSyDu9Wjy9ov_AYQn0eBtUZ_2sMJZ7TEFl4w")

GEMINI_MODEL = "gemini-2.5-flash"

# MiniLM embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faiss_index = None
text_chunks = []
metadata_store = []
# FastAPI init
app = FastAPI(title="RAG API ")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
FAISS_FILE = "metadata/faiss_index.index"
CHUNKS_FILE = "metadata/text_chunks.json"
METADATA_FILE = "metadata/metadata.json"



# FAISS index
embedding_dimension = 384  

# Load function
def load_index():
    global faiss_index, text_chunks, metadata_store
    if os.path.exists(FAISS_FILE) and os.path.exists(CHUNKS_FILE):
        faiss_index = faiss.read_index(FAISS_FILE)
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            text_chunks = json.load(f)
    else:
        faiss_index = faiss.IndexFlatL2(embedding_dimension)
        text_chunks = []

    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata_store = json.load(f)
    else:
        metadata_store = []

# Save function
def save_index():
    faiss.write_index(faiss_index, FAISS_FILE)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(text_chunks, f, ensure_ascii=False, indent=2)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, ensure_ascii=False, indent=2)

async def ask_gemini(prompt: str) -> str:
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text

#Loading existing index
load_index()

#   ENDPOINTS
class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

# Upload endpoint
@app.post("/upload")
async def upload_files(files: List[UploadFile]):
    new_chunks = []
    new_metadata = []

    for file in files:
        file_location = f"{UPLOAD_FOLDER}/{file.filename}"
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Extract text
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_location)
        elif file.filename.endswith(".docx"):
            text = extract_text_from_docx(file_location)
        elif file.filename.endswith(".txt"):
            text = extract_text_from_txt(file_location)
        else:
            raise HTTPException(400, f"Unsupported format: {file.filename}")

        # Chunk text
        chunks = chunk_text(text)
        new_chunks.extend(chunks)

        # Add metadata for each chunk
        new_metadata.extend([{"file": file.filename}] * len(chunks))

    # Append new chunks + metadata
    text_chunks.extend(new_chunks)
    metadata_store.extend(new_metadata)

    # Embed and add to FAISS
    embeddings = embed_text(new_chunks)
    faiss_index.add(embeddings)

    # Save to disk for persistence
    save_index()

    return {"status": "success", "chunks_added": len(new_chunks), "files_uploaded": [f.filename for f in files]}


# Chat endpoint
@app.post("/chat")
async def chat_with_rag(req: ChatRequest):
    if faiss_index.ntotal == 0:
        raise HTTPException(400, "No knowledge base loaded")

    # Embed query
    query_embedding = embedding_model.encode([req.query])
    distances, indices = faiss_index.search(query_embedding, req.top_k)

    # Retrieve relevant chunks
    retrieved_texts = [text_chunks[i] for i in indices[0]]
    retrieved_sources = [metadata_store[i]["filename"] for i in indices[0]]
    context = "\n\n".join(retrieved_texts)

    # Build Gemini prompt
    prompt = f"""
You are an AI assistant. Use the following context to answer the question.

Context:
{context}

Question: {req.query}

Answer clearly and concisely:
"""

    response = await ask_gemini(prompt)

    return {
        "response": response,
        "context_used": retrieved_texts,
        "sources": retrieved_sources
    }
