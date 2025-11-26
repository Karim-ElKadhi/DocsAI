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
import google.generativeai as genai

# Gemini setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-1.5-flash"

# MiniLM embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# =============================
#   UTILS
# =============================



def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size)]


def embed_text(chunks: List[str]):
    return embedding_model.encode(chunks, convert_to_numpy=True)