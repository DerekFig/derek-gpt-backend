# main.py

import io
import json
import os
import base64
from uuid import UUID
from typing import Optional, List

import fitz  # PyMuPDF
from PIL import Image
from pypdf import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
from openpyxl import load_workbook
from io import BytesIO

from fastapi import (
    FastAPI,
    Depends,
    UploadFile,
    File,
    Form,
    HTTPException,
    Header,
    Query,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from openai import OpenAI, RateLimitError
from db import get_db

# --------------------------------------------------------
# Environment-backed keys
# --------------------------------------------------------

INTERNAL_BACKEND_KEY = os.getenv("INTERNAL_BACKEND_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "changeme-reset-key")

# --------------------------------------------------------
# Guards
# --------------------------------------------------------

def verify_internal_key(x_internal_key: str = Header(None)):
    if INTERNAL_BACKEND_KEY is None:
        raise HTTPException(status_code=500, detail="Internal key not configured")
    if x_internal_key != INTERNAL_BACKEND_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def verify_admin(x_admin_api_key: str = Header(None)):
    if x_admin_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

# --------------------------------------------------------
# FastAPI app
# --------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://derek-gpt-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------
# OpenAI config
# --------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 1024

client = OpenAI()
vision_client = OpenAI()

# --------------------------------------------------------
# Helpers
# --------------------------------------------------------

def embedding_to_pgvector_literal(vec: list[float]) -> str:
    return "[" + ", ".join(f"{x:.8f}" for x in vec) + "]"


def embed_text(text: str) -> list[float]:
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIM,
        )
        return response.data[0].embedding
    except RateLimitError:
        raise HTTPException(status_code=503, detail="OpenAI quota exceeded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e!r}")

# --------------------------------------------------------
# Pydantic models
# --------------------------------------------------------

class QueryRequest(BaseModel):
    tenant_id: str
    workspace_id: str
    query: str
    top_k: int = 12


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(QueryRequest):
    history: Optional[List[ChatMessage]] = None


class QueryResult(BaseModel):
    document_id: str
    chunk_index: int
    score: float
    chunk_text: str

# --------------------------------------------------------
# QUERY ENDPOINT (FIXED)
# --------------------------------------------------------

@app.post("/query", dependencies=[Depends(verify_internal_key)])
def query_embeddings(payload: QueryRequest, db: Session = Depends(get_db)):
    query_embedding = embed_text(payload.query)
    query_vec = embedding_to_pgvector_literal(query_embedding)

    sql = text("""
        SELECT
            e.document_id,
            e.chunk_index,
            e.chunk_text,
            1 - (e.embedding <=> CAST(:query_embedding AS vector(1024))) AS score
        FROM embeddings e
        JOIN workspaces w ON w.id = e.workspace_id
        WHERE w.tenant_id = :tenant_id
          AND e.workspace_id = :workspace_id
        ORDER BY e.embedding <=> CAST(:query_embedding AS vector(1024))
        LIMIT :top_k;
    """)

    rows = db.execute(
        sql,
        {
            "tenant_id": UUID(payload.tenant_id),
            "workspace_id": UUID(payload.workspace_id),
            "query_embedding": query_vec,
            "top_k": payload.top_k,
        },
    ).fetchall()

    return {
        "query": payload.query,
        "results": [
            QueryResult(
                document_id=str(r.document_id),
                chunk_index=r.chunk_index,
                score=float(r.score),
                chunk_text=r.chunk_text,
            ).dict()
            for r in rows
        ],
    }

# --------------------------------------------------------
# CHAT ENDPOINT (FIXED)
# --------------------------------------------------------

@app.post("/chat", dependencies=[Depends(verify_internal_key)])
def chat_with_workspace(payload: ChatRequest, db: Session = Depends(get_db)):
    query_embedding = embed_text(payload.query)
    query_vec = embedding_to_pgvector_literal(query_embedding)

    sql = text("""
        SELECT
            e.chunk_text,
            1 - (e.embedding <=> CAST(:query_embedding AS vector(1024))) AS score
        FROM embeddings e
        JOIN workspaces w ON w.id = e.workspace_id
        WHERE w.tenant_id = :tenant_id
          AND e.workspace_id = :workspace_id
        ORDER BY e.embedding <=> CAST(:query_embedding AS vector(1024))
        LIMIT :top_k;
    """)

    rows = db.execute(
        sql,
        {
            "tenant_id": UUID(payload.tenant_id),
            "workspace_id": UUID(payload.workspace_id),
            "query_embedding": query_vec,
            "top_k": payload.top_k,
        },
    ).fetchall()

    if not rows:
        return {"answer": "No relevant documents found.", "results": []}

    context = "\n\n---\n\n".join(r.chunk_text for r in rows)

    prompt = (
        "Answer strictly using the context below.\n\n"
        f"{context}\n\n"
        f"Question: {payload.query}"
    )

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return {
        "answer": completion.choices[0].message.content,
        "results": [{"chunk_text": r.chunk_text, "score": float(r.score)} for r in rows],
    }

# --------------------------------------------------------
# STREAMING CHAT (FIXED)
# --------------------------------------------------------

@app.post("/chat-stream", dependencies=[Depends(verify_internal_key)])
def chat_stream(payload: ChatRequest, db: Session = Depends(get_db)):
    query_embedding = embed_text(payload.query)
    query_vec = embedding_to_pgvector_literal(query_embedding)

    sql = text("""
        SELECT chunk_text
        FROM embeddings
        WHERE tenant_id = :tenant_id
          AND workspace_id = :workspace_id
        ORDER BY embedding <=> CAST(:query_embedding AS vector(1024))
        LIMIT :top_k;
    """)

    rows = db.execute(
        sql,
        {
            "tenant_id": UUID(payload.tenant_id),
            "workspace_id": UUID(payload.workspace_id),
            "query_embedding": query_vec,
            "top_k": payload.top_k,
        },
    ).fetchall()

    if not rows:
        def empty():
            yield "No relevant context found."
        return StreamingResponse(empty(), media_type="text/plain")

    context = "\n\n---\n\n".join(r.chunk_text for r in rows)

    def stream():
        stream = client.chat.completions.create(
            model="gpt-4.1-mini",
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": f"{context}\n\nQuestion: {payload.query}",
                }
            ],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content

    return StreamingResponse(stream(), media_type="text/plain")

# --------------------------------------------------------
# HEALTH + WHOAMI
# --------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/whoami")
def whoami():
    return {"running_file": "MAIN_PY_IS_RUNNING_FIXED_VECTOR_CAST"}
