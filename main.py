# main.py

# CORS settings updated for localhost frontend

import io
import json
import os
import fitz  # PyMuPDF
from openai import OpenAI
import base64
from io import BytesIO
from uuid import UUID
from typing import Optional, List
from PIL import Image

import base64
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

# For extracting text from files
from pypdf import PdfReader
from docx import Document as DocxDocument


# --------------------------------------------------------
# FastAPI app & CORS
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
# --- OCR HELPERS USING OPENAI VISION ---
# --------------------------------------------------------

vision_client = OpenAI()  # uses OPENAI_API_KEY from env


def _image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ocr_pages_with_openai(page_images):
    """
    Given a list of PIL Images (one per PDF page),
    call OpenAI Vision to extract ALL visible text from each page.
    Returns a single big string.
    """
    ocr_chunks = []

    for idx, img in enumerate(page_images):
        b64 = _image_to_base64(img)

        resp = vision_client.chat.completions.create(
            model="gpt-4.1-mini",  # or "gpt-4.1" if you prefer
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract ALL text visible on this slide/page. "
                                "Return plain text only, no commentary."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=800,
        )

        text = (resp.choices[0].message.content or "").strip()
        if text:
            ocr_chunks.append(f"[PAGE {idx+1} OCR]\n{text}")

    return "\n\n".join(ocr_chunks)


def extract_text_from_pdf_bytes_with_ocr(file_bytes: bytes) -> str:
    """
    Use PyMuPDF to render each page of the PDF (from bytes) to an image,
    then run OCR using OpenAI Vision on each page.
    Returns all OCR text as one string.
    """
    try:
        # Open the PDF from bytes using PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        print(f"PyMuPDF failed to open PDF: {e}")
        return ""

    page_images = []
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            # Render the page to a pixmap (image)
            pix = page.get_pixmap(dpi=200)
            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_images.append(img)
    except Exception as e:
        print(f"Error rendering PDF pages to images: {e}")
        return ""

    if not page_images:
        return ""

    # Run OCR with OpenAI Vision on the rendered page images
    ocr_text = ocr_pages_with_openai(page_images)
    return ocr_text

# --------------------------------------------------------
# Admin reset config
# --------------------------------------------------------

class ResetWorkspaceRequest(BaseModel):
    tenant_id: UUID
    workspace_id: UUID


ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "changeme-reset-key")


def verify_admin(x_admin_api_key: str = Header(None)):
    if x_admin_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


# --------------------------------------------------------
# OpenAI + embedding model config
# --------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-large"
# Must match Supabase: embedding vector(1024)
EMBEDDING_DIM = 1024

# OpenAI client (uses OPENAI_API_KEY from env)
client = OpenAI()


# --------------------------------------------------------
# Pydantic models
# --------------------------------------------------------

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    tenant_id: str
    workspace_id: str
    query: str
    top_k: int = 5
    history: Optional[List[ChatMessage]] = None


class TenantCreate(BaseModel):
    name: str
    type: str = "client"


class WorkspaceCreate(BaseModel):
    tenant_id: str
    name: str


class WorkspaceUpdate(BaseModel):
    system_prompt: Optional[str] = None
    default_model: Optional[str] = None
    temperature: Optional[float] = None


class DocumentIn(BaseModel):
    tenant_id: str
    workspace_id: str
    original_filename: str
    content: str
    metadata: Optional[dict] = None


class QueryRequest(BaseModel):
    tenant_id: str
    workspace_id: str
    query: str
    top_k: int = 5


class QueryResult(BaseModel):
    document_id: str
    chunk_index: int
    score: float
    chunk_text: str
    original_filename: Optional[str] = None
    source_type: Optional[str] = None


# --------------------------------------------------------
# Helpers
# --------------------------------------------------------

def get_workspace_config(db: Session, workspace_id: str):
    """
    Load persona/config for a workspace.
    Falls back to sane defaults if not set.
    """
    row = db.execute(
        text("""
            SELECT system_prompt, default_model, temperature
            FROM workspaces
            WHERE id = :wid
        """),
        {"wid": UUID(workspace_id)},
    ).fetchone()

    if not row:
        return {
            "system_prompt": None,
            "default_model": "gpt-4.1-mini",
            "temperature": 0.2,
        }

    return {
        "system_prompt": row.system_prompt,
        "default_model": row.default_model or "gpt-4.1-mini",
        "temperature": float(row.temperature) if row.temperature is not None else 0.2,
    }


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """Simple sliding-window chunking of long text."""
    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


SOURCE_TYPES = [
    "deck",
    "email",
    "transcript",
    "notes",
    "document",
    "interview",
    "call",
]


def classify_source_type(filename: str, content: str) -> str:
    """
    Lightweight classifier that combines simple heuristics with an optional
    LLM call. Returns one of SOURCE_TYPES.
    """
    lower_name = (filename or "").lower()

    # --- Heuristics first (cheap / fast) ---

    # Deck
    if any(ext in lower_name for ext in [".ppt", ".pptx", " deck", " slides"]):
        return "deck"

    # Transcript / call
    if "transcript" in lower_name or "otter" in lower_name:
        return "transcript"
    if any(word in lower_name for word in ["call", "zoom", "meeting"]):
        return "call"

    # Notes
    if "notes" in lower_name:
        return "notes"

    # Interview
    if "interview" in lower_name:
        return "interview"

    # Email
    if any(ext in lower_name for ext in [".eml", ".msg"]) or "email" in lower_name:
        return "email"

    # --- Optional: short LLM classification if still ambiguous ---

    snippet = (content or "")[:2000]

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            max_tokens=16,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a classifier. "
                        "Given a filename and text, return exactly ONE of: "
                        "deck, email, transcript, notes, document, interview, call."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Filename: {filename}\n\nText snippet:\n{snippet}",
                },
            ],
        )
        label = (completion.choices[0].message.content or "").strip().lower()
        if label in SOURCE_TYPES:
            return label
        for t in SOURCE_TYPES:
            if t in label:
                return t
    except Exception as e:
        print("WARN: classify_source_type fallback due to error:", repr(e))

    # Fallback
    return "document"


def embed_text(text: str) -> list[float]:
    """
    Create an embedding for text using the global EMBEDDING_MODEL constant.
    Uses EMBEDDING_DIM (1024) to match the pgvector column.
    Handles OpenAI rate limit / quota errors cleanly.
    """
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIM,
        )
        return response.data[0].embedding

    except RateLimitError as e:
        print("OpenAI RateLimitError in embed_text:", repr(e))
        raise HTTPException(
            status_code=503,
            detail="Embedding service unavailable: OpenAI rate limit / quota exceeded.",
        )

    except Exception as e:
        print("Unexpected error in embed_text:", repr(e))
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while creating embedding.",
        )


def ingest_document_text(
    db: Session,
    tenant_id: str,
    workspace_id: str,
    original_filename: str,
    content: str,
    metadata: Optional[dict] = None,
):
    """
    Core ingestion logic:
    - classify source_type
    - insert into documents
    - chunk content
    - embed each chunk
    - insert into embeddings
    """
    metadata = metadata or {}

    # 0) Decide source_type (allow metadata override if present)
    source_type = metadata.get("source_type")
    if not source_type:
        source_type = classify_source_type(original_filename, content)

    # 1) Insert document row
    result = db.execute(
        text("""
            INSERT INTO documents (
                workspace_id,
                original_filename,
                content_text,
                metadata,
                source_type,
                primary_embedding_model
            )
            VALUES (
                :workspace_id,
                :original_filename,
                :content_text,
                :metadata,
                :source_type,
                :primary_embedding_model
            )
            RETURNING id;
        """),
        {
            "workspace_id": UUID(workspace_id),
            "original_filename": original_filename,
            "content_text": content,
            "metadata": json.dumps(metadata) if metadata else None,
            "source_type": source_type,
            "primary_embedding_model": EMBEDDING_MODEL,
        },
    )
    doc_row = result.fetchone()
    document_id = doc_row.id

    # 2) Chunk
    chunks = chunk_text(content)

    # 3) For each chunk: embed + insert into embeddings
    for idx, chunk in enumerate(chunks):
        embedding = embed_text(chunk)

        db.execute(
            text("""
                INSERT INTO embeddings (
                    tenant_id,
                    workspace_id,
                    document_id,
                    chunk_index,
                    chunk_text,
                    embedding,
                    embedding_model
                )
                VALUES (
                    :tenant_id,
                    :workspace_id,
                    :document_id,
                    :chunk_index,
                    :chunk_text,
                    :embedding,
                    :embedding_model
                );
            """),
            {
                "tenant_id": UUID(tenant_id),
                "workspace_id": UUID(workspace_id),
                "document_id": document_id,
                "chunk_index": idx,
                "chunk_text": chunk,
                "embedding": embedding,
                "embedding_model": EMBEDDING_MODEL,
            },
        )

    db.commit()

    return {
        "document_id": str(document_id),
        "num_chunks": len(chunks),
        "source_type": source_type,
    }


def extract_text_from_upload(file: UploadFile) -> str:
    """Extract plain text from an uploaded file based on its extension."""
    filename = file.filename or "unnamed"
    lower_name = filename.lower()

    content_bytes = file.file.read()

    if lower_name.endswith(".txt") or lower_name.endswith(".md"):
        return content_bytes.decode("utf-8", errors="ignore")

    if lower_name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content_bytes))
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        return "\n".join(pages_text)

    if lower_name.endswith(".docx"):
        doc = DocxDocument(io.BytesIO(content_bytes))
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type for '{filename}'. Supported: .txt, .md, .pdf, .docx",
    )


# --------------------------------------------------------
# Routes
# --------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/admin/reset-workspace")
def reset_workspace(
    payload: ResetWorkspaceRequest,
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    tenant_id = str(payload.tenant_id)
    workspace_id = str(payload.workspace_id)

    workspace_row = db.execute(
        text("""
            SELECT id, tenant_id
            FROM workspaces
            WHERE id = :workspace_id
            LIMIT 1
        """),
        {"workspace_id": workspace_id},
    ).fetchone()

    if workspace_row is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    if str(workspace_row.tenant_id) != tenant_id:
        print(
            "WARNING: reset_workspace called with tenant_id",
            tenant_id,
            "but workspace actually belongs to",
            str(workspace_row.tenant_id),
        )

    try:
        db.execute(
            text("""
                DELETE FROM embeddings
                WHERE workspace_id = :workspace_id
            """),
            {"workspace_id": workspace_id},
        )

        db.execute(
            text("""
                DELETE FROM documents
                WHERE workspace_id = :workspace_id
            """),
            {"workspace_id": workspace_id},
        )

        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to reset workspace: {e}")

    return {
        "status": "ok",
        "message": "Workspace reset. Documents and embeddings deleted.",
        "tenant_id": tenant_id,
        "workspace_id": workspace_id,
    }


@app.get("/tenants")
def list_tenants(db: Session = Depends(get_db)):
    result = db.execute(
        text("SELECT id, name, type, created_at FROM tenants ORDER BY created_at ASC;")
    )
    rows = result.fetchall()

    return [
        {
            "id": str(row.id),
            "name": row.name,
            "type": row.type,
            "created_at": row.created_at,
        }
        for row in rows
    ]


@app.post("/tenants")
def create_tenant(tenant: TenantCreate, db: Session = Depends(get_db)):
    stmt = text("""
        INSERT INTO tenants (name, type)
        VALUES (:name, :type)
        RETURNING id, name, type, created_at
    """)

    result = db.execute(
        stmt,
        {"name": tenant.name, "type": tenant.type},
    )

    row = result.fetchone()
    db.commit()

    return {
        "id": str(row.id),
        "name": row.name,
        "type": row.type,
        "created_at": row.created_at.isoformat(),
    }


@app.get("/workspaces")
def list_workspaces(tenant_id: str, db: Session = Depends(get_db)):
    stmt = text("""
        SELECT id, tenant_id, name, created_at
        FROM workspaces
        WHERE tenant_id = :tenant_id
        ORDER BY created_at ASC
    """)

    rows = db.execute(stmt, {"tenant_id": tenant_id}).fetchall()

    return [
        {
            "id": str(row.id),
            "tenant_id": str(row.tenant_id),
            "name": row.name,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]


@app.post("/workspaces")
def create_workspace(payload: WorkspaceCreate, db: Session = Depends(get_db)):
    stmt = text("""
        INSERT INTO workspaces (tenant_id, name)
        VALUES (:tenant_id, :name)
        RETURNING id, tenant_id, name, created_at
    """)

    result = db.execute(
        stmt,
        {
            "tenant_id": str(payload.tenant_id),
            "name": payload.name,
        },
    )

    row = result.fetchone()
    db.commit()

    return {
        "id": str(row.id),
        "tenant_id": str(row.tenant_id),
        "name": row.name,
        "created_at": row.created_at.isoformat(),
    }


@app.get("/documents")
def list_documents(tenant_id: str, workspace_id: str, db: Session = Depends(get_db)):
    """
    List documents for a given tenant + workspace, including source_type.
    """
    sql = text("""
        SELECT
            d.id,
            d.original_filename,
            d.created_at,
            d.source_type,
            COALESCE(c.chunk_count, 0) AS chunk_count
        FROM documents d
        LEFT JOIN (
            SELECT document_id, COUNT(*) AS chunk_count
            FROM embeddings
            GROUP BY document_id
        ) c ON c.document_id = d.id
        WHERE d.workspace_id = :workspace_id
        ORDER BY d.created_at DESC;
    """)

    rows = db.execute(
        sql,
        {"workspace_id": UUID(workspace_id)},
    ).fetchall()

    return [
        {
            "id": str(row.id),
            "original_filename": row.original_filename,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "chunk_count": int(row.chunk_count),
            "source_type": row.source_type,
        }
        for row in rows
    ]


@app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    tenant_id: str = Query(...),
    workspace_id: str = Query(...),
    db: Session = Depends(get_db),
):
    """
    Delete a single document and all of its embeddings for a given tenant + workspace.
    """
    check_sql = text("""
        SELECT id
        FROM documents
        WHERE id = :document_id
          AND workspace_id = :workspace_id
        LIMIT 1
    """)

    doc_row = db.execute(
        check_sql,
        {
            "document_id": UUID(document_id),
            "workspace_id": UUID(workspace_id),
        },
    ).fetchone()

    if doc_row is None:
        raise HTTPException(status_code=404, detail="Document not found in workspace")

    try:
        db.execute(
            text("""
                DELETE FROM embeddings
                WHERE document_id = :document_id
                  AND workspace_id = :workspace_id
                  AND tenant_id = :tenant_id
            """),
            {
                "document_id": UUID(document_id),
                "workspace_id": UUID(workspace_id),
                "tenant_id": UUID(tenant_id),
            },
        )

        db.execute(
            text("""
                DELETE FROM documents
                WHERE id = :document_id
                  AND workspace_id = :workspace_id
            """),
            {
                "document_id": UUID(document_id),
                "workspace_id": UUID(workspace_id),
            },
        )

        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {e}",
        )

    return


@app.get("/debug/db")
def debug_db(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT current_database(), current_user, current_schema();"))
    db_name, user, schema = result.fetchone()

    ws_result = db.execute(text("SELECT id, tenant_id, name FROM workspaces"))
    workspaces = [
        {"id": str(r.id), "tenant_id": str(r.tenant_id), "name": r.name}
        for r in ws_result.fetchall()
    ]

    return {
        "database": db_name,
        "user": user,
        "schema": schema,
        "workspaces": workspaces,
    }


@app.get("/debug/openai")
def debug_openai():
    """
    Quick health check for OpenAI embeddings.
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="ping",
        )
        dim = len(response.data[0].embedding)
        return {
            "status": "ok",
            "model": "text-embedding-3-small",
            "dimension": dim,
        }

    except RateLimitError:
        raise HTTPException(
            status_code=503,
            detail="OpenAI quota / billing issue: please add credits or update your plan.",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected OpenAI error: {e!r}",
        )


@app.post("/query")
def query_embeddings(payload: QueryRequest, db: Session = Depends(get_db)):
    """
    Vector search endpoint for RAG.
    """
    query_embedding = embed_text(payload.query)

    sql = text("""
        SELECT
            e.document_id,
            e.chunk_index,
            e.chunk_text,
            1 - (e.embedding <=> CAST(:query_embedding AS vector(1024))) AS score,
            d.original_filename,
            d.source_type
        FROM embeddings e
        JOIN documents d ON d.id = e.document_id
        WHERE e.tenant_id = :tenant_id
          AND e.workspace_id = :workspace_id
        ORDER BY e.embedding <=> CAST(:query_embedding AS vector(1024))
        LIMIT :top_k;
    """)

    rows = db.execute(
        sql,
        {
            "tenant_id": payload.tenant_id,
            "workspace_id": payload.workspace_id,
            "query_embedding": query_embedding,
            "top_k": payload.top_k,
        },
    ).fetchall()

    results: List[QueryResult] = []
    for row in rows:
        results.append(
            QueryResult(
                document_id=str(row.document_id),
                chunk_index=row.chunk_index,
                score=float(row.score),
                chunk_text=row.chunk_text,
                original_filename=row.original_filename,
                source_type=row.source_type,
            )
        )

    return {
        "query": payload.query,
        "results": [r.dict() for r in results],
    }


@app.post("/documents")
def ingest_document(payload: DocumentIn, db: Session = Depends(get_db)):
    """
    JSON-based ingestion: caller passes raw text.
    """
    print("DEBUG workspace_id repr:", repr(payload.workspace_id))
    print("DEBUG workspace_id type:", type(payload.workspace_id))

    ws_sql = text("""
        SELECT id, tenant_id, name
        FROM workspaces
        WHERE id = :wid
    """)

    ws_row = db.execute(ws_sql, {"wid": str(payload.workspace_id)}).fetchone()
    print("DEBUG workspace row from DB:", ws_row)

    if ws_row is None:
        raise HTTPException(
            status_code=400,
            detail=f"Workspace {payload.workspace_id} not found (pre-insert check)",
        )

    return ingest_document_text(
        db=db,
        tenant_id=payload.tenant_id,
        workspace_id=payload.workspace_id,
        original_filename=payload.original_filename,
        content=payload.content,
        metadata=payload.metadata,
    )


@app.post("/upload-file")
def upload_file(
    tenant_id: str = Form(...),
    workspace_id: str = Form(...),
    metadata: Optional[str] = Form(None),
    file: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """
    File-based ingestion:
    - Client uploads one or more files
    - We extract text for each
    - Then call the same ingestion pipeline
    """
    # ---- Parse metadata JSON (unchanged) ----
    metadata_dict = None
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="metadata must be valid JSON")

    if not file or len(file) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # IMPORTANT: define results once, before the loop
    results: list = []

    # ---- Handle each uploaded file ----
    for f in file:
        filename_lower = (f.filename or "").lower()
        text_content: Optional[str] = None

        if filename_lower.endswith(".pdf"):
            # Read bytes for OCR
            file_bytes = f.file.read()
            # Reset pointer so other functions can still read if needed
            f.file.seek(0)

            # Run OCR on the PDF bytes
            ocr_text = extract_text_from_pdf_bytes_with_ocr(file_bytes)
            print("DEBUG: OCR text length for", f.filename, "=", len(ocr_text or ""))

            if ocr_text and ocr_text.strip():
                # Add visible marker so you can confirm OCR in Sources
                text_content = "[USING_OCR]\n" + ocr_text
            else:
                # Fallback to your existing extraction if OCR somehow fails
                text_content = extract_text_from_upload(f)
        else:
            # Non-PDF files use existing extraction logic
            text_content = extract_text_from_upload(f)

        # Skip if we still have no text
        if not text_content or not text_content.strip():
            continue

        # Call your existing ingestion pipeline
        ingest_result = ingest_document_text(
            db=db,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            original_filename=f.filename or "uploaded_file",
            content=text_content,
            metadata=metadata_dict,
        )
        results.append(ingest_result)

    if not results:
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from any file.",
        )

    return {"results": results}


@app.post("/chat")
def chat_with_workspace(payload: ChatRequest, db: Session = Depends(get_db)):
    """
    RAG-style multi-turn chat (non-streaming).
    """
    query_embedding = embed_text(payload.query)

    sql = text("""
        SELECT
            document_id,
            chunk_index,
            chunk_text,
            1 - (embedding <=> CAST(:query_embedding AS vector(1024))) AS score
        FROM embeddings
        WHERE tenant_id = :tenant_id
          AND workspace_id = :workspace_id
        ORDER BY embedding <=> CAST(:query_embedding AS vector(1024))
        LIMIT :top_k;
    """)

    rows = db.execute(
        sql,
        {
            "tenant_id": payload.tenant_id,
            "workspace_id": payload.workspace_id,
            "query_embedding": query_embedding,
            "top_k": payload.top_k,
        },
    ).fetchall()

    contexts = [row.chunk_text for row in rows]

    if not contexts:
        return {
            "answer": "I couldn't find any relevant documents in this workspace yet.",
            "results": [],
        }

    history_lines: List[str] = []
    if payload.history:
        recent = payload.history[-10:]
        for m in recent:
            role = m.role.lower()
            if role not in ("user", "assistant"):
                continue
            tag = "User" if role == "user" else "Assistant"
            history_lines.append(f"{tag}: {m.content.strip()}")

    history_block = "\n".join(history_lines) if history_lines else "None so far."

    base_context_prompt = (
        "You are DerekGPT, a helpful assistant answering questions based on the context.\n\n"
        "Context (from documents):\n"
        + "\n\n---\n\n".join(contexts)
        + "\n\nConversation so far:\n"
        + history_block
        + "\n\nUser's new question: "
        + payload.query
        + "\n\nAnswer concisely based only on the context above. "
          "If something is not in the context, say you don't know."
    )

    cfg = get_workspace_config(db, payload.workspace_id)
    system_prompt = (
        cfg["system_prompt"]
        or "You are DerekGPT, a helpful assistant answering questions based on the provided context."
    )
    model_name = cfg["default_model"]
    temperature = cfg["temperature"]

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": base_context_prompt},
        ],
        temperature=temperature,
    )

    answer = completion.choices[0].message.content

    results = []
    for row in rows:
        results.append(
            {
                "document_id": str(row.document_id),
                "chunk_index": row.chunk_index,
                "score": float(row.score),
                "chunk_text": row.chunk_text,
            }
        )

    return {"answer": answer, "results": results}


@app.post("/chat-stream")
def chat_with_workspace_stream(
    payload: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Streaming RAG chat endpoint.
    """
    query_embedding = embed_text(payload.query)

    sql = text("""
        SELECT
            document_id,
            chunk_index,
            chunk_text,
            1 - (embedding <=> CAST(:query_embedding AS vector(1024))) AS score
        FROM embeddings
        WHERE tenant_id = :tenant_id
          AND workspace_id = :workspace_id
        ORDER BY embedding <=> CAST(:query_embedding AS vector(1024))
        LIMIT :top_k;
    """)

    rows = db.execute(
        sql,
        {
            "tenant_id": payload.tenant_id,
            "workspace_id": payload.workspace_id,
            "query_embedding": query_embedding,
            "top_k": payload.top_k,
        },
    ).fetchall()

    contexts = [row.chunk_text for row in rows]

    if not contexts:
        def empty_gen():
            yield "I couldn't find any relevant documents in this workspace yet."
        return StreamingResponse(empty_gen(), media_type="text/plain")

    history_lines: List[str] = []
    if payload.history:
        for m in payload.history[-10:]:
            role = m.role.lower()
            if role not in ("user", "assistant"):
                continue
            tag = "User" if role == "user" else "Assistant"
            history_lines.append(f"{tag}: {m.content.strip()}")

    history_block = "\n".join(history_lines) if history_lines else "None so far."

    base_context_prompt = (
        "You are DerekGPT, a helpful assistant answering questions based on the context.\n\n"
        "Context (from documents):\n"
        + "\n\n---\n\n".join(contexts)
        + "\n\nConversation so far:\n"
        + history_block
        + "\n\nUser's new question: "
        + payload.query
        + "\n\nAnswer concisely based only on the context above. "
          "If something is not in the context, say you don't know."
    )

    cfg = get_workspace_config(db, payload.workspace_id)
    system_prompt = (
        cfg["system_prompt"]
        or "You are DerekGPT, a helpful assistant answering questions based on the provided context."
    )
    model_name = cfg["default_model"]
    temperature = cfg["temperature"]

    def token_stream():
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": base_context_prompt},
                ],
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None) or ""
                if content:
                    yield content
        except Exception as e:
            print("Streaming error:", repr(e))
            yield "\n[Error while streaming answer]"

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.patch("/workspaces/{workspace_id}")
def update_workspace(
    workspace_id: str,
    payload: WorkspaceUpdate,
    db: Session = Depends(get_db),
):
    fields = []
    params = {"workspace_id": UUID(workspace_id)}

    if payload.system_prompt is not None:
        fields.append("system_prompt = :system_prompt")
        params["system_prompt"] = payload.system_prompt

    if payload.default_model is not None:
        fields.append("default_model = :default_model")
        params["default_model"] = payload.default_model

    if payload.temperature is not None:
        fields.append("temperature = :temperature")
        params["temperature"] = payload.temperature

    if not fields:
        return {"status": "ok"}

    sql = text(f"""
        UPDATE workspaces
        SET {", ".join(fields)}
        WHERE id = :workspace_id
        RETURNING id;
    """)

    row = db.execute(sql, params).fetchone()
    db.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Workspace not found")

    return {"status": "ok", "workspace_id": str(row.id)}
