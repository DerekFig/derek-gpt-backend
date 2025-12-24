# main.py

import io
import json
import os
import hashlib
from uuid import UUID
from typing import Optional, List

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

from openai import RateLimitError

from db import get_db
from security import verify_internal_key

# --------------------------------------------------------
# Import ingestion + extraction helpers from ingest_helpers
# (This breaks the circular import between main.py and ingest_routes.py)
# --------------------------------------------------------

from ingest_helpers import (
    SUPABASE_STORAGE_BUCKET,
    download_from_supabase_storage,
    extract_text_from_pdf_bytes_with_ocr,
    extract_text_from_pptx_bytes,
    extract_text_from_xlsx_bytes,
    extract_text_from_upload,
    ingest_document_text,
    client,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)

# --------------------------------------------------------
# Environment-backed keys
# --------------------------------------------------------

INTERNAL_BACKEND_KEY = os.getenv("INTERNAL_BACKEND_KEY")

# --------------------------------------------------------
# Admin API key + guard
# --------------------------------------------------------

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "changeme-reset-key")


def verify_admin(x_admin_api_key: str = Header(None)):
    """
    Simple admin guard for /admin endpoints.
    Callers must send X-Admin-Api-Key header that matches ADMIN_API_KEY.
    """
    if x_admin_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


# --------------------------------------------------------
# FastAPI app & CORS
# --------------------------------------------------------

app = FastAPI()

# IMPORTANT: main imports ingest_routes, but ingest_routes MUST NOT import main.
from ingest_routes import router as ingest_router
app.include_router(ingest_router)

from rag_routes import router as rag_router
app.include_router(rag_router)

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
# DEBUG: Ingest tables (MOVED BELOW app = FastAPI())
# --------------------------------------------------------

@app.get("/debug/ingest-tables", dependencies=[Depends(verify_internal_key)])
def debug_ingest_tables(db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN ('ingest_jobs','ingest_job_items')
        ORDER BY table_name;
    """)).fetchall()
    return {"tables": [r.table_name for r in rows]}


# --------------------------------------------------------
# Admin reset config
# --------------------------------------------------------

class ResetWorkspaceRequest(BaseModel):
    tenant_id: UUID
    workspace_id: UUID


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
    top_k: int = 12
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


class StorageIngestRequest(BaseModel):
    tenant_id: str
    workspace_id: str
    original_filename: str
    storage_path: str
    metadata: Optional[dict] = None


class QueryRequest(BaseModel):
    tenant_id: str
    workspace_id: str
    query: str
    top_k: int = 12


class QueryResult(BaseModel):
    document_id: str
    chunk_index: int
    score: float
    chunk_text: str
    original_filename: Optional[str] = None
    source_type: Optional[str] = None


# --------------------------------------------------------
# Routes
# --------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/admin/reset-workspace",
    dependencies=[Depends(verify_internal_key)],
)
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


@app.get("/tenants", dependencies=[Depends(verify_internal_key)])
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


@app.post("/tenants", dependencies=[Depends(verify_internal_key)])
def create_tenant(tenant: TenantCreate, db: Session = Depends(get_db)):
    stmt = text("""
        INSERT INTO tenants (name, type)
        VALUES (:name, :type)
        RETURNING id, name, type, created_at
    """)

    result = db.execute(stmt, {"name": tenant.name, "type": tenant.type})
    row = result.fetchone()
    db.commit()

    return {
        "id": str(row.id),
        "name": row.name,
        "type": row.type,
        "created_at": row.created_at.isoformat(),
    }


@app.get("/workspaces", dependencies=[Depends(verify_internal_key)])
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


@app.post("/workspaces", dependencies=[Depends(verify_internal_key)])
def create_workspace(payload: WorkspaceCreate, db: Session = Depends(get_db)):
    stmt = text("""
        INSERT INTO workspaces (tenant_id, name)
        VALUES (:tenant_id, :name)
        RETURNING id, tenant_id, name, created_at
    """)

    result = db.execute(
        stmt,
        {"tenant_id": str(payload.tenant_id), "name": payload.name},
    )

    row = result.fetchone()
    db.commit()

    return {
        "id": str(row.id),
        "tenant_id": str(row.tenant_id),
        "name": row.name,
        "created_at": row.created_at.isoformat(),
    }


@app.get("/documents", dependencies=[Depends(verify_internal_key)])
def list_documents(
    tenant_id: str,
    workspace_id: str,
    db: Session = Depends(get_db),
):
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

    rows = db.execute(sql, {"workspace_id": UUID(workspace_id)}).fetchall()

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


@app.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_internal_key)],
)
def delete_document(
    document_id: str,
    tenant_id: str = Query(...),
    workspace_id: str = Query(...),
    db: Session = Depends(get_db),
):
    check_sql = text("""
        SELECT id
        FROM documents
        WHERE id = :document_id
          AND workspace_id = :workspace_id
        LIMIT 1
    """)

    doc_row = db.execute(
        check_sql,
        {"document_id": UUID(document_id), "workspace_id": UUID(workspace_id)},
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
            {"document_id": UUID(document_id), "workspace_id": UUID(workspace_id)},
        )

        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")

    return


@app.get("/debug/db", dependencies=[Depends(verify_internal_key)])
def debug_db(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT current_database(), current_user, current_schema();"))
    db_name, user, schema = result.fetchone()

    ws_result = db.execute(text("SELECT id, tenant_id, name FROM workspaces"))
    workspaces = [
        {"id": str(r.id), "tenant_id": str(r.tenant_id), "name": r.name}
        for r in ws_result.fetchall()
    ]

    return {"database": db_name, "user": user, "schema": schema, "workspaces": workspaces}


@app.get("/debug/openai", dependencies=[Depends(verify_internal_key)])
def debug_openai():
    try:
        response = client.embeddings.create(model="text-embedding-3-small", input="ping")
        dim = len(response.data[0].embedding)
        return {"status": "ok", "model": "text-embedding-3-small", "dimension": dim}

    except RateLimitError:
        raise HTTPException(
            status_code=503,
            detail="OpenAI quota / billing issue: please add credits or update your plan.",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected OpenAI error: {e!r}")


# NOTE: The rest of your routes (/query, /documents POST, /upload-file, /ingest-from-storage, /chat, /chat-stream,
# /workspaces PATCH, /whoami) remain unchanged below this point in your original file.
# Keep them as-is.
#
# IMPORTANT:
# - If any of those routes referenced helpers that used to be defined in main.py (download_from_supabase_storage,
#   extract_text_from_upload, ingest_document_text, etc.), they will still work because those names are now imported
#   from ingest_helpers at the top of this file.
