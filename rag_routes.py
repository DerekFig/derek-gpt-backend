# rag_routes.py

from __future__ import annotations

import os
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from db import get_db
from security import verify_internal_key

# Reuse your existing, working helpers (do NOT reimplement)
from ingest_helpers import (
    client,  # OpenAI() client
    embed_text,
    embedding_to_pgvector_literal,
)

router = APIRouter(tags=["rag"])


# ----------------------------
# Schemas
# ----------------------------

class QueryIn(BaseModel):
    tenant_id: UUID
    workspace_id: UUID
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


class QueryHit(BaseModel):
    document_id: UUID
    original_filename: Optional[str] = None
    chunk_index: int
    chunk_text: str
    embedding_model: Optional[str] = None
    distance: float


class QueryOut(BaseModel):
    query: str
    top_k: int
    matches: list[QueryHit]


class ChatIn(BaseModel):
    tenant_id: UUID
    workspace_id: UUID
    message: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    model: Optional[str] = None  # optional override


class ChatOut(BaseModel):
    answer: str
    sources: list[QueryHit]


# ----------------------------
# Core retrieval (pgvector)
# ----------------------------

def _retrieve(
    db: Session,
    tenant_id: UUID,
    workspace_id: UUID,
    query: str,
    top_k: int,
) -> list[QueryHit]:
    # 1) Embed the query using the same model/dim as ingestion
    q_embedding = embed_text(query)

    # 2) Convert to pgvector literal so we can safely cast in SQL
    q_vec_literal = embedding_to_pgvector_literal(q_embedding)

    # 3) Vector similarity search (lower distance = closer)
    # NOTE: embeddings table schema confirmed from Supabase screenshot
    rows = db.execute(
        text("""
            SELECT
                e.document_id,
                d.original_filename,
                e.chunk_index,
                e.chunk_text,
                e.embedding_model,
                (e.embedding <-> (:qvec)::vector) AS distance
            FROM embeddings e
            JOIN documents d
              ON d.id = e.document_id
            WHERE
                e.tenant_id = :tenant_id
                AND e.workspace_id = :workspace_id
            ORDER BY e.embedding <-> (:qvec)::vector
            LIMIT :top_k;
        """),
        {
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "qvec": q_vec_literal,
            "top_k": top_k,
        },
    ).fetchall()

    return [
        QueryHit(
            document_id=r.document_id,
            original_filename=getattr(r, "original_filename", None),
            chunk_index=r.chunk_index,
            chunk_text=r.chunk_text,
            embedding_model=getattr(r, "embedding_model", None),
            distance=float(r.distance),
        )
        for r in rows
    ]


# ----------------------------
# POST /query  (RAG retrieval)
# ----------------------------

@router.post(
    "/query",
    response_model=QueryOut,
    dependencies=[Depends(verify_internal_key)],
)
def query_endpoint(payload: QueryIn, db: Session = Depends(get_db)) -> QueryOut:
    try:
        matches = _retrieve(
            db=db,
            tenant_id=payload.tenant_id,
            workspace_id=payload.workspace_id,
            query=payload.query,
            top_k=payload.top_k,
        )
        return QueryOut(query=payload.query, top_k=payload.top_k, matches=matches)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


# ----------------------------
# POST /chat  (RAG + completion)
# ----------------------------

@router.post(
    "/chat",
    response_model=ChatOut,
    dependencies=[Depends(verify_internal_key)],
)
def chat_endpoint(payload: ChatIn, db: Session = Depends(get_db)) -> ChatOut:
    matches = _retrieve(
        db=db,
        tenant_id=payload.tenant_id,
        workspace_id=payload.workspace_id,
        query=payload.message,
        top_k=payload.top_k,
    )

    context_blocks: list[str] = []
    for m in matches:
        fname = m.original_filename or str(m.document_id)
        context_blocks.append(
            f"[Source: {fname} | doc_id={m.document_id} | chunk={m.chunk_index} | dist={m.distance:.4f}]\n"
            f"{m.chunk_text}"
        )

    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant context found."

    model = payload.model or os.getenv("CHAT_MODEL", "gpt-4o-mini")

    system_msg = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the context is insufficient, say what is missing and ask a concise follow-up question. "
        "When you use context, cite sources by referencing doc_id and chunk index."
    )

    user_msg = (
        f"USER QUESTION:\n{payload.message}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        "INSTRUCTIONS:\n"
        "- Provide a clear, direct answer.\n"
        "- Include citations like: (doc_id=..., chunk=...)\n"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        answer = resp.choices[0].message.content or ""
        return ChatOut(answer=answer, sources=matches)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


# ----------------------------
# POST /chat-stream  (streaming)
# ----------------------------

@router.post(
    "/chat-stream",
    dependencies=[Depends(verify_internal_key)],
)
def chat_stream_endpoint(payload: ChatIn, db: Session = Depends(get_db)):
    matches = _retrieve(
        db=db,
        tenant_id=payload.tenant_id,
        workspace_id=payload.workspace_id,
        query=payload.message,
        top_k=payload.top_k,
    )

    context_blocks: list[str] = []
    for m in matches:
        fname = m.original_filename or str(m.document_id)
        context_blocks.append(
            f"[Source: {fname} | doc_id={m.document_id} | chunk={m.chunk_index} | dist={m.distance:.4f}]\n"
            f"{m.chunk_text}"
        )

    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant context found."
    model = payload.model or os.getenv("CHAT_MODEL", "gpt-4o-mini")

    system_msg = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the context is insufficient, say what is missing and ask a concise follow-up question. "
        "When you use context, cite sources by referencing doc_id and chunk index."
    )

    user_msg = (
        f"USER QUESTION:\n{payload.message}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        "INSTRUCTIONS:\n"
        "- Provide a clear, direct answer.\n"
        "- Include citations like: (doc_id=..., chunk=...)\n"
    )

    def event_stream():
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                stream=True,
            )

            for part in stream:
                delta = part.choices[0].delta
                token = getattr(delta, "content", None)
                if token:
                    # plain text stream
                    yield token

        except Exception as e:
            yield f"\n\n[stream-error] {e}"

    # Frontends typically accept plain text streaming
    return StreamingResponse(event_stream(), media_type="text/plain")
