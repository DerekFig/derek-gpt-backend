# ingest_routes.py

from __future__ import annotations

import os
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import text, select
from sqlalchemy.orm import Session

from db import get_db, SessionLocal
from security import verify_internal_key
from models_ingest import IngestJob, IngestJobItem

# IMPORTANT: Only import helpers from ingest_helpers to avoid circular imports.
from ingest_helpers import (
    download_from_supabase_storage,
    extract_text_from_pdf_bytes_with_ocr,
    extract_text_from_pptx_bytes,
    extract_text_from_xlsx_bytes,
    extract_text_from_docx_bytes,
    ingest_document_text,
    sha256_bytes,
)

router = APIRouter()

# ----------------------------
# Schemas
# ----------------------------

class IngestFileIn(BaseModel):
    original_filename: str
    storage_path: str

class IngestJobCreateIn(BaseModel):
    tenant_id: UUID
    workspace_id: UUID
    files: list[IngestFileIn]

class IngestJobCreateOut(BaseModel):
    job_id: UUID
    total_files: int
    status: str

class IngestJobItemOut(BaseModel):
    id: UUID
    original_filename: str
    storage_path: str
    status: str
    error: str | None
    attempts: int
    max_attempts: int
    created_at: datetime
    updated_at: datetime

class IngestJobOut(BaseModel):
    id: UUID
    tenant_id: UUID
    workspace_id: UUID
    status: str
    total_files: int
    processed_files: int
    failed_files: int
    created_at: datetime
    updated_at: datetime
    items: list[IngestJobItemOut]

# ----------------------------
# Background processor
# ----------------------------

def _bucket() -> str:
    return os.getenv("SUPABASE_STORAGE_BUCKET", "documents")

def _sanitize_text_for_db(text_in: str) -> str:
    # Postgres TEXT cannot contain NUL bytes
    return (text_in or "").replace("\x00", "")

def process_ingest_job(job_id: UUID) -> None:
    # Stamp to confirm the correct processor is running
    print("[STAMP] ingest_routes.process_ingest_job is running", flush=True)

    db: Session = SessionLocal()
    try:
        job = db.execute(select(IngestJob).where(IngestJob.id == job_id)).scalars().first()
        if not job:
            return

        job.status = "processing"
        job.updated_at = datetime.utcnow()
        db.commit()

        items = db.execute(
            select(IngestJobItem)
            .where(IngestJobItem.job_id == job_id)
            .order_by(IngestJobItem.created_at.asc())
        ).scalars().all()

        bucket = _bucket()

        for item in items:
            if item.status in ("completed", "failed"):
                continue

            max_attempts = item.max_attempts or 3
            attempts = item.attempts or 0

            if attempts >= max_attempts:
                item.status = "failed"
                item.error = item.error or "Max attempts reached"
                item.updated_at = datetime.utcnow()
                db.commit()

                job.failed_files = (job.failed_files or 0) + 1
                job.updated_at = datetime.utcnow()
                db.commit()
                continue

            item.status = "processing"
            item.attempts = attempts + 1
            item.updated_at = datetime.utcnow()
            db.commit()

            try:
                file_bytes = download_from_supabase_storage(bucket=bucket, path=item.storage_path)

                # Compute hash from raw bytes (source of truth)
                file_hash = sha256_bytes(file_bytes)
                print(
                    f"[truth] computed sha256={file_hash} len_bytes={len(file_bytes)} filename={item.original_filename}",
                    flush=True,
                )

                path_lower = (item.storage_path or "").lower()

                # -------------------------------------------------
                # Option 1: Reject legacy binary Office formats
                # -------------------------------------------------
                if path_lower.endswith(".doc") and not path_lower.endswith(".docx"):
                    raise ValueError(
                        "Unsupported legacy Word format (.doc). "
                        "Please open the file in Microsoft Word (or Google Docs) and save it as .docx, "
                        "then upload again."
                    )

                if path_lower.endswith(".xls") and not path_lower.endswith(".xlsx"):
                    raise ValueError(
                        "Unsupported legacy Excel format (.xls). "
                        "Please open the file in Microsoft Excel (or Google Sheets) and save it as .xlsx, "
                        "then upload again."
                    )

                # -------------------------------------------------
                # Extract text for supported formats
                # -------------------------------------------------
                if path_lower.endswith(".pdf"):
                    content = extract_text_from_pdf_bytes_with_ocr(file_bytes)

                elif path_lower.endswith(".pptx"):
                    content = extract_text_from_pptx_bytes(file_bytes)

                elif path_lower.endswith(".xlsx"):
                    content = extract_text_from_xlsx_bytes(file_bytes)

                elif path_lower.endswith(".docx"):
                    content = extract_text_from_docx_bytes(file_bytes)

                elif path_lower.endswith(".txt") or path_lower.endswith(".md"):
                    content = file_bytes.decode("utf-8", errors="ignore")

                else:
                    raise ValueError(
                        f"Unsupported file type for storage_path '{item.storage_path}'. "
                        "Supported: .pdf, .pptx, .xlsx, .docx, .txt, .md "
                        "(Legacy .doc/.xls not supported. Convert to .docx/.xlsx.)"
                    )

                content = _sanitize_text_for_db(content)

                ingest_document_text(
                    db=db,
                    tenant_id=str(job.tenant_id),
                    workspace_id=str(job.workspace_id),
                    original_filename=item.original_filename,
                    content=content,
                    metadata={
                        "storage_path": item.storage_path,
                        "ingest_job_id": str(job.id),
                        "bucket": bucket,
                    },
                    file_hash=file_hash,
                )

                item.status = "completed"
                item.error = None
                item.updated_at = datetime.utcnow()
                db.commit()

                job.processed_files = (job.processed_files or 0) + 1
                job.updated_at = datetime.utcnow()
                db.commit()

            except Exception as e:
                item.error = str(e)
                item.updated_at = datetime.utcnow()

                # Retry logic: if still attempts left, re-queue. Otherwise mark failed.
                if item.attempts < max_attempts:
                    item.status = "queued"
                else:
                    item.status = "failed"
                    job.failed_files = (job.failed_files or 0) + 1

                db.commit()
                job.updated_at = datetime.utcnow()
                db.commit()

        remaining = db.execute(
            select(IngestJobItem)
            .where(IngestJobItem.job_id == job_id)
            .where(IngestJobItem.status.in_(["queued", "processing"]))
        ).scalars().first()

        job.status = "processing" if remaining else ("failed" if (job.failed_files or 0) > 0 else "completed")
        job.updated_at = datetime.utcnow()
        db.commit()

    finally:
        db.close()

# ----------------------------
# POST /ingest-jobs (create + kickoff)
# ----------------------------

@router.post(
    "/ingest-jobs",
    response_model=IngestJobCreateOut,
    dependencies=[Depends(verify_internal_key)],
)
def create_ingest_job(
    payload: IngestJobCreateIn,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    ws = db.execute(
        text("""
            SELECT id, tenant_id
            FROM workspaces
            WHERE id = :wid
            LIMIT 1
        """),
        {"wid": str(payload.workspace_id)},
    ).fetchone()

    if not ws:
        raise HTTPException(status_code=400, detail="Workspace not found")

    if str(ws.tenant_id) != str(payload.tenant_id):
        raise HTTPException(status_code=400, detail="Workspace does not belong to tenant")

    if not payload.files:
        raise HTTPException(status_code=400, detail="No files provided")

    job = IngestJob(
        tenant_id=payload.tenant_id,
        workspace_id=payload.workspace_id,
        status="queued",
        total_files=len(payload.files),
        processed_files=0,
        failed_files=0,
    )
    db.add(job)
    db.flush()

    items = [
        IngestJobItem(
            job_id=job.id,
            tenant_id=payload.tenant_id,
            workspace_id=payload.workspace_id,
            original_filename=f.original_filename,
            storage_path=f.storage_path,
            status="queued",
            attempts=0,
            max_attempts=3,
        )
        for f in payload.files
    ]

    db.add_all(items)
    db.commit()

    background_tasks.add_task(process_ingest_job, job.id)

    return IngestJobCreateOut(job_id=job.id, total_files=job.total_files, status=job.status)

# ----------------------------
# GET /ingest-jobs/{job_id}
# ----------------------------

@router.get(
    "/ingest-jobs/{job_id}",
    response_model=IngestJobOut,
    dependencies=[Depends(verify_internal_key)],
)
def get_ingest_job_status(
    job_id: UUID,
    db: Session = Depends(get_db),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    job = db.execute(select(IngestJob).where(IngestJob.id == job_id)).scalars().first()
    if not job:
        raise HTTPException(status_code=404, detail="Ingest job not found")

    items = db.execute(
        select(IngestJobItem)
        .where(IngestJobItem.job_id == job_id)
        .order_by(IngestJobItem.created_at.asc())
        .limit(limit)
        .offset(offset)
    ).scalars().all()

    return IngestJobOut(
        id=job.id,
        tenant_id=job.tenant_id,
        workspace_id=job.workspace_id,
        status=job.status,
        total_files=job.total_files,
        processed_files=job.processed_files,
        failed_files=job.failed_files,
        created_at=job.created_at,
        updated_at=job.updated_at,
        items=[
            IngestJobItemOut(
                id=i.id,
                original_filename=i.original_filename,
                storage_path=i.storage_path,
                status=i.status,
                error=i.error,
                attempts=i.attempts,
                max_attempts=i.max_attempts,
                created_at=i.created_at,
                updated_at=i.updated_at,
            )
            for i in items
        ],
    )
