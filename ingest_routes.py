# ingest_routes.py

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import text, select
from sqlalchemy.orm import Session

from db import get_db
from security import verify_internal_key
from models_ingest import IngestJob, IngestJobItem

# ----------------------------
# Helper imports (download, OCR/extract, ingest)
# ----------------------------
# Preferred: keep these in ingest_helpers.py so ingest_routes.py stays clean.
# Fallback: try importing from main.py if helpers currently live there.
try:
    from ingest_helpers import (
        download_from_supabase_storage,
        extract_text_from_pdf_bytes_with_ocr,
        extract_text_from_pptx_bytes,
        extract_text_from_xlsx_bytes,
        ingest_document_text,
    )
except Exception:
    from main import (  # type: ignore
        download_from_supabase_storage,
        extract_text_from_pdf_bytes_with_ocr,
        extract_text_from_pptx_bytes,
        extract_text_from_xlsx_bytes,
        ingest_document_text,
    )

router = APIRouter()

# ----------------------------
# Schemas (kept local to avoid import churn)
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

def process_ingest_job(job_id: UUID) -> None:
    """
    MVP background worker:
    - Uses existing helper logic for download/extract/ingest
    - Updates per-item status and job counters
    """

    # IMPORTANT: Background tasks must NOT reuse request DB session.
    db_gen = get_db()
    db: Session = next(db_gen)

    try:
        job = db.execute(select(IngestJob).where(IngestJob.id == job_id)).scalars().first()
        if not job:
            return

        # Mark job processing
        job.status = "processing"
        job.updated_at = datetime.utcnow()
        db.commit()

        # Load items in order
        items = db.execute(
            select(IngestJobItem)
            .where(IngestJobItem.job_id == job_id)
            .order_by(IngestJobItem.created_at.asc())
        ).scalars().all()

        for item in items:
            # Skip already-finished items
            if item.status in ("completed", "failed"):
                continue

            # If attempts exhausted, mark failed and continue
            max_attempts = item.max_attempts or 3
            attempts = item.attempts or 0
            if attempts >= max_attempts:
                item.status = "failed"
                if not item.error:
                    item.error = "Max attempts reached"
                item.updated_at = datetime.utcnow()
                db.commit()

                job.failed_files = (job.failed_files or 0) + 1
                job.updated_at = datetime.utcnow()
                db.commit()
                continue

            # Mark item processing + bump attempts
            item.status = "processing"
            item.attempts = attempts + 1
            item.updated_at = datetime.utcnow()
            db.commit()

            try:
                # 1) Download file bytes from Supabase Storage
                # Your helper should accept bucket/path. If your helper signature differs,
                # adjust only this call.
                file_bytes = download_from_supabase_storage(
                    bucket=None,  # many implementations default internally; change if needed
                    path=item.storage_path,
                )

                # 2) Extract text based on file extension
                path_lower = (item.storage_path or "").lower()

                if path_lower.endswith(".pdf"):
                    text_content = extract_text_from_pdf_bytes_with_ocr(file_bytes)
                elif path_lower.endswith(".pptx"):
                    text_content = extract_text_from_pptx_bytes(file_bytes)
                elif path_lower.endswith(".xlsx"):
                    text_content = extract_text_from_xlsx_bytes(file_bytes)
                else:
                    # Default: treat as text
                    text_content = file_bytes.decode("utf-8", errors="ignore")

                # 3) Ingest text (your existing chunk/embed/dedupe logic)
                # Adjust argument names if your helper uses different names.
                ingest_document_text(
                    tenant_id=job.tenant_id,
                    workspace_id=job.workspace_id,
                    text=text_content,
                    source_path=item.storage_path,
                    original_filename=item.original_filename,
                )

                # 4) Mark item completed
                item.status = "completed"
                item.error = None
                item.updated_at = datetime.utcnow()
                db.commit()

                # 5) Update job counters
                job.processed_files = (job.processed_files or 0) + 1
                job.updated_at = datetime.utcnow()
                db.commit()

            except Exception as e:
                # Mark item failed for now; if attempts remain we re-queue it
                item.error = str(e)
                item.updated_at = datetime.utcnow()

                # Re-queue if we still have attempts left
                if item.attempts < max_attempts:
                    item.status = "queued"
                else:
                    item.status = "failed"
                    job.failed_files = (job.failed_files or 0) + 1

                db.commit()

                job.updated_at = datetime.utcnow()
                db.commit()

        # Finalize job status
        remaining = db.execute(
            select(IngestJobItem)
            .where(IngestJobItem.job_id == job_id)
            .where(IngestJobItem.status.in_(["queued", "processing"]))
        ).scalars().first()

        if remaining:
            job.status = "processing"
        else:
            # simple rule: any failures => failed, otherwise completed
            if (job.failed_files or 0) > 0:
                job.status = "failed"
            else:
                job.status = "completed"

        job.updated_at = datetime.utcnow()
        db.commit()

    finally:
        try:
            db.close()
        except Exception:
            pass
        try:
            next(db_gen)
        except Exception:
            pass

# ----------------------------
# Step 1: Create job (FAST) + kick off ingestion
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
    # Validate workspace exists and belongs to tenant (uses your existing tables)
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

    # Create job
    job = IngestJob(
        tenant_id=payload.tenant_id,
        workspace_id=payload.workspace_id,
        status="queued",
        total_files=len(payload.files),
        processed_files=0,
        failed_files=0,
    )
    db.add(job)
    db.flush()  # ensures job.id is available without commit

    # Create items
    items = []
    for f in payload.files:
        items.append(
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
        )

    db.add_all(items)
    db.commit()

    # Kick off background ingestion
    background_tasks.add_task(process_ingest_job, job.id)

    return IngestJobCreateOut(job_id=job.id, total_files=job.total_files, status=job.status)

# ----------------------------
# Step 2: Get job status (polling)
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
