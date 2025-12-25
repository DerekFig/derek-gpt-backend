# ingest_job_processor.py
from __future__ import annotations

from datetime import datetime
import os
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from db import SessionLocal
from models_ingest import IngestJob, IngestJobItem

# IMPORTANT:
# Keep using your existing helpers, but we also need sha256_bytes for file_hash.
# If you truly must import from main, keep it. Otherwise, ingest_helpers is cleaner.
from main import (
    download_from_supabase_storage,
    ingest_document_text,
    extract_text_from_pdf_bytes_with_ocr,
    extract_text_from_pptx_bytes,
    extract_text_from_xlsx_bytes,
)

# This is required to compute file_hash from bytes
from ingest_helpers import sha256_bytes

DEFAULT_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "documents")


def _utcnow():
    return datetime.utcnow()


def _set_job_counts(db: Session, job: IngestJob):
    total = db.query(IngestJobItem).filter(IngestJobItem.job_id == job.id).count()
    completed = db.query(IngestJobItem).filter(
        IngestJobItem.job_id == job.id, IngestJobItem.status == "completed"
    ).count()
    failed = db.query(IngestJobItem).filter(
        IngestJobItem.job_id == job.id, IngestJobItem.status == "failed"
    ).count()

    if hasattr(job, "total_items"):
        job.total_items = total
    if hasattr(job, "completed_items"):
        job.completed_items = completed
    if hasattr(job, "failed_items"):
        job.failed_items = failed


def _find_existing_document_id(db: Session, workspace_id: str, file_hash: str) -> str | None:
    """
    Returns an existing documents.id if a doc in this workspace already has the same file_hash.
    This enables strict dedupe by replacing instead of creating a new row.
    """
    if not file_hash:
        return None

    row = db.execute(
        text(
            """
            SELECT id
            FROM documents
            WHERE workspace_id = :workspace_id
              AND file_hash = :file_hash
            LIMIT 1;
            """
        ),
        {"workspace_id": UUID(workspace_id), "file_hash": file_hash},
    ).fetchone()

    if not row:
        return None

    # row may be tuple-like or have attribute access depending on driver
    existing_id = getattr(row, "id", None) or row[0]
    return str(existing_id)


def process_ingest_job(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.query(IngestJob).filter(IngestJob.id == job_id).one()

        # Mark job running
        job.status = "running"
        if hasattr(job, "started_at"):
            job.started_at = _utcnow()
        if hasattr(job, "updated_at"):
            job.updated_at = _utcnow()
        db.commit()

        items = (
            db.query(IngestJobItem)
            .filter(IngestJobItem.job_id == job_id)
            .order_by(
                IngestJobItem.created_at.asc()
                if hasattr(IngestJobItem, "created_at")
                else IngestJobItem.id.asc()
            )
            .all()
        )

        for item in items:
            try:
                item.status = "running"
                if hasattr(item, "started_at"):
                    item.started_at = _utcnow()
                if hasattr(item, "updated_at"):
                    item.updated_at = _utcnow()
                db.commit()

                bucket = getattr(item, "bucket", None) or DEFAULT_BUCKET
                storage_path = item.storage_path

                # Be robust to naming differences in your model
                original_filename = (
                    getattr(item, "original_filename", None)
                    or getattr(item, "filename", None)
                    or storage_path.split("/")[-1]
                )

                # 1) Download bytes
                file_bytes = download_from_supabase_storage(bucket, storage_path)

                # 2) Compute file_hash from raw bytes (this is the dedupe key)
                file_hash = sha256_bytes(file_bytes)

                # 3) Extract text
                name = (original_filename or "").lower()
                if name.endswith(".pdf"):
                    content = extract_text_from_pdf_bytes_with_ocr(file_bytes)
                elif name.endswith(".pptx"):
                    content = extract_text_from_pptx_bytes(file_bytes)
                elif name.endswith(".xlsx"):
                    content = extract_text_from_xlsx_bytes(file_bytes)
                else:
                    # Conservative fallback; you can expand later
                    content = file_bytes.decode("utf-8", errors="ignore")

                # Optional: pass through metadata if you store it on the item/job
                metadata = getattr(item, "metadata", None) or getattr(job, "metadata", None) or None

                # 4) Strict dedupe: if same hash already exists in workspace, REPLACE it
                # (no duplicate rows, no extra embeddings)
                existing_document_id = _find_existing_document_id(
                    db=db,
                    workspace_id=str(item.workspace_id),
                    file_hash=file_hash,
                )

                # 5) Ingest using the correct signature + pass file_hash
                # IMPORTANT: your ingest_helpers.py expects:
                # ingest_document_text(db, tenant_id, workspace_id, original_filename, content, ...)
                ingest_document_text(
                    db=db,
                    tenant_id=str(item.tenant_id),
                    workspace_id=str(item.workspace_id),
                    original_filename=original_filename,
                    content=content,
                    metadata=metadata,
                    document_id=existing_document_id,  # replace if dup exists
                    file_hash=file_hash,
                )

                item.status = "completed"
                if hasattr(item, "completed_at"):
                    item.completed_at = _utcnow()
                if hasattr(item, "updated_at"):
                    item.updated_at = _utcnow()
                if hasattr(item, "error"):
                    item.error = None
                db.commit()

            except Exception as e:
                item.status = "failed"
                if hasattr(item, "error"):
                    item.error = str(e)[:2000]
                if hasattr(item, "completed_at"):
                    item.completed_at = _utcnow()
                if hasattr(item, "updated_at"):
                    item.updated_at = _utcnow()
                db.commit()

        # Finalize job
        _set_job_counts(db, job)

        any_failed = (
            db.query(IngestJobItem)
            .filter(IngestJobItem.job_id == job.id, IngestJobItem.status == "failed")
            .count()
            > 0
        )

        job.status = "completed_with_errors" if any_failed else "completed"
        if hasattr(job, "completed_at"):
            job.completed_at = _utcnow()
        if hasattr(job, "updated_at"):
            job.updated_at = _utcnow()
        db.commit()

    except Exception as e:
        # If job-level failure, mark job failed
        try:
            job = db.query(IngestJob).filter(IngestJob.id == job_id).one()
            job.status = "failed"
            if hasattr(job, "error"):
                job.error = str(e)[:2000]
            if hasattr(job, "completed_at"):
                job.completed_at = _utcnow()
            if hasattr(job, "updated_at"):
                job.updated_at = _utcnow()
            db.commit()
        except Exception:
            pass
        raise
    finally:
        db.close()
