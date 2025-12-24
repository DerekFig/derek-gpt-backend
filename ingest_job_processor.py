# ingest_job_processor.py
from __future__ import annotations

from datetime import datetime
import os

from sqlalchemy.orm import Session

from db import SessionLocal
from models_ingest import IngestJob, IngestJobItem

# Reuse existing helpers exactly as requested.
# If these helpers are currently defined in main.py, import them from there.
# If that causes circular imports, move ONLY the helper functions to a pure module
# (e.g., ingest_helpers.py) and import from that module instead.
from main import (
    download_from_supabase_storage,
    ingest_document_text,
    extract_text_from_pdf_bytes_with_ocr,
    extract_text_from_pptx_bytes,
    extract_text_from_xlsx_bytes,
)

# Optional: bucket default if not stored per-item
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

    # Only set these if your model/table has them. If not, remove these lines.
    if hasattr(job, "total_items"):
        job.total_items = total
    if hasattr(job, "completed_items"):
        job.completed_items = completed
    if hasattr(job, "failed_items"):
        job.failed_items = failed


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
            .order_by(IngestJobItem.created_at.asc() if hasattr(IngestJobItem, "created_at") else IngestJobItem.id.asc())
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
                filename = item.filename or storage_path.split("/")[-1]

                file_bytes = download_from_supabase_storage(bucket, storage_path)

                name = filename.lower()
                if name.endswith(".pdf"):
                    text = extract_text_from_pdf_bytes_with_ocr(file_bytes)
                elif name.endswith(".pptx"):
                    text = extract_text_from_pptx_bytes(file_bytes)
                elif name.endswith(".xlsx"):
                    text = extract_text_from_xlsx_bytes(file_bytes)
                else:
                    # If you already support more formats elsewhere, call that existing helper here.
                    # This fallback is intentionally conservative.
                    text = file_bytes.decode("utf-8", errors="ignore")

                # This is the core: reuse your existing ingestion pipeline.
                # Make sure these argument names match your ingest_document_text signature.
                ingest_document_text(
                    tenant_id=item.tenant_id,
                    workspace_id=item.workspace_id,
                    filename=filename,
                    text=text,
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

        any_failed = db.query(IngestJobItem).filter(
            IngestJobItem.job_id == job.id, IngestJobItem.status == "failed"
        ).count() > 0

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
