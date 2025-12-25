# ingest_job_processor.py
from __future__ import annotations

from datetime import datetime
import os

from sqlalchemy.orm import Session
from sqlalchemy import text

from db import SessionLocal
from models_ingest import IngestJob, IngestJobItem

# IMPORTANT: import from ingest_helpers (not main) so file_hash + correct insert logic is used
from ingest_helpers import (
    download_from_supabase_storage,
    ingest_document_text,
    extract_text_from_pdf_bytes_with_ocr,
    extract_text_from_pptx_bytes,
    extract_text_from_xlsx_bytes,
    sha256_bytes,
)

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


def process_ingest_job(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.query(IngestJob).filter(IngestJob.id == job_id).one()

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

                filename = (
                    getattr(item, "original_filename", None)
                    or getattr(item, "filename", None)
                    or storage_path.split("/")[-1]
                )

                file_bytes = download_from_supabase_storage(bucket, storage_path)

                # Compute hash from actual bytes
                file_hash = sha256_bytes(file_bytes)
                print(f"[truth] computed sha256={file_hash} storage_path={storage_path}", flush=True)

                print(f"[INGEST] job={job_id} file={filename} hash={file_hash} bytes={len(file_bytes)}", flush=True)
                
                # --- DEBUG (1): persist evidence on the ingest job item itself (safe, small) ---
                # If your IngestJobItem table has a metadata/json field you can store this there.
                # If it doesn't, skip this block. We keep it defensive.
                if hasattr(item, "debug"):
                    try:
                        item.debug = f"hash={file_hash[:16]} size={len(file_bytes)}"
                    except Exception:
                        pass
                db.commit()

                name = (filename or "").lower()
                if name.endswith(".pdf"):
                    content_text = extract_text_from_pdf_bytes_with_ocr(file_bytes)
                elif name.endswith(".pptx"):
                    content_text = extract_text_from_pptx_bytes(file_bytes)
                elif name.endswith(".xlsx"):
                    content_text = extract_text_from_xlsx_bytes(file_bytes)
                else:
                    content_text = file_bytes.decode("utf-8", errors="ignore")

                metadata = {
                    "bucket": bucket,
                    "storage_path": storage_path,
                }

                result = ingest_document_text(
                    db=db,
                    tenant_id=str(item.tenant_id),
                    workspace_id=str(item.workspace_id),
                    original_filename=filename,
                    content=content_text,
                    metadata=metadata,
                    file_hash=file_hash,
                )

                # --- CRITICAL SAFETY NET ---
                # If ingest_document_text deduped or inserted but somehow file_hash didn’t persist,
                # force it onto the documents row we ended up with.
                document_id = result.get("document_id")
                if document_id:
                    db.execute(
                        text("""
                            UPDATE documents
                            SET file_hash = :file_hash
                            WHERE id = :document_id
                              AND workspace_id = :workspace_id
                        """),
                        {
                            "file_hash": file_hash,
                            "document_id": document_id,
                            "workspace_id": str(item.workspace_id),
                        },
                    )
                    db.commit()

                # --- DEBUG (2): confirm what the DB currently has for that doc_id ---
                if document_id:
                    row = db.execute(
                        text("""
                            SELECT file_hash
                            FROM documents
                            WHERE id = :document_id
                              AND workspace_id = :workspace_id
                            LIMIT 1
                        """),
                        {"document_id": document_id, "workspace_id": str(item.workspace_id)},
                    ).fetchone()
                    db_hash = row[0] if row else None
                    print(f"[ingest_job_processor] doc_id={document_id} computed_hash={file_hash} db_hash={db_hash}")

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
