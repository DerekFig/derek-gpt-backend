# ingest_job_processor.py
from __future__ import annotations

from datetime import datetime
import os

from uuid import UUID
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
    extract_text_from_docx_bytes,
    sha256_bytes,
    SUPABASE_STORAGE_BUCKET,
)

# --------------------------------------------------------
# Background processor entrypoint
# --------------------------------------------------------


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


def _mark_job_done(db: Session, job: IngestJob, status: str):
    job.status = status
    job.completed_at = _utcnow()
    _set_job_counts(db, job)
    db.commit()


def _mark_item(db: Session, item: IngestJobItem, status: str, *, error: str | None = None, result: dict | None = None):
    item.status = status
    item.updated_at = _utcnow()
    if hasattr(item, "error"):
        item.error = error
    if hasattr(item, "result"):
        item.result = result
    db.commit()


def _safe_extract_text(filename: str, file_bytes: bytes) -> str:
    lower_name = (filename or "").lower()

    # PDF: try OCR path you’ve built (vision-based OCR on rendered pages)
    if lower_name.endswith(".pdf"):
        return extract_text_from_pdf_bytes_with_ocr(file_bytes)

    # PPTX
    if lower_name.endswith(".pptx"):
        return extract_text_from_pptx_bytes(file_bytes)

    # XLSX
    if lower_name.endswith(".xlsx"):
        return extract_text_from_xlsx_bytes(file_bytes)

    # DOCX
    if lower_name.endswith(".docx"):
        return extract_text_from_docx_bytes(file_bytes)

    # Plain text fallback
    if lower_name.endswith(".txt") or lower_name.endswith(".md"):
        try:
            return file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    # If unsupported, return empty (caller will treat as failure)
    return ""


def process_ingest_job(job_id: str):
    print("[STAMP] ingest_job_processor.process_ingest_job is running", flush=True)
    """
    Run ingestion for all items in a job:
      - download from Supabase Storage
      - compute sha256(file_bytes)
      - extract text
      - call ingest_document_text (documents + chunks/embeddings)
    """
    db = SessionLocal()
    try:
        job = db.query(IngestJob).filter(IngestJob.id == job_id).first()
        if not job:
            print(f"[INGEST] job not found: {job_id}", flush=True)
            return

        if job.status not in ("queued", "running"):
            print(f"[INGEST] job status is {job.status}; not processing: {job_id}", flush=True)
            return

        job.status = "running"
        job.started_at = _utcnow()
        db.commit()

        items = (
            db.query(IngestJobItem)
            .filter(IngestJobItem.job_id == job_id)
            .order_by(IngestJobItem.created_at.asc())
            .all()
        )

        print(f"[INGEST] job={job_id} items={len(items)}", flush=True)

        for item in items:
            # Skip already completed
            if item.status == "completed":
                continue

            try:
                bucket = getattr(item, "storage_bucket", None) or SUPABASE_STORAGE_BUCKET
                path = getattr(item, "storage_path", None) or getattr(item, "storage_key", None)
                filename = getattr(item, "original_filename", None) or getattr(item, "filename", None) or "unnamed"

                if not path:
                    raise RuntimeError("Missing storage_path on ingest item")

                print(f"[INGEST] downloading: bucket={bucket} path={path} filename={filename}", flush=True)
                file_bytes = download_from_supabase_storage(bucket, path)

                # --- DEBUG (truth): compute hash from raw bytes ---
                file_hash = sha256_bytes(file_bytes)
                print(
                    f"[truth] computed sha256={file_hash} len_bytes={len(file_bytes)} filename={filename}",
                    flush=True,
                )

                content_text = _safe_extract_text(filename, file_bytes)
                if not content_text or not content_text.strip():
                    raise RuntimeError("Text extraction returned empty content")

                metadata = getattr(item, "metadata", None) or {}
                # Ensure provenance metadata has storage info if useful
                if isinstance(metadata, dict):
                    metadata.setdefault("storage_bucket", bucket)
                    metadata.setdefault("storage_path", path)

                # --- IMPORTANT: pass file_hash into ingest_document_text ---
                result = ingest_document_text(
                    db=db,
                    tenant_id=str(item.tenant_id),
                    workspace_id=str(item.workspace_id),
                    original_filename=filename,
                    content=content_text,
                    metadata=metadata,
                    file_hash=file_hash,
                )

                document_id = result.get("document_id")

                # --- SAFETY NET: ensure documents.file_hash is set even if helper path failed ---
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
                            "document_id": UUID(str(document_id)),
                            "workspace_id": UUID(str(item.workspace_id)),
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
                        {"document_id": UUID(str(document_id)), "workspace_id": UUID(str(item.workspace_id))},
                    ).fetchone()
                    db_hash = row[0] if row else None
                    print(
                        f"[DEBUG] db readback: document_id={document_id} file_hash_db={db_hash} file_hash_computed={file_hash}",
                        flush=True,
                    )

                _mark_item(db, item, "completed", result=result)

            except Exception as e:
                print(f"[INGEST] item failed: id={item.id} err={repr(e)}", flush=True)
                _mark_item(db, item, "failed", error=str(e))

        # Final job status
        _set_job_counts(db, job)
        if getattr(job, "failed_items", 0) and getattr(job, "failed_items", 0) > 0:
            _mark_job_done(db, job, "completed_with_errors")
        else:
            _mark_job_done(db, job, "completed")

        print(f"[INGEST] job complete: {job_id} status={job.status}", flush=True)

    finally:
        db.close()
