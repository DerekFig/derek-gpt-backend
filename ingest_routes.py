from db import SessionLocal

def process_ingest_job(job_id: UUID) -> None:
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

                path_lower = (item.storage_path or "").lower()
                if path_lower.endswith(".pdf"):
                    content = extract_text_from_pdf_bytes_with_ocr(file_bytes)
                elif path_lower.endswith(".pptx"):
                    content = extract_text_from_pptx_bytes(file_bytes)
                elif path_lower.endswith(".xlsx"):
                    content = extract_text_from_xlsx_bytes(file_bytes)
                else:
                    content = file_bytes.decode("utf-8", errors="ignore")

                # If ingest_document_text expects a db session, keep db=db.
                # If not, remove db=db.
                ingest_document_text(
                    db=db,
                    tenant_id=str(job.tenant_id),
                    workspace_id=str(job.workspace_id),
                    original_filename=item.original_filename,
