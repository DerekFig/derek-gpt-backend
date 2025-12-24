# ingest_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select
from uuid import UUID

from schemas_ingest import IngestJobOut, IngestJobItemOut
from main import IngestJob, IngestJobItem
from db import get_db  # adjust import to your SessionLocal dependency
from security import require_internal_key  # adjust to your X-Internal-Key dependency

router = APIRouter()

@router.get("/ingest-jobs/{job_id}", response_model=IngestJobOut, dependencies=[Depends(require_internal_key)])
def get_ingest_job_status(
    job_id: UUID,
    db: Session = Depends(get_db),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    # 1) Fetch job
    job = db.execute(select(IngestJob).where(IngestJob.id == job_id)).scalars().first()
    if not job:
        raise HTTPException(status_code=404, detail="Ingest job not found")

    # 2) Fetch items (paginated)
    items = db.execute(
        select(IngestJobItem)
        .where(IngestJobItem.job_id == job_id)
        .order_by(IngestJobItem.created_at.asc())
        .limit(limit)
        .offset(offset)
    ).scalars().all()

    # 3) Return response
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
