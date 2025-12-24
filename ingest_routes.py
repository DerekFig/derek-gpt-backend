# ingest_routes.py
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import text, select
from sqlalchemy.orm import Session

from db import get_db
from security import verify_internal_key
from models_ingest import IngestJob, IngestJobItem

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
# Step 1: Create job (FAST, no ingestion)
# ----------------------------

@router.post(
    "/ingest-jobs",
    response_model=IngestJobCreateOut,
    dependencies=[Depends(verify_internal_key)],
)
def create_ingest_job(payload: IngestJobCreateIn, db: Session = Depends(get_db)):
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
