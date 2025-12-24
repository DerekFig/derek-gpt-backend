# schemas_ingest.py
from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel

class IngestJobItemOut(BaseModel):
    id: UUID
    original_filename: str
    storage_path: str
    status: str
    error: Optional[str]
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
    items: List[IngestJobItemOut]
