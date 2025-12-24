# models_ingest.py
import uuid
from datetime import datetime

from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, DateTime, Integer, String, Text, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

Base = declarative_base()

class IngestJob(Base):
    __tablename__ = "ingest_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    workspace_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    status = Column(String, nullable=False, default="queued")
    total_files = Column(Integer, nullable=False, default=0)
    processed_files = Column(Integer, nullable=False, default=0)
    failed_files = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    items = relationship("IngestJobItem", back_populates="job", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("status IN ('queued','processing','completed','failed')", name="ingest_jobs_status_check"),
    )

class IngestJobItem(Base):
    __tablename__ = "ingest_job_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    job_id = Column(UUID(as_uuid=True), ForeignKey("ingest_jobs.id", ondelete="CASCADE"), nullable=False, index=True)

    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    workspace_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    original_filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)

    status = Column(String, nullable=False, default="queued")
    error = Column(Text, nullable=True)

    attempts = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=3)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    job = relationship("IngestJob", back_populates="items")

    __table_args__ = (
        CheckConstraint("status IN ('queued','processing','completed','failed')", name="ingest_job_items_status_check"),
    )
