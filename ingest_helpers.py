# ingest_helpers.py

from __future__ import annotations

from typing import Optional
from sqlalchemy.orm import Session

# IMPORTANT:
# Copy the real implementations of these functions from main.py into this file.
# This avoids ingest_routes importing main.py (which causes circular import).

def download_from_supabase_storage(bucket: str, path: str) -> bytes:
    raise NotImplementedError("Move implementation from main.py into ingest_helpers.py")

def extract_text_from_pdf_bytes_with_ocr(pdf_bytes: bytes) -> str:
    raise NotImplementedError("Move implementation from main.py into ingest_helpers.py")

def extract_text_from_pptx_bytes(pptx_bytes: bytes) -> str:
    raise NotImplementedError("Move implementation from main.py into ingest_helpers.py")

def extract_text_from_xlsx_bytes(xlsx_bytes: bytes) -> str:
    raise NotImplementedError("Move implementation from main.py into ingest_helpers.py")

def ingest_document_text(
    db: Session,
    tenant_id: str,
    workspace_id: str,
    original_filename: str,
    content: str,
    metadata: Optional[dict] = None,
    *,
    document_id: Optional[str] = None,
    file_hash: Optional[str] = None,
):
    raise NotImplementedError("Move implementation from main.py into ingest_helpers.py")
