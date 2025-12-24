# security.py
import os
from fastapi import Header, HTTPException

INTERNAL_BACKEND_KEY = os.getenv("INTERNAL_BACKEND_KEY")

def verify_internal_key(x_internal_key: str = Header(None)):
    """
    Require X-Internal-Key for protected endpoints.
    """
    if INTERNAL_BACKEND_KEY is None:
        raise HTTPException(status_code=500, detail="Internal key not configured")

    if x_internal_key != INTERNAL_BACKEND_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing X-Internal-Key")
