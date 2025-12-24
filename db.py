# db.py
import os
from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load local .env if present (safe in production too)
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

# Normalize scheme (some platforms use postgres://)
DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Force psycopg3 driver for Supabase pooler compatibility
# Result: postgresql+psycopg://... ?sslmode=require
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    # pgBouncer/Supabase pooler friendly: only use universally supported libpq params
    connect_args={
        "sslmode": "require",
        # optional: harmless, widely supported; avoids surprises with timeouts
        "options": "-c statement_timeout=0",
    },
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """FastAPI dependency to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
