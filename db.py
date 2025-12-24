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

# Normalize scheme (Railway sometimes uses postgres://)
DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Force psycopg3 dialect for Supabase pooler compatibility
# Example final form:
# postgresql+psycopg://user:pass@host:6543/postgres?sslmode=require
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# Engine settings tuned for pgBouncer / Supabase pooler
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={
        # Ensure SSL for Supabase
        "sslmode": "require",
        # psycopg3 + pgBouncer: avoid prepared statement caching issues
        "statement_cache_size": 0,
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
