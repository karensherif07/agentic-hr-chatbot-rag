"""
database.py
SQLAlchemy connection setup. Import get_db() wherever you need a DB session.
"""

import os
from contextlib import contextmanager
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

_url = os.getenv("DB_URL")
if not _url:
    raise RuntimeError("DB_URL not set in .env")

engine = create_engine(
    _url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,   # reconnects if connection dropped
    echo=False            # set True to log all SQL (debug only)
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

@contextmanager
def get_db():
    """
    Usage:
        with get_db() as db:
            result = db.execute(text("SELECT ..."), {...})
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def test_connection():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✓ Database connection OK")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()