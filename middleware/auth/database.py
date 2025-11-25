from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from core.config import settings
from typing import Generator

# Use the DATABASE_URL from the centralized config
DATABASE_URL = settings.DATABASE_URL

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for model definitions
Base = declarative_base()

# Dependency function to get a database session
def get_db() -> Generator:
    """Provides a database session for use in FastAPI dependencies."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()