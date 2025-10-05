from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional, List, Dict, Any
import os

Base = declarative_base()

class FileAnalysis(Base):
    """Database model for storing file analysis results."""
    __tablename__ = 'file_analyses'
    
    id = Column(Integer, primary_key=True)
    file_path = Column(String, unique=True, nullable=False, index=True)
    file_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer)  # in bytes
    mtime = Column(Float)  # Last modified timestamp
    content_hash = Column(String, nullable=True)
    
    # Analysis results
    summary = Column(String)
    tags = Column(JSON)  # Stored as JSON array
    embedding = Column(JSON)  # Stored as JSON array of floats
    extra_metadata = Column(JSON)  # Additional metadata specific to file type
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    @classmethod
    def from_analysis(
        cls, 
        file_path: str, 
        analysis: 'FileAnalysis',
        file_type: str,
        file_size: Optional[int] = None,
        mtime: Optional[float] = None,
        agent_name: Optional[str] = None,
        content_hash: Optional[str] = None,
    ) -> 'FileAnalysis':
        """Create a FileAnalysis instance from analysis results."""
        extra_metadata = analysis.metadata or {}
        if agent_name:
            extra_metadata = {**extra_metadata, 'agent': agent_name}
        if content_hash:
            extra_metadata = {**extra_metadata, 'content_hash': content_hash}

        return cls(
            file_path=os.path.abspath(file_path),
            file_name=os.path.basename(file_path),
            file_type=file_type,
            file_size=file_size or os.path.getsize(file_path) if os.path.exists(file_path) else None,
            mtime=mtime or os.path.getmtime(file_path) if os.path.exists(file_path) else None,
            summary=analysis.summary,
            tags=analysis.tags,
            embedding=analysis.embedding,
            extra_metadata=extra_metadata,
            content_hash=content_hash
        )

class Database:
    """Database connection and session management."""
    
    def __init__(self, db_url: str = 'sqlite:///neuroexplorer.db'):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def init_db(self) -> None:
        """Initialize the database tables."""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def close(self) -> None:
        """Close the database connection."""
        self.engine.dispose()
        
# Global database instance
db = Database()
