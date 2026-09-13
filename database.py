import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wardrobe.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# For SQLite with FastAPI, check_same_thread=False allows multiple threads
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class GarmentItem(Base):
    __tablename__ = "inventory"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True, default="default_user")  # Ready for multi-user wardrobes
    dress_name = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    garment_type = Column(String, index=True)
    shade_name = Column(String)
    pattern_type = Column(String)
    dominant_lab = Column(Text)   # JSON string of [L, a, b]
    pattern_emb = Column(Text)    # JSON string of 512-D float list
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Initializes tables cleanly on startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI Dependency: Yields an auto-closing pooled DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()