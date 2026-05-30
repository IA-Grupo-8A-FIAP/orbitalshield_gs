# db/connection.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from db.models import Base

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./orbitalshield.db"
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db():
    """Cria todas as tabelas se não existirem."""
    Base.metadata.create_all(bind=engine)
    print("Banco inicializado com sucesso.")


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()