from sqlalchemy import String, Integer, Column
from sqlalchemy.orm import mapped_column, DeclarativeBase
from pgvector.sqlalchemy import Vector
from db import create_session

class Base(DeclarativeBase):
    pass


# CREATE TABLE documents (id SERIAL PRIMARY KEY, filename TEXT, embedding VECTOR(1536), chunk_index INTEGER);
class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    embedding = mapped_column(Vector(1536))
    chunk_index = Column(Integer)

    def __repr__(self):
        return f"<Document(filename='{self.filename}', embedding='{self.embedding}', chunk_index='{self.chunk_index}')>"


class Search:
    def __init__(self):
        session = create_session()

    def execute(session, query_embedding, limit=10):
        query = session.query(Document).order_by(
            Document.embedding.cosine_distance(query_embedding)
        ).limit(limit)
        return query.all()
