from openai import OpenAI
from db import create_session
from document import Document

client = OpenAI()
session = create_session()


class Search:
    def __init__(self):
        self.session = create_session()

    def execute(self, query_embedding, limit=10):
        query = self.session.query(Document).order_by(
            Document.embedding.cosine_distance(query_embedding)
        ).limit(limit)
        return query.all()
