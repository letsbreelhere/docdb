from openai import OpenAI


class EmbeddingClient:
    @classmethod
    def instance(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.client = OpenAI()

    def create(self, text):
        response = self.client.embeddings.create(
            input=text,
            model='text-embedding-ada-002'
        )
        return response.data[0].embedding
