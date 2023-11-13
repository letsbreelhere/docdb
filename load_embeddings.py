import os
from openai import OpenAI
import tiktoken
from document import Document
from db import create_session
import json
from embedding import EmbeddingClient
encoding = tiktoken.get_encoding("cl100k_base")


def content_chunks(text, chunk_size=8192):
    tokens = encoding.encode(text)
    for i in range(0, len(tokens), chunk_size):
        yield (encoding.decode(tokens[i:i+chunk_size]), i)


def token_length(text):
    return len(encoding.encode(text))


def insert_document(session, filename, contents, index):
    json_contents = json.dumps({"text": contents, "filename": filename})
    embedding = EmbeddingClient.instance().create(json_contents)

    document = Document(
        filename=filename,
        embedding=embedding,
        chunk_index=index
    )
    session.add(document)
    session.commit()


if __name__ == "__main__":
    files = os.listdir('docs')
    client = OpenAI()
    session = create_session()

    for filename in files:
        print(f"Processing {filename}...", end="")
        query = session.query(Document).filter(Document.filename == filename)
        if query.count() > 0:
            print("⊝")
            continue

        contents = open('docs/' + filename, 'r').read()
        for chunk, ix in content_chunks(contents, chunk_size=5000):
            insert_document(session, filename, chunk, ix)
        print("✓")
