import json
import os
import chromadb
from dotenv import load_dotenv

load_dotenv()
client = chromadb.PersistentClient(path=os.getenv('CHROMA_PATH'))
collection = client.get_or_create_collection(name="hansard_speeches")

embeddings = []
metadatas = []
documents = []

with open("./cache/speech_embeddings-8b.jsonl", "r", encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data["embedding"])
        metadatas.append({"speaker": data["speaker"], "section_title": data["section_title"], "file": data["file"]})
        documents.append(data["text_to_embed"])

batch_size = 1000
for i in range(0, len(embeddings), batch_size):
    if len(embeddings) - i < batch_size:
        collection.add(
            embeddings=embeddings[i:],
            metadatas=metadatas[i:],
            ids=[str(e) for e in list(range(i, len(embeddings)))],
            documents=documents[i:]
        )
    else:
        collection.add(
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=[str(e) for e in list(range(i,i + batch_size))],
            documents=documents[i:i + batch_size]
        )
    print("batch of " + str(batch_size) + " done")