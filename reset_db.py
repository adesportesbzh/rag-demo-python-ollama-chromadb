"""
Run this to wipe the Chroma DB and reingest cleanly:
  source venv/bin/activate
  python reset_db.py
  python vector.py
"""
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectordb = Chroma(
    collection_name="my_multisource_db",
    embedding_function=embeddings,
    persist_directory="./chroma_store"
)

collection = vectordb._collection
all_ids = collection.get(include=[])["ids"]
print(f"Found {len(all_ids)} items in DB")

if all_ids:
    collection.delete(ids=all_ids)
    vectordb.persist()
    print("✓ All items deleted. DB is now empty.")
else:
    print("DB is already empty.")
