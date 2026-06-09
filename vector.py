"""
vector.py — Document ingestion pipeline

Reads files listed in datafiles.txt, splits them into chunks,
embeds them using Ollama, and stores them in a local ChromaDB collection.

Run this once before starting main.py:
    python vector.py

Re-running is safe: files are hashed and skipped if already ingested.
"""

from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from pypdf import PdfReader
from pathlib import Path
import csv
import hashlib


# ---------------------------------------------------------------------------
# Shared embedding model + ChromaDB instance
# ---------------------------------------------------------------------------

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vectordb = Chroma(
    collection_name="my_multisource_db",
    embedding_function=embeddings,
    persist_directory="./chroma_store"
)

# Chunk size and overlap are tunable. Larger chunks preserve more context per
# piece; more overlap reduces the chance of splitting a key sentence across chunks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)


# ---------------------------------------------------------------------------
# Ingestion functions — one per supported file type
# ---------------------------------------------------------------------------

def ingest_pdf(path, hash_value):
    """Extract text page by page and store each page's chunks."""
    reader = PdfReader(path)
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        chunks = splitter.split_text(text)
        vectordb.add_texts(
            texts=chunks,
            metadatas=[{
                "source": "pdf",
                "filename": path,
                "page": page_num,
                "file_hash": hash_value
            }] * len(chunks)
        )


def ingest_csv(path, hash_value):
    """Each CSV row becomes one document, formatted as 'column: value' pairs."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader):
            text = "\n".join([f"{k}: {v}" for k, v in row.items()])
            vectordb.add_texts(
                texts=[text],
                metadatas=[{
                    "source": "csv",
                    "filename": path,
                    "row": row_num,
                    "file_hash": hash_value
                }]
            )


def ingest_txt(path, hash_value):
    """Read the full file and chunk it. Batched to avoid memory issues on large files."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = splitter.split_text(text)
    print(f"   {len(chunks)} chunks created")

    metadatas = [{
        "source": "txt",
        "filename": path,
        "file_hash": hash_value
    }] * len(chunks)

    # Add in batches to avoid sending too many items at once
    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        vectordb.add_texts(
            texts=chunks[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def file_hash(path):
    """Return the SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def already_ingested(hash_value):
    """Return True if any chunk with this file hash exists in the vector store."""
    results = vectordb._collection.get(
        where={"file_hash": hash_value},
        include=[]
    )
    return len(results["ids"]) > 0


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------

with open("datafiles.txt", "r") as f:
    for line in f:
        filepath = line.strip()
        if not filepath:
            continue

        ext = Path(filepath).suffix.lower()
        hash_value = file_hash(filepath)

        if already_ingested(hash_value):
            print(f"Skipping {filepath} — already ingested.")
            continue

        print(f"Ingesting {filepath} ({ext})")

        if ext == ".csv":
            ingest_csv(filepath, hash_value)
        elif ext == ".pdf":
            ingest_pdf(filepath, hash_value)
        elif ext == ".txt":
            ingest_txt(filepath, hash_value)
        else:
            print(f"Unsupported file type: {ext} — skipping.")

vectordb.persist()
print("Ingestion complete.")


# ---------------------------------------------------------------------------
# Retriever — used by main.py
# ---------------------------------------------------------------------------

# k controls how many chunks are returned per query.
# Higher k = more context for the LLM but slower responses.
# 5 is a good default for most use cases.
retriever = vectordb.as_retriever(
    search_kwargs={"k": 5}
)
