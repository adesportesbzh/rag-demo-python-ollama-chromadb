# RAG Demo — Python, Ollama & ChromaDB

A local Retrieval-Augmented Generation (RAG) application that lets you chat with your own documents — PDFs, CSVs, and text files — entirely on your machine with no cloud API required.

Built with Python, [LangChain](https://www.langchain.com/), [Ollama](https://ollama.com/), and [ChromaDB](https://www.trychroma.com/).

---

## How it works

```
Your documents (PDF / CSV / TXT)
        │
        ▼
  [ vector.py ]  ←  splits text into chunks, embeds them, stores in ChromaDB
        │
        ▼
  [ ChromaDB ]  ←  local vector store (persisted on disk)
        │
        ▼
  [ main.py ]   ←  Flask API: retrieves relevant chunks, sends to LLM
        │
        ▼
  [ Ollama / llama3.2 ]  ←  local LLM, answers using only the retrieved context
        │
        ▼
  [ index.html ]  ←  simple chat UI served by Flask
```

**Why RAG?** LLMs are trained on general data. RAG lets you ground answers in *your* documents, making responses accurate and traceable to a source — without fine-tuning a model.

---

## Prerequisites

### 1. Install Ollama

Ollama runs LLMs locally. Download it from [ollama.com](https://ollama.com/download) and follow the installer for your OS.

Then pull the two models this project uses:

```bash
# LLM — answers questions
ollama pull llama3.2

# Embedding model — converts text to vectors for semantic search
ollama pull mxbai-embed-large
```

Verify Ollama is running:
```bash
ollama list
```

### 2. Python 3.9+

Check your version:
```bash
python3 --version
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/adesportesbzh/rag-demo-python-ollama-chromadb.git
cd rag-demo-python-ollama-chromadb
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Add your documents

List the files you want to ingest in `datafiles.txt`, one path per line:

```
data/restaurant_reviews.csv
data/my_report.pdf
data/notes.txt
```

Supported formats: `.csv`, `.pdf`, `.txt`

### Step 2 — Ingest documents into ChromaDB

```bash
python vector.py
```

This embeds your documents and stores them in `chroma_store/`. Files are hashed so re-running is safe — already-ingested files are skipped automatically.

To start fresh (wipe and reingest):
```bash
python reset_db.py
python vector.py
```

### Step 3 — Start the API server

```bash
python main.py
```

The server starts on `http://127.0.0.1:5005`.

### Step 4 — Open the chat UI

Navigate to [http://127.0.0.1:5005](http://127.0.0.1:5005) in your browser and start asking questions about your documents.

---

## Code overview

### `vector.py` — Document ingestion pipeline

Handles everything related to loading documents into the vector store:

- Reads the file list from `datafiles.txt`
- Splits each document into overlapping chunks (800 chars, 150-char overlap) using LangChain's `RecursiveCharacterTextSplitter`
- Embeds chunks using Ollama's `mxbai-embed-large` model
- Stores embeddings in a local ChromaDB collection (`chroma_store/`)
- Uses SHA-256 file hashing to skip files that are already ingested

Supports three ingestion strategies depending on file type:
- **CSV** — each row becomes one document, formatted as `key: value` pairs
- **PDF** — each page is split into chunks independently
- **TXT** — the full file is chunked as a single document, batched to avoid memory issues

### `main.py` — Flask API server

Exposes a REST API and serves the chat UI:

- `GET /` — serves the chat interface (`index.html`)
- `GET /answer_question?q=your+question` — the main RAG endpoint:
  1. Retrieves the top matching chunks from ChromaDB
  2. Injects them into a prompt that instructs the LLM to answer *only* from the provided context
  3. Sends the prompt to `llama3.2` via Ollama
  4. Returns the answer as JSON
- `GET /debug/chroma` — returns all documents and metadata in the vector store (useful for development)

---

## Web UI

The frontend is a minimal single-page chat interface (`index.html` + `index.js` + `index.css`) with no framework dependencies. It sends questions to the Flask API and displays responses. A "Thinking…" indicator is shown while the LLM processes the request.

Since we ingested a CSV file about a pizza restaurant reviews, we can ask a few questions

<img width="1055" height="705" alt="web-ui" src="https://github.com/user-attachments/assets/d284dde7-ed91-4107-8cf7-b454a55ce552" />

You can see that our server receives the query and returns a result coming from our CSV file.

<img width="1479" height="430" alt="terminal-logs" src="https://github.com/user-attachments/assets/353dfcc9-6272-440e-b785-be1d51cf9f27" />


---

## Project structure

```
.
├── main.py              # Flask API server
├── vector.py            # Document ingestion pipeline
├── reset_db.py          # Utility to wipe the vector store
├── requirements.txt     # Python dependencies
├── datafiles.txt        # List of files to ingest
├── data/                # Your source documents go here
├── chroma_store/        # ChromaDB persisted on disk (auto-created, gitignored)
├── index.html           # Chat UI
├── index.js             # Chat UI logic
└── index.css            # Chat UI styles
```

---

## Notes & honest limitations

- **Response time** — local LLMs are slow. Expect 10–40 seconds per answer depending on your hardware. Streaming responses would improve the perceived speed and is a natural next step.
- **Retrieval count (`k=5`)** — the retriever returns the 5 most relevant chunks per query. You can increase this in `vector.py` for broader recall at the cost of slower responses.
- **No authentication** — the `/debug/chroma` endpoint exposes all stored content. This is fine for local development but should be removed or protected before any public deployment.
- **Flask dev server** — `main.py` uses Flask's built-in development server, which is not suitable for production. For a real deployment, use [Gunicorn](https://gunicorn.org/) or similar.
- **Local only by design** — all processing happens on your machine. No data leaves, no API key required. This is a feature for privacy-sensitive use cases, but means performance depends entirely on your hardware.

---

## Dependencies

| Library | Role |
|---|---|
| `langchain` | Orchestration framework for the RAG pipeline |
| `langchain-ollama` | LangChain integration for Ollama (LLM + embeddings) |
| `langchain-chroma` | LangChain integration for ChromaDB |
| `langchain-community` | Community loaders and utilities |
| `chromadb` | Local vector database |
| `pypdf` | PDF text extraction |
| `flask` | Web API server |
| `flask-cors` | Cross-origin request headers |
