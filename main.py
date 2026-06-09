"""
main.py — Flask API server

Serves the chat UI and exposes a REST endpoint that answers questions
using the RAG pipeline (ChromaDB retrieval + Ollama LLM).

Start the server:
    python main.py

Then open http://127.0.0.1:5005 in your browser.
"""

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, vectordb
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


app = Flask(__name__, static_folder=".")
CORS(app)

model = OllamaLLM(model="llama3.2")

# The prompt explicitly forbids the LLM from using its training knowledge.
# Without this constraint, the model may answer from memory instead of
# from the retrieved documents, making RAG verification impossible.
template = """
You are a helpful assistant. Answer the question using ONLY the context provided below.
Do not use any prior knowledge or training data.
If the answer cannot be found in the context, say: "I don't have that information in my documents."

Context:
{answers}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory(".", "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(".", path)


@app.route("/answer_question", methods=["GET"])
def answer_question():
    """
    RAG endpoint.

    Query params:
        q (str): the user's question

    Returns:
        JSON { "answer": "..." }

    Steps:
        1. Retrieve the top-k most semantically similar chunks from ChromaDB
        2. Inject them into the prompt as context
        3. Send to the LLM and return the answer
    """
    question = request.args.get("q", "").strip()
    if not question:
        return jsonify({"error": "Missing query parameter 'q'"}), 400

    print(f"Question: {question}")
    context_chunks = retriever.invoke(question)
    answer = chain.invoke({"answers": context_chunks, "question": question})
    print(f"Answer: {answer}")

    return jsonify({"answer": answer})


@app.route("/debug/chroma")
def debug_chroma():
    """
    Development helper — returns all documents and metadata stored in ChromaDB.
    Remove or protect this endpoint before any public deployment.
    """
    collection = vectordb._collection
    items = collection.get(include=["metadatas", "documents"])
    return jsonify(items)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
