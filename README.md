# RAG Arena

Aiming for the easiest frameworkless implementation of different RAG approaches, including **Native RAG**, **Reranking RAG**, **Graph-RAG**, **LLM-wiki**, and **Agentic-RAG**.

> **Important Note:** The data used in this experiment is already chunked.

## Setup

Use `uv` to create a virtual environment and install dependencies (Windows):

```bash
uv venv --python 3.11 .venv
.venv\Scripts\activate # source .venv/bin/activate for linux/mac
uv pip install -r requirements.txt
```

Create a `.env` file in the root directory (refer to .env.example) and add your API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage
You must build the vector database first before running the evaluations. Execute the scripts from the root directory in the following order:
```bash
# 1. Build the Vector Database (ChromaDB) first
python src/build_vector_db.py

# 2. Run Native RAG evaluation
python src/rag.py

# 3. Run Reranking RAG evaluation
python src/rag_rerank.py
```