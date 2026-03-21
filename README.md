# Legal RAG System 🏛️

A production-ready **Retrieval Augmented Generation (RAG)** system built from scratch for legal document intelligence. Upload any PDF, ask questions, and get grounded answers with source citations — powered by vector search and LLMs.

> Built to demonstrate production ML engineering: clean architecture, Docker deployment, CI/CD, and comprehensive testing.

---

## What It Does

```
User uploads PDF → Extract Text → Chunk → Embed → Store in Qdrant
                                                          ↓
       Grounded Answer ← LLM (Groq/Llama3) ← Similar Chunks ← User Query
```

1. **Upload a PDF** → system extracts, chunks, and embeds it into a vector database
2. **Ask a question** → system finds the most relevant chunks using semantic search
3. **Get a grounded answer** → LLM answers using only content from your document

---

## Architecture

```
legal-rag-system/
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI endpoints (/upload, /user_query)
│   ├── core/
│   │   ├── config.py          # Environment config (.env loading)
│   │   └── logging_config.py  # Daily rotating logs
│   └── services/
│       ├── pdf_service.py         # PDF text extraction (PyMuPDF)
│       ├── chunking_service.py    # Token-based chunking with overlap
│       ├── embedding_service.py   # Local embeddings (sentence-transformers)
│       ├── vector_store_service.py # Qdrant vector DB operations
│       └── llm_service.py         # Groq LLM integration
├── tests/
│   └── test_chunking.py       # 24 tests for core services
├── Dockerfile                 # FastAPI container
├── compose.yaml               # Docker Compose (API + Qdrant)
├── .github/workflows/
│   └── ci.yml                 # GitHub Actions CI pipeline
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| API | FastAPI | Async, fast, auto Swagger docs |
| PDF Parsing | PyMuPDF | Fast, handles complex PDFs |
| Chunking | tiktoken (cl100k_base) | Token-accurate, matches LLM tokenization |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, local, 384-dim, no API needed |
| Vector DB | Qdrant | Production-grade, Docker-ready, cosine similarity |
| LLM | Groq (Llama 3.3 70B) | Free tier, fast inference |
| Containerization | Docker + Docker Compose | One command deployment |
| CI/CD | GitHub Actions | Auto-test on every push |
| Testing | pytest | 24 tests across all core services |
| Logging | Python logging + TimedRotatingFileHandler | Daily log rotation, 7-day retention |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker Desktop
- Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone the repository

```bash
git clone https://github.com/DevilSyfer/legal-rag-system-.git
cd legal-rag-system-
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=http://localhost:6333
```

### 3. Run locally (development)

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Start Qdrant (Docker required)
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Start FastAPI
uvicorn app.main:app --reload
```

API available at: `http://localhost:8000/docs`

### 4. Run with Docker Compose (recommended)

```bash
docker-compose up --build
```

This starts both FastAPI and Qdrant together. API available at `http://localhost:8000/docs`

For subsequent runs (no code changes):
```bash
docker-compose up
```

To stop:
```bash
docker-compose down
```

---

## API Endpoints

### POST `/upload`
Upload a PDF and process it into the vector database.

**Request:** Multipart form with PDF file

**Response:**
```json
{
  "filename": "document.pdf",
  "status": "uploaded",
  "total_chunks": 41,
  "collection_name": "document"
}
```

**Validations:**
- Only `.pdf` files accepted
- Maximum file size: 10MB
- Returns meaningful error messages for all failure cases

---

### POST `/user_query`
Ask a question about an uploaded document.

**Parameters:**
- `collection_name` — name of the uploaded document (filename without .pdf)
- `userquery` — your question
- `limit` — number of chunks to retrieve (default: 5)

**Response:**
```json
"Based on the document, the main challenges in agricultural exports include poor infrastructure, lack of technology acceptance, and insufficient awareness of quality standards..."
```

---

### GET `/`
Health check endpoint.

```json
{"status": "running"}
```

---

## How RAG Works (Under the Hood)

### Indexing Phase (Upload)
1. **Extract** — PyMuPDF reads PDF, extracts raw text with page markers
2. **Chunk** — tiktoken splits text into 500-token chunks with 50-token overlap. Overlap ensures context isn't lost at boundaries
3. **Embed** — sentence-transformers converts each chunk to 384 float values representing semantic meaning
4. **Store** — Qdrant stores vectors with cosine similarity for retrieval

### Query Phase (Ask)
1. **Embed query** — user question converted to 384-dim vector
2. **Search** — Qdrant finds top-k chunks with highest cosine similarity to query
3. **Ground** — retrieved chunks passed as context to LLM with strict prompt: "answer only from provided context"
4. **Answer** — LLM returns grounded answer, hallucination-resistant

---

## Key Engineering Decisions

**Why token-based chunking over sentence-based?**
Sentence chunking breaks on long legal sentences. Token-based with overlap preserves context at boundaries and matches exactly what LLMs see.

**Why local embeddings over OpenAI?**
No API cost, no latency dependency, runs offline. `all-MiniLM-L6-v2` is 90MB, produces 384-dim vectors, fast on CPU.

**Why Qdrant over ChromaDB?**
Production-grade, used in real companies, better performance at scale, official Docker image, cloud deployment option.

**Why streaming file save (`shutil.copyfileobj`) over `file.read()`?**
`file.read()` loads entire file into RAM. For large PDFs this spikes memory. Stream-copy keeps memory constant regardless of file size.

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/ -v -k "TestChunkingService"

# Run with output
pytest tests/ -v -s
```

**Test coverage:**
- `TestChunkingService` — 12 tests (return types, keys, token counts, overlap, edge cases)
- `TestEmbeddingService` — 8 tests (dimensions, types, semantic similarity, numpy conversion)
- `TestPdfService` — 4 tests (missing files, return types, page markers)

---

## CI/CD Pipeline

Every push to `master` automatically:
1. Spins up Ubuntu Linux on GitHub servers
2. Installs Python 3.11
3. Installs all dependencies
4. Runs all 24 pytest tests
5. Reports pass/fail

If any test fails → push is flagged immediately before it can cause production issues.

---

## Logging

Logs are written to `logs/app.log` with daily rotation:

```
2026-03-19 22:23:57 - app.main - INFO - Legal RAG System started successfully
2026-03-19 22:24:10 - app.api.routes - INFO - File Path: uploads/document.pdf
2026-03-19 22:24:15 - app.api.routes - INFO - Successfully processed document.pdf — 41 chunks stored
```

- New log file created every day at midnight
- Last 7 days retained, older logs auto-deleted
- Logs to both file and terminal simultaneously

---

## Production Considerations

- **Secrets** — all via `.env`, never in code or Docker images
- **Error handling** — every endpoint wrapped with specific error messages
- **File validation** — extension check + size limit (10MB)
- **Batch upsert** — all chunks stored in single Qdrant call, not 41 separate calls
- **Model loading** — embedding model loaded once at startup, not per request
- **Layer caching** — Dockerfile optimized so pip install only runs when requirements change

---

## What's Next (Phase 2)

- LangChain orchestration layer
- LangGraph for multi-step agent workflows
- LangSmith for tracing and evaluation
- Drift detection and model monitoring
- Jenkins CI/CD for enterprise deployment
- Claude SDK integration

---

## Project Status

| Feature | Status |
|---------|--------|
| PDF Upload & Processing | ✅ Complete |
| Vector Storage (Qdrant) | ✅ Complete |
| Semantic Search | ✅ Complete |
| LLM Answer Generation | ✅ Complete |
| Error Handling | ✅ Complete |
| Logging | ✅ Complete |
| Docker Deployment | ✅ Complete |
| CI/CD Pipeline | ✅ Complete |
| Unit Tests (24) | ✅ Complete |
| Monitoring & Drift Detection | 🔄 Phase 2 |
| Agent Layer | 🔄 Phase 2 |

---

*Built as a learning project to master production ML engineering — RAG, MLOps, Docker, CI/CD, and LLM engineering.*
