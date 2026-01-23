# Agentic RAG Pipeline - Ingest Module

PDF ingestion pipeline using docling, Gemini 2.0 Flash, and Qdrant.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY
```

## Quick Start

### 1. Start Infrastructure (Qdrant)
Ensure Docker is running, then start Qdrant:
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
    -v qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### 2. Start Backend
Run the FastAPI server (Ingestion & RAG API):
```powershell
# In a new terminal
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload --port 8000 --host 0.0.0.0
```

### 3. Start Frontend
Run the React UI:
```powershell
# In a separate terminal
cd frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) to verify.

## Architecture

- **docling**: PDF parsing with table/figure extraction
- **Gemini 2.0 Flash**: Transcription + self-correction verification
- **Qdrant**: Vector storage with Shadow Text metadata

