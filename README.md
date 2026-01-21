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

## Usage

```powershell
python -m backend.IngestScript.ingest --pdf-path <path_to_pdf> --output-dir ./output
```

## Architecture

- **docling**: PDF parsing with table/figure extraction
- **Gemini 2.0 Flash**: Transcription + self-correction verification
- **Qdrant**: Vector storage with Shadow Text metadata

