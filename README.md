<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-18+-61DAFB?logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC382D" alt="Qdrant">
  <img src="https://img.shields.io/badge/LangGraph-Agentic-8B5CF6" alt="LangGraph">
  <img src="https://img.shields.io/badge/Gemini-2.0_Flash-4285F4?logo=google&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/Groq-Llama_3.3-F55036" alt="Groq">
</p>

<h1 align="center">🔍 Waldo - Multimodal Agentic RAG Pipeline</h1>

<p align="center">
  <strong>A production-grade Retrieval-Augmented Generation system with multimodal support for PDFs containing text, images, tables, and charts.</strong>
</p>

<p align="center">
  <em>Built with Docling • Gemini VLM • Groq LLMs • Qdrant Vector DB • LangGraph</em>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-api-reference">API</a>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Agentic Workflow](#-agentic-workflow)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Anti-Hallucination Safeguards](#-anti-hallucination-safeguards)
- [Performance & Limitations](#-performance--limitations)
- [License](#-license)

---

## 🎯 Overview

**Waldo** is a state-of-the-art **Agentic RAG (Retrieval-Augmented Generation) pipeline** designed to ingest, understand, and intelligently query complex PDF documents containing:

- 📄 **Text content** - Paragraphs, headings, lists
- 📊 **Tables** - Structured data with rows and columns
- 📈 **Charts & Graphs** - Bar charts, line graphs, pie charts
- 🖼️ **Figures & Diagrams** - Medical images, technical diagrams, illustrations

### What Makes This Different?

| Traditional RAG | Waldo Multimodal RAG |
|-----------------|---------------------|
| Text extraction only | Text, tables, AND figures |
| Static retrieval | Agentic self-correction with retries |
| Single-pass answers | Query rewriting for better results |
| No visual understanding | Gemini VLM transcription for images |
| Basic keyword search | Semantic + metadata hybrid search |
| May hallucinate | Strict anti-hallucination guardrails |

---

## ✨ Key Features

### 🔄 Intelligent Document Processing

| Feature | Description |
|---------|-------------|
| **PDF Parsing** | IBM's Docling library for high-fidelity document structure extraction |
| **Figure Extraction** | Identifies and extracts individual figures with bounding boxes |
| **Table Detection** | Preserves table structure and exports to Markdown |
| **OCR Support** | RapidOCR for text extraction from scanned documents |
| **Chunking** | LangChain's RecursiveCharacterTextSplitter (1000 chars, 200 overlap) |

### 🤖 Multimodal Understanding

| Feature | Description |
|---------|-------------|
| **Gemini VLM** | Transcribes figures, diagrams, and charts to searchable semantic text |
| **Rich Metadata** | Captures figure numbers, captions, and surrounding context |
| **Shadow Text** | Creates semantic representations for visual content |
| **Fallback System** | Uses Docling captions when Gemini quota is exhausted |

### 🧩 Agentic RAG with LangGraph

| Feature | Description |
|---------|-------------|
| **Adaptive Routing** | Decides between direct response vs. retrieval |
| **Document Grading** | LLM-based relevance scoring for each document |
| **Visual Query Detection** | Automatically includes figures when semantically appropriate |
| **Query Rewriting** | Transforms failed queries for better retrieval (max 2 retries) |
| **Anti-Hallucination** | Refuses to answer out-of-scope questions |

### 💾 Vector Storage & Retrieval

| Feature | Description |
|---------|-------------|
| **Qdrant** | High-performance vector similarity search (in-memory) |
| **Sentence Transformers** | all-MiniLM-L6-v2 embeddings (384 dimensions) |
| **TOP-K Retrieval** | Fetches 10 most similar documents |
| **Metadata Filtering** | Filter by element_type, page_number, etc. |

### 🎨 Modern Web Interface

| Feature | Description |
|---------|-------------|
| **React + Vite** | Fast, responsive chat interface |
| **Real-time Progress** | Live ingestion status with polling |
| **Inline Images** | Figures and tables displayed directly in chat |
| **TailwindCSS** | Modern, clean styling |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React + Vite)                            │
│                                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐│
│  │    Upload    │  │     Chat     │  │    Reset     │  │   Inline Images      ││
│  │  Component   │  │  Interface   │  │    Button    │  │ (Figures + Tables)   ││
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────────────────────┘│
└─────────┼─────────────────┼─────────────────┼───────────────────────────────────┘
          │                 │                 │
          │ POST /ingest    │ POST /chat      │ DELETE /reset
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND (FastAPI)                                  │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                           API Layer (main.py)                              │ │
│  │                                                                            │ │
│  │   POST /ingest    POST /chat    DELETE /reset    GET /ingestion-status    │ │
│  └────────┬──────────────┬──────────────┬─────────────────────────────────────┘ │
│           │              │              │                                       │
│           ▼              ▼              ▼                                       │
│  ┌─────────────────┐  ┌─────────────────────────────────────────────────────┐   │
│  │  IngestScript   │  │                   GraphBrain                        │   │
│  │                 │  │                  (LangGraph)                        │   │
│  │  ┌───────────┐  │  │                                                     │   │
│  │  │PDF Parser │  │  │   ┌─────────┐   ┌─────────┐   ┌─────────────────┐   │   │
│  │  │ (Docling) │  │  │   │ Router  │──▶│Retrieve │──▶│     Grade       │   │   │
│  │  └─────┬─────┘  │  │   └─────────┘   └─────────┘   │ (LLM Relevance) │   │   │
│  │        │        │  │                               └────────┬────────┘   │   │
│  │  ┌─────▼─────┐  │  │                                        │            │   │
│  │  │  Gemini   │  │  │   ┌─────────┐   ┌─────────┐   ┌────────▼────────┐   │   │
│  │  │Transcribe │  │  │   │ Rewrite │◀──│ Decide  │◀──│    Generate     │   │   │
│  │  │   (VLM)   │  │  │   │  Query  │   │         │   │  (Groq LLM)     │   │   │
│  │  └─────┬─────┘  │  │   └─────────┘   └─────────┘   └─────────────────┘   │   │
│  │        │        │  │                                                     │   │
│  │  ┌─────▼─────┐  │  └─────────────────────────────────────────────────────┘   │
│  │  │ Embed &   │  │                                                            │
│  │  │  Store    │──┼──────────────────────────────────────────┐                 │
│  │  └───────────┘  │                                          │                 │
│  └─────────────────┘                                          │                 │
│                                                               ▼                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         VectorStore (Qdrant)                               │ │
│  │                                                                            │ │
│  │   Collection: pdf_documents                                                │ │
│  │   ┌──────────────────────────────────────────────────────────────────┐    │ │
│  │   │ Fields:                                                           │    │ │
│  │   │   • shadow_text (str)         - Searchable content               │    │ │
│  │   │   • element_type (str)        - "text", "figure", "table"        │    │ │
│  │   │   • page_number (int)         - Source page                      │    │ │
│  │   │   • original_image_path (str) - Path to extracted image          │    │ │
│  │   │   • source_pdf (str)          - Original PDF filename            │    │ │
│  │   │   • heading (str)             - Figure/Table number              │    │ │
│  │   └──────────────────────────────────────────────────────────────────┘    │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
   │  Groq API   │            │ Gemini API  │            │   Qdrant    │
   │ (LLM Chat)  │            │   (VLM)     │            │ (In-Memory) │
   │             │            │             │            │             │
   │ llama-3.3   │            │ gemini-2.0  │            │ Collection: │
   │ 70b-versat  │            │   flash     │            │pdf_documents│
   └─────────────┘            └─────────────┘            └─────────────┘
```

---

## 🔄 Agentic Workflow

The system follows a **cognitive architecture** with self-correction capabilities:

```
                              ┌─────────────────┐
                              │   User Query    │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │     Router      │
                              │ RAG or Direct?  │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
           ┌────────────────┐                    ┌────────────────┐
           │    Greeting    │                    │   Retrieval    │
           │ Direct Answer  │                    │ Top-10 Chunks  │
           └────────────────┘                    └───────┬────────┘
                                                         │
                                                         ▼
                                                ┌────────────────┐
                                                │     Grade      │
                                                │  Each Document │
                                                │  (LLM-based)   │
                                                └───────┬────────┘
                                                        │
                                     ┌──────────────────┴──────────────────┐
                                     │                                     │
                            Has Relevant?                           No Relevant?
                                     │                                     │
                                     ▼                                     ▼
                            ┌────────────────┐                    ┌────────────────┐
                            │    Generate    │                    │  Rewrite Query │
                            │ Grounded Answer│                    │  (Max 2 tries) │
                            └───────┬────────┘                    └───────┬────────┘
                                    │                                     │
                                    │                      ┌──────────────┴──────────────┐
                                    │                      │                             │
                                    │             Retry Available?                 Max Retries?
                                    │                      │                             │
                                    │                      ▼                             ▼
                                    │             ┌────────────────┐            ┌────────────────┐
                                    │             │ Back to        │            │    Refuse      │
                                    │             │ Retrieval      │            │ "I don't have  │
                                    │             └────────────────┘            │  information"  │
                                    │                                           └───────┬────────┘
                                    │                                                   │
                                    └───────────────────────┬───────────────────────────┘
                                                            │
                                                            ▼
                                                   ┌────────────────┐
                                                   │   Response     │
                                                   │ + Inline Image │
                                                   └────────────────┘
```

### Workflow Steps Explained

| Step | Description |
|------|-------------|
| **1. Router** | Classifies query as greeting (skip retrieval) or RAG query (retrieve documents) |
| **2. Retrieve** | Semantic search in Qdrant, returns top-10 most similar chunks |
| **3. Grade** | LLM evaluates each document's relevance (strict for figures, lenient for text) |
| **4. Decide** | If relevant docs found → Generate. If not → Rewrite or Refuse |
| **5. Rewrite** | Transforms query for better retrieval (e.g., "Fig 1" → "Figure 1 diagram") |
| **6. Generate** | Synthesizes answer from relevant documents only, refuses if no context |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | React 18 + Vite | Modern, fast UI framework |
| **Styling** | TailwindCSS | Utility-first CSS |
| **Backend** | FastAPI | Async Python web framework |
| **PDF Processing** | IBM Docling | Document structure extraction |
| **OCR** | RapidOCR | Text recognition for images |
| **Vision LLM** | Gemini 2.0 Flash | Figure/chart transcription |
| **Chat LLM** | Groq (Llama 3.3 70B) | Fast inference, answer generation |
| **Embeddings** | Sentence Transformers | all-MiniLM-L6-v2 (384 dims) |
| **Vector DB** | Qdrant | In-memory vector storage |
| **Orchestration** | LangGraph | Agentic state machine |
| **Text Splitting** | LangChain | RecursiveCharacterTextSplitter |

---

## 📦 Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| Node.js | 18+ |
| GPU | CUDA-capable (optional, for faster inference) |

### 1. Clone the Repository

```bash
git clone https://github.com/Vasu-DevS/Waldo.git
cd Waldo
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Configure Environment Variables

```bash
# Copy example config
cp .env.example .env

# Edit .env with your API keys
```

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# Required API Keys
GROQ_API_KEY=gsk_your_groq_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here

# Optional: Qdrant (defaults to in-memory)
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Getting API Keys

| Service | How to Get Key |
|---------|----------------|
| **Groq** | Sign up at [console.groq.com](https://console.groq.com) → API Keys |
| **Gemini** | Sign up at [aistudio.google.com](https://aistudio.google.com) → Get API Key |

### Rate Limits (Free Tier)

| Service | Free Tier Limit |
|---------|-----------------|
| **Groq** | 30 requests/min, 6000 tokens/min |
| **Gemini** | 15 requests/min, 1,500 requests/day |

---

## 🚀 Usage

### Start the Backend

```bash
# From project root
.\.venv\Scripts\python -m uvicorn backend.main:app --port 8000 --host 0.0.0.0
```

### Start the Frontend

```bash
cd frontend
npm run dev
```

### Access the Application

Open **http://localhost:5173** in your browser.

### Workflow

1. **📤 Upload PDF** - Drag and drop or click to upload
2. **⏳ Wait for Ingestion** - Progress indicator shows status (~30s for text, ~2-5min with figures)
3. **💬 Ask Questions** - Chat naturally about the document
4. **🖼️ View Results** - Relevant figures/tables appear inline

---

## 📡 API Reference

### `POST /ingest`

Upload and process a PDF document.

**Request:**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.pdf" \
  -F "filename=document.pdf"
```

**Response:**
```json
{
  "status": "processing",
  "filename": "document.pdf",
  "message": "Ingestion started in background"
}
```

---

### `GET /ingestion-status/{filename}`

Check ingestion progress.

**Response:**
```json
{
  "status": "complete",
  "filename": "document.pdf",
  "stats": {
    "total_elements": 15,
    "text_chunks": 10,
    "tables": 2,
    "figures": 3,
    "stored": 16
  }
}
```

---

### `POST /chat`

Send a query and get a response.

**Request:**
```json
{
  "message": "What are the key findings in Table 1?"
}
```

**Response:**
```json
{
  "response": "### Key Findings\n\n* **Revenue increased** by 25% in Q4\n* **Customer satisfaction** rated 4.5/5\n...",
  "relevant_documents": [
    {
      "id": "abc123",
      "element_type": "table",
      "page_number": 3,
      "original_image_path": "E:/output/table_1_page_3.png",
      "shadow_text": "Table 1 | Q4 Financial Results..."
    }
  ]
}
```

---

### `DELETE /reset`

Clear the vector database for a new document.

**Response:**
```json
{
  "status": "success",
  "message": "Knowledge base reset complete"
}
```

---

### `GET /images/{filename}`

Serve extracted figure/table images.

**Example:**
```
http://localhost:8000/images/figure_1_page_1.png
```

---

## 📁 Project Structure

```
Waldo/
├── backend/
│   ├── main.py                        # FastAPI app & endpoints
│   ├── __init__.py
│   │
│   ├── IngestScript/
│   │   ├── ingest.py                  # Main ingestion orchestrator
│   │   ├── __init__.py
│   │   └── services/
│   │       ├── pdf_parser.py          # Docling PDF extraction
│   │       ├── gemini_transcriber.py  # VLM transcription + verification
│   │       └── vector_store.py        # Qdrant operations + embeddings
│   │
│   └── GraphBrain/
│       ├── graph.py                   # LangGraph RAG pipeline
│       └── __init__.py
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                    # Main React component
│   │   ├── main.jsx                   # Entry point
│   │   └── index.css                  # TailwindCSS styles
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── output/                            # Extracted images (gitignored)
├── TestingDATA/                       # Sample PDFs for testing
│
├── .env.example                       # Environment template
├── .gitignore
├── requirements.txt
├── README.md
└── SYSTEM_CARD.md                     # Detailed system documentation
```

---

## 🛡️ Anti-Hallucination Safeguards

Waldo implements multiple layers to prevent hallucination:

### 1. Zero-Context Refusal
```python
if len(relevant_documents) == 0:
    return "I don't have information about that in the uploaded document."
```

### 2. Grounded Generation Prompt
The system prompt explicitly forbids external knowledge:
```
CRITICAL RULE - NO HALLUCINATION:
* You can ONLY answer based on the document fragments provided.
* If the question asks about something NOT in the documents, refuse.
* NEVER make up information. NEVER use external knowledge.
```

### 3. Low Temperature
```python
temperature=0.3  # Reduced from 0.7 for factual responses
```

### 4. Document Grading
Each retrieved document is evaluated by an LLM before use:
```python
prompt = f"Is this document relevant to: {query}? Answer 'yes' or 'no'."
```

### 5. Figure Limiting
Only the most relevant figure is shown per response to avoid information overload.

---

## 📊 Performance & Limitations

### Performance

| Metric | Value |
|--------|-------|
| **Text Ingestion** | ~5-10 seconds per PDF |
| **Figure Transcription** | ~65 seconds per figure (rate limited) |
| **Query Response** | 1-3 seconds |
| **Embedding Speed** | ~30 docs/sec on GPU |

### Current Limitations

| Limitation | Workaround |
|------------|------------|
| **Gemini Rate Limits** | 65s delay between calls, fallback to Docling captions |
| **Single Document** | Reset required for new document |
| **In-Memory Storage** | Data lost on restart |
| **Table OCR** | Some complex tables may not extract perfectly |

### Future Improvements

- [ ] Multi-document support with source attribution
- [ ] Persistent Qdrant with Docker
- [ ] Streaming responses
- [ ] Table-to-SQL conversion
- [ ] Citation highlighting in PDF viewer

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Vasudev Siddh** ([@Vasu-DevS](https://github.com/Vasu-DevS))

Built for SOS 42 Technical Assessment • January 2026

---

<p align="center">
  <strong>⭐ Star this repo if you found it useful!</strong>
</p>
