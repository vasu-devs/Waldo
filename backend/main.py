import shutil
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

# Suppress annoying asyncio ConnectionResetError warnings
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


app = FastAPI(title="SOS 42 API", version="1.0.0")

# Mount static files for serving extracted images
output_dir = Path(__file__).parent.parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(output_dir)), name="images")
logger.info(f"Mounted static images from: {output_dir}")


# CORS Middleware
origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Lazy-loaded services (initialized on first use) ---
_settings = None
_rag_graph = None
_transcriber = None
_vector_store = None

def get_settings():
    """Lazy-load settings."""
    global _settings
    if _settings is None:
        from backend.IngestScript.config.settings import get_settings as _get_settings
        _settings = _get_settings()
    return _settings

def get_rag_graph():
    """Lazy-load the RAG graph on first use, using SHARED VectorStore."""
    global _rag_graph
    if _rag_graph is None:
        logger.info("Initializing RAG graph (first request)...")
        from backend.GraphBrain.graph import build_graph_with_vector_store
        settings = get_settings()
        
        # CRITICAL: Use the SAME VectorStore instance as ingestion
        shared_vector_store = get_vector_store()
        
        _rag_graph = build_graph_with_vector_store(
            groq_api_key=settings.groq_api_key,
            vector_store=shared_vector_store,
            model_name=settings.groq_model,
        )
        logger.info("RAG graph initialized successfully with SHARED VectorStore")
    return _rag_graph


def get_transcriber():
    """Lazy-load the Gemini transcriber on first use."""
    global _transcriber
    if _transcriber is None:
        logger.info("Initializing Gemini transcriber...")
        from backend.IngestScript.services.gemini_transcriber import GeminiTranscriber
        settings = get_settings()
        _transcriber = GeminiTranscriber(
            api_key=settings.google_api_key,
            model_name=settings.gemini_model,
        )
        logger.info("Gemini transcriber initialized")
    return _transcriber

def get_vector_store():
    """Lazy-load the VectorStore on first use."""
    global _vector_store
    if _vector_store is None:
        logger.info("Initializing VectorStore (loading embedding model)...")
        from backend.IngestScript.services.vector_store import VectorStore
        settings = get_settings()
        
        # FORCE in-memory storage to avoid [WinError 10061] connection refused
        # This ensures the backend works self-contained without external Qdrant
        logger.info("Initializing VectorStore...")
        
        _vector_store = VectorStore(
            host=None, # Let VectorStore logic handle path="./qdrant_data"
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name,
        )
        logger.info("VectorStore initialized")
    return _vector_store

# --- Models ---

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    documents: list[dict]
    rewritten_query: str | None = None

# --- Ingestion tracking ---
ingestion_status = {}

# --- Ingestion Logic ---

async def handle_ingestion(file_path: Path, filename: str):
    """Background task to process the PDF."""
    try:
        logger.info(f"Starting background ingestion for {filename}")
        ingestion_status[filename] = {"status": "processing", "message": "Starting ingestion..."}
        
        # Lazy import
        from backend.IngestScript.ingest import process_pdf
        
        settings = get_settings()
        output_dir = settings.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ingestion_status[filename]["message"] = "Parsing PDF..."
        
        def update_progress(data: dict):
            if filename in ingestion_status:
                ingestion_status[filename].update(data)
        
        stats = await process_pdf(
            pdf_path=file_path,
            output_dir=output_dir,
            transcriber=get_transcriber(),
            vector_store=get_vector_store(),
            progress_callback=update_progress,
        )
        logger.info(f"Ingestion complete for {filename}: {stats}")
        ingestion_status[filename] = {
            "status": "completed", 
            "message": "Ingestion complete!", 
            "stats": stats
        }
        
    except Exception as e:
        print(f"\n{'='*60}\nINGESTION ERROR for {filename}: {e}\n{'='*60}\n")
        logger.error(f"Ingestion failed for {filename}: {e}", exc_info=True)
        ingestion_status[filename] = {"status": "error", "message": str(e)}
    finally:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up temp file {file_path}")

# --- Endpoints ---

@app.get("/")
async def root():
    """Root endpoint for quick reachability test."""
    return {"status": "ok", "message": "SOS 42 API is running"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Server is running"}


@app.delete("/reset")
async def reset_system():
    """Reset the entire knowledge base for a fresh demo."""
    try:
        logger.info("Resetting knowledge base...")
        settings = get_settings()
        vs = get_vector_store()
        
        vs.client.delete_collection(
            collection_name=settings.qdrant_collection_name
        )
        logger.info(f"Deleted collection: {settings.qdrant_collection_name}")
        
        vs._ensure_collection()
        logger.info(f"Recreated collection: {settings.qdrant_collection_name}")
        
        ingestion_status.clear()
        
        return {"status": "success", "message": "Knowledge base cleared. Ready for new document."}
        
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingestion-status/{filename}")
async def get_ingestion_status(filename: str):
    return ingestion_status.get(filename, {"status": "unknown", "message": "File not found"})

@app.post("/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        suffix = Path(file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix, prefix="upload_") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
            
        logger.info(f"File uploaded: {file.filename} -> {tmp_path}")
        
        ingestion_status[file.filename] = {"status": "queued", "message": "Queued for processing..."}
        
        background_tasks.add_task(handle_ingestion, tmp_path, file.filename)
        
        return {
            "message": f"Ingestion started for {file.filename}",
            "status": "processing",
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Chat Request: {request.query}")
        
        rag_graph = get_rag_graph()
        
        initial_state = {
            "query": request.query,
            "documents": [],
            "relevant_documents": [],
            "generation": None,
            "retry_count": 0,
            "rewritten_query": None,
        }
        
        result = await rag_graph.ainvoke(initial_state)
        
        return {
            "response": result.get("generation") or "I couldn't generate a response.",
            "documents": result.get("relevant_documents", []),
            "rewritten_query": result.get("rewritten_query")
        }
        
    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
