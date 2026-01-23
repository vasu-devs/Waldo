import shutil
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.IngestScript.config.settings import get_settings
from backend.IngestScript.services.gemini_transcriber import GeminiTranscriber
from backend.IngestScript.services.vector_store import VectorStore
from backend.IngestScript.ingest import process_pdf
from backend.GraphBrain.graph import create_rag_graph_from_settings

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(title="SOS 42 API", version="1.0.0")

# CORS Middleware
origins = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:5174",  # Vite fallback port
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

# --- Global Services ---
settings = get_settings()
rag_graph = create_rag_graph_from_settings()

# Initialize Ingest Services
transcriber = GeminiTranscriber(
    api_key=settings.google_api_key,
    model_name=settings.gemini_model,
)

vector_store = VectorStore(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
    collection_name=settings.qdrant_collection_name,
)

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
        
        # Ensure output dir exists
        output_dir = settings.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ingestion_status[filename]["message"] = "Parsing PDF..."
        
        def update_progress(data: dict):
            # Update status safely
            if filename in ingestion_status:
                ingestion_status[filename].update(data)
        
        stats = await process_pdf(
            pdf_path=file_path,
            output_dir=output_dir,
            transcriber=transcriber,
            vector_store=vector_store,
            progress_callback=update_progress,
        )
        logger.info(f"Ingestion complete for {filename}: {stats}")
        ingestion_status[filename] = {
            "status": "completed", 
            "message": "Ingestion complete!", 
            "stats": stats
        }
        
    except Exception as e:
        # Print to console for guaranteed visibility + log for structured output
        print(f"\n{'='*60}\nINGESTION ERROR for {filename}: {e}\n{'='*60}\n")
        logger.error(f"Ingestion failed for {filename}: {e}", exc_info=True)
        ingestion_status[filename] = {"status": "error", "message": str(e)}
    finally:
        # Cleanup temp file
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up temp file {file_path}")

# --- Endpoints ---

@app.get("/")
async def root():
    """Root endpoint for quick reachability test."""
    return {"status": "ok"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "cuda": "enabled" if vector_store else "unknown"}

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
        # Save upload to a temp file
        suffix = Path(file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix, prefix="upload_") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
            
        logger.info(f"File uploaded: {file.filename} -> {tmp_path}")
        
        # Initialize status
        ingestion_status[file.filename] = {"status": "queued", "message": "Queued for processing..."}
        
        # Run ingestion in background to not block the response
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
        
        # Invoke LangGraph Pipeline
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
