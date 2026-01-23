"""
Ingest Script - Main CLI entry point.

Parses PDFs, transcribes tables/charts with Gemini, and stores in Qdrant.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.IngestScript.config.settings import get_settings
from backend.IngestScript.services.pdf_parser import PDFParser, ElementType
from backend.IngestScript.services.gemini_transcriber import GeminiTranscriber
from backend.IngestScript.services.vector_store import VectorStore, DocumentMetadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


from typing import Callable, Any

async def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    transcriber: GeminiTranscriber,
    vector_store: VectorStore,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict:
    """
    Process a single PDF file.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory for extracted images.
        transcriber: Gemini transcriber service.
        vector_store: Qdrant vector store service.
        progress_callback: Optional callback for progress updates.

    Returns:
        Processing statistics.
    """
    stats = {
        "total_elements": 0,
        "text_chunks": 0,
        "tables": 0,
        "figures": 0,
        "corrections": 0,
        "stored": 0,
    }

    # Collect all text for summary generation
    all_text_content: list[str] = []

    # Parse PDF (Step 1)
    if progress_callback:
        progress_callback({"status": "parsing", "message": "Parsing PDF structure...", "progress": 5})

    parser = PDFParser(output_dir=output_dir)
    # Run synchronous parsing in a thread to avoid blocking the event loop
    elements = await asyncio.to_thread(parser.parse, pdf_path)
    stats["total_elements"] = len(elements)
    total = len(elements)

    logger.info(f"Processing {len(elements)} elements from {pdf_path.name}")

    if total == 0:
        logger.warning(f"No elements extracted from PDF! Check if docling parsed correctly.")
        if progress_callback:
            progress_callback({"status": "completed", "message": "No content extracted from PDF", "progress": 100})
        return stats

    for i, element in enumerate(elements):
        # Calculate progress (10% to 85%)
        current_progress = 10 + int((i / max(total, 1)) * 75)
        
        if progress_callback:
            progress_callback({
                "status": "processing", 
                "message": f"Processing element {i+1}/{total} (Page {element.page_number})", 
                "progress": current_progress,
                "current": i + 1,
                "total": total
            })

        if element.element_type == ElementType.TABLE:
            stats["tables"] += 1
        elif element.element_type == ElementType.FIGURE:
            stats["figures"] += 1

        # Handle Text Elements (No Transcription/Image needed)
        if element.element_type == ElementType.TEXT:
            if not element.content:
                continue
                
            logger.info(f"Processing Text Element (Page {element.page_number})")
            
            # Collect text for summary
            all_text_content.append(element.content)
            
            # Store directly in Qdrant
            metadata = DocumentMetadata(
                shadow_text=element.content,
                original_image_path=None, # Text elements have no image
                element_type=element.element_type.value,
                source_pdf=str(pdf_path.absolute()),
                page_number=element.page_number,
            )

            try:
                doc_id = vector_store.upsert_document(metadata)
                stats["stored"] += 1
                stats["text_chunks"] += 1
                logger.info(f"Stored Text Chunk in Qdrant with ID: {doc_id}")
            except Exception as e:
                logger.error(f"Failed to store text in Qdrant: {e}")
            
            continue # Done with this element

        # Skip non-text elements without images
        if element.image_path is None:
            logger.warning(f"Skipping element without image: {element}")
            continue

        # Transcribe with verification
        logger.info(f"Transcribing {element.element_type.value} on page {element.page_number}...")
        
        try:
            result = await transcriber.transcribe_with_verification(element.image_path)
        except Exception as e:
            logger.error(f"Transcription failed for {element.image_path}: {e}")
            continue

        if result.was_corrected:
            stats["corrections"] += 1
            logger.info("✓ Self-correction applied")

        # Collect transcription for summary
        all_text_content.append(result.verified_transcription)

        # Log the shadow text
        logger.info(f"--- Shadow Text ({element.element_type.value}) ---")
        logger.info(f"\n{result.verified_transcription}\n")
        logger.info("--- End Shadow Text ---")

        # Store in Qdrant
        metadata = DocumentMetadata(
            shadow_text=result.verified_transcription,
            original_image_path=str(element.image_path.absolute()),
            element_type=element.element_type.value,
            source_pdf=str(pdf_path.absolute()),
            page_number=element.page_number,
        )

        try:
            doc_id = vector_store.upsert_document(metadata)
            stats["stored"] += 1
            logger.info(f"Stored in Qdrant with ID: {doc_id}")
        except Exception as e:
            logger.error(f"Failed to store in Qdrant: {e}")

    # Generate and store global summary (Step 2)
    if all_text_content:
        if progress_callback:
            progress_callback({"status": "summarizing", "message": "Generating document summary...", "progress": 90})
        
        try:
            full_text = "\n\n".join(all_text_content)
            summary = await transcriber.generate_summary(full_text)
            
            # Store summary as a special chunk
            summary_metadata = DocumentMetadata(
                shadow_text=summary,
                original_image_path=None,
                element_type="global_summary",
                source_pdf=str(pdf_path.absolute()),
                page_number=0,  # 0 indicates document-level
                keywords="summary, overview, what is this, about, describe, explain",
            )
            
            doc_id = vector_store.upsert_document(summary_metadata)
            stats["stored"] += 1
            logger.info(f"Stored global summary in Qdrant with ID: {doc_id}")
            logger.info(f"--- Global Summary ---\n{summary}\n--- End Global Summary ---")
            
        except Exception as e:
            logger.error(f"Failed to generate/store summary: {e}")

    return stats


async def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pdf-path",
        type=Path,
        required=True,
        help="Path to the PDF file to ingest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for extracted images (default: from settings)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load settings
    try:
        settings = get_settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        logger.error("Make sure you have a .env file with GOOGLE_API_KEY set")
        sys.exit(1)

    # Resolve output directory
    output_dir = args.output_dir or settings.output_dir
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Agentic RAG Pipeline - PDF Ingestion")
    logger.info("=" * 60)
    logger.info(f"PDF Path: {args.pdf_path}")
    logger.info(f"Output Dir: {output_dir}")
    logger.info(f"Gemini Model: {settings.gemini_model}")
    logger.info(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    logger.info("=" * 60)

    # Initialize services
    transcriber = GeminiTranscriber(
        api_key=settings.google_api_key,
        model_name=settings.gemini_model,
    )

    vector_store = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_name,
    )

    # Process PDF
    try:
        stats = await process_pdf(
            pdf_path=args.pdf_path.resolve(),
            output_dir=output_dir,
            transcriber=transcriber,
            vector_store=vector_store,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        sys.exit(1)

    # Print summary
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total Elements: {stats['total_elements']}")
    logger.info(f"  Tables: {stats['tables']}")
    logger.info(f"  Figures: {stats['figures']}")
    logger.info(f"Self-Corrections: {stats['corrections']}")
    logger.info(f"Stored in Qdrant: {stats['stored']}")
    logger.info(f"Total in Collection: {vector_store.count_documents()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
