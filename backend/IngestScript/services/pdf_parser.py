"""
PDF Parser service using docling.

Extracts text, tables, and figures from PDF documents.
Uses simple character-based chunking to preserve document coherence.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

logger = logging.getLogger(__name__)


class ElementType(str, Enum):
    """Types of document elements."""
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


@dataclass
class ExtractedElement:
    """Represents an extracted element from a PDF."""
    element_type: ElementType
    content: str | None
    image_path: Path | None
    page_number: int
    heading: str | None = None  # Section heading for context
    bbox: tuple[float, float, float, float] | None = None


class PDFParser:
    """Service for parsing PDF documents using docling."""

    def __init__(self, output_dir: Path) -> None:
        """
        Initialize the PDF parser.

        Args:
            output_dir: Directory to save extracted images and markdown.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure docling pipeline
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
        
        # Chunking configuration
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.min_doc_size_for_chunking = 4000  # ~2 pages: keep as single chunk

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks using simple character-based chunking.
        
        Args:
            text: The text to chunk.
            
        Returns:
            List of text chunks.
        """
        text = text.strip()
        
        # Keep as single chunk if document is small
        if len(text) < self.min_doc_size_for_chunking:
            logger.info(f"Document is small ({len(text)} chars), keeping as single chunk")
            return [text] if text else []
        
        # Character-based chunking with overlap
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Don't break mid-word if possible
            if end < len(text):
                # Look for a good break point (space, newline)
                for i in range(min(50, end - start)):
                    if text[end - i] in ' \n\t':
                        end = end - i
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start forward, accounting for overlap
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
            # Prevent infinite loop
            if start >= len(text) - self.chunk_overlap:
                break
                
        logger.info(f"Created {len(chunks)} chunks from {len(text)} chars (size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks

    def parse(self, pdf_path: Path) -> list[ExtractedElement]:
        """
        Parse a PDF document and extract all elements.

        Uses simple character-based chunking to preserve document coherence.
        Small documents (< 2000 chars) are kept as a single chunk.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of extracted elements (text chunks, tables, figures).
        """
        logger.info(f"Parsing PDF: {pdf_path}")

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Convert PDF using docling
        result = self.converter.convert(pdf_path)
        doc = result.document
        elements: list[ExtractedElement] = []
        markdown_text = ""

        # Step 1: Export and save markdown for reference
        try:
            markdown_text = doc.export_to_markdown()
            logger.info(f"Exported markdown: {len(markdown_text)} chars")
            
            # Save markdown file
            md_filename = pdf_path.stem + ".md"
            md_path = self.output_dir / md_filename
            md_path.write_text(markdown_text, encoding="utf-8")
            logger.info(f"Saved markdown to: {md_path}")
            
        except Exception as e:
            logger.error(f"Failed to export markdown: {e}")

        # Step 2: Simple character-based chunking
        try:
            chunks = self._chunk_text(markdown_text)
            
            for i, chunk_text in enumerate(chunks):
                if chunk_text.strip():
                    elements.append(ExtractedElement(
                        element_type=ElementType.TEXT,
                        content=chunk_text.strip(),
                        image_path=None,
                        page_number=1,  # Simple chunking doesn't track pages
                        heading=None,
                    ))
                    logger.debug(f"Chunk {i+1}: {len(chunk_text)} chars")
                    
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            # Fallback: keep entire text as single chunk
            if markdown_text.strip():
                elements.append(ExtractedElement(
                    element_type=ElementType.TEXT,
                    content=markdown_text.strip(),
                    image_path=None,
                    page_number=1,
                ))
        
        # Step 3: Extract tables and figures with images
        try:
            items_list = list(doc.iterate_items())
            logger.info(f"Found {len(items_list)} items via iterate_items()")
            
            for item in items_list:
                # Check if this item has an image (table or figure)
                if hasattr(item, 'image') and item.image is not None:
                    page_no = getattr(item, 'page_no', 1) or 1
                    
                    # Determine type
                    element_type = ElementType.TABLE
                    label = str(getattr(item, 'label', '')).lower()
                    if 'figure' in label or 'picture' in label or 'image' in label:
                        element_type = ElementType.FIGURE
                    
                    # Save image
                    image_filename = f"{element_type.value}_{page_no}_{len(elements)}.png"
                    image_path = self.output_dir / image_filename
                    
                    try:
                        item.image.pil_image.save(image_path)
                        logger.info(f"Saved {element_type.value} image: {image_path}")
                        
                        elements.append(ExtractedElement(
                            element_type=element_type,
                            content=None,
                            image_path=image_path,
                            page_number=page_no,
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to save image: {e}")
                        
        except Exception as e:
            logger.warning(f"Error iterating items for images: {e}")

        logger.info(f"Extracted {len(elements)} total elements from PDF")
        return elements
