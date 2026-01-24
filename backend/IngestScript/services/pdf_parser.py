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
        pipeline_options.generate_page_images = True  # Enable page images as fallback
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
        logger.info(f"PDFParser initialized with output_dir: {self.output_dir}")
        
        # Chunking configuration
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.min_doc_size_for_chunking = 4000  # ~2 pages: keep as single chunk

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks using recursive character splitting.
        
        Attempts to split on semantic boundaries in order of priority:
        1. Double newlines (paragraphs)
        2. Single newlines (lines)
        3. Sentence endings (. )
        4. Spaces
        
        Args:
            text: The text to chunk.
            
        Returns:
            List of text chunks with overlap.
        """
        text = text.strip()
        
        if not text:
            return []
        
        # Try to use langchain-text-splitters if available
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                keep_separator=True,
            )
            chunks = splitter.split_text(text)
            logger.info(f"Used LangChain splitter: {len(chunks)} chunks from {len(text)} chars")
            return chunks
            
        except ImportError:
            logger.info("langchain-text-splitters not installed, using pure Python implementation")
        
        # Pure Python recursive character splitting
        return self._smart_split(text, self.chunk_size, self.chunk_overlap)
    
    def _smart_split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """
        Pure Python implementation of recursive character splitting.
        
        Args:
            text: Text to split.
            chunk_size: Maximum chunk size.
            chunk_overlap: Overlap between chunks.
            
        Returns:
            List of text chunks.
        """
        separators = ["\n\n", "\n", ". ", " "]
        
        def split_recursive(text: str, sep_idx: int = 0) -> list[str]:
            """Recursively split text using separators in priority order."""
            if len(text) <= chunk_size:
                return [text.strip()] if text.strip() else []
            
            if sep_idx >= len(separators):
                # No more separators, hard split
                chunks = []
                start = 0
                while start < len(text):
                    end = min(start + chunk_size, len(text))
                    chunk = text[start:end].strip()
                    if chunk:
                        chunks.append(chunk)
                    start = end - chunk_overlap if end < len(text) else len(text)
                return chunks
            
            sep = separators[sep_idx]
            parts = text.split(sep)
            
            if len(parts) == 1:
                # Separator not found, try next
                return split_recursive(text, sep_idx + 1)
            
            # Merge parts into chunks respecting size limit
            chunks = []
            current_chunk = ""
            
            for i, part in enumerate(parts):
                # Re-add separator (except for last part)
                part_with_sep = part + sep if i < len(parts) - 1 else part
                
                if not current_chunk:
                    current_chunk = part_with_sep
                elif len(current_chunk) + len(part_with_sep) <= chunk_size:
                    current_chunk += part_with_sep
                else:
                    # Current chunk is full
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap from previous
                    if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                        overlap_text = current_chunk[-chunk_overlap:]
                        current_chunk = overlap_text + part_with_sep
                    else:
                        current_chunk = part_with_sep
                    
                    # If single part exceeds chunk_size, split it recursively
                    if len(current_chunk) > chunk_size:
                        sub_chunks = split_recursive(current_chunk, sep_idx + 1)
                        if sub_chunks:
                            chunks.extend(sub_chunks[:-1])
                            current_chunk = sub_chunks[-1] if sub_chunks else ""
            
            # Don't forget the last chunk
            if current_chunk.strip():
                if len(current_chunk) > chunk_size:
                    chunks.extend(split_recursive(current_chunk, sep_idx + 1))
                else:
                    chunks.append(current_chunk.strip())
            
            return chunks
        
        result = split_recursive(text)
        logger.info(f"Smart split: {len(result)} chunks from {len(text)} chars (size={chunk_size}, overlap={chunk_overlap})")
        return result

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
            
            for i, item in enumerate(items_list):
                # Debug logging for item
                has_image = hasattr(item, 'image') and item.image is not None
                if has_image:
                    logger.info(f"Item {i} has image! Label: {getattr(item, 'label', 'N/A')}")
                
                # Check if this item has an image (table or figure)
                if has_image:
                    page_no = getattr(item, 'page_no', 1) or 1
                    
                    # Determine type
                    element_type = ElementType.TABLE
                    label = str(getattr(item, 'label', '')).lower()
                    if 'figure' in label or 'picture' in label or 'image' in label:
                        element_type = ElementType.FIGURE
                    # Fallback: if it has an image but no clear label, call it a FIGURE
                    elif 'table' not in label:
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

        # Step 4: Fallback - Extract Page Images if enabled
        if not any(e.element_type in [ElementType.TABLE, ElementType.FIGURE] for e in elements):
            logger.info("No figures/tables found via iterate_items. checking doc.pages...")
            try:
                for i, page in enumerate(doc.pages.values()): # doc.pages is a dict {page_no: Page}
                    if hasattr(page, 'image') and page.image is not None:
                        logger.info(f"Processing image for page {page.page_no}")
                        # Save page image
                        image_filename = f"page_{page.page_no}_{len(elements)}.png"
                        image_path = self.output_dir / image_filename
                        
                        try:
                            # docling ImageRef handling
                            img = None
                            if hasattr(page.image, 'pil_image'):
                                img = page.image.pil_image
                            elif hasattr(page.image, 'image'): # old version?
                                img = page.image.image
                            else:
                                img = page.image # Fallback if it is already PIL image
                                
                            if img is None:
                                logger.warning(f"Page {page.page_no} image is None")
                                continue
                                
                            img.save(image_path)
                            logger.info(f"Saved PAGE image: {image_path}")
                            
                            elements.append(ExtractedElement(
                                element_type=ElementType.FIGURE, # Treat page as figure
                                content=f"Page {page.page_no} Screenshot",
                                image_path=image_path,
                                page_number=page.page_no,
                            ))
                        except Exception as e:
                            logger.warning(f"Failed to save page image {page.page_no}: {e}")
            except Exception as e:
                logger.warning(f"Error extracting page images: {e}")

        logger.info(f"Extracted {len(elements)} total elements from PDF")
        return elements
