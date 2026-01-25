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

        # Step 4: Extract individual pictures using doc.pictures (PREFERRED)
        # This extracts only the figure regions, not full pages
        # Also extracts figure labels and captions for rich metadata
        try:
            if hasattr(doc, 'pictures') and doc.pictures:
                logger.info(f"Found {len(doc.pictures)} pictures via doc.pictures")
                
                # Track figure numbering
                figure_number = 1
                
                for i, picture in enumerate(doc.pictures):
                    # Get page number from provenance
                    page_no = 1
                    if hasattr(picture, 'prov') and picture.prov:
                        for prov_item in picture.prov:
                            if hasattr(prov_item, 'page_no'):
                                page_no = prov_item.page_no
                                break
                    
                    # Get the image
                    if hasattr(picture, 'image') and picture.image is not None:
                        # Use figure number in filename for proper identification
                        image_filename = f"figure_{figure_number}_page_{page_no}.png"
                        image_path = self.output_dir / image_filename
                        
                        try:
                            # Get PIL image from ImageRef
                            pil_img = None
                            if hasattr(picture.image, 'pil_image'):
                                pil_img = picture.image.pil_image
                            
                            if pil_img:
                                pil_img.save(image_path)
                                logger.info(f"Saved Figure {figure_number}: {image_path} (from page {page_no})")
                                
                                # Build RICH caption with all available context
                                caption_parts = []
                                
                                # 1. Add figure label (Figure X)
                                caption_parts.append(f"Figure {figure_number}")
                                
                                # 2. Try to get caption from Docling
                                if hasattr(picture, 'caption_text'):
                                    try:
                                        docling_caption = picture.caption_text(doc)
                                        if docling_caption and docling_caption.strip():
                                            caption_parts.append(docling_caption.strip())
                                    except Exception:
                                        pass
                                
                                # 3. Try to get text from annotations
                                if hasattr(picture, 'annotations'):
                                    for ann in picture.annotations or []:
                                        if hasattr(ann, 'text') and ann.text:
                                            caption_parts.append(ann.text)
                                
                                # 4. Try to get any label property
                                if hasattr(picture, 'label') and picture.label:
                                    label_text = str(picture.label)
                                    if label_text not in ' '.join(caption_parts):
                                        caption_parts.append(label_text)
                                
                                # 5. Look for figure references in markdown near this page
                                # Search for "Figure X:" patterns in the markdown
                                import re
                                figure_pattern = rf"Figure\s*{figure_number}\s*[:\.]?\s*([^.]*(?:\.[^.]*)?)"
                                matches = re.findall(figure_pattern, markdown_text, re.IGNORECASE)
                                for match in matches:
                                    if match.strip() and match.strip() not in ' '.join(caption_parts):
                                        caption_parts.append(match.strip())
                                
                                # Combine all caption parts
                                full_caption = " | ".join(filter(None, caption_parts))
                                logger.info(f"Figure {figure_number} caption: {full_caption[:100]}...")
                                
                                elements.append(ExtractedElement(
                                    element_type=ElementType.FIGURE,
                                    content=full_caption,
                                    image_path=image_path,
                                    page_number=page_no,
                                    heading=f"Figure {figure_number}",  # Store figure number in heading
                                ))
                                
                                figure_number += 1
                        except Exception as e:
                            logger.warning(f"Failed to save picture {i}: {e}")
            else:
                logger.info("No pictures found in doc.pictures, skipping figure extraction")
        except Exception as e:
            logger.warning(f"Error extracting pictures: {e}")

        # Step 5: Extract tables using doc.tables (for BOTH text and images)
        # Tables are stored as:
        # 1. Text content (markdown) for semantic search
        # 2. Image for visual display to user
        try:
            if hasattr(doc, 'tables') and doc.tables:
                logger.info(f"Found {len(doc.tables)} tables via doc.tables")
                
                table_number = 1
                
                for i, table in enumerate(doc.tables):
                    # Get page number from provenance
                    page_no = 1
                    if hasattr(table, 'prov') and table.prov:
                        for prov_item in table.prov:
                            if hasattr(prov_item, 'page_no'):
                                page_no = prov_item.page_no
                                break
                    
                    # Try to get table image
                    image_path = None
                    if hasattr(table, 'image') and table.image is not None:
                        image_filename = f"table_{table_number}_page_{page_no}.png"
                        img_path = self.output_dir / image_filename
                        
                        try:
                            pil_img = None
                            if hasattr(table.image, 'pil_image'):
                                pil_img = table.image.pil_image
                            
                            if pil_img:
                                pil_img.save(img_path)
                                image_path = img_path
                                logger.info(f"Saved Table {table_number} image: {img_path} (from page {page_no})")
                        except Exception as e:
                            logger.warning(f"Failed to save table image: {e}")
                    
                    # Get table content as markdown
                    table_content = ""
                    
                    # Try export_to_markdown method
                    if hasattr(table, 'export_to_markdown'):
                        try:
                            table_content = table.export_to_markdown()
                        except Exception:
                            pass
                    
                    # Try to get caption
                    caption = ""
                    if hasattr(table, 'caption_text'):
                        try:
                            caption = table.caption_text(doc) or ""
                        except Exception:
                            pass
                    
                    # Build rich table metadata
                    content_parts = [f"Table {table_number}"]
                    if caption:
                        content_parts.append(caption)
                    if table_content:
                        content_parts.append(table_content[:2000])  # Limit for embedding
                    
                    full_content = " | ".join(filter(None, content_parts))
                    
                    if full_content.strip() or image_path:
                        elements.append(ExtractedElement(
                            element_type=ElementType.TABLE,
                            content=full_content if full_content.strip() else f"Table {table_number} from page {page_no}",
                            image_path=image_path,
                            page_number=page_no,
                            heading=f"Table {table_number}",
                        ))
                        logger.info(f"Table {table_number} content: {full_content[:100]}...")
                        table_number += 1
            else:
                logger.info("No tables found in doc.tables, skipping table extraction")
        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")

        logger.info(f"Extracted {len(elements)} total elements from PDF")
        return elements


