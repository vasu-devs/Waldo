"""
PDF Parser service using docling.

Extracts text, tables, and figures from PDF documents.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import ConversionResult

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
    content: str | None  # Text content if available
    image_path: Path | None  # Path to extracted image
    page_number: int
    bbox: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1


class PDFParser:
    """Service for parsing PDF documents using docling."""

    def __init__(self, output_dir: Path) -> None:
        """
        Initialize the PDF parser.

        Args:
            output_dir: Directory to save extracted images.
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

    def parse(self, pdf_path: Path) -> list[ExtractedElement]:
        """
        Parse a PDF document and extract all elements.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of extracted elements (text, tables, figures).
        """
        logger.info(f"Parsing PDF: {pdf_path}")

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        result: ConversionResult = self.converter.convert(pdf_path)
        elements: list[ExtractedElement] = []

        # Process document elements
        doc = result.document

        for item in doc.iterate_items():
            element = item  # The actual document element
            page_no = getattr(element, 'page_no', 1) or 1

            # Handle tables
            if hasattr(element, 'image') and element.image is not None:
                # Determine element type
                element_type = ElementType.TABLE
                if hasattr(element, 'label') and 'figure' in str(getattr(element, 'label', '')).lower():
                    element_type = ElementType.FIGURE

                # Save image
                image_filename = f"{element_type.value}_{page_no}_{len(elements)}.png"
                image_path = self.output_dir / image_filename

                # Save the PIL image
                element.image.pil_image.save(image_path)
                logger.info(f"Saved {element_type.value} image: {image_path}")

                # Get text content if available
                text_content = None
                if hasattr(element, 'text'):
                    text_content = element.text

                elements.append(ExtractedElement(
                    element_type=element_type,
                    content=text_content,
                    image_path=image_path,
                    page_number=page_no,
                ))

        logger.info(f"Extracted {len(elements)} elements from PDF")
        return elements
