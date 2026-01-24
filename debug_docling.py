
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_docling")

def debug_pdf(pdf_path: Path):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
    
    logger.info(f"Converting {pdf_path}...")
    result = converter.convert(pdf_path)
    doc = result.document
    
    logger.info(f"Iterating items...")
    count = 0
    for item in doc.iterate_items():
        if hasattr(item, 'image') and item.image is not None:
            logger.info(f"Item found with image! Label: {getattr(item, 'label', 'N/A')}")
            count += 1
            
    logger.info(f"Total items with images: {count}")
    
    logger.info(f"Checking pages...")
    for i, page in doc.pages.items():
        logger.info(f"Page {i}: has_image={hasattr(page, 'image')}, image_is_none={page.image is None}")
        if page.image:
             logger.info(f"Page image type: {type(page.image)}")
             logger.info(f"Page image dir: {dir(page.image)}")

if __name__ == "__main__":
    debug_pdf(Path("TestingDATA/somatosensory.pdf"))
