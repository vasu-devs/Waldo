"""
Gemini Transcriber service.

Uses Gemini 2.0 Flash to transcribe tables/charts to Markdown with self-correction.
"""

import base64
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Prompts for transcription and verification
TRANSCRIPTION_PROMPT = """Analyze this image and provide a DETAILED description for document retrieval.

Your description should include:
1. **Type**: What kind of visual is this? (diagram, chart, table, photograph, illustration, anatomical figure, etc.)
2. **Main Subject**: What is the primary topic or content shown?
3. **Key Elements**: List ALL labeled components, structures, or data points visible
4. **Relationships**: Describe how elements relate to each other (connections, layers, hierarchies)
5. **Text/Labels**: Transcribe ALL visible text, labels, captions, and annotations
6. **Context**: What concept or information does this visual explain?

For tables: Use proper Markdown table syntax with headers.
For diagrams: Describe the structure and all labeled parts in detail.
For anatomy figures: Name ALL anatomical structures shown and their locations.

Be EXHAUSTIVE - your description will be used for semantic search to find this image when users ask questions about its content."""

VERIFICATION_PROMPT = """Verify this description against the image and ensure it's comprehensive for retrieval.

DESCRIPTION TO VERIFY:
{transcription}

Instructions:
1. Check if ALL visible elements, labels, and structures are mentioned.
2. Ensure technical terms and proper names are correctly spelled.
3. Add any missing details that would help users find this image when searching.
4. If there are errors or omissions, provide the CORRECTED version.
5. If the description is complete, return it unchanged.

Return ONLY the final description (corrected if needed), no explanations."""



@dataclass
class TranscriptionResult:
    """Result of transcription with verification."""
    original_transcription: str
    verified_transcription: str
    was_corrected: bool
    image_path: Path


class GeminiTranscriber:
    """Service for transcribing images using Gemini 2.0 Flash."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash") -> None:
        """
        Initialize the Gemini transcriber.

        Args:
            api_key: Google AI API key.
            model_name: Gemini model to use.
        """
        # Lazy-load google.genai to prevent blocking server startup
        from google import genai
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.genai_types = None  # Will be loaded on first use
        logger.info(f"Initialized Gemini transcriber with model: {model_name}")

    def _load_image_as_base64(self, image_path: Path) -> str:
        """Load an image file and return base64 encoded string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_mime_type(self, image_path: Path) -> str:
        """Get MIME type based on file extension."""
        suffix = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        return mime_types.get(suffix, "image/png")

    async def transcribe(self, image_path: Path) -> str:
        """
        Transcribe an image to Markdown.

        Args:
            image_path: Path to the image file.

        Returns:
            Markdown transcription of the image content.
        """
        logger.info(f"Transcribing image: {image_path}")

        # Lazy-load types
        from google.genai import types

        image_data = self._load_image_as_base64(image_path)
        mime_type = self._get_mime_type(image_path)

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            data=base64.b64decode(image_data),
                            mime_type=mime_type,
                        ),
                        types.Part.from_text(text=TRANSCRIPTION_PROMPT),
                    ],
                )
            ],
        )

        transcription = response.text.strip()
        logger.debug(f"Initial transcription:\n{transcription}")
        return transcription

    async def verify(self, image_path: Path, transcription: str) -> str:
        """
        Verify and correct a transcription against the original image.

        Args:
            image_path: Path to the original image.
            transcription: The transcription to verify.

        Returns:
            Verified (and possibly corrected) transcription.
        """
        logger.info(f"Verifying transcription for: {image_path}")

        # Lazy-load types
        from google.genai import types

        image_data = self._load_image_as_base64(image_path)
        mime_type = self._get_mime_type(image_path)

        prompt = VERIFICATION_PROMPT.format(transcription=transcription)

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            data=base64.b64decode(image_data),
                            mime_type=mime_type,
                        ),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ],
        )

        verified = response.text.strip()
        logger.debug(f"Verified transcription:\n{verified}")
        return verified

    async def transcribe_with_verification(
        self, image_path: Path
    ) -> TranscriptionResult:
        """
        Transcribe an image and verify the result with self-correction.

        This implements a two-pass approach:
        1. Initial transcription
        2. Verification pass that can correct errors

        Args:
            image_path: Path to the image file.

        Returns:
            TranscriptionResult with original and verified transcriptions.
        """
        # Phase 1: Initial transcription
        original = await self.transcribe(image_path)

        # Phase 2: Self-correction verification
        verified = await self.verify(image_path, original)

        # Detect if correction was made
        was_corrected = original.strip() != verified.strip()

        if was_corrected:
            logger.info(f"Transcription was corrected for: {image_path}")
        else:
            logger.info(f"Transcription verified (no corrections) for: {image_path}")

        return TranscriptionResult(
            original_transcription=original,
            verified_transcription=verified,
            was_corrected=was_corrected,
            image_path=image_path,
        )

    async def generate_summary(self, full_text: str) -> str:
        """
        Generate a preview summary of the document text (LOCAL, no API call).

        Args:
            full_text: The concatenated text content of the entire document.

        Returns:
            A preview summary using the first 1,500 characters of the document.
        """
        logger.info("Generating local preview summary (no API call)...")

        # Take first 1,500 characters as preview
        preview = full_text[:1500].strip()
        
        summary = f"DOCUMENT PREVIEW / EXECUTIVE SUMMARY:\n{preview}"
        
        logger.info(f"Generated local summary: {len(summary)} chars")
        return summary
