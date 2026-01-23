"""
Configuration settings using pydantic-settings.

Loads environment variables from .env file with strict validation.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Google AI Configuration (for image transcription only)
    google_api_key: str = Field(
        ...,
        description="Google AI API key for Gemini access",
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model to use for transcription",
    )

    # Groq Configuration (for chat/RAG)
    groq_api_key: str = Field(
        ...,
        description="Groq API key for chat/RAG",
    )
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model to use for chat",
    )

    # Qdrant Configuration
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant server host",
    )
    qdrant_port: int = Field(
        default=6333,
        description="Qdrant server port",
    )
    qdrant_collection_name: str = Field(
        default="pdf_documents",
        description="Qdrant collection name for storing documents",
    )

    # Output Configuration
    output_dir: Path = Field(
        default=Path("./output"),
        description="Directory for extracted images and artifacts",
    )


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
