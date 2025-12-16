"""FastAPI server for Constitutional Audio.

Provides REST API endpoints for prompt/audio classification,
voice enrollment, and management.

Usage:
    Start via CLI:
        constitutional-audio serve --host 0.0.0.0 --port 8000

    Or directly:
        uvicorn decrescendo.constitutional_audio.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        "FastAPI is required for the API server. "
        "Install with: pip install decrescendo[serve]"
    ) from e


# Global state for the loaded models
_pipeline = None
_voice_database = None
_voice_enroller = None
_config = None


# -----------------------------------------------------------------------------
# Pydantic Models for Request/Response
# -----------------------------------------------------------------------------


class PromptRequest(BaseModel):
    """Request body for prompt classification."""

    prompt: str = Field(..., description="Text prompt to classify")


class AudioUploadResponse(BaseModel):
    """Response for audio classification."""

    decision: str
    harm_scores: dict[str, float]
    voice_matches: list[dict[str, Any]]
    decision_reasons: list[str]


class VoiceEnrollRequest(BaseModel):
    """Request body for voice enrollment."""

    name: str = Field(..., description="Name for the voice (e.g., artist name)")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata"
    )


class VoiceEntry(BaseModel):
    """Voice entry in the database."""

    voice_id: int
    name: str
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    input_classifier_loaded: bool
    output_classifier_loaded: bool
    voice_database_loaded: bool
    num_protected_voices: int


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None


# -----------------------------------------------------------------------------
# API Factory
# -----------------------------------------------------------------------------


def create_app(
    input_checkpoint: str | Path | None = None,
    output_checkpoint: str | Path | None = None,
    voice_database_path: str | Path | None = None,
) -> FastAPI:
    """Create a FastAPI app with the given configuration.

    Args:
        input_checkpoint: Path to input classifier checkpoint.
        output_checkpoint: Path to output classifier checkpoint.
        voice_database_path: Path to voice database.

    Returns:
        Configured FastAPI application.
    """
    global _pipeline, _voice_database, _voice_enroller, _config

    app = FastAPI(
        title="Constitutional Audio API",
        description="Safety classification API for AI-generated audio",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.on_event("startup")
    async def startup_event() -> None:
        """Load models on startup."""
        global _pipeline, _voice_database, _voice_enroller

        from .pipeline import PipelineConfig, load_constitutional_audio

        config = PipelineConfig(
            enable_input_classifier=input_checkpoint is not None,
            enable_output_classifier=output_checkpoint is not None,
            enable_voice_matching=voice_database_path is not None,
        )

        if input_checkpoint or output_checkpoint:
            _pipeline = load_constitutional_audio(
                input_checkpoint=input_checkpoint,
                output_checkpoint=output_checkpoint,
                voice_database_path=voice_database_path,
                config=config,
            )

        if voice_database_path and Path(voice_database_path).exists():
            from .output_classifier.voice_database import VoiceDatabase

            _voice_database = VoiceDatabase.load(voice_database_path)

        if output_checkpoint and _pipeline and _pipeline.output_classifier:
            from .output_classifier.voice_enrollment import (
                create_voice_enroller_from_inference,
            )

            _voice_enroller = create_voice_enroller_from_inference(
                _pipeline.output_classifier
            )

    # -------------------------------------------------------------------------
    # Health Endpoint
    # -------------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        num_voices = 0
        if _voice_database is not None:
            num_voices = len(_voice_database)

        return HealthResponse(
            status="ok",
            input_classifier_loaded=_pipeline is not None
            and _pipeline.input_classifier is not None,
            output_classifier_loaded=_pipeline is not None
            and _pipeline.output_classifier is not None,
            voice_database_loaded=_voice_database is not None,
            num_protected_voices=num_voices,
        )

    # -------------------------------------------------------------------------
    # Classification Endpoints
    # -------------------------------------------------------------------------

    @app.post("/v1/classify/prompt", tags=["Classification"])
    async def classify_prompt(request: PromptRequest) -> JSONResponse:
        """Classify a text prompt for safety concerns.

        Returns classification results including intent, artist/voice requests,
        policy violations, and a final decision.
        """
        if _pipeline is None or _pipeline.input_classifier is None:
            raise HTTPException(
                status_code=503,
                detail="Input classifier not loaded. "
                "Start server with --input-checkpoint.",
            )

        result = _pipeline.classify_prompt(request.prompt)
        return JSONResponse(content=result.to_dict())

    @app.post("/v1/classify/audio", tags=["Classification"])
    async def classify_audio(file: UploadFile = File(...)) -> JSONResponse:
        """Classify an audio file for safety concerns.

        Accepts audio files (WAV, MP3, FLAC, etc.) and returns harm scores,
        voice matches, and a final decision.
        """
        if _pipeline is None or _pipeline.output_classifier is None:
            raise HTTPException(
                status_code=503,
                detail="Output classifier not loaded. "
                "Start server with --output-checkpoint.",
            )

        # Save uploaded file to temp location
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            result = _pipeline.classify_audio(tmp_path)
            return JSONResponse(content=result.to_dict())
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.post("/v1/classify/generation", tags=["Classification"])
    async def classify_generation(
        prompt: str | None = None,
        file: UploadFile | None = File(default=None),
    ) -> JSONResponse:
        """Classify a full generation (prompt + audio).

        Runs both input and output classifiers and aggregates the decisions.
        At least one of prompt or file must be provided.
        """
        if _pipeline is None:
            raise HTTPException(
                status_code=503, detail="Pipeline not loaded."
            )

        if prompt is None and file is None:
            raise HTTPException(
                status_code=400,
                detail="At least one of 'prompt' or 'file' must be provided.",
            )

        # Handle audio file if provided
        tmp_path: Path | None = None
        if file is not None:
            suffix = Path(file.filename).suffix if file.filename else ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = Path(tmp.name)

        try:
            result = _pipeline.classify_generation(
                prompt=prompt,
                audio=tmp_path,
            )
            return JSONResponse(content=result.to_dict())
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    # -------------------------------------------------------------------------
    # Voice Management Endpoints
    # -------------------------------------------------------------------------

    @app.get("/v1/voices", response_model=list[VoiceEntry], tags=["Voices"])
    async def list_voices() -> list[VoiceEntry]:
        """List all enrolled protected voices."""
        if _voice_database is None:
            raise HTTPException(
                status_code=503,
                detail="Voice database not loaded. "
                "Start server with --voice-db.",
            )

        entries = _voice_database.list_voices()
        return [
            VoiceEntry(
                voice_id=e.voice_id,
                name=e.name,
                metadata=e.metadata,
            )
            for e in entries
        ]

    @app.post("/v1/voices/enroll", tags=["Voices"])
    async def enroll_voice(
        name: str,
        file: UploadFile = File(...),
        metadata: str | None = None,
    ) -> JSONResponse:
        """Enroll a new protected voice.

        Upload an audio file of the voice to protect. Multiple enrollments
        with different audio samples for the same voice will improve matching.

        Args:
            name: Name for the voice (e.g., artist name)
            file: Audio file for voice enrollment
            metadata: Optional JSON string with metadata
        """
        if _voice_database is None:
            raise HTTPException(
                status_code=503,
                detail="Voice database not loaded. "
                "Start server with --voice-db.",
            )

        if _voice_enroller is None:
            raise HTTPException(
                status_code=503,
                detail="Voice enroller not available. "
                "Start server with --output-checkpoint.",
            )

        # Parse metadata if provided
        parsed_metadata: dict[str, Any] | None = None
        if metadata:
            import json

            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid metadata JSON: {e}"
                ) from e

        # Save uploaded file to temp location
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            result = _voice_enroller.enroll_from_file(
                database=_voice_database,
                name=name,
                path=tmp_path,
                metadata=parsed_metadata,
            )

            # Save database if enrollment succeeded
            if result.success and voice_database_path:
                _voice_database.save(voice_database_path)

            response = {
                "success": result.success,
                "voice_id": result.voice_id,
                "name": result.name,
                "embedding_dim": result.embedding_dim,
                "num_samples_used": result.num_samples_used,
                "error": result.error,
            }

            if not result.success:
                return JSONResponse(content=response, status_code=400)

            return JSONResponse(content=response, status_code=201)

        finally:
            tmp_path.unlink(missing_ok=True)

    @app.delete("/v1/voices/{voice_id}", tags=["Voices"])
    async def delete_voice(voice_id: int) -> JSONResponse:
        """Remove a protected voice from the database.

        Args:
            voice_id: ID of the voice to remove
        """
        if _voice_database is None:
            raise HTTPException(
                status_code=503,
                detail="Voice database not loaded. "
                "Start server with --voice-db.",
            )

        from .output_classifier.voice_database import VoiceNotFoundError

        try:
            entry = _voice_database.remove_voice(voice_id)

            # Save database after removal
            if voice_database_path:
                _voice_database.save(voice_database_path)

            return JSONResponse(
                content={
                    "voice_id": entry.voice_id,
                    "name": entry.name,
                    "metadata": entry.metadata,
                    "message": "Voice removed successfully",
                }
            )

        except VoiceNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Voice with ID {voice_id} not found"
            ) from None

    @app.get("/v1/voices/{voice_id}", response_model=VoiceEntry, tags=["Voices"])
    async def get_voice(voice_id: int) -> VoiceEntry:
        """Get details for a specific voice.

        Args:
            voice_id: ID of the voice to retrieve
        """
        if _voice_database is None:
            raise HTTPException(
                status_code=503,
                detail="Voice database not loaded. "
                "Start server with --voice-db.",
            )

        from .output_classifier.voice_database import VoiceNotFoundError

        try:
            entry, _ = _voice_database.get_voice(voice_id)
            return VoiceEntry(
                voice_id=entry.voice_id,
                name=entry.name,
                metadata=entry.metadata,
            )
        except VoiceNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Voice with ID {voice_id} not found"
            ) from None

    return app


# Default app instance for direct uvicorn usage
# This will be created without any checkpoints loaded
# Use create_app() with arguments for production
app = create_app()
