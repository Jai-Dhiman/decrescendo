"""Command-line interface for Constitutional Audio.

Provides commands for classifying prompts and audio, enrolling protected voices,
and running an HTTP API server.

Usage:
    constitutional-audio classify-prompt "Generate music like Drake"
    constitutional-audio classify-audio output.wav --output-checkpoint ./checkpoints/output
    constitutional-audio enroll-voice "Artist Name" sample1.wav sample2.wav --voice-db ./voices
    constitutional-audio list-voices --voice-db ./voices
    constitutional-audio serve --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Output format constants
FORMAT_JSON = "json"
FORMAT_TABLE = "table"
FORMAT_TEXT = "text"


class CLIError(Exception):
    """Base exception for CLI errors."""

    pass


class CheckpointNotProvidedError(CLIError):
    """Raised when a required checkpoint path is not provided."""

    pass


# -----------------------------------------------------------------------------
# Output Formatters
# -----------------------------------------------------------------------------


def format_json(data: dict[str, Any]) -> str:
    """Format data as JSON."""
    return json.dumps(data, indent=2, default=str)


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format data as a simple text table."""
    if not rows:
        return "(empty)"

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Build table
    lines = []

    # Header
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # Rows
    for row in rows:
        row_line = "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        lines.append(row_line)

    return "\n".join(lines)


def format_prompt_result_text(result: Any) -> str:
    """Format prompt classification result as human-readable text."""
    lines = [
        f"Decision: {result.decision.value}",
        "",
        f"Intent: {result.input_result.intent.name} "
        f"(confidence: {result.input_result.intent_confidence:.2%})",
        f"Artist Request: {result.input_result.artist_request.name}",
        f"Voice Request: {result.input_result.voice_request.name}",
    ]

    if result.input_result.policy_flags:
        lines.append("")
        lines.append("Policy Violations:")
        for flag in result.input_result.policy_flags:
            score = result.input_result.policy_violations[flag]
            lines.append(f"  - {flag}: {score:.2%}")

    if result.input_result.decision_reasons:
        lines.append("")
        lines.append("Reasons:")
        for reason in result.input_result.decision_reasons:
            lines.append(f"  - {reason}")

    return "\n".join(lines)


def format_audio_result_text(result: Any) -> str:
    """Format audio classification result as human-readable text."""
    lines = [
        f"Decision: {result.decision.value}",
        "",
        "Harm Scores:",
    ]

    for category, score in result.output_result.harm_scores.items():
        bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
        lines.append(f"  {category:25s} [{bar}] {score:.2%}")

    if result.output_result.voice_matches:
        lines.append("")
        lines.append("Voice Matches:")
        for match in result.output_result.voice_matches:
            lines.append(f"  - {match.name} (similarity: {match.similarity:.2%})")

    if result.output_result.decision_reasons:
        lines.append("")
        lines.append("Reasons:")
        for reason in result.output_result.decision_reasons:
            lines.append(f"  - {reason}")

    return "\n".join(lines)


def format_enrollment_result_text(result: Any) -> str:
    """Format enrollment result as human-readable text."""
    if result.success:
        lines = [
            f"Successfully enrolled voice '{result.name}'",
            f"Voice ID: {result.voice_id}",
            f"Embedding dimension: {result.embedding_dim}",
            f"Samples used: {result.num_samples_used}",
        ]

        if result.quality_results:
            lines.append("")
            lines.append("Quality Checks:")
            for i, qr in enumerate(result.quality_results):
                status = "PASS" if qr.passed else "FAIL"
                lines.append(f"  Sample {i + 1}: {status}")
                lines.append(f"    Duration: {qr.duration_sec:.1f}s")
                lines.append(f"    RMS: {qr.rms_db:.1f} dB")
                if qr.snr_db is not None:
                    lines.append(f"    SNR: {qr.snr_db:.1f} dB")
                if qr.issues:
                    for issue in qr.issues:
                        lines.append(f"    Issue: {issue}")
    else:
        lines = [
            f"Failed to enroll voice '{result.name}'",
            f"Error: {result.error}",
        ]

        if result.duplicate_match:
            lines.append(
                f"Similar to: {result.duplicate_match.name} "
                f"(similarity: {result.duplicate_match.similarity:.2%})"
            )

    return "\n".join(lines)


def format_voice_list_table(entries: list[Any]) -> str:
    """Format voice entries as a table."""
    if not entries:
        return "No protected voices enrolled."

    headers = ["ID", "Name", "Metadata"]
    rows = []
    for entry in entries:
        metadata_str = json.dumps(entry.metadata) if entry.metadata else ""
        rows.append([str(entry.voice_id), entry.name, metadata_str])

    return format_table(headers, rows)


# -----------------------------------------------------------------------------
# Command Handlers
# -----------------------------------------------------------------------------


def cmd_classify_prompt(args: argparse.Namespace) -> int:
    """Handle classify-prompt command."""
    # Lazy imports to speed up CLI startup
    from .pipeline import PipelineConfig, load_constitutional_audio

    if not args.input_checkpoint:
        raise CheckpointNotProvidedError(
            "Input classifier checkpoint is required. "
            "Use --input-checkpoint to specify the path."
        )

    # Load pipeline (input classifier only)
    config = PipelineConfig(
        enable_input_classifier=True,
        enable_output_classifier=False,
        enable_voice_matching=False,
    )

    print("Loading input classifier...", file=sys.stderr)
    pipeline = load_constitutional_audio(
        input_checkpoint=args.input_checkpoint,
        config=config,
    )

    # Classify prompt
    result = pipeline.classify_prompt(args.prompt)

    # Format output
    if args.format == FORMAT_JSON:
        print(format_json(result.to_dict()))
    elif args.format == FORMAT_TABLE:
        # For prompt results, use text format since table doesn't fit well
        print(format_prompt_result_text(result))
    else:  # text
        print(format_prompt_result_text(result))

    # Return non-zero exit code if blocked
    from .pipeline import PipelineDecision

    if result.decision == PipelineDecision.BLOCK:
        return 1
    return 0


def cmd_classify_audio(args: argparse.Namespace) -> int:
    """Handle classify-audio command."""
    from .pipeline import PipelineConfig, load_constitutional_audio

    if not args.output_checkpoint:
        raise CheckpointNotProvidedError(
            "Output classifier checkpoint is required. "
            "Use --output-checkpoint to specify the path."
        )

    # Validate file exists
    audio_path = Path(args.file)
    if not audio_path.exists():
        raise CLIError(f"Audio file not found: {audio_path}")

    # Load pipeline
    config = PipelineConfig(
        enable_input_classifier=bool(args.input_checkpoint),
        enable_output_classifier=True,
        enable_voice_matching=bool(args.voice_db),
    )

    print("Loading classifiers...", file=sys.stderr)
    pipeline = load_constitutional_audio(
        input_checkpoint=args.input_checkpoint,
        output_checkpoint=args.output_checkpoint,
        voice_database_path=args.voice_db,
        config=config,
    )

    # Classify audio
    result = pipeline.classify_audio(audio_path)

    # Format output
    if args.format == FORMAT_JSON:
        print(format_json(result.to_dict()))
    else:
        print(format_audio_result_text(result))

    # Return non-zero exit code if blocked
    from .pipeline import PipelineDecision

    if result.decision == PipelineDecision.BLOCK:
        return 1
    return 0


def cmd_enroll_voice(args: argparse.Namespace) -> int:
    """Handle enroll-voice command."""
    from .output_classifier.checkpointing import load_output_classifier
    from .output_classifier.voice_database import VoiceDatabase
    from .output_classifier.voice_enrollment import VoiceEnroller

    if not args.output_checkpoint:
        raise CheckpointNotProvidedError(
            "Output classifier checkpoint is required for voice enrollment. "
            "Use --output-checkpoint to specify the path."
        )

    if not args.voice_db:
        raise CLIError(
            "Voice database path is required. Use --voice-db to specify the path."
        )

    # Validate audio files exist
    audio_paths = [Path(f) for f in args.files]
    for path in audio_paths:
        if not path.exists():
            raise CLIError(f"Audio file not found: {path}")

    # Load or create voice database
    voice_db_path = Path(args.voice_db)
    if voice_db_path.exists():
        print(f"Loading voice database from {voice_db_path}...", file=sys.stderr)
        database = VoiceDatabase.load(voice_db_path)
    else:
        print(f"Creating new voice database at {voice_db_path}...", file=sys.stderr)
        database = VoiceDatabase()

    # Load model for embedding extraction
    print("Loading output classifier...", file=sys.stderr)
    model, variables, config = load_output_classifier(args.output_checkpoint)

    # Create enroller
    enroller = VoiceEnroller(model, variables, config)

    # Parse metadata if provided
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            raise CLIError(f"Invalid metadata JSON: {e}") from e

    # Enroll voice
    print(f"Enrolling voice '{args.name}' from {len(audio_paths)} file(s)...", file=sys.stderr)
    result = enroller.enroll_from_files(
        database=database,
        name=args.name,
        paths=audio_paths,
        metadata=metadata,
    )

    # Save database if enrollment succeeded
    if result.success:
        database.save(voice_db_path)
        print(f"Voice database saved to {voice_db_path}", file=sys.stderr)

    # Format output
    if args.format == FORMAT_JSON:
        output = {
            "success": result.success,
            "voice_id": result.voice_id,
            "name": result.name,
            "embedding_dim": result.embedding_dim,
            "num_samples_used": result.num_samples_used,
            "error": result.error,
        }
        print(format_json(output))
    else:
        print(format_enrollment_result_text(result))

    return 0 if result.success else 1


def cmd_list_voices(args: argparse.Namespace) -> int:
    """Handle list-voices command."""
    from .output_classifier.voice_database import VoiceDatabase

    if not args.voice_db:
        raise CLIError(
            "Voice database path is required. Use --voice-db to specify the path."
        )

    voice_db_path = Path(args.voice_db)
    if not voice_db_path.exists():
        raise CLIError(f"Voice database not found: {voice_db_path}")

    # Load database
    database = VoiceDatabase.load(voice_db_path)
    entries = database.list_voices()

    # Format output
    if args.format == FORMAT_JSON:
        output = [
            {
                "voice_id": e.voice_id,
                "name": e.name,
                "metadata": e.metadata,
            }
            for e in entries
        ]
        print(format_json(output))
    else:
        print(f"Protected Voices ({len(entries)} total):")
        print(format_voice_list_table(entries))

    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Handle serve command."""
    try:
        import uvicorn
    except ImportError:
        raise CLIError(
            "FastAPI and uvicorn are required for the serve command. "
            "Install with: pip install decrescendo[serve]"
        ) from None

    # Import the API app (will be created next)
    from .api import create_app

    # Create app with configuration
    app = create_app(
        input_checkpoint=args.input_checkpoint,
        output_checkpoint=args.output_checkpoint,
        voice_database_path=args.voice_db,
    )

    print(f"Starting server at http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port)

    return 0


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="constitutional-audio",
        description="Constitutional Audio - Safety classification for AI-generated audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Classify a text prompt:
    constitutional-audio classify-prompt "Generate music like Drake" \\
      --input-checkpoint ./checkpoints/input

  Classify an audio file:
    constitutional-audio classify-audio output.wav \\
      --output-checkpoint ./checkpoints/output \\
      --voice-db ./voices

  Enroll a protected voice:
    constitutional-audio enroll-voice "Artist Name" sample1.wav sample2.wav \\
      --output-checkpoint ./checkpoints/output \\
      --voice-db ./voices

  List protected voices:
    constitutional-audio list-voices --voice-db ./voices

  Start the HTTP API server:
    constitutional-audio serve --host 0.0.0.0 --port 8000 \\
      --input-checkpoint ./checkpoints/input \\
      --output-checkpoint ./checkpoints/output
""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # classify-prompt
    prompt_parser = subparsers.add_parser(
        "classify-prompt",
        help="Classify a text prompt for safety concerns",
    )
    prompt_parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt to classify",
    )
    prompt_parser.add_argument(
        "--input-checkpoint",
        type=str,
        required=True,
        help="Path to input classifier checkpoint",
    )
    prompt_parser.add_argument(
        "--format",
        type=str,
        choices=[FORMAT_JSON, FORMAT_TABLE, FORMAT_TEXT],
        default=FORMAT_TEXT,
        help="Output format (default: text)",
    )

    # classify-audio
    audio_parser = subparsers.add_parser(
        "classify-audio",
        help="Classify an audio file for safety concerns",
    )
    audio_parser.add_argument(
        "file",
        type=str,
        help="Path to audio file to classify",
    )
    audio_parser.add_argument(
        "--input-checkpoint",
        type=str,
        help="Path to input classifier checkpoint (optional)",
    )
    audio_parser.add_argument(
        "--output-checkpoint",
        type=str,
        required=True,
        help="Path to output classifier checkpoint",
    )
    audio_parser.add_argument(
        "--voice-db",
        type=str,
        help="Path to voice database for protected voice matching",
    )
    audio_parser.add_argument(
        "--format",
        type=str,
        choices=[FORMAT_JSON, FORMAT_TABLE, FORMAT_TEXT],
        default=FORMAT_TEXT,
        help="Output format (default: text)",
    )

    # enroll-voice
    enroll_parser = subparsers.add_parser(
        "enroll-voice",
        help="Enroll a protected voice",
    )
    enroll_parser.add_argument(
        "name",
        type=str,
        help="Name for the voice (e.g., artist name)",
    )
    enroll_parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="Audio file(s) for voice enrollment",
    )
    enroll_parser.add_argument(
        "--output-checkpoint",
        type=str,
        required=True,
        help="Path to output classifier checkpoint",
    )
    enroll_parser.add_argument(
        "--voice-db",
        type=str,
        required=True,
        help="Path to voice database",
    )
    enroll_parser.add_argument(
        "--metadata",
        type=str,
        help="JSON metadata for the voice (e.g., '{\"genre\": \"pop\"}')",
    )
    enroll_parser.add_argument(
        "--format",
        type=str,
        choices=[FORMAT_JSON, FORMAT_TABLE, FORMAT_TEXT],
        default=FORMAT_TEXT,
        help="Output format (default: text)",
    )

    # list-voices
    list_parser = subparsers.add_parser(
        "list-voices",
        help="List enrolled protected voices",
    )
    list_parser.add_argument(
        "--voice-db",
        type=str,
        required=True,
        help="Path to voice database",
    )
    list_parser.add_argument(
        "--format",
        type=str,
        choices=[FORMAT_JSON, FORMAT_TABLE, FORMAT_TEXT],
        default=FORMAT_TEXT,
        help="Output format (default: text)",
    )

    # serve
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the HTTP API server",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    serve_parser.add_argument(
        "--input-checkpoint",
        type=str,
        help="Path to input classifier checkpoint",
    )
    serve_parser.add_argument(
        "--output-checkpoint",
        type=str,
        help="Path to output classifier checkpoint",
    )
    serve_parser.add_argument(
        "--voice-db",
        type=str,
        help="Path to voice database",
    )

    return parser


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Command dispatch
    handlers = {
        "classify-prompt": cmd_classify_prompt,
        "classify-audio": cmd_classify_audio,
        "enroll-voice": cmd_enroll_voice,
        "list-voices": cmd_list_voices,
        "serve": cmd_serve,
    }

    handler = handlers.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    try:
        return handler(args)
    except CLIError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
