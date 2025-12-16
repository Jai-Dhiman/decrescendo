"""Tests for CLI and API."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from decrescendo.constitutional_audio.cli import (
    CLIError,
    CheckpointNotProvidedError,
    create_parser,
    format_json,
    format_table,
    format_voice_list_table,
    main,
)


class TestOutputFormatters:
    """Test output formatting functions."""

    def test_format_json_simple(self):
        """Test JSON formatting with simple data."""
        data = {"key": "value", "number": 42}
        result = format_json(data)

        assert json.loads(result) == data
        assert '"key": "value"' in result

    def test_format_json_nested(self):
        """Test JSON formatting with nested data."""
        data = {"outer": {"inner": [1, 2, 3]}}
        result = format_json(data)

        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == [1, 2, 3]

    def test_format_table_basic(self):
        """Test table formatting."""
        headers = ["ID", "Name", "Score"]
        rows = [
            ["1", "Alice", "95"],
            ["2", "Bob", "87"],
        ]
        result = format_table(headers, rows)

        assert "ID" in result
        assert "Name" in result
        assert "Alice" in result
        assert "Bob" in result
        assert "---" in result  # separator

    def test_format_table_empty(self):
        """Test table formatting with empty data."""
        result = format_table(["A", "B"], [])
        assert result == "(empty)"

    def test_format_table_varying_widths(self):
        """Test table formatting adjusts column widths."""
        headers = ["Short", "Very Long Header"]
        rows = [["x", "value"]]
        result = format_table(headers, rows)

        lines = result.split("\n")
        # Header and data should be aligned
        assert len(lines) == 3  # header, separator, data

    def test_format_voice_list_table_empty(self):
        """Test voice list with no entries."""
        result = format_voice_list_table([])
        assert "No protected voices" in result

    def test_format_voice_list_table_with_entries(self):
        """Test voice list with entries."""
        # Create mock entries
        entry1 = MagicMock()
        entry1.voice_id = 1
        entry1.name = "Artist A"
        entry1.metadata = {"genre": "pop"}

        entry2 = MagicMock()
        entry2.voice_id = 2
        entry2.name = "Artist B"
        entry2.metadata = {}

        result = format_voice_list_table([entry1, entry2])

        assert "Artist A" in result
        assert "Artist B" in result
        assert "1" in result
        assert "2" in result


class TestArgumentParser:
    """Test CLI argument parsing."""

    def test_parser_creation(self):
        """Test parser can be created."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "constitutional-audio"

    def test_classify_prompt_args(self):
        """Test classify-prompt argument parsing."""
        parser = create_parser()
        args = parser.parse_args([
            "classify-prompt",
            "Test prompt",
            "--input-checkpoint", "./checkpoint",
            "--format", "json",
        ])

        assert args.command == "classify-prompt"
        assert args.prompt == "Test prompt"
        assert args.input_checkpoint == "./checkpoint"
        assert args.format == "json"

    def test_classify_prompt_requires_checkpoint(self):
        """Test classify-prompt requires --input-checkpoint."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["classify-prompt", "Test prompt"])

    def test_classify_audio_args(self):
        """Test classify-audio argument parsing."""
        parser = create_parser()
        args = parser.parse_args([
            "classify-audio",
            "audio.wav",
            "--output-checkpoint", "./checkpoint",
            "--voice-db", "./voices",
        ])

        assert args.command == "classify-audio"
        assert args.file == "audio.wav"
        assert args.output_checkpoint == "./checkpoint"
        assert args.voice_db == "./voices"

    def test_enroll_voice_args(self):
        """Test enroll-voice argument parsing."""
        parser = create_parser()
        args = parser.parse_args([
            "enroll-voice",
            "Artist Name",
            "sample1.wav",
            "sample2.wav",
            "--output-checkpoint", "./checkpoint",
            "--voice-db", "./voices",
            "--metadata", '{"genre": "pop"}',
        ])

        assert args.command == "enroll-voice"
        assert args.name == "Artist Name"
        assert args.files == ["sample1.wav", "sample2.wav"]
        assert args.metadata == '{"genre": "pop"}'

    def test_list_voices_args(self):
        """Test list-voices argument parsing."""
        parser = create_parser()
        args = parser.parse_args([
            "list-voices",
            "--voice-db", "./voices",
            "--format", "json",
        ])

        assert args.command == "list-voices"
        assert args.voice_db == "./voices"
        assert args.format == "json"

    def test_serve_args(self):
        """Test serve argument parsing."""
        parser = create_parser()
        args = parser.parse_args([
            "serve",
            "--host", "0.0.0.0",
            "--port", "9000",
            "--input-checkpoint", "./input",
            "--output-checkpoint", "./output",
        ])

        assert args.command == "serve"
        assert args.host == "0.0.0.0"
        assert args.port == 9000
        assert args.input_checkpoint == "./input"
        assert args.output_checkpoint == "./output"

    def test_serve_defaults(self):
        """Test serve command defaults."""
        parser = create_parser()
        args = parser.parse_args(["serve"])

        assert args.host == "127.0.0.1"
        assert args.port == 8000

    def test_format_choices(self):
        """Test format argument accepts valid choices."""
        parser = create_parser()

        for fmt in ["json", "table", "text"]:
            args = parser.parse_args([
                "list-voices",
                "--voice-db", "./voices",
                "--format", fmt,
            ])
            assert args.format == fmt

    def test_invalid_format_rejected(self):
        """Test invalid format is rejected."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "list-voices",
                "--voice-db", "./voices",
                "--format", "invalid",
            ])


class TestListVoicesCommand:
    """Test list-voices command."""

    def test_list_voices_missing_db_raises(self):
        """Test list-voices raises when voice-db not provided."""
        from decrescendo.constitutional_audio.cli import cmd_list_voices

        args = MagicMock()
        args.voice_db = None

        with pytest.raises(CLIError, match="Voice database path is required"):
            cmd_list_voices(args)

    def test_list_voices_db_not_found_raises(self):
        """Test list-voices raises when database doesn't exist."""
        from decrescendo.constitutional_audio.cli import cmd_list_voices

        args = MagicMock()
        args.voice_db = "/nonexistent/path"

        with pytest.raises(CLIError, match="not found"):
            cmd_list_voices(args)

    def test_list_voices_json_output(self, capsys):
        """Test list-voices JSON output."""
        from decrescendo.constitutional_audio.cli import cmd_list_voices
        from decrescendo.constitutional_audio.output_classifier.voice_database import (
            VoiceDatabase,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "voices"

            # Create and save a database
            db = VoiceDatabase(embedding_dim=192)
            embedding = np.random.randn(192).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            db.add_voice("Test Artist", embedding, metadata={"genre": "rock"})
            db.save(db_path)

            args = MagicMock()
            args.voice_db = str(db_path)
            args.format = "json"

            result = cmd_list_voices(args)

            assert result == 0
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert len(output) == 1
            assert output[0]["name"] == "Test Artist"


class TestEnrollVoiceCommand:
    """Test enroll-voice command."""

    def test_enroll_voice_missing_checkpoint(self):
        """Test enroll-voice raises when checkpoint not provided."""
        from decrescendo.constitutional_audio.cli import cmd_enroll_voice

        args = MagicMock()
        args.output_checkpoint = None
        args.voice_db = "./voices"

        with pytest.raises(CheckpointNotProvidedError):
            cmd_enroll_voice(args)

    def test_enroll_voice_missing_voice_db(self):
        """Test enroll-voice raises when voice-db not provided."""
        from decrescendo.constitutional_audio.cli import cmd_enroll_voice

        args = MagicMock()
        args.output_checkpoint = "./checkpoint"
        args.voice_db = None

        with pytest.raises(CLIError, match="Voice database path is required"):
            cmd_enroll_voice(args)

    def test_enroll_voice_file_not_found(self):
        """Test enroll-voice raises when audio file doesn't exist."""
        from decrescendo.constitutional_audio.cli import cmd_enroll_voice

        args = MagicMock()
        args.output_checkpoint = "./checkpoint"
        args.voice_db = "./voices"
        args.name = "Test"
        args.files = ["/nonexistent/audio.wav"]

        with pytest.raises(CLIError, match="Audio file not found"):
            cmd_enroll_voice(args)


class TestClassifyPromptCommand:
    """Test classify-prompt command."""

    def test_classify_prompt_missing_checkpoint(self):
        """Test classify-prompt raises when checkpoint not provided."""
        from decrescendo.constitutional_audio.cli import cmd_classify_prompt

        args = MagicMock()
        args.input_checkpoint = None

        with pytest.raises(CheckpointNotProvidedError):
            cmd_classify_prompt(args)


class TestClassifyAudioCommand:
    """Test classify-audio command."""

    def test_classify_audio_missing_checkpoint(self):
        """Test classify-audio raises when checkpoint not provided."""
        from decrescendo.constitutional_audio.cli import cmd_classify_audio

        args = MagicMock()
        args.output_checkpoint = None
        args.file = "test.wav"

        with pytest.raises(CheckpointNotProvidedError):
            cmd_classify_audio(args)

    def test_classify_audio_file_not_found(self):
        """Test classify-audio raises when file doesn't exist."""
        from decrescendo.constitutional_audio.cli import cmd_classify_audio

        args = MagicMock()
        args.output_checkpoint = "./checkpoint"
        args.file = "/nonexistent/audio.wav"

        with pytest.raises(CLIError, match="Audio file not found"):
            cmd_classify_audio(args)


class TestServeCommand:
    """Test serve command."""

    def test_serve_missing_fastapi(self):
        """Test serve raises when FastAPI not installed."""
        from decrescendo.constitutional_audio.cli import cmd_serve

        args = MagicMock()
        args.host = "127.0.0.1"
        args.port = 8000
        args.input_checkpoint = None
        args.output_checkpoint = None
        args.voice_db = None

        # Mock uvicorn not being available
        with patch.dict("sys.modules", {"uvicorn": None}):
            with pytest.raises(CLIError, match="FastAPI and uvicorn are required"):
                cmd_serve(args)


class TestMainEntryPoint:
    """Test main entry point."""

    def test_main_with_unknown_command(self, monkeypatch):
        """Test main handles unknown command gracefully."""
        # This should be caught by argparse
        monkeypatch.setattr("sys.argv", ["constitutional-audio", "unknown-cmd"])

        with pytest.raises(SystemExit):
            main()

    def test_main_with_cli_error(self, monkeypatch, capsys):
        """Test main handles CLIError gracefully."""
        monkeypatch.setattr(
            "sys.argv",
            ["constitutional-audio", "list-voices", "--voice-db", "/nonexistent"],
        )

        result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_main_help(self, monkeypatch):
        """Test main shows help."""
        monkeypatch.setattr("sys.argv", ["constitutional-audio", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0


class TestPromptResultFormatter:
    """Test prompt result text formatting."""

    def test_format_prompt_result_text(self):
        """Test formatting prompt classification result."""
        from decrescendo.constitutional_audio.cli import format_prompt_result_text
        from decrescendo.constitutional_audio.pipeline import PipelineDecision

        # Create mock result
        input_result = MagicMock()
        input_result.intent.name = "BENIGN"
        input_result.intent_confidence = 0.95
        input_result.artist_request.name = "NONE"
        input_result.voice_request.name = "NONE"
        input_result.policy_flags = []
        input_result.policy_violations = {}
        input_result.decision_reasons = ["No safety concerns detected"]

        result = MagicMock()
        result.decision = PipelineDecision.ALLOW
        result.input_result = input_result

        output = format_prompt_result_text(result)

        assert "Decision: ALLOW" in output
        assert "Intent: BENIGN" in output
        assert "95" in output  # confidence percentage

    def test_format_prompt_result_with_violations(self):
        """Test formatting with policy violations."""
        from decrescendo.constitutional_audio.cli import format_prompt_result_text
        from decrescendo.constitutional_audio.pipeline import PipelineDecision

        input_result = MagicMock()
        input_result.intent.name = "SUSPICIOUS"
        input_result.intent_confidence = 0.7
        input_result.artist_request.name = "NAMED_ARTIST"
        input_result.voice_request.name = "NONE"
        input_result.policy_flags = ["COPYRIGHT_IP", "VOICE_CLONING"]
        input_result.policy_violations = {
            "COPYRIGHT_IP": 0.85,
            "VOICE_CLONING": 0.65,
        }
        input_result.decision_reasons = [
            "Named artist reference detected",
            "Policy violations: COPYRIGHT_IP, VOICE_CLONING",
        ]

        result = MagicMock()
        result.decision = PipelineDecision.BLOCK
        result.input_result = input_result

        output = format_prompt_result_text(result)

        assert "Decision: BLOCK" in output
        assert "Policy Violations:" in output
        assert "COPYRIGHT_IP" in output


class TestAudioResultFormatter:
    """Test audio result text formatting."""

    def test_format_audio_result_text(self):
        """Test formatting audio classification result."""
        from decrescendo.constitutional_audio.cli import format_audio_result_text
        from decrescendo.constitutional_audio.pipeline import PipelineDecision

        output_result = MagicMock()
        output_result.harm_scores = {
            "copyright_ip": 0.1,
            "voice_cloning": 0.05,
        }
        output_result.voice_matches = []
        output_result.decision_reasons = ["No safety concerns"]

        result = MagicMock()
        result.decision = PipelineDecision.ALLOW
        result.output_result = output_result

        output = format_audio_result_text(result)

        assert "Decision: ALLOW" in output
        assert "Harm Scores:" in output
        assert "copyright_ip" in output

    def test_format_audio_result_with_voice_match(self):
        """Test formatting with voice matches."""
        from decrescendo.constitutional_audio.cli import format_audio_result_text
        from decrescendo.constitutional_audio.pipeline import PipelineDecision

        voice_match = MagicMock()
        voice_match.name = "Protected Artist"
        voice_match.similarity = 0.92

        output_result = MagicMock()
        output_result.harm_scores = {"voice_cloning": 0.95}
        output_result.voice_matches = [voice_match]
        output_result.decision_reasons = ["Voice match detected"]

        result = MagicMock()
        result.decision = PipelineDecision.BLOCK
        result.output_result = output_result

        output = format_audio_result_text(result)

        assert "Voice Matches:" in output
        assert "Protected Artist" in output
        assert "92" in output  # similarity percentage


class TestEnrollmentResultFormatter:
    """Test enrollment result formatting."""

    def test_format_enrollment_success(self):
        """Test formatting successful enrollment."""
        from decrescendo.constitutional_audio.cli import format_enrollment_result_text

        quality = MagicMock()
        quality.passed = True
        quality.duration_sec = 10.5
        quality.rms_db = -20.0
        quality.snr_db = 35.0
        quality.issues = []

        result = MagicMock()
        result.success = True
        result.name = "Test Artist"
        result.voice_id = 42
        result.embedding_dim = 192
        result.num_samples_used = 2
        result.quality_results = [quality]

        output = format_enrollment_result_text(result)

        assert "Successfully enrolled" in output
        assert "Test Artist" in output
        assert "42" in output  # voice_id
        assert "192" in output  # embedding_dim

    def test_format_enrollment_failure(self):
        """Test formatting failed enrollment."""
        from decrescendo.constitutional_audio.cli import format_enrollment_result_text

        duplicate = MagicMock()
        duplicate.name = "Existing Artist"
        duplicate.similarity = 0.98

        result = MagicMock()
        result.success = False
        result.name = "Test Artist"
        result.error = "Duplicate voice detected"
        result.duplicate_match = duplicate

        output = format_enrollment_result_text(result)

        assert "Failed to enroll" in output
        assert "Duplicate voice detected" in output
        assert "Existing Artist" in output
