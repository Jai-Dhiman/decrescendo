# Constitutional Audio: Classifier Tasks

## Overview

This document tracks remaining tasks to complete the classifier components of Constitutional Audio.

**Completed:**
- Input Classifier (text prompts): model, pretrained loading, training pipeline, inference, tests
- Output Classifier (audio): model, preprocessing, inference, tests
- Checkpointing: Input/Output classifier save/load, voice database persistence, tests
- Output Classifier Training: audio dataset/dataloader, losses (harm BCE + speaker contrastive), trainer with BatchNorm, pretrained audio encoder support (MERT/WavLM), 48 tests
- Unified Pipeline: ConstitutionalAudio class, classify_prompt/audio/generation methods, PipelineDecision enum, load_constitutional_audio factory

---

## 1. Output Classifier Training

### 1.1 Audio Dataset Pipeline
**File:** `src/decrescendo/constitutional_audio/data/audio_dataset.py`

- [x] `AudioClassificationSample` dataclass (audio path, harm labels, speaker ID)
- [x] `AudioDataset` class with `from_directory()` and `from_manifest()` loaders
- [x] `AudioDataLoader` with batching, shuffling, and on-the-fly preprocessing
- [x] Support for common formats (WAV, MP3, FLAC)
- [x] Data augmentation options (noise, pitch shift, time stretch, speed, RIR)

### 1.2 Training Losses
**File:** `src/decrescendo/constitutional_audio/training/audio_losses.py`

- [x] `harm_classification_loss`: Binary cross-entropy for multi-label harm categories
- [x] `speaker_verification_loss`: Contrastive (InfoNCE) and triplet loss for speaker embeddings
- [x] `combined_audio_loss`: Weighted combination of harm + speaker losses
- [x] Loss weighting configuration (`AudioLossWeights`)

### 1.3 Audio Trainer
**File:** `src/decrescendo/constitutional_audio/training/audio_trainer.py`

- [x] `AudioTrainState` with batch_stats handling for BatchNorm
- [x] `audio_train_step`: JIT-compiled training step with `mutable=['batch_stats']`
- [x] `audio_eval_step`: JIT-compiled evaluation step
- [x] `AudioTrainer` class with training loop
- [x] Audio-specific metrics (per-category F1, speaker EER)
- [x] Gradient clipping and mixed precision support

### 1.4 Pretrained Audio Encoders
**File:** `src/decrescendo/constitutional_audio/output_classifier/pretrained_audio.py`

- [x] `load_mert_encoder`: Load MERT encoder weights from HuggingFace
- [x] `load_wavlm_encoder`: Load WavLM encoder weights
- [x] `HybridAudioClassifier`: Adapter with projection layers for pretrained encoders
- [x] Option to freeze pretrained layers during fine-tuning

---

## 2. Protected Voice System

### 2.1 Voice Database
**File:** `src/decrescendo/constitutional_audio/output_classifier/voice_database.py`

- [x] `VoiceEntry` dataclass (voice_id, name, embedding, metadata)
- [x] `VoiceDatabase` class for managing protected voices
- [x] File-based storage (JSON/NPZ for embeddings)
- [x] `add_voice`, `remove_voice`, `get_voice`, `list_voices` methods
- [x] Batch similarity search against all protected voices
- [x] `search`, `find_match`, `check_duplicate` methods
- [x] `update_metadata`, `clear`, `get_all_embeddings` methods
- [x] File persistence with `save()` and `load()` class methods

### 2.2 Voice Enrollment Pipeline
**File:** `src/decrescendo/constitutional_audio/output_classifier/voice_enrollment.py`

- [x] `VoiceEnroller` class
- [x] `enroll_from_file`: Extract embedding from single audio file
- [x] `enroll_from_files`: Average embeddings from multiple samples
- [x] Quality checks (audio length, RMS level, SNR estimation)
- [x] Duplicate detection (warn if similar voice already enrolled)
- [x] `extract_embedding`, `extract_embedding_from_array` methods
- [x] `create_voice_enroller`, `create_voice_enroller_from_inference` helpers

---

## 3. Checkpointing

### 3.1 Input Classifier Checkpointing
**File:** `src/decrescendo/constitutional_audio/input_classifier/checkpointing.py`

- [x] `save_input_classifier`: Save params + config with Orbax
- [x] `load_input_classifier`: Load and reconstruct model + inference pipeline
- [x] `load_input_classifier_inference`: Load ready-to-use inference pipeline
- [x] Version compatibility checking
- [x] Support for checkpoint directory structure
- [x] `InputClassifierCheckpointer` class with `max_to_keep` support

### 3.2 Output Classifier Checkpointing
**File:** `src/decrescendo/constitutional_audio/output_classifier/checkpointing.py`

- [x] `save_output_classifier`: Save params + batch_stats + config
- [x] `load_output_classifier`: Load and reconstruct model + inference pipeline
- [x] `load_output_classifier_inference`: Load inference with optional voice database
- [x] `save_voice_database`: Save protected voices (NPZ + JSON manifest)
- [x] `load_voice_database`: Load protected voices
- [x] `VoiceEntry` dataclass for voice metadata
- [x] `OutputClassifierCheckpointer` class with `max_to_keep` support

---

## 4. Integration

### 4.1 Unified Pipeline
**File:** `src/decrescendo/constitutional_audio/pipeline.py`

- [x] `ConstitutionalAudio` class combining both classifiers
- [x] `classify_prompt(text)`: Run input classifier on text prompt
- [x] `classify_audio(audio)`: Run output classifier on audio
- [x] `classify_generation(prompt, audio)`: Full pipeline for generated audio
- [x] Unified `PipelineDecision` enum and result types
- [x] Configuration for which checks to enable/disable (`PipelineConfig`)

### 4.2 End-to-End Tests
**File:** `tests/test_pipeline.py`

- [ ] Test input classifier standalone
- [ ] Test output classifier standalone
- [ ] Test combined pipeline with mock audio generation
- [ ] Test decision aggregation (input + output decisions)
- [ ] Test with protected voice matching
- [ ] Performance/latency benchmarks

### 4.3 CLI Interface
**File:** `src/decrescendo/constitutional_audio/cli.py`

- [ ] `classify-prompt` command: Classify text prompts
- [ ] `classify-audio` command: Classify audio files
- [ ] `enroll-voice` command: Add voice to protected database
- [ ] `list-voices` command: Show enrolled protected voices
- [ ] `serve` command: Start HTTP API server (optional)
- [ ] Output formats: JSON, table, human-readable

**Entry point in pyproject.toml:**
```toml
[project.scripts]
constitutional-audio = "decrescendo.constitutional_audio.cli:main"
```

---

## 5. Tests

### 5.1 Additional Test Files Needed

- [x] `tests/test_audio_losses.py`: Audio loss function tests (19 tests)
- [x] `tests/test_audio_dataset.py`: Audio data loading tests (19 tests)
- [x] `tests/test_audio_training.py`: Training loop tests (10 tests)
- [x] `tests/test_voice_database.py`: Voice database and enrollment tests (43 tests)
- [x] `tests/test_checkpointing.py`: Save/load tests (22 tests)
- [ ] `tests/test_pipeline.py`: Integration tests
- [ ] `tests/test_cli.py`: CLI command tests

---

## Priority Order

| Priority | Category | Status | Rationale |
|----------|----------|--------|-----------|
| 1 | Checkpointing | Done | Enables saving trained models |
| 2 | Output Classifier Training | Done | Core training functionality |
| 3 | Protected Voice System | Done | Key safety feature |
| 4 | Integration | In Progress | Pipeline done, tests pending |
| 5 | CLI | Todo | User-facing interface |

---

## File Structure After Completion

```
src/decrescendo/constitutional_audio/
    __init__.py
    pipeline.py                      # [done] Unified pipeline
    cli.py                           # [todo] Command-line interface

    input_classifier/
        __init__.py
        config.py                    # [done]
        model.py                     # [done]
        pretrained.py                # [done]
        inference.py                 # [done]
        checkpointing.py             # [done]

    output_classifier/
        __init__.py
        config.py                    # [done]
        model.py                     # [done]
        audio_preprocessing.py       # [done]
        inference.py                 # [done]
        checkpointing.py             # [done] (includes VoiceEntry, voice database save/load)
        pretrained_audio.py          # [done] (MERT/WavLM loading, HybridAudioClassifier)
        voice_database.py            # [done] (VoiceDatabase class with add/remove/search)
        voice_enrollment.py          # [done] (VoiceEnroller with quality checks)

    data/
        __init__.py
        dataset.py                   # [done] (text)
        audio_dataset.py             # [done] (AudioDataset, AudioDataLoader, augmentation)

    training/
        __init__.py
        train_state.py               # [done]
        losses.py                    # [done] (text)
        metrics.py                   # [done]
        trainer.py                   # [done] (text)
        audio_losses.py              # [done] (harm BCE, speaker contrastive/triplet)
        audio_metrics.py             # [done] (harm metrics, speaker EER)
        audio_trainer.py             # [done] (AudioTrainState, AudioTrainer)

tests/
    test_checkpointing.py            # [done] (22 tests)
    test_audio_losses.py             # [done] (19 tests)
    test_audio_dataset.py            # [done] (19 tests)
    test_audio_training.py           # [done] (10 tests)
    test_voice_database.py           # [done] (43 tests)
```

---

## Notes

- All models use JAX/Flax (Linen API)
- Package management: uv
- Checkpointing: Orbax
- Config management: Hydra (optional for training scripts)
- Target: Local/research deployment
