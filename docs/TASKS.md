# MusiCritic: Implementation Tasks

## Overview

**MusiCritic** is a unified evaluation framework for AI-generated music that assesses both **quality** (dimensions 1-4) and **safety** (dimensions 5-8) in a single pipeline.

**Target Metrics:**
- Quality: >0.75 human correlation (vs FAD's 0.52)
- Safety: <2% attack success rate
- Latency: <2 seconds for 30-second audio (GPU)

**Repository:** `decrescendo`
**Package:** `decrescendo.musicritic`
**CLI:** `musicritic`

---

## Current State

### Completed Infrastructure

| Component | Status | Tests | Location |
|-----------|--------|-------|----------|
| Package refactoring | Done | 221 | `src/decrescendo/musicritic/` |
| Dimension base classes | Done | 32 | `src/decrescendo/musicritic/dimensions/base.py` |
| Input classifier (text) | Done | - | `src/decrescendo/musicritic/input_classifier/` |
| Output classifier (audio) | Done | - | `src/decrescendo/musicritic/output_classifier/` |
| Voice database | Done | 43 | `src/decrescendo/musicritic/output_classifier/voice_database.py` |
| Voice enrollment | Done | - | `src/decrescendo/musicritic/output_classifier/voice_enrollment.py` |
| Unified pipeline | Done | 38 | `src/decrescendo/musicritic/pipeline.py` |
| CLI (5 commands) | Done | 36 | `src/decrescendo/musicritic/cli.py` |
| FastAPI server | Done | - | `src/decrescendo/musicritic/api.py` |
| Checkpointing (Orbax) | Done | 22 | Various `checkpointing.py` files |

**Total Tests:** 253 passing

### What Exists vs What's Needed

The existing codebase implements **Constitutional Audio** (safety-focused classification). MusiCritic extends this with:

| Dimension | Exists | Needs Building |
|-----------|--------|----------------|
| 1. Prompt Adherence | No | CLAP integration, similarity scoring |
| 2. Musical Coherence | No | Structure, harmony, rhythm, melody analysis |
| 3. Audio Quality | No | Artifact detection, loudness analysis |
| 4. Musicality | No | TIS computation, tension-resolution |
| 5. Copyright | No | Chromaprint, melody similarity |
| 6. Voice Cloning | Partial | Migrate existing voice database |
| 7. Cultural Sensitivity | No | Pattern detection, flagging |
| 8. Content Safety | Partial | Migrate existing harm classifier |

---

## Tech Stack

### Core Dependencies (Already Installed)
- **JAX/Flax**: ML framework
- **transformers**: HuggingFace models (MERT, CLAP available)
- **librosa**: Audio processing
- **Orbax**: Checkpointing

### Quality Dependencies (Optional Install)
```bash
# CLAP for prompt adherence
uv pip install laion-clap

# Audio fingerprinting
uv pip install pyacoustid

# Music analysis (requires build deps)
pip install Cython numpy
pip install madmom essentia
```

### Model Requirements

| Model | Purpose | Dimension | Size | License |
|-------|---------|-----------|------|---------|
| MERT-v1-95M | Music embeddings | 2,3,4 | 0.4GB | CC-BY-NC-4.0 |
| CLAP (laion-music) | Text-audio alignment | 1 | 0.8GB | Apache 2.0 |
| madmom DBNBeatTracker | Beat tracking | 2 | Small | BSD-3 |
| Essentia | Chords, key, clicks | 2,3 | Small | AGPL-3.0 |
| WavLM-Large | Speaker embeddings | 6 | 1.2GB | MIT |
| ECAPA-TDNN | Voice fingerprinting | 6 | Small | Apache 2.0 |
| Chromaprint | Audio fingerprinting | 5 | Small | LGPL 2.1 |

---

## Directory Structure

```
src/decrescendo/musicritic/
    __init__.py                  # Exports MusiCritic, load_musicritic
    pipeline.py                  # Legacy ConstitutionalAudio (to be extended)
    scoring.py                   # [NEW] Quality/Safety score aggregation
    cli.py                       # CLI commands
    api.py                       # FastAPI server
    config.py                    # [NEW] Unified configuration

    dimensions/
        __init__.py              # [DONE] Exports base classes
        base.py                  # [DONE] DimensionEvaluator, DimensionResult, Registry

        prompt_adherence/        # [NEW] Dimension 1
            __init__.py
            clap_encoder.py      # CLAP model wrapper
            evaluator.py         # PromptAdherenceEvaluator

        musical_coherence/       # [NEW] Dimension 2
            __init__.py
            structure.py         # Segment/section detection
            harmony.py           # Chord progression, key detection
            rhythm.py            # Beat tracking, tempo stability
            melody.py            # Pitch tracking, contour analysis
            evaluator.py         # MusicalCoherenceEvaluator

        audio_quality/           # [NEW] Dimension 3
            __init__.py
            artifacts.py         # Click detection, AI fingerprints
            loudness.py          # LUFS, LRA, True Peak
            perceptual.py        # Spectral quality metrics
            evaluator.py         # AudioQualityEvaluator

        musicality/              # [NEW] Dimension 4
            __init__.py
            tis.py               # Tonal Interval Space computation
            tension.py           # Tension-resolution curves
            expression.py        # Dynamic variation, groove
            evaluator.py         # MusicalityEvaluator

        copyright/               # [NEW] Dimension 5
            __init__.py
            fingerprint.py       # Chromaprint wrapper
            similarity.py        # MERT-based melody similarity
            evaluator.py         # CopyrightEvaluator

        voice_cloning/           # [MIGRATE] Dimension 6
            __init__.py
            database.py          # From output_classifier/voice_database.py
            enrollment.py        # From output_classifier/voice_enrollment.py
            evaluator.py         # VoiceCloningEvaluator

        cultural_sensitivity/    # [NEW] Dimension 7
            __init__.py
            detector.py          # Sacred/ceremonial pattern detection
            evaluator.py         # CulturalSensitivityEvaluator

        content_safety/          # [MIGRATE] Dimension 8
            __init__.py
            harm_classifier.py   # From output_classifier/
            evaluator.py         # ContentSafetyEvaluator

    encoders/                    # [NEW] Shared encoder infrastructure
        __init__.py
        mert.py                  # MERT-v1-95M wrapper
        clap.py                  # CLAP wrapper
        wavlm.py                 # WavLM wrapper (from pretrained_audio.py)
        cache.py                 # Embedding cache with TTL

    preprocessing/               # [EXISTS] Shared preprocessing
        __init__.py
        audio.py                 # From output_classifier/audio_preprocessing.py

    input_classifier/            # [EXISTS] Text prompt classification
    output_classifier/           # [EXISTS] Audio harm classification
    data/                        # [EXISTS] Dataset loading
    training/                    # [EXISTS] Training infrastructure
```

---

## Task Breakdown

### Phase 1: Infrastructure [DONE]

- [x] Rename package from `constitutional_audio` to `musicritic`
- [x] Update all imports and CLI entry point
- [x] Create `dimensions/base.py` with core abstractions
- [x] Create `DimensionEvaluator` protocol
- [x] Create `DimensionRegistry` for managing evaluators
- [x] Create `DimensionResult`, `QualityResult`, `SafetyResult` dataclasses
- [x] Add optional dependencies to pyproject.toml
- [x] Verify all 253 tests pass

---

### Phase 2: Quality Dimensions

#### 2.1 Prompt Adherence (Dimension 1) [DONE]
**Priority:** High (quick win, CLAP is straightforward)
**Dependencies:** torch (uses HuggingFace transformers CLAP)
**Files:** `dimensions/prompt_adherence/`

- [x] Create `CLAPEncoder` wrapper class
  - Load `laion/larger_clap_music` model via HuggingFace transformers
  - Implement `encode_text(prompt) -> embedding` with caching
  - Implement `encode_audio(audio, sr) -> embedding` with auto-resampling
  - Add text embedding caching for repeated prompts

- [x] Create `PromptAdherenceEvaluator`
  - Implement `_evaluate_impl()` method
  - Compute cosine similarity between text and audio embeddings
  - Return score with thresholds:
    - >0.7: Strong adherence
    - 0.5-0.7: Moderate adherence
    - <0.5: Poor adherence
  - Generate explanation based on score

- [ ] Add genre/mood classification (optional enhancement)
  - Use CLAP zero-shot classification
  - Compare detected vs expected from prompt

- [x] Write tests (27 tests, target was 15+)
  - Test encoder loading and lazy initialization
  - Test text embedding extraction (single and batch)
  - Test audio embedding extraction with resampling
  - Test similarity computation
  - Test threshold classification
  - Test evaluator integration

---

#### 2.2 Musical Coherence (Dimension 2)
**Priority:** Medium (most complex, multiple sub-systems)
**Dependencies:** madmom, essentia, librosa
**Files:** `dimensions/musical_coherence/`

##### 2.2.1 Structure Detection (`structure.py`)
- [ ] Implement segment boundary detection
  - Use self-similarity matrix from MERT embeddings
  - Detect major structural transitions
- [ ] Create verse/chorus/bridge identifier
  - Use repetition detection
  - Label sections by function
- [ ] Add section timing extraction
  - Return list of (start, end, label) tuples
- [ ] Compute structure score
  - Reward clear section boundaries
  - Penalize lack of structure

##### 2.2.2 Harmonic Analysis (`harmony.py`)
- [ ] Wrap Essentia `ChordsDetection`
  - Extract chord sequence with timestamps
- [ ] Implement chord progression quality scoring
  - Analyze functional harmony
  - Detect common progressions vs random chords
- [ ] Add key detection (Krumhansl-Schmuckler)
  - Use Essentia or librosa
- [ ] Create key consistency metric
  - Measure tonal center stability over time
  - Penalize excessive modulation without resolution

##### 2.2.3 Rhythmic Analysis (`rhythm.py`)
- [ ] Wrap madmom `DBNBeatTracker`
  - Extract beat positions
  - Compute tempo (BPM)
- [ ] Implement tempo stability metric
  - Measure tempo drift over time
  - Penalize erratic tempo changes
- [ ] Add beat grid alignment scoring
  - Check if events align to beat grid
- [ ] Create rhythmic consistency score
  - Combine tempo stability + beat alignment

##### 2.2.4 Melodic Analysis (`melody.py`)
- [ ] Add pitch tracking (librosa pyin)
  - Extract fundamental frequency contour
- [ ] Implement melodic contour analysis
  - Detect phrase boundaries
  - Analyze contour shapes
- [ ] Create phrase completion scoring
  - Check for melodic resolution
  - Penalize hanging/unresolved phrases

##### 2.2.5 Integration (`evaluator.py`)
- [ ] Create `MusicalCoherenceEvaluator`
  - Combine structure, harmony, rhythm, melody scores
  - Use configurable weights (default: equal)
  - Return sub_scores for each component
- [ ] Write tests (target: 25+ tests)

---

#### 2.3 Audio Quality (Dimension 3)
**Priority:** High (Essentia integrations)
**Dependencies:** essentia, librosa
**Files:** `dimensions/audio_quality/`

##### 2.3.1 Artifact Detection (`artifacts.py`)
- [ ] Wrap Essentia `ClickDetector`
  - Detect transient artifacts
  - Return artifact count and timestamps
- [ ] Implement AI artifact detection
  - Spectral peak analysis for generative fingerprints
  - Detect unnatural spectral patterns
- [ ] Add clipping detection
  - Check for samples at maximum amplitude
- [ ] Create artifact location timestamps
  - Mark where artifacts occur for debugging

##### 2.3.2 Loudness Analysis (`loudness.py`)
- [ ] Implement LUFS measurement
  - Target: -14 LUFS for streaming
  - Use ITU-R BS.1770-4 algorithm
- [ ] Add Loudness Range (LRA) calculation
  - Measure dynamic range
- [ ] Implement True Peak measurement
  - Target: <-1 dBTP
  - Check for intersample peaks
- [ ] Create dynamic range scoring
  - Reward appropriate dynamics
  - Penalize over-compression or excessive dynamics

##### 2.3.3 Perceptual Quality (`perceptual.py`)
- [ ] Add spectral centroid/flatness metrics
  - Measure tonal vs noisy content
- [ ] Implement frequency balance analysis
  - Check bass/mid/treble balance
- [ ] Create overall audio quality score
  - Combine all perceptual metrics

##### 2.3.4 Integration (`evaluator.py`)
- [ ] Create `AudioQualityEvaluator`
  - Combine artifacts, loudness, perceptual scores
  - Add pass/fail thresholds for streaming compliance
  - Return detailed breakdown in metadata
- [ ] Write tests (target: 20+ tests)

---

#### 2.4 Musicality (Dimension 4)
**Priority:** Medium (depends on coherence features)
**Dependencies:** Chord detection from 2.2.2
**Files:** `dimensions/musicality/`

##### 2.4.1 Tonal Interval Space (`tis.py`)
- [ ] Implement TIS computation from chord sequences
  - Map chords to pitch class sets
  - Compute cloud centroids in TIS
- [ ] Calculate cloud diameter (harmonic clarity)
  - Smaller = more consonant
- [ ] Calculate cloud momentum (rate of harmonic change)
  - Rate of movement through TIS
- [ ] Calculate tensile strain (deviation from tonal center)
  - Distance from reference key center

##### 2.4.2 Tension-Resolution (`tension.py`)
- [ ] Create tension curve computation
  - Combine TIS metrics over time
- [ ] Implement resolution detection at cadences
  - Detect tension drops at phrase endings
- [ ] Add tension arc quality scoring
  - Reward proper build-up and resolution
  - Penalize flat or unresolved tension

##### 2.4.3 Expressive Features (`expression.py`)
- [ ] Implement dynamic variation analysis
  - Measure loudness changes over time
- [ ] Add micro-timing/groove analysis (optional)
  - Detect swing/groove patterns
- [ ] Create genre authenticity scoring (optional)
  - Compare to genre-specific corpus

##### 2.4.4 Integration (`evaluator.py`)
- [ ] Create `MusicalityEvaluator`
  - Combine TIS, tension, expression scores
  - Weight based on genre (optional)
- [ ] Write tests (target: 20+ tests)

---

### Phase 3: Safety Dimensions

#### 3.1 Copyright & Originality (Dimension 5)
**Priority:** High (key safety addition)
**Dependencies:** pyacoustid (Chromaprint), MERT
**Files:** `dimensions/copyright/`

- [ ] Integrate Chromaprint
  - Create `ChromaprintEncoder` wrapper
  - Generate fingerprints from audio

- [ ] Create fingerprint database schema
  - Store fingerprints with metadata
  - Support similarity search

- [ ] Implement exact match detection
  - Query fingerprint database
  - Return matches with confidence

- [ ] Add MERT-based semantic similarity
  - Compare embeddings to reference corpus
  - Detect melodic/rhythmic similarity

- [ ] Create melody similarity detection
  - Focus on pitch + rhythm combination
  - Harmony alone insufficient (many songs share progressions)

- [ ] Distinguish style vs copying
  - "Sounds like" vs "copies from"
  - Quantify stylistic similarity separately

- [ ] Add evidence generation
  - Provide matched sections/timestamps
  - Include similarity scores for review

- [ ] Create `CopyrightEvaluator`
  - Return ALLOW/FLAG/BLOCK decision
  - Include evidence in metadata

- [ ] Write tests (target: 20+ tests)

---

#### 3.2 Voice Cloning Detection (Dimension 6)
**Priority:** Low (mostly migration)
**Dependencies:** Existing voice_database.py
**Files:** `dimensions/voice_cloning/`

- [ ] Migrate `VoiceDatabase` class
  - Copy from `output_classifier/voice_database.py`
  - Adapt interface for dimension system

- [ ] Migrate `VoiceEnroller` class
  - Copy from `output_classifier/voice_enrollment.py`
  - Maintain quality checks

- [ ] Add protected voice database option
  - Pre-loaded politicians/celebrities (optional)

- [ ] Improve similarity threshold configuration
  - Per-voice configurable thresholds
  - Different thresholds for different use cases

- [ ] Add deepfake confidence scoring
  - Distinguish natural vs synthetic voice
  - Use spectral analysis

- [ ] Create `VoiceCloningEvaluator`
  - Wrap existing functionality
  - Return ALLOW/FLAG/BLOCK decision

- [ ] Write integration tests (target: 10+ new tests)

---

#### 3.3 Cultural Sensitivity (Dimension 7)
**Priority:** Low (flagging only)
**Dependencies:** MERT embeddings
**Files:** `dimensions/cultural_sensitivity/`

- [ ] Define cultural sensitivity taxonomy
  - Categories: sacred, ceremonial, indigenous, etc.
  - Reference materials for each category

- [ ] Create sacred/ceremonial music detector
  - Train classifier on labeled examples
  - Or use zero-shot with CLAP

- [ ] Add cultural context flagging
  - Flag for human review (not automatic blocking)
  - Include context in explanation

- [ ] Implement stereotyping pattern detection (optional)
  - Detect potentially offensive stereotypes
  - Requires careful definition and review

- [ ] Create `CulturalSensitivityEvaluator`
  - NOTE: Flagging only, not adjudication
  - Always requires human review

- [ ] Write tests (target: 15+ tests)

---

#### 3.4 Content Safety (Dimension 8)
**Priority:** Low (mostly migration)
**Dependencies:** Existing harm classifier
**Files:** `dimensions/content_safety/`

- [ ] Migrate existing harm classification
  - Copy from `output_classifier/`
  - Adapt for dimension system

- [ ] Add ASR integration for lyrics (optional)
  - Extract lyrics via speech recognition
  - Pass to NLP pipeline

- [ ] Implement hate speech/slur detection
  - Use ASR + NLP pipeline
  - Flag for review

- [ ] Add harmful instruction detection
  - Detect audio describing dangerous activities

- [ ] Create physical safety checks
  - Harmful frequencies detection
  - Volume spike detection
  - Epileptogenic pattern detection (optional)

- [ ] Create `ContentSafetyEvaluator`
  - Return ALLOW/FLAG/BLOCK decision
  - Include detected issues in metadata

- [ ] Write integration tests (target: 10+ new tests)

---

### Phase 4: Unified Scoring & API

#### 4.1 Quality Score Aggregation
**File:** `src/decrescendo/musicritic/scoring.py`

- [ ] Create `QualityScorer` class
  - Take results from dimensions 1-4
  - Compute weighted average (0-100 scale)
  - Use configurable weights per dimension

- [ ] Add genre-specific weighting profiles
  - Different weights for different genres
  - E.g., rhythm more important for EDM

- [ ] Create confidence intervals
  - Aggregate confidence from dimensions
  - Report overall confidence

- [ ] Add sub-score breakdown
  - Include all dimension scores in result

---

#### 4.2 Safety Decision Aggregation
**File:** `src/decrescendo/musicritic/scoring.py`

- [ ] Create `SafetyScorer` class
  - Take results from dimensions 5-8
  - Apply threshold-based decision logic

- [ ] Implement decision logic
  - BLOCK if any dimension > block_threshold
  - FLAG if any dimension > flag_threshold
  - ALLOW otherwise

- [ ] Add per-dimension evidence
  - Include reasons for each flag/block

- [ ] Create configurable threshold overrides
  - Allow custom thresholds per dimension

---

#### 4.3 Unified Evaluation Pipeline
**File:** `src/decrescendo/musicritic/pipeline.py`

- [ ] Create `MusiCritic` class (extend existing)
  - Add `evaluate(audio, prompt=None)` method
  - Return `EvaluationResult` with quality + safety

- [ ] Add dimension selection
  - `evaluate_quality()` for dimensions 1-4 only
  - `evaluate_safety()` for dimensions 5-8 only
  - `evaluate()` for all dimensions

- [ ] Create streaming-friendly mode
  - Safety-only, faster processing
  - Target: <500ms latency

- [ ] Add batch evaluation support
  - Process multiple audio files
  - Parallel dimension evaluation

- [ ] Add embedding caching
  - Cache MERT/CLAP embeddings
  - TTL-based expiration

---

#### 4.4 Updated CLI
**File:** `src/decrescendo/musicritic/cli.py`

- [ ] Update `evaluate` command
  - Full 8-dimension analysis
  - JSON/table/text output

- [ ] Add `evaluate-quality` command
  - Dimensions 1-4 only
  - Faster than full evaluation

- [ ] Add `evaluate-safety` command
  - Dimensions 5-8 only
  - Faster than full evaluation

- [ ] Keep existing voice management commands
  - `enroll-voice`, `list-voices`

- [ ] Update output formatters
  - Add quality score display
  - Add safety decision display
  - Show dimension breakdown

- [ ] Add progress indicators
  - Show which dimension is being evaluated
  - Useful for long evaluations

---

#### 4.5 API Response Schema
**File:** `src/decrescendo/musicritic/api.py`

- [ ] Define unified evaluation response
  ```json
  {
    "quality_score": 78.5,
    "safety_decision": "ALLOW",
    "confidence": 0.92,
    "quality_dimensions": {...},
    "safety_dimensions": {...},
    "processing_time_ms": 1450,
    "explanation": "..."
  }
  ```

- [ ] Update FastAPI endpoints
  - `POST /v1/evaluate` for full evaluation
  - `POST /v1/evaluate/quality` for quality only
  - `POST /v1/evaluate/safety` for safety only

- [ ] Add human-readable explanation generator
  - Summarize results in natural language

---

### Phase 5: Validation & Testing

#### 5.1 Benchmark Integration
- [ ] Download AIME dataset (6,500 tracks)
- [ ] Create AIME-compatible evaluation protocol
- [ ] Implement pairwise comparison evaluation
- [ ] Add MusicCaps validation for prompt adherence

#### 5.2 Correlation Validation
- [ ] Create human correlation measurement tools
- [ ] Target: >0.75 Spearman correlation
- [ ] Add statistical significance testing

#### 5.3 Performance Optimization
- [ ] Profile full pipeline latency
- [ ] Target: <2s for 30-second audio
- [ ] Implement parallel encoder execution
- [ ] Add FP16 inference option
- [ ] Create embedding cache with TTL

#### 5.4 Test Coverage
- [ ] Maintain 253 existing tests
- [ ] Target: 150+ new tests for quality dimensions
- [ ] Target: 50+ new tests for safety extensions
- [ ] Add integration tests for full pipeline
- [ ] Add performance benchmarks

---

## Priority Order

| Priority | Task Group | Rationale |
|----------|------------|-----------|
| 1 | Prompt Adherence (2.1) | Quick win, CLAP is straightforward |
| 2 | Audio Quality (2.3) | Essentia integrations, useful standalone |
| 3 | Musical Coherence (2.2) | Most complex, foundational for musicality |
| 4 | Musicality (2.4) | Depends on coherence features |
| 5 | Copyright (3.1) | Key safety addition |
| 6 | Voice Cloning (3.2) | Mostly migration |
| 7 | Content Safety (3.4) | Mostly migration |
| 8 | Cultural Sensitivity (3.3) | Flagging only, lower priority |
| 9 | Unified API (4.x) | Ties everything together |
| 10 | Validation (5.x) | Final benchmarking |

---

## Success Criteria

- [ ] All 8 dimensions implemented and tested
- [ ] Quality score (0-100) operational
- [ ] Safety decision (ALLOW/FLAG/BLOCK) operational
- [ ] Processing latency <2s for 30-second audio (GPU)
- [ ] Test coverage: 400+ tests total
- [ ] CLI commands: `evaluate`, `evaluate-quality`, `evaluate-safety`
- [ ] AIME benchmark compatibility
- [ ] Human correlation target: >0.75

---

## Notes for Agents

### Getting Started
```bash
# Clone and setup
cd decrescendo
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Import the package
from decrescendo.musicritic import MusiCritic, load_musicritic
from decrescendo.musicritic.dimensions import (
    DimensionEvaluator,
    DimensionResult,
    QualityDimension,
    SafetyDimension,
)
```

### Creating a New Dimension Evaluator
```python
from decrescendo.musicritic.dimensions.base import (
    BaseDimensionEvaluator,
    DimensionCategory,
    DimensionResult,
    QualityDimension,
)

class MyEvaluator(BaseDimensionEvaluator):
    dimension = QualityDimension.PROMPT_ADHERENCE
    category = DimensionCategory.QUALITY

    def _evaluate_impl(self, audio, sample_rate, prompt=None, **kwargs):
        # Your evaluation logic here
        score = 0.75  # 0.0-1.0
        return DimensionResult(
            dimension=self.dimension,
            score=score,
            confidence=0.9,
            explanation="Evaluation explanation",
            sub_scores={"component1": 0.8, "component2": 0.7},
        )
```

### Key Files to Reference
- `docs/PRD.md` - Product requirements and success metrics
- `docs/Architecture.md` - Technical architecture and model choices
- `src/decrescendo/musicritic/dimensions/base.py` - Base classes and protocols
- `tests/test_dimensions_base.py` - Example tests for dimensions

### Conventions
- Use JAX/Flax for neural network components
- Use dataclasses for configuration (frozen=True)
- Scores are always 0.0-1.0 (scaled to 0-100 for display)
- Safety decisions are ALLOW/FLAG/BLOCK
- All evaluators must handle mono audio (stereo is converted in base class)
- Cache embeddings when possible for efficiency
