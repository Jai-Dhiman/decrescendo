# System Architecture Document

## MusiCritic

**Author:** Jai Dhiman
**Version:** 2.0
**Last Updated:** December 2024

---

## Architecture Overview

MusiCritic is a unified evaluation framework for AI-generated music that combines **quality assessment** and **safety evaluation** in a single pipeline. It evaluates audio across 8 dimensions (4 quality + 4 safety) using shared infrastructure for efficient processing.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MUSICRITIC UNIFIED PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Audio Input + Text Prompt                                                  │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PREPROCESSING & EMBEDDING                         │   │
│  │  (Resampling, Chunking, MERT-v1-95M, CLAP, WavLM, madmom, Essentia) │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ├─────────────────────────────────────────────┐                    │
│         ▼                                             ▼                    │
│  ┌─────────────────────────────┐       ┌─────────────────────────────────┐ │
│  │    QUALITY DIMENSIONS (4)   │       │     SAFETY DIMENSIONS (4)       │ │
│  ├─────────────────────────────┤       ├─────────────────────────────────┤ │
│  │ 1. Prompt Adherence (CLAP)  │       │ 5. Copyright & Originality      │ │
│  │ 2. Musical Coherence        │       │ 6. Voice Cloning Detection      │ │
│  │ 3. Audio Quality            │       │ 7. Cultural Sensitivity         │ │
│  │ 4. Musicality (TIS)         │       │ 8. Content Safety               │ │
│  └─────────────────────────────┘       └─────────────────────────────────┘ │
│         │                                             │                    │
│         └─────────────────────────────────────────────┘                    │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      UNIFIED SCORING MODULE                          │   │
│  │  Quality Score (0-100) + Safety Decision (ALLOW/FLAG/BLOCK)         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                 │                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SERVING LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  REST API │ Python SDK │ Hugging Face Integration │ Webhook Callbacks       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack Summary

### Core Dependencies

| Component | Technology | Purpose | License |
|-----------|------------|---------|---------|
| **Primary Audio Encoder** | MERT-v1-95M | Music understanding embeddings (smaller for latency) | CC-BY-NC-4.0 |
| **Text-Audio Alignment** | CLAP (laion/larger_clap_music) | Prompt adherence scoring | Apache 2.0 |
| **Beat/Tempo Analysis** | madmom | Beat tracking (95%+ accuracy), tempo detection | BSD-3-Clause |
| **Chord/Key Detection** | Essentia | ChordsDetection, Key estimation, ClickDetector | AGPL-3.0 |
| **Melody Analysis** | CREPE / pYIN | Pitch tracking, melody extraction | MIT / GPL |
| **FAD Calculation** | fadtk | Frechet Audio Distance computation | MIT |
| **Speaker Analysis** | WavLM-Large | Voice/speaker embeddings (voice cloning detection) | MIT |
| **Speaker Verification** | ECAPA-TDNN | Voice fingerprinting | Apache 2.0 |
| **Audio Fingerprinting** | Chromaprint | Content identification, originality checking | LGPL 2.1 |
| **Vector Database** | FAISS / Qdrant | Embedding storage & search | MIT / Apache 2.0 |
| **ML Framework** | JAX/Flax | Model training & inference | Apache 2.0 |
| **Experiment Tracking** | Weights & Biases | Training monitoring | Commercial |
| **Serving** | Modal / FastAPI | API deployment | Various |

**License Restrictions (Important):**
- **MERT-v1-95M**: CC-BY-NC-4.0 - Non-commercial use only
- **Essentia**: AGPL-3.0 - Copyleft, source code disclosure required for derivative works
- All other core dependencies are Apache 2.0 / MIT / BSD compatible

### Development Environment

| Tool | Purpose |
|------|---------|
| Python 3.11+ | Primary language |
| JAX/Flax | Primary ML framework (consistent with Constitutional Audio) |
| torchaudio | Audio I/O, resampling (PyTorch-based libraries via interop) |
| Docker | Containerization |
| pytest | Testing |
| ruff | Linting |
| pre-commit | Code quality |

**Note:** Some libraries (madmom, Essentia, CREPE) are PyTorch/NumPy-based. JAX interoperability is handled via NumPy arrays at boundaries.

---

## Shared Infrastructure

### Audio Preprocessing Pipeline

All audio inputs pass through a standardized preprocessing pipeline before embedding extraction:

```
Raw Audio File
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                  PREPROCESSING PIPELINE                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Format Detection                                     │
│     └─→ Supported: WAV, MP3, FLAC, OGG, M4A             │
│                                                          │
│  2. Resampling                                           │
│     └─→ Target: 24kHz (MERT), 48kHz (CLAP)              │
│     └─→ Library: torchaudio.transforms.Resample         │
│                                                          │
│  3. Channel Handling                                     │
│     └─→ Stereo → Mono (mean of channels)                │
│     └─→ Preserve stereo flag for downstream             │
│                                                          │
│  4. Normalization                                        │
│     └─→ Peak normalization to [-1, 1]                   │
│     └─→ Optional: RMS normalization                     │
│                                                          │
│  5. Chunking                                             │
│     └─→ Window: 5 seconds                               │
│     └─→ Hop: 2.5 seconds (50% overlap)                  │
│     └─→ Padding: Zero-pad final chunk                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
     │
     ▼
Preprocessed Chunks [N × 120000 samples @ 24kHz]
```

### Embedding Extraction Layer

Multiple encoders and analyzers run to extract complementary representations for AI music evaluation:

```
Preprocessed Chunks
     │
     ├──────────────────┬──────────────────┬──────────────────┐
     ▼                  ▼                  ▼                  ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   MERT      │  │   CLAP      │  │   madmom    │  │  Essentia   │
│   v1-95M    │  │ laion-music │  │             │  │             │
├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤
│ 95M params  │  │ Audio enc.  │  │ Beat track  │  │ Chords/Key  │
│ 1024-dim    │  │ 1024-dim    │  │ 95%+ acc    │  │ Click det.  │
│ 75 Hz       │  │ Pooled      │  │             │  │ HPCP        │
│             │  │             │  │             │  │             │
│ Music       │  │ Prompt      │  │ Rhythm      │  │ Harmonic    │
│ Features    │  │ Adherence   │  │ Features    │  │ Features    │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
     │                  │                  │                  │
     └──────────────────┴──────────────────┴──────────────────┘
                                 │
                                 ▼
                    Combined Feature Tensor
                    [N × T × D] per encoder
```

**Additional Encoders (Safety Dimensions):**
- WavLM-Large: Speaker/voice embeddings for voice cloning detection
- ECAPA-TDNN: Voice fingerprinting for artist protection

**Encoder Selection by Task:**

| Task | Primary Encoder | Secondary Tools | Rationale |
|------|-----------------|-----------------|-----------|
| Prompt adherence | CLAP (laion-music) | — | Text-audio alignment |
| Musical coherence | MERT-v1-95M | madmom, Essentia | Structure, harmony, rhythm |
| Audio quality | MERT-v1-95M | Essentia ClickDetector | Artifact detection |
| Musicality | MERT-v1-95M | TIS computation | Tension-resolution |
| Originality | MERT-v1-95M | Chromaprint | Fingerprinting + similarity |
| Voice detection | WavLM | ECAPA-TDNN | Speaker-optimized (Safety Dim. 6) |

### Vector Storage Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      VECTOR DATABASE (FAISS/Qdrant)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Collections:                                                    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  artist_fingerprints                                     │    │
│  │  ├─ Vectors: ECAPA-TDNN embeddings (192-dim)            │    │
│  │  ├─ Metadata: artist_id, consent_status, created_at     │    │
│  │  └─ Index: IVF4096,PQ32 (for 1M+ vectors)               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  reference_corpus                                        │    │
│  │  ├─ Vectors: MERT embeddings (1024-dim)                 │    │
│  │  ├─ Metadata: genre, era, quality_label                 │    │
│  │  └─ Index: HNSW (for fast approximate search)           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  audio_fingerprints                                      │    │
│  │  ├─ Vectors: Chromaprint hashes                         │    │
│  │  ├─ Metadata: source_id, rights_holder                  │    │
│  │  └─ Index: Exact match (LSH)                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Unified Evaluation Pipeline

### System Design

MusiCritic evaluates AI-generated music across **8 dimensions**: 4 quality dimensions and 4 safety dimensions.

```
Audio Input + Text Prompt
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MUSICRITIC UNIFIED PIPELINE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              FEATURE EXTRACTION LAYER                     │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │                                                           │   │
│  │  MERT Branch (Music Understanding):                       │   │
│  │  └─→ Frame embeddings → Layer selection → 1024-dim       │   │
│  │                                                           │   │
│  │  CLAP Branch (Prompt Adherence):                          │   │
│  │  └─→ Audio encoder + Text encoder → Cosine similarity    │   │
│  │                                                           │   │
│  │  Beat/Tempo Branch (Rhythm):                              │   │
│  │  └─→ madmom DBNBeatTracker → Beat times + Tempo          │   │
│  │                                                           │   │
│  │  Harmonic Branch (Chords/Key):                            │   │
│  │  └─→ Essentia ChordsDetection → TIS computation          │   │
│  │                                                           │   │
│  │  Quality Branch (Artifacts):                              │   │
│  │  └─→ Essentia ClickDetector + Spectral analysis          │   │
│  │                                                           │   │
│  │  Copyright Branch (Fingerprinting):                       │   │
│  │  └─→ Chromaprint → Copyright database similarity search  │   │
│  │                                                           │   │
│  │  Voice Branch (Speaker Verification):                     │   │
│  │  └─→ WavLM → ECAPA-TDNN → Protected voice matching       │   │
│  │                                                           │   │
│  │  Content Branch (Safety):                                 │   │
│  │  └─→ ASR + NLP → Harm category classification            │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           HIERARCHICAL TEMPORAL AGGREGATION               │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │                                                           │   │
│  │  Frame (10ms) ──BiLSTM──→ Beat (500ms)                   │   │
│  │       │                        │                          │   │
│  │       └─── Attention ──────────┘                          │   │
│  │                                │                          │   │
│  │  Beat (500ms) ──BiLSTM──→ Phrase (4-16 bars)             │   │
│  │       │                        │                          │   │
│  │       └─── Attention ──────────┘                          │   │
│  │                                │                          │   │
│  │  Phrase ──BiLSTM──→ Section ──BiLSTM──→ Piece            │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         MULTI-TASK PREDICTION HEADS (8 Dimensions)        │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │                                                           │   │
│  │  Shared Trunk (cross-attention fusion)                    │   │
│  │       │                                                   │   │
│  │       │  QUALITY HEADS (Dimensions 1-4):                  │   │
│  │       ├──→ Prompt Adherence ──→ Score + Confidence       │   │
│  │       ├──→ Musical Coherence ──→ Score + Confidence      │   │
│  │       ├──→ Audio Quality ──→ Score + Confidence          │   │
│  │       ├──→ Musicality ──→ Score + Confidence             │   │
│  │       │                                                   │   │
│  │       │  SAFETY HEADS (Dimensions 5-8):                   │   │
│  │       ├──→ Copyright/Originality ──→ Score + Flag        │   │
│  │       ├──→ Voice Cloning ──→ Score + Flag                │   │
│  │       ├──→ Cultural Sensitivity ──→ Score + Flag         │   │
│  │       ├──→ Content Safety ──→ Score + Flag               │   │
│  │       │                                                   │   │
│  │       │  AGGREGATE OUTPUTS:                               │   │
│  │       ├──→ Overall Quality Score (0-100)                 │   │
│  │       └──→ Safety Decision (ALLOW / FLAG / BLOCK)        │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
Unified Evaluation Output (Quality Score + Safety Decision)
```

### Model Architecture Details

**Feature Fusion Module:**

The system fuses multiple complementary feature streams using cross-attention:

**Quality Streams:**

| Stream | Source | Dimension | Information Captured |
|--------|--------|-----------|---------------------|
| Acoustic | MERT-v1-95M layers 18-24 | 1024 | Timbre, rhythm, pitch |
| Text-Audio | CLAP (laion-music) | 1024 | Prompt adherence |
| Rhythm | madmom DBNBeatTracker | Variable | Beat positions, tempo |
| Harmonic | Essentia + TIS | Variable | Chords, key, tension |
| Quality | Essentia ClickDetector | Variable | Artifacts, clicks |

**Safety Streams:**

| Stream | Source | Dimension | Information Captured |
|--------|--------|-----------|---------------------|
| Copyright | Chromaprint + MERT | 1024 | Melody/sample similarity |
| Speaker | WavLM + ECAPA-TDNN | 192 | Voice identity |
| Content | MERT + ASR + NLP | Variable | Harm categories |

Fusion uses learned cross-attention where acoustic features attend to text-audio alignment features for quality, and speaker features attend to the protected voice database for safety.

**Temporal Aggregation:**

```
Frame Features (75 Hz from MERT)
        │
        ▼
┌───────────────────────────────────────┐
│  BiLSTM Layer 1 (hidden=512)          │
│  + Multi-Head Attention (8 heads)     │
│  + Beat-aligned pooling               │
└───────────────────────────────────────┘
        │
        ▼ Beat Features (~2 Hz)
┌───────────────────────────────────────┐
│  BiLSTM Layer 2 (hidden=256)          │
│  + Relative Position Attention        │
│  + Phrase boundary detection          │
└───────────────────────────────────────┘
        │
        ▼ Phrase Features (~0.1 Hz)
┌───────────────────────────────────────┐
│  BiLSTM Layer 3 (hidden=128)          │
│  + Section-level attention            │
└───────────────────────────────────────┘
        │
        ▼ Piece-level Representation (1 × 128)
```

**Prediction Heads:**

Each musicality dimension has a dedicated head with shared lower layers:

- **Architecture:** 2-layer MLP with residual connection
- **Output:** Score (0-100) + confidence (0-1)
- **Loss:** Combination of MSE (regression) and pairwise ranking loss
- **Regularization:** Dropout (0.3) + label smoothing

**Training Objective:**

```
L_total = λ₁ · L_regression + λ₂ · L_ranking + λ₃ · L_consistency + λ₄ · L_clap + λ₅ · L_triplet

Where:
- L_regression: MSE between predicted and annotated scores
- L_ranking: Pairwise margin ranking loss (AIME dataset compatible)
- L_consistency: Correlation loss ensuring dimension scores are coherent
- L_clap: CLAP cosine similarity loss for prompt adherence
- L_triplet: Triplet loss for originality (melody/rhythm plagiarism detection)
```

### Tension-Resolution Module

Implements Tonal Interval Space (TIS) analysis:

```
Audio → Chord Detection (madmom) → Chord Sequence
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │   TIS COMPUTATION   │
                              ├─────────────────────┤
                              │                     │
                              │  For each chord:    │
                              │  1. Map to pitch    │
                              │     class set       │
                              │  2. Compute cloud   │
                              │     centroid        │
                              │  3. Compute cloud   │
                              │     diameter        │
                              │                     │
                              │  Across sequence:   │
                              │  1. Cloud momentum  │
                              │  2. Tensile strain  │
                              │  3. Resolution      │
                              │     strength        │
                              │                     │
                              └─────────────────────┘
                                        │
                                        ▼
                              Tension Curve [T × 3]
```

**TIS Features:**

| Feature | Description | Range |
|---------|-------------|-------|
| Cloud Diameter | Harmonic complexity/dissonance | 0-1 |
| Cloud Momentum | Rate of harmonic change | 0-1 |
| Tensile Strain | Distance from tonal center | 0-1 |

### Prompt Adherence Module

Measures how well AI-generated audio matches the input text prompt using CLAP embeddings.

```
Text Prompt                           Audio Input
     │                                      │
     ▼                                      ▼
┌─────────────────────┐           ┌─────────────────────┐
│  CLAP Text Encoder  │           │  CLAP Audio Encoder │
│  (laion-music)      │           │  (laion-music)      │
└─────────────────────┘           └─────────────────────┘
     │                                      │
     ▼                                      ▼
   Text Embedding (1024-dim)        Audio Embedding (1024-dim)
     │                                      │
     └──────────────┬───────────────────────┘
                    │
                    ▼
            Cosine Similarity
                    │
                    ▼
         Prompt Adherence Score (0-1)
```

**Thresholds:**
- >0.7: Strong prompt adherence
- 0.5-0.7: Moderate adherence
- <0.5: Poor adherence (flag for review)

### AI Artifact Detection Module

Detects artifacts specific to AI-generated audio using spectral analysis and dedicated detectors.

```
Audio Input
     │
     ├──────────────────┬──────────────────┐
     ▼                  ▼                  ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Spectral   │  │  Essentia   │  │  Loudness   │
│  Analysis   │  │ ClickDetect │  │  Analysis   │
├─────────────┤  ├─────────────┤  ├─────────────┤
│ Peak detect │  │ Transient   │  │ LUFS meas.  │
│ for AI      │  │ artifact    │  │ LRA calc.   │
│ fingerprint │  │ detection   │  │ True Peak   │
└─────────────┘  └─────────────┘  └─────────────┘
     │                  │                  │
     └──────────────────┴──────────────────┘
                        │
                        ▼
              Audio Quality Score
```

**Quality Targets:**
- Loudness: -14 LUFS (streaming standard)
- True Peak: <-1 dBTP
- Artifacts: 0 detected (ideal)

### Originality Detection Module

Detects potential plagiarism or excessive similarity to existing music using fingerprinting and embedding similarity.

```
Audio Input
     │
     ├──────────────────┬──────────────────┐
     ▼                  ▼                  ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Chromaprint │  │ MERT        │  │  Melody     │
│ Fingerprint │  │ Embeddings  │  │  Extraction │
├─────────────┤  ├─────────────┤  ├─────────────┤
│ Exact match │  │ Semantic    │  │ Pitch track │
│ database    │  │ similarity  │  │ + contour   │
│ lookup      │  │ search      │  │ comparison  │
└─────────────┘  └─────────────┘  └─────────────┘
     │                  │                  │
     └──────────────────┴──────────────────┘
                        │
                        ▼
              Originality Score + Matches
```

**Key Insight:** Melody + rhythm combination is prioritized (harmony alone insufficient for plagiarism detection since many songs share common progressions).

### Performance Budget

Target: **<2 seconds** for 30-second audio clip.

| Component | Time (GPU) | Time (CPU) | Notes |
|-----------|------------|------------|-------|
| MERT-v1-95M | 600ms | 1.5s | 6 chunks x 100ms |
| CLAP | 240ms | 600ms | 3 chunks x 80ms |
| madmom | 150ms | 200ms | Beat tracking |
| Essentia | 200ms | 500ms | Chords, key, clicks |
| Aggregation | 50ms | 100ms | Feature fusion |
| **Total** | **~1.2s** | **~2.9s** | GPU recommended |

**Optimization Strategies:**
- FP16 inference: 50% memory reduction, ~1.5-2x speedup
- Batch processing: Group multiple clips for GPU efficiency
- Embedding caching: Redis with 7-day TTL for repeated evaluations
- Parallel extraction: Run MERT, CLAP, madmom, Essentia concurrently

---

## Safety Dimensions Architecture

The safety dimensions (5-8) use specialized modules for detecting harmful content in AI-generated music.

### Safety Detection Modules

| Dimension | Detection Approach | Model |
|-----------|-------------------|-------|
| Copyright/Originality | Fingerprint matching + embedding similarity | Chromaprint + MERT |
| Voice Cloning | Speaker verification against protected database | WavLM + ECAPA-TDNN |
| Cultural Sensitivity | Pattern matching + flagging (human review required) | Fine-tuned classifier |
| Content Safety | Hate/harmful content via ASR + NLP | Audio + ASR + NLP pipeline |

### Voice Cloning Detection Module

```
Audio Input
     │
     ▼
┌─────────────────────┐
│  SPEAKER ANALYZER   │
├─────────────────────┤
│                     │
│  WavLM → ECAPA-TDNN │
│        │            │
│        ▼            │
│  Speaker Embedding  │
│  (192-dim)          │
│        │            │
│        ▼            │
│  FAISS Similarity   │
│  Search             │
│        │            │
│        ▼            │
│  Protected Voice    │
│  Match Score        │
│                     │
└─────────────────────┘
     │
     ▼
Decision: ALLOW / FLAG / BLOCK
```

### Safety Threshold Configuration

| Threshold | Action | Use Case |
|-----------|--------|----------|
| p > 0.95 | BLOCK | High-confidence safety violation |
| 0.7 < p < 0.95 | FLAG | Requires human review |
| p < 0.7 | ALLOW | No safety concerns detected |

**Artist Protection System:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARTIST PROTECTION SYSTEM                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Registration Flow:                                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Artist uploads voice samples (5-10 minutes)            │    │
│  │       │                                                  │    │
│  │       ▼                                                  │    │
│  │  Extract ECAPA-TDNN embeddings                          │    │
│  │       │                                                  │    │
│  │       ▼                                                  │    │
│  │  Generate centroid + variance                           │    │
│  │       │                                                  │    │
│  │       ▼                                                  │    │
│  │  Store in protected_voices collection                   │    │
│  │       │                                                  │    │
│  │       ▼                                                  │    │
│  │  Set similarity threshold (default: 0.85 cosine)        │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Detection Flow:                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Incoming audio                                          │    │
│  │       │                                                  │    │
│  │       ▼                                                  │    │
│  │  Extract speaker embedding                              │    │
│  │       │                                                  │    │
│  │       ▼                                                  │    │
│  │  FAISS approximate nearest neighbor search              │    │
│  │       │                                                  │    │
│  │       ▼                                                  │    │
│  │  If similarity > threshold:                             │    │
│  │       │                                                  │    │
│  │       ├──→ Flag for review                              │    │
│  │       ├──→ Log match details                            │    │
│  │       └──→ Trigger notification workflow                │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Safety Training Strategy

**Training the Safety Classifiers (Following Anthropic's Constitutional AI Approach):**

1. **Generate Synthetic Training Data**
   - Define safety constitution (natural language rules)
   - Generate adversarial prompts covering edge cases
   - Create safe variations for contrastive learning
   - Label with safety categories

2. **Train Safety Heads**
   - Fine-tune safety prediction heads on MERT features
   - Multi-label binary cross-entropy loss
   - Optimize for high recall (minimize false negatives)
   - Calibrate thresholds on validation set

3. **Adversarial Red-Teaming**
   - Continuous testing with jailbreak prompts
   - Evasion technique detection
   - Attack success rate monitoring (<2% target)

---

## Deployment Architecture

### Inference Serving

```
┌─────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT OPTIONS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Option 1: Modal (Serverless GPU)                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Auto-scaling based on demand                         │    │
│  │  • Cold start: ~10-15 seconds                           │    │
│  │  • GPU: A10G / A100                                     │    │
│  │  • Cost: Pay per second of compute                      │    │
│  │  │                                                       │    │
│  │  modal deploy musicritic/serve.py                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Option 2: Self-Hosted (Docker + K8s)                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Full control over infrastructure                     │    │
│  │  • Persistent GPU allocation                            │    │
│  │  • Lower latency (no cold start)                        │    │
│  │  │                                                       │    │
│  │  docker-compose up -d                                   │    │
│  │  kubectl apply -f k8s/                                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Option 3: Hugging Face Inference Endpoints                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Managed hosting                                      │    │
│  │  • Automatic scaling                                    │    │
│  │  • Easy integration with HF ecosystem                   │    │
│  │  │                                                       │    │
│  │  huggingface-cli endpoint create                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### API Design

**RESTful Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/evaluate` | POST | Full unified evaluation (quality + safety) |
| `/v1/evaluate/quality` | POST | Quality dimensions only (faster) |
| `/v1/evaluate/safety` | POST | Safety dimensions only (faster) |
| `/v1/compare` | POST | Pairwise comparison |
| `/v1/dimensions/{dim}` | POST | Single dimension score |
| `/v1/voices/register` | POST | Artist voice registration |
| `/v1/voices/check` | POST | Voice similarity check |
| `/health` | GET | Service health check |

**Request/Response Schema (Unified Evaluation):**

```
POST /v1/evaluate

Request:
{
  "audio": "<base64-encoded-audio>",
  "format": "wav",
  "sample_rate": 44100,
  "prompt": "An upbeat electronic dance track with synth leads",
  "dimensions": ["all"],  // or specific: ["prompt_adherence", "coherence", "copyright"]
  "return_temporal": false,
  "reference_genre": "electronic"  // optional
}

Response:
{
  "quality_score": 78.5,
  "safety_decision": "ALLOW",
  "confidence": 0.92,

  "quality_dimensions": {
    "prompt_adherence": {
      "score": 82.3,
      "confidence": 0.91,
      "clap_similarity": 0.73
    },
    "musical_coherence": {
      "score": 76.5,
      "confidence": 0.88,
      "structure": {
        "verse_chorus_detected": true,
        "sections": [{"type": "intro", "start": 0.0, "end": 8.2}]
      },
      "harmony": {"key": "C major", "chord_progression_quality": 0.81},
      "rhythm": {"tempo_bpm": 128, "stability": 0.94}
    },
    "audio_quality": {
      "score": 85.1,
      "confidence": 0.93,
      "artifacts_detected": 2,
      "loudness_lufs": -13.2,
      "true_peak_dbtp": -0.8
    },
    "musicality": {
      "score": 74.2,
      "confidence": 0.85,
      "tension_resolution": 0.72
    }
  },

  "safety_dimensions": {
    "copyright_originality": {
      "score": 0.12,
      "decision": "ALLOW",
      "fingerprint_matches": []
    },
    "voice_cloning": {
      "score": 0.05,
      "decision": "ALLOW",
      "matched_voices": []
    },
    "cultural_sensitivity": {
      "score": 0.08,
      "decision": "ALLOW",
      "flags": []
    },
    "content_safety": {
      "score": 0.03,
      "decision": "ALLOW",
      "detected_issues": []
    }
  },

  "explanation": "Good prompt adherence with clear electronic elements. Structure is coherent with proper verse-chorus form. Minor artifacts detected. No safety concerns.",
  "processing_time_ms": 1450
}
```

**Safety Decision Values:**
- `ALLOW`: No safety concerns detected (all safety scores < 0.7)
- `FLAG`: Requires human review (any safety score 0.7-0.95)
- `BLOCK`: High-confidence safety violation (any safety score > 0.95)

---

## Data Architecture

### Training Data Sources

**Quality Dimensions (1-4):**

| Dataset | Size | Use | License |
|---------|------|-----|---------|
| AIME | 6,500 tracks, 15,600 pairwise comparisons | Primary validation benchmark (AI-generated music) | CC BY-4.0 |
| MusicPrefs | 7 models, pairwise preferences | Secondary validation | Open-source |
| MTG-Jamendo | 55,000 tracks | Genre diversity, reference corpus | CC variants |
| MusicCaps | 5,521 clips | Text-audio alignment training | CC BY-SA 4.0 |
| MARBLE | 18 tasks, 12 datasets | Music understanding benchmarking | Various |

**Safety Dimensions (5-8):**

| Dataset | Size | Use | License |
|---------|------|-----|---------|
| VoxCeleb2 | 1M utterances | Speaker verification training | Research only |
| ASVspoof 2019 | 54,000 samples | Deepfake/voice clone detection | CC BY 4.0 |
| Custom adversarial | ~50,000 prompts | Safety classifier training | Proprietary |
| Artist registry | Growing | Protected voice matching | Artist consent |

### Annotation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    ANNOTATION PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Audio Selection                                              │
│     └─→ Stratified sampling across genres, quality levels        │
│                                                                  │
│  2. Annotation Interface (Label Studio)                          │
│     └─→ Pairwise comparison: "Which is more musical?"           │
│     └─→ Dimension ratings: Likert scale (1-7)                   │
│     └─→ Free-text explanation                                    │
│                                                                  │
│  3. Quality Control                                              │
│     └─→ Minimum 3 annotators per sample                         │
│     └─→ Inter-annotator agreement (Krippendorff's α > 0.7)      │
│     └─→ Expert review for edge cases                            │
│                                                                  │
│  4. Dataset Export                                               │
│     └─→ Hugging Face Datasets format                            │
│     └─→ Train/validation/test splits (70/15/15)                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Monitoring & Observability

### Metrics

**Quality Dimensions:**

- Human correlation (weekly validation study)
- Dimension calibration (predicted vs. actual distribution)
- Inference latency (p50, p95, p99)
- Throughput (requests/minute)

**Safety Dimensions:**

- Attack success rate (adversarial benchmark, <2% target)
- False positive rate (over-refusal)
- False negative rate (missed harmful content)
- Voice match precision/recall
- Copyright detection accuracy

### Logging

```
┌─────────────────────────────────────────────────────────────────┐
│                        OBSERVABILITY STACK                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Metrics:      Prometheus + Grafana                             │
│  Logging:      Structured JSON → Loki                           │
│  Tracing:      OpenTelemetry → Jaeger                           │
│  Experiments:  Weights & Biases                                  │
│  Alerts:       PagerDuty integration                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Security Considerations

### Data Security

- Audio uploads encrypted in transit (TLS 1.3) and at rest (AES-256)
- Artist voice fingerprints stored with access controls
- PII scrubbing in logs
- GDPR-compliant data retention policies

### Model Security

- Model weights signed and verified
- Input validation to prevent adversarial audio attacks
- Rate limiting to prevent abuse
- Audit logging for all classifications

### Access Control

- API key authentication
- Role-based access (admin, developer, read-only)
- Separate keys for production vs. development
- IP allowlisting option for enterprise

---

## Appendix: Technology Evaluation

### Audio Encoder Comparison

| Model | Music Performance | License | Inference Speed | Memory |
|-------|------------------|---------|-----------------|--------|
| **MERT-v1-95M** | Strong on MIR tasks, fits latency budget | CC-BY-NC-4.0 | 100ms/5s chunk | 0.4 GB |
| MERT-v1-330M | SOTA on 14 MIR tasks | CC-BY-NC-4.0 | 45ms/5s chunk | 1.5 GB |
| CLAP (laion-music) | Best text-audio alignment | Apache 2.0 | 80ms/10s chunk | 0.8 GB |
| Wav2Vec2-Large | Good general audio | MIT | 35ms/5s chunk | 1.2 GB |
| HuBERT-Large | Strong speech | MIT | 38ms/5s chunk | 1.2 GB |
| WavLM-Large | Best speaker tasks | MIT | 40ms/5s chunk | 1.2 GB |

**Recommendation:**
- **Quality Dimensions (1-4):** MERT-v1-95M (latency) + CLAP laion-music (prompt adherence) + madmom/Essentia (rhythm/harmony)
- **Safety Dimensions (5-8):** WavLM for speaker analysis, ECAPA-TDNN for voice fingerprinting, Chromaprint for copyright

### Vector Database Comparison

| Database | Speed | Scalability | Features | License |
|----------|-------|-------------|----------|---------|
| FAISS | Fastest | Excellent | Basic | MIT |
| Qdrant | Fast | Excellent | Filtering, payloads | Apache 2.0 |
| Milvus | Fast | Excellent | Distributed | Apache 2.0 |
| Pinecone | Fast | Excellent | Managed | Commercial |

**Recommendation:** FAISS for initial development (simplicity), Qdrant for production (filtering capabilities, metadata support).

### Serving Framework Comparison

| Framework | GPU Support | Scaling | Cold Start | Cost Model |
|-----------|-------------|---------|------------|------------|
| Modal | Excellent | Auto | 10-15s | Per-second |
| BentoML | Excellent | Manual | None | Self-hosted |
| Ray Serve | Excellent | Auto | Configurable | Self-hosted |
| TorchServe | Excellent | Manual | None | Self-hosted |
| HF Endpoints | Good | Auto | Configurable | Per-hour |

**Recommendation:** Modal for development and initial deployment (simplicity, cost-effective), migrate to self-hosted Ray Serve for high-volume production.
