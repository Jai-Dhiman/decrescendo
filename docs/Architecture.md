# System Architecture Document

## MusiCritic & Constitutional Audio

**Author:** Jai Dhiman  
**Version:** 1.0  
**Last Updated:** December 2024

---

## Architecture Overview

Both MusiCritic and Constitutional Audio share a common foundation: audio embedding extraction, vector storage, and inference serving. They diverge at the task-specific layers—MusiCritic focuses on multi-dimensional quality prediction while Constitutional Audio focuses on harm classification and content moderation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SHARED INFRASTRUCTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Audio Input → Preprocessing → Embedding Extraction → Vector Storage        │
│                     │                   │                                    │
│              (Resampling,         (MERT, CLAP,                              │
│               Chunking)           WavLM, EnCodec)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                                    │
                    ▼                                    ▼
┌─────────────────────────────────┐    ┌─────────────────────────────────────┐
│         MUSICRITIC              │    │       CONSTITUTIONAL AUDIO          │
├─────────────────────────────────┤    ├─────────────────────────────────────┤
│ • Musicality Prediction Heads   │    │ • Input Classifier                  │
│ • Temporal Aggregation          │    │ • Output Classifier (Streaming)     │
│ • Tension-Resolution Module     │    │ • Artist Fingerprint Matching       │
│ • Comparative Ranking           │    │ • Harm Category Classification      │
└─────────────────────────────────┘    └─────────────────────────────────────┘
                    │                                    │
                    ▼                                    ▼
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
| **Primary Audio Encoder** | MERT-v1-330M | Music understanding embeddings | CC-BY-NC-4.0 |
| **Text-Audio Alignment** | LAION-CLAP | Evaluation metrics, zero-shot | Apache 2.0 |
| **Speaker Analysis** | WavLM-Large | Voice/speaker embeddings | MIT |
| **Audio Tokenization** | EnCodec 24kHz | Discrete audio representation | MIT |
| **Speaker Verification** | ECAPA-TDNN | Voice fingerprinting | Apache 2.0 |
| **Audio Fingerprinting** | Chromaprint | Content identification | LGPL 2.1 |
| **Vector Database** | FAISS / Qdrant | Embedding storage & search | MIT / Apache 2.0 |
| **ML Framework** | PyTorch 2.x | Model training & inference | BSD |
| **Experiment Tracking** | Weights & Biases | Training monitoring | Commercial |
| **Serving** | Modal / FastAPI | API deployment | Various |

### Development Environment

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Primary language |
| JAX/Flax | Alternative training (DPO) |
| Docker | Containerization |
| pytest | Testing |
| ruff | Linting |
| pre-commit | Code quality |

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

Multiple encoders run in parallel to extract complementary representations:

```
Preprocessed Chunks
     │
     ├──────────────────┬──────────────────┬──────────────────┐
     ▼                  ▼                  ▼                  ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   MERT      │  │   CLAP      │  │   WavLM     │  │  EnCodec    │
│   330M      │  │   HTSAT     │  │   Large     │  │   24kHz     │
├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤
│ 24 layers   │  │ Audio enc.  │  │ 24 layers   │  │ 4 codebooks │
│ 1024-dim    │  │ 1024-dim    │  │ 1024-dim    │  │ 50 Hz       │
│ 75 Hz       │  │ Pooled      │  │ 50 Hz       │  │             │
│             │  │             │  │             │  │             │
│ Music       │  │ Text-Audio  │  │ Speaker     │  │ Discrete    │
│ Features    │  │ Alignment   │  │ Features    │  │ Tokens      │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
     │                  │                  │                  │
     └──────────────────┴──────────────────┴──────────────────┘
                                 │
                                 ▼
                    Combined Feature Tensor
                    [N × T × D] per encoder
```

**Encoder Selection by Task:**

| Task | Primary Encoder | Secondary Encoder | Rationale |
|------|-----------------|-------------------|-----------|
| Musicality scoring | MERT | CLAP | MERT captures music; CLAP for evaluation |
| Voice detection | WavLM | ECAPA-TDNN | Speaker-optimized representations |
| Content fingerprinting | MERT + Chromaprint | — | Music similarity + exact match |
| Harmonic analysis | MERT | — | Pre-trained on CQT features |

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

## MusiCritic Architecture

### System Design

```
Audio Input
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MUSICRITIC PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              FEATURE EXTRACTION LAYER                     │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │                                                           │   │
│  │  MERT Branch:                                             │   │
│  │  └─→ Frame embeddings → Layer selection → Projection      │   │
│  │                                                           │   │
│  │  Symbolic Branch:                                         │   │
│  │  └─→ Audio-to-MIDI (basic-pitch) → MusicBERT → Embeddings│   │
│  │                                                           │   │
│  │  Harmonic Branch:                                         │   │
│  │  └─→ Chord detection (madmom) → TIS computation          │   │
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
│  │              MULTI-TASK PREDICTION HEADS                  │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │                                                           │   │
│  │  Shared Trunk (cross-attention fusion)                    │   │
│  │       │                                                   │   │
│  │       ├──→ Pitch/Intonation Head ──→ Score + Confidence  │   │
│  │       ├──→ Rhythm/Timing Head ──→ Score + Confidence     │   │
│  │       ├──→ Dynamics Head ──→ Score + Confidence          │   │
│  │       ├──→ Phrasing Head ──→ Score + Confidence          │   │
│  │       ├──→ Articulation Head ──→ Score + Confidence      │   │
│  │       ├──→ Harmonic Head ──→ Score + Confidence          │   │
│  │       ├──→ Structure Head ──→ Score + Confidence         │   │
│  │       ├──→ Expression Head ──→ Score + Confidence        │   │
│  │       └──→ Overall Musicality ──→ Score + Confidence     │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
Multi-Dimensional Score Output
```

### Model Architecture Details

**Feature Fusion Module:**

The system fuses three complementary feature streams using cross-attention:

| Stream | Source | Dimension | Information Captured |
|--------|--------|-----------|---------------------|
| Acoustic | MERT layers 18-24 | 1024 | Timbre, rhythm, pitch |
| Symbolic | MusicBERT | 768 | Harmony, structure |
| Tension | TIS computation | 3 | Harmonic tension curve |

Fusion uses learned cross-attention where acoustic features attend to symbolic features, enabling the model to ground abstract harmonic concepts in acoustic reality.

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
L_total = λ₁ · L_regression + λ₂ · L_ranking + λ₃ · L_consistency

Where:
- L_regression: MSE between predicted and annotated scores
- L_ranking: Pairwise margin ranking loss for comparative pairs
- L_consistency: Correlation loss ensuring dimension scores are coherent
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

---

## Constitutional Audio Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                 CONSTITUTIONAL AUDIO PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │                  INPUT CLASSIFIER                       │     │
│  ├────────────────────────────────────────────────────────┤     │
│  │                                                         │     │
│  │  Text Prompt ──→ Intent Analysis ──→ Policy Check      │     │
│  │       │              │                    │             │     │
│  │       │              │                    ▼             │     │
│  │       │              │         ┌─────────────────┐     │     │
│  │       │              │         │ BLOCK / ALLOW / │     │     │
│  │       │              │         │ FLAG FOR REVIEW │     │     │
│  │       │              │         └─────────────────┘     │     │
│  │       │              │                                  │     │
│  │       ▼              ▼                                  │     │
│  │  Artist Request  Voice Request                          │     │
│  │  Detection       Detection                              │     │
│  │                                                         │     │
│  └────────────────────────────────────────────────────────┘     │
│                          │                                       │
│                          ▼ (if allowed)                          │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              AUDIO GENERATION MODEL                     │     │
│  │         (External - not part of this system)           │     │
│  └────────────────────────────────────────────────────────┘     │
│                          │                                       │
│                          ▼                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              OUTPUT CLASSIFIER (STREAMING)              │     │
│  ├────────────────────────────────────────────────────────┤     │
│  │                                                         │     │
│  │  Audio Chunks ──→ Parallel Analysis:                   │     │
│  │       │                                                 │     │
│  │       ├──→ Speaker Similarity (ECAPA-TDNN)             │     │
│  │       │         └─→ Match against protected voices     │     │
│  │       │                                                 │     │
│  │       ├──→ Content Fingerprint (Chromaprint)           │     │
│  │       │         └─→ Match against copyright database   │     │
│  │       │                                                 │     │
│  │       ├──→ Harm Classification (Multi-label)           │     │
│  │       │         └─→ 7 harm categories                  │     │
│  │       │                                                 │     │
│  │       └──→ Cumulative Probability Aggregation          │     │
│  │                    │                                    │     │
│  │                    ▼                                    │     │
│  │         ┌─────────────────────────────────┐            │     │
│  │         │ Threshold Check:                 │            │     │
│  │         │ • Hard block (p > 0.95)         │            │     │
│  │         │ • Flag for review (p > 0.7)     │            │     │
│  │         │ • Allow (p < 0.7)               │            │     │
│  │         └─────────────────────────────────┘            │     │
│  │                                                         │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Classifier Architectures

**Input Classifier:**

```
Text Prompt
     │
     ▼
┌─────────────────────────────────────────┐
│        TEXT ENCODER (RoBERTa-base)       │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│         CLASSIFICATION HEADS             │
├─────────────────────────────────────────┤
│                                          │
│  Intent Head:                            │
│  └─→ benign / suspicious / malicious    │
│                                          │
│  Artist Request Head:                    │
│  └─→ None / Named artist / Style ref    │
│                                          │
│  Voice Request Head:                     │
│  └─→ None / Celebrity / Politician      │
│                                          │
│  Policy Violation Head:                  │
│  └─→ Multi-label (7 harm categories)    │
│                                          │
└─────────────────────────────────────────┘
     │
     ▼
Decision: BLOCK / ALLOW / MODIFY_PROMPT
```

**Output Classifier (Streaming):**

Processes audio in real-time during generation:

```
Audio Stream (chunked at 1-second intervals)
     │
     ├─────────────────────────────────────────────────────┐
     │                                                      │
     ▼                                                      ▼
┌─────────────────────┐                          ┌─────────────────────┐
│  SPEAKER ANALYZER   │                          │  CONTENT ANALYZER   │
├─────────────────────┤                          ├─────────────────────┤
│                     │                          │                     │
│  WavLM → ECAPA-TDNN │                          │  MERT → Harm Heads  │
│        │            │                          │        │            │
│        ▼            │                          │        ▼            │
│  Speaker Embedding  │                          │  Content Embedding  │
│  (192-dim)          │                          │  (1024-dim)         │
│        │            │                          │        │            │
│        ▼            │                          │        ▼            │
│  FAISS Similarity   │                          │  Multi-label        │
│  Search             │                          │  Classification     │
│        │            │                          │        │            │
│        ▼            │                          │        ▼            │
│  Protected Voice    │                          │  Harm Probabilities │
│  Match Score        │                          │  (7 categories)     │
│                     │                          │                     │
└─────────────────────┘                          └─────────────────────┘
     │                                                      │
     └──────────────────────┬───────────────────────────────┘
                            │
                            ▼
                 ┌─────────────────────────┐
                 │  AGGREGATION MODULE     │
                 ├─────────────────────────┤
                 │                         │
                 │  Cumulative scoring:    │
                 │  • Max pooling          │
                 │  • Exponential decay    │
                 │  • Threshold tracking   │
                 │                         │
                 └─────────────────────────┘
                            │
                            ▼
                 Decision: CONTINUE / INTERVENE / STOP
```

### Harm Classification Details

**Multi-Label Classifier Architecture:**

| Harm Category | Detection Approach | Model |
|---------------|-------------------|-------|
| Copyright/IP | Fingerprint matching + embedding similarity | Chromaprint + MERT |
| Voice Cloning | Speaker verification | ECAPA-TDNN |
| Cultural | Pattern matching + flagging | Fine-tuned classifier |
| Misinformation | Synthetic speech detection | SKA-TDNN |
| Emotional | Subliminal pattern detection | Specialized analyzer |
| Content Safety | Hate/harmful content | Audio + ASR + NLP |
| Physical Safety | Frequency analysis | Signal processing |

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

### Training Strategy

**Constitutional Classifier Training (Following Anthropic's Approach):**

```
Phase 1: Generate Synthetic Training Data
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  1. Define Constitution (natural language safety rules)         │
│                                                                  │
│  2. Generate harmful prompts using Claude API:                  │
│     • Diverse attack vectors                                    │
│     • Edge cases and adversarial examples                       │
│     • Multi-turn conversations                                  │
│                                                                  │
│  3. Generate safe variations of similar prompts                 │
│                                                                  │
│  4. Label with harm categories                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Phase 2: Train Input Classifier
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  • Fine-tune RoBERTa on prompt classification                   │
│  • Multi-label binary cross-entropy loss                        │
│  • Optimize for high recall (catch harmful content)             │
│  • Accept higher false positive rate (can be reviewed)          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Phase 3: Train Output Classifier
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  • Collect audio outputs from generation model                  │
│  • Label for harm categories (automated + human)                │
│  • Fine-tune MERT heads for harm detection                      │
│  • Calibrate thresholds on validation set                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Phase 4: DPO Alignment (Optional - for generation model)
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  • Build preference dataset (90K+ triplets):                    │
│    - Prompt                                                     │
│    - Preferred output (safe, high quality)                      │
│    - Dispreferred output (harmful or low quality)               │
│                                                                  │
│  • Multi-reward DPO with three dimensions:                      │
│    - Text alignment (CLAP score)                                │
│    - Audio quality (MusiCritic score)                           │
│    - Safety compliance (Constitutional classifier)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

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
│  │  modal deploy constitutional_audio/serve.py             │    │
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
| `/v1/musicritic/evaluate` | POST | Full musicality evaluation |
| `/v1/musicritic/compare` | POST | Pairwise comparison |
| `/v1/musicritic/dimensions/{dim}` | POST | Single dimension score |
| `/v1/constitutional/classify` | POST | Full harm classification |
| `/v1/constitutional/stream` | WS | Streaming classification |
| `/v1/constitutional/register-voice` | POST | Artist voice registration |
| `/v1/constitutional/check-voice` | POST | Voice similarity check |
| `/health` | GET | Service health check |

**Request/Response Schema (MusiCritic):**

```
POST /v1/musicritic/evaluate

Request:
{
  "audio": "<base64-encoded-audio>",
  "format": "wav",
  "sample_rate": 44100,
  "dimensions": ["all"],  // or specific: ["pitch", "rhythm", "phrasing"]
  "return_temporal": false,
  "reference_genre": "classical_piano"  // optional
}

Response:
{
  "overall_score": 78.5,
  "confidence": 0.92,
  "dimensions": {
    "pitch_accuracy": {"score": 85.2, "confidence": 0.95},
    "rhythmic_precision": {"score": 72.1, "confidence": 0.89},
    "phrasing": {"score": 81.0, "confidence": 0.87},
    "dynamics": {"score": 76.3, "confidence": 0.91},
    "articulation": {"score": 79.8, "confidence": 0.88},
    "harmonic_sophistication": {"score": 74.5, "confidence": 0.85},
    "structural_coherence": {"score": 80.2, "confidence": 0.86},
    "expression": {"score": 77.9, "confidence": 0.84}
  },
  "explanation": "Strong technical foundation with room for improvement in rhythmic precision. Phrasing shows musical understanding.",
  "processing_time_ms": 1450
}
```

**Request/Response Schema (Constitutional Audio):**

```
POST /v1/constitutional/classify

Request:
{
  "audio": "<base64-encoded-audio>",
  "format": "wav",
  "prompt": "Generate a song in the style of...",  // optional
  "check_categories": ["all"]
}

Response:
{
  "decision": "FLAG_FOR_REVIEW",
  "harm_scores": {
    "copyright_ip": {"score": 0.72, "details": "High similarity to protected work"},
    "voice_cloning": {"score": 0.15, "details": null},
    "cultural": {"score": 0.08, "details": null},
    "misinformation": {"score": 0.05, "details": null},
    "emotional_manipulation": {"score": 0.12, "details": null},
    "content_safety": {"score": 0.03, "details": null},
    "physical_safety": {"score": 0.01, "details": null}
  },
  "matched_fingerprints": [
    {"source": "Song Title - Artist", "similarity": 0.72, "segment": "0:15-0:45"}
  ],
  "matched_voices": [],
  "recommended_action": "Human review required for copyright concern",
  "processing_time_ms": 320
}
```

---

## Data Architecture

### Training Data Sources

**MusiCritic:**

| Dataset | Size | Use | License |
|---------|------|-----|---------|
| MAESTRO v3.0.0 | 200 hours | Piano evaluation training | CC BY-NC-SA 4.0 |
| PercePiano | 12,736 annotations | Perceptual feature labels | Academic |
| MTG-Jamendo | 55,000 tracks | Genre diversity | CC variants |
| MusicCaps | 5,521 clips | Text descriptions | CC BY-SA 4.0 |
| Custom annotations | ~10,000 pairs | Pairwise preferences | Proprietary |

**Constitutional Audio:**

| Dataset | Size | Use | License |
|---------|------|-----|---------|
| VoxCeleb2 | 1M utterances | Speaker verification training | Research only |
| ASVspoof 2019 | 54,000 samples | Deepfake detection | CC BY 4.0 |
| Custom adversarial | ~50,000 prompts | Input classifier training | Proprietary |
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

**MusiCritic:**

- Human correlation (weekly validation study)
- Dimension calibration (predicted vs. actual distribution)
- Inference latency (p50, p95, p99)
- Throughput (requests/minute)

**Constitutional Audio:**

- Attack success rate (adversarial benchmark)
- False positive rate (over-refusal)
- False negative rate (missed harmful content)
- Artist match precision/recall
- Latency for streaming classification

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
| MERT-v1-330M | SOTA on 14 MIR tasks | CC-BY-NC-4.0 | 45ms/5s chunk | 1.5 GB |
| Wav2Vec2-Large | Good general audio | MIT | 35ms/5s chunk | 1.2 GB |
| HuBERT-Large | Strong speech | MIT | 38ms/5s chunk | 1.2 GB |
| WavLM-Large | Best speaker tasks | MIT | 40ms/5s chunk | 1.2 GB |
| Jukebox | Music generation | MIT | 500ms/5s chunk | 5+ GB |

**Recommendation:** MERT for music understanding, WavLM for speaker analysis, both for comprehensive coverage.

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
