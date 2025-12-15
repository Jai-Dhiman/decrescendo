# Product Requirements Document

## MusiCritic & Constitutional Audio

**Author:** Jai Dhiman  
**Version:** 1.0  
**Last Updated:** December 2024

---

## Executive Summary

This PRD defines two interconnected open-source products that address critical gaps in the AI music ecosystem:

1. **MusiCritic** — An automated musicality evaluation framework that measures what current metrics miss: phrase coherence, tension-resolution, expressive timing, and stylistic authenticity.

2. **Constitutional Audio** — A safety framework implementing Constitutional AI principles for audio generation, addressing copyright, voice cloning, cultural harms, and content moderation.

Both projects share infrastructure and target the same user base: AI music researchers, generative audio companies, and music technology developers.

---

## Problem Statement

### MusiCritic: The Evaluation Gap

Current music AI evaluation is broken. The industry standard metric (Fréchet Audio Distance) achieves only **0.52 correlation with human perception**. CLAP scores fare even worse at **~0.30 correlation**. These metrics measure acoustic similarity but completely miss musicality—the qualities that make music feel intentional, expressive, and alive.

**Consequences:**

- Music AI labs (Google, Meta, Stability, Suno, Udio) cannot objectively compare models on dimensions that matter to musicians
- Papers report metrics that don't correlate with actual quality
- Progress is bottlenecked by lack of principled evaluation
- No benchmark captures phrase coherence, harmonic sophistication, or expressive timing

### Constitutional Audio: The Safety Gap

Audio AI safety is fragmented and inadequate. The UMG lawsuits against Suno and Udio revealed that these platforms processed ~864,000 files daily using copyrighted recordings. ADL investigations found extensive hate content bypass on Suno using coded language. Voice cloning has already enabled election interference (Slovak election 2023) and $25M CEO fraud.

**Consequences:**

- No systematic taxonomy of audio-specific harms exists
- Artist voice cloning lacks reliable detection
- Cultural appropriation in music remains undefined computationally
- Regulation is coming (EU AI Act, Tennessee ELVIS Act) but industry lacks compliance tools

---

## Target Users

### Primary Users

| User Type | MusiCritic Use Case | Constitutional Audio Use Case |
|-----------|---------------------|-------------------------------|
| **Music AI Researchers** | Benchmark model improvements, ablation studies, paper evaluation | Red-team models, measure safety metrics |
| **Generative Audio Companies** | A/B test generation quality, optimize for musicality | Content moderation, copyright detection, regulatory compliance |
| **Music Educators** | Automated performance feedback, practice tool evaluation | N/A |
| **Audio Platform Trust & Safety** | N/A | Content screening, artist protection, policy enforcement |

### Secondary Users

- Music information retrieval (MIR) researchers
- Streaming platform recommendation teams
- Music licensing and rights management companies
- Regulatory bodies and standards organizations

---

## Product Vision

### MusiCritic Vision

**"The definitive open-source framework for evaluating AI-generated music on dimensions that actually matter to musicians."**

MusiCritic will become the industry standard for music AI evaluation, analogous to how BLEU/ROUGE became standard for NLP or FID for image generation—but with strong correlation to human musical judgment.

### Constitutional Audio Vision

**"The first systematic framework for identifying and mitigating harms unique to audio AI generation."**

Constitutional Audio will provide the safety infrastructure that enables responsible development of audio AI, protecting artists, preventing misuse, and establishing industry best practices before regulation mandates crude solutions.

---

## MusiCritic: Product Definition

### Core Capabilities

#### 1. Multi-Dimensional Musicality Scoring

Evaluate audio across 16+ dimensions organized in four tiers:

**Tier 1: Highly Objective (Directly Measurable)**

- Pitch accuracy (Hz deviation from target)
- Rhythmic precision (onset timing deviation)
- Tempo stability (BPM variance over time)
- Dynamic range (dB measurement)
- Intonation quality (harmonic spectrum analysis)

**Tier 2: Moderately Objective (Feature Extraction Required)**

- Phrasing coherence (phrase boundary detection + contour analysis)
- Articulation clarity (attack/decay characteristics)
- Rubato appropriateness (tempo deviation patterns)
- Voicing balance (spectral energy distribution)
- Pedaling effectiveness (sustain/resonance analysis)

**Tier 3: Semi-Subjective (Reference Model Required)**

- Stylistic authenticity (corpus-based similarity)
- Structural coherence (form adherence, motif development)
- Emotional expression (arousal/valence trajectory)
- Harmonic sophistication (chord vocabulary, voice leading)
- Groove/swing feel (micro-timing patterns)

**Tier 4: Aggregate Scores**

- Overall musicality score (weighted combination)
- Genre-specific subscores
- Comparative ranking against reference corpus

#### 2. Hierarchical Temporal Analysis

Analyze music at multiple time scales:

- **Frame level** (10ms): Note onsets, spectral features
- **Beat level** (500ms): Rhythmic patterns, micro-timing
- **Phrase level** (4-16 bars): Melodic contour, harmonic progression
- **Section level** (verse/chorus): Structural coherence
- **Piece level**: Overall narrative arc, tension-resolution

#### 3. Tension-Resolution Modeling

Compute harmonic and rhythmic tension curves using Tonal Interval Space (TIS):

- Cloud diameter (harmonic clarity)
- Cloud momentum (rate of harmonic change)
- Tensile strain (deviation from tonal center)
- Resolution strength at cadential points

#### 4. Comparative Evaluation

- Pairwise ranking: "Which performance is more musical?"
- Reference-based scoring: "How does this compare to professional recordings?"
- Style transfer evaluation: "Does this sound like the target artist/genre?"

### Output Formats

**API Response:**

- JSON with dimension scores, confidence intervals, and explanations
- Embedding vectors for downstream tasks
- Temporal analysis (scores over time)

**Human-Readable Report:**

- Overall assessment with natural language explanation
- Dimension breakdown with visualizations
- Specific feedback for improvement (educational use case)

**Benchmark Integration:**

- Leaderboard-compatible metrics
- Reproducible evaluation protocol
- Statistical significance testing

### Success Metrics

| Metric | Current State | Target |
|--------|---------------|--------|
| Human correlation (Spearman) | 0.52 (FAD) | >0.75 |
| Inter-annotator agreement | N/A | >0.80 Krippendorff's α |
| Dimension coverage | 3-4 metrics | 16+ dimensions |
| Processing speed | Varies | <2s per 30s audio |
| Genre coverage | Western-centric | 10+ genre families |

### Non-Goals (V1)

- Real-time evaluation during live performance
- Symbolic (MIDI-only) evaluation without audio
- Composition evaluation (lyrics, arrangement choices)
- Subjective "star quality" or commercial viability prediction

---

## Constitutional Audio: Product Definition

### Core Capabilities

#### 1. Harm Classification

Multi-label classification across seven harm categories:

**Copyright & IP (Critical)**

- Artist voice detection (is this trying to sound like a specific artist?)
- Melody similarity scoring (does this copy an existing melody?)
- Sample detection (does this contain copyrighted samples?)
- Style mimicry assessment (problematic imitation vs. legitimate influence)

**Voice Cloning (Critical)**

- Known voice detection (politician, celebrity, protected individual)
- Consent verification integration (does this voice have usage rights?)
- Deepfake confidence scoring

**Cultural Harms (High)**

- Sacred/ceremonial music detection
- Cultural context flagging (requires human review)
- Stereotyping pattern detection

**Misinformation (Critical)**

- Synthetic speech detection
- Manipulation artifact analysis
- Provenance verification (watermark detection)

**Emotional Manipulation (Medium)**

- Subliminal pattern detection
- Addiction design pattern flagging
- Extreme emotional content warning

**Content Safety (High)**

- Hate speech/slur detection in lyrics
- Harmful instruction detection
- Age-inappropriate content flagging

**Physical Safety (High)**

- Harmful frequency detection
- Volume spike detection
- Epileptogenic pattern detection

#### 2. Constitutional Classifier Pipeline

Two-stage classification following Anthropic's Constitutional Classifiers architecture:

**Input Classifier (Pre-Generation)**

- Prompt analysis for harmful intent
- Artist/voice request detection
- Policy violation prediction

**Output Classifier (Streaming)**

- Real-time content analysis during generation
- Cumulative harm probability scoring
- Intervention triggers at configurable thresholds

#### 3. Artist Protection System

**Voice Fingerprint Registry**

- Artists register voice embeddings
- Similarity threshold configuration
- Automated takedown workflow integration

**Style Protection**

- Distinguish "sounds like" from "copies from"
- Quantify stylistic similarity vs. direct copying
- Evidence generation for rights claims

#### 4. Red-Teaming Infrastructure

**Automated Adversarial Testing**

- Jailbreak prompt generation
- Evasion technique detection
- Classifier robustness measurement

**Continuous Evaluation**

- Attack success rate monitoring
- False positive/negative tracking
- Model drift detection

### Output Formats

**API Response:**

- Harm category scores with confidence
- Specific policy violations detected
- Recommended actions (block, flag, allow)
- Evidence for human review

**Moderation Dashboard:**

- Queue management for flagged content
- Appeal workflow support
- Audit logging

**Compliance Reporting:**

- EU AI Act transparency reports
- Training data summaries
- Watermark verification logs

### Success Metrics

| Metric | Current State | Target |
|--------|---------------|--------|
| Attack success rate | ~16% (baseline) | <2% |
| Over-refusal rate | N/A | <0.5% increase |
| Artist mimicry detection | Unknown | >95% precision @ 90% recall |
| Voice clone detection | ~85% | >98% accuracy |
| Processing latency | N/A | <100ms (streaming) |
| False positive rate | N/A | <1% |

### Non-Goals (V1)

- Legal determination of copyright infringement (requires human/legal judgment)
- Cultural appropriation adjudication (flagging only, not ruling)
- Content generation (detection and classification only)
- Speech-to-text transcription for lyric analysis (use existing ASR)

---

## Shared Requirements

### Open Source Commitment

Both projects will be released under permissive open-source licenses:

- **Code:** Apache 2.0
- **Models:** Model weights with appropriate license (considering MERT's CC-BY-NC-4.0)
- **Datasets:** CC-BY-SA 4.0 for new annotations
- **Documentation:** CC-BY 4.0

### Integration Requirements

**Hugging Face Integration**

- Model hosting on Hugging Face Hub
- Transformers/Datasets compatibility
- Spaces demo deployment

**API Design**

- RESTful endpoints with OpenAPI specification
- Python SDK with type hints
- Batch processing support
- Webhook callbacks for async processing

**Deployment Options**

- Self-hosted (Docker containers)
- Cloud functions (Modal, AWS Lambda)
- Hugging Face Inference Endpoints

### Performance Requirements

| Requirement | MusiCritic | Constitutional Audio |
|-------------|------------|---------------------|
| Latency (30s audio) | <2 seconds | <500ms |
| Throughput | 100 req/min | 1000 req/min |
| Max audio length | 10 minutes | 5 minutes |
| Supported formats | WAV, MP3, FLAC, OGG | WAV, MP3, FLAC, OGG |
| Sample rates | 16kHz-48kHz | 16kHz-48kHz |

### Quality Requirements

- **Reliability:** 99.9% uptime for hosted inference
- **Reproducibility:** Deterministic outputs given same input and model version
- **Explainability:** All scores accompanied by contributing factors
- **Auditability:** Complete logging for compliance requirements

---

## Competitive Positioning

### MusiCritic Differentiation

| Capability | FAD | CLAP | Stability Metrics | MusiCritic |
|------------|-----|------|-------------------|------------|
| Human correlation | 0.52 | 0.30 | ~0.5 | >0.75 (target) |
| Musicality dimensions | 1 | 1 | 3-4 | 16+ |
| Hierarchical analysis | ❌ | ❌ | ❌ | ✅ |
| Tension modeling | ❌ | ❌ | ❌ | ✅ |
| Explainable scores | ❌ | ❌ | Partial | ✅ |
| Open source | ✅ | ✅ | ✅ | ✅ |

### Constitutional Audio Differentiation

| Capability | Suno | Udio | Stability | Google | Constitutional Audio |
|------------|------|------|-----------|--------|---------------------|
| Copyright detection | ❌ | ⚠️ | ✅ | Unknown | ✅ |
| Voice cloning protection | ❌ | ❌ | ✅ | ✅ | ✅ |
| Cultural harm detection | ❌ | ❌ | ❌ | ❌ | ✅ |
| Coded language detection | ❌ | ❌ | ❌ | ❌ | ✅ |
| Open source | ❌ | ❌ | Partial | ❌ | ✅ |
| Watermarking | ❌ | ⚠️ | ✅ | ✅ | ✅ |

---

## Release Strategy

### Phase 1: Foundation (Months 1-2)

- Core evaluation pipeline (MusiCritic)
- Basic safety classifiers (Constitutional Audio)
- Hugging Face model release
- Technical blog post + paper preprint

### Phase 2: Expansion (Months 3-4)

- Multi-dimensional scoring (MusiCritic)
- Constitutional Classifier pipeline (Constitutional Audio)
- Python SDK release
- Community feedback integration

### Phase 3: Production (Months 5-6)

- Full 16-dimension evaluation (MusiCritic)
- Artist protection system (Constitutional Audio)
- Hosted inference API
- Industry partnership outreach

### Phase 4: Ecosystem (Months 7+)

- Leaderboard and benchmark establishment
- Integration with major generation frameworks
- Continuous model improvement
- Community contribution program

---

## Success Criteria

### MusiCritic Success

1. **Adoption:** 1,000+ monthly active users within 6 months
2. **Citation:** Referenced in 10+ music AI papers within 12 months
3. **Accuracy:** Achieve >0.75 human correlation validated by independent study
4. **Industry:** Adopted by at least 2 major music AI companies for internal evaluation

### Constitutional Audio Success

1. **Effectiveness:** Reduce attack success rate to <2% on standard adversarial benchmark
2. **Adoption:** Integrated by 3+ audio platforms within 12 months
3. **Recognition:** Cited in regulatory guidance or industry standards documents
4. **Protection:** 100+ artists registered in voice protection system

---

## Appendix: User Stories

### MusiCritic User Stories

**As a music AI researcher**, I want to evaluate my model's outputs on musicality dimensions so that I can demonstrate improvements over baselines in my paper.

**As a generative music company**, I want to A/B test different model versions so that I can ship the one that produces more musical outputs.

**As a music educator**, I want to provide automated feedback on student performances so that they can practice more effectively between lessons.

**As a benchmark maintainer**, I want standardized evaluation metrics so that the community can fairly compare different approaches.

### Constitutional Audio User Stories

**As a platform trust & safety lead**, I want to detect copyrighted content before publication so that we avoid legal liability.

**As an artist**, I want to register my voice so that unauthorized clones can be detected and removed.

**As a compliance officer**, I want to generate transparency reports so that we meet EU AI Act requirements.

**As a red team researcher**, I want to test model robustness so that we can improve safety before deployment.

**As a content moderator**, I want flagged content with evidence so that I can make informed decisions efficiently.
