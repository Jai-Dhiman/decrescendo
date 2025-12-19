# Product Requirements Document

## MusiCritic

**Author:** Jai Dhiman
**Version:** 2.0
**Last Updated:** December 2024

---

## Executive Summary

**MusiCritic** is a comprehensive evaluation framework for **AI-generated music** that unifies quality assessment and safety evaluation into a single system. It measures what current metrics miss across two complementary dimensions:

**Quality Dimensions:**
- Prompt adherence, musical coherence, audio quality, and musicality

**Safety Dimensions:**
- Copyright/originality, voice cloning detection, cultural sensitivity, and content safety

**Targets:**
- **>0.75 human correlation** (vs. FAD's 0.52) for quality metrics
- **<2% attack success rate** for safety classifiers
- **<2 second processing** per 30-second clip

MusiCritic provides a single API call that returns both "Is this AI music good?" and "Is this AI music safe?" — enabling AI music companies to evaluate quality and ensure compliance in one integrated workflow.

---

## Problem Statement

### The Quality Gap

Current AI music evaluation is broken. The industry standard metric (Fréchet Audio Distance) achieves only **0.52 correlation with human perception**. CLAP scores fare even worse at **0.26-0.30 correlation**. These metrics measure acoustic similarity but completely miss what makes AI-generated music sound good—coherent structure, prompt adherence, and absence of artifacts.

**Recent Progress (Still Insufficient):**

- **Human-CLAP** (Takano et al., 2025): Fine-tuned CLAP improved correlation to 0.512
- **MAD** (Mauve Audio Divergence): Achieves 0.62 correlation on MusicPrefs benchmark
- **FAD-CLAP-MA**: Using music-trained CLAP embeddings shows best correlation

**Consequences:**

- Music AI labs (Google, Meta, Stability, Suno, Udio) cannot objectively compare models on dimensions that matter
- Papers report metrics that don't correlate with actual quality
- Progress is bottlenecked by lack of principled evaluation
- No benchmark captures prompt adherence, structural coherence, or audio quality

### The Safety Gap

Audio AI safety is fragmented and inadequate. The UMG lawsuits against Suno and Udio revealed that these platforms processed ~864,000 files daily using copyrighted recordings. ADL investigations found extensive hate content bypass on Suno using coded language. Voice cloning has already enabled election interference (Slovak election 2023) and $25M CEO fraud.

**Consequences:**

- No systematic taxonomy of audio-specific harms exists
- Artist voice cloning lacks reliable detection
- Cultural appropriation in music remains undefined computationally
- Regulation is coming (EU AI Act, Tennessee ELVIS Act) but industry lacks compliance tools

### The Integration Gap

Currently, companies must use **separate tools** for quality evaluation and safety assessment, leading to:

- Fragmented workflows with multiple API calls
- Inconsistent scoring frameworks
- Duplicate audio processing (wasted compute)
- No unified view of "good and safe" AI music

**Validation Standards:**
- **Quality:** AIME dataset (6,500 AI-generated tracks, 15,600 pairwise human comparisons)
- **Safety:** ASVspoof for voice cloning, custom adversarial benchmarks for content safety

---

## Target Users

### Primary Users

| User Type | Quality Use Cases | Safety Use Cases |
|-----------|-------------------|------------------|
| **Music AI Researchers** | Benchmark model improvements, ablation studies, paper evaluation | Red-team models, measure safety metrics |
| **Generative Audio Companies** | A/B test generation quality, CI/CD quality gates, prompt adherence testing | Content moderation, copyright detection, regulatory compliance |
| **AI Music Startups** | Compare against Suno/Udio baselines on standardized benchmarks | Pre-release safety screening |
| **Audio Platform Trust & Safety** | Quality-based content ranking | Content screening, artist protection, policy enforcement |

### Secondary Users

- Music information retrieval (MIR) researchers
- Streaming platform recommendation teams
- Music licensing and rights management companies
- Regulatory bodies and standards organizations
- Artists seeking voice protection

---

## Product Vision

**"The comprehensive open-source framework for evaluating AI-generated music — measuring both quality and safety in a single, unified assessment."**

MusiCritic will become the industry standard for AI music evaluation, analogous to how BLEU/ROUGE became standard for NLP or FID for image generation—but specifically designed for the unique challenges of AI-generated audio:

- **Quality:** Detecting artifacts, measuring prompt alignment, assessing structural coherence
- **Safety:** Protecting artists, preventing misuse, ensuring regulatory compliance

One API. One score framework. Complete evaluation.

---

## Product Definition

### Core Capabilities

MusiCritic evaluates AI-generated audio across **8 dimensions** organized into Quality and Safety categories:

#### Quality Dimensions (4)

Evaluate AI-generated audio quality with >0.75 target human correlation:

**Dimension 1: Prompt Adherence**

Measures how well the generated audio matches the text prompt.

- CLAP cosine similarity using `laion/larger_clap_music` embeddings
- Genre classification accuracy (CNN+LSTM, 95.4% accuracy)
- Mood/instrumentation alignment scoring
- Style transfer fidelity assessment

**Dimension 2: Musical Coherence**

Assesses structural and compositional quality.

- Structure detection: verse/chorus/bridge identification (SpecTNT, 87% accuracy)
- Harmonic quality: chord progression analysis (Essentia ChordsDetection)
- Melodic coherence: phrase completion and contour analysis (CREPE/pYIN)
- Rhythmic stability: beat tracking and tempo drift detection (madmom DBNBeatTracker, 95%+ accuracy)
- Key consistency: tonal center stability (Krumhansl-Schmuckler profiles)

**Dimension 3: Audio Quality**

Detects artifacts and production issues specific to AI generation.

- AI artifact detection: spectral peak analysis for generative fingerprints (99%+ accuracy)
- Click/transient artifacts: Essentia ClickDetector
- Loudness compliance: LUFS measurement (-14 LUFS target for streaming)
- Dynamic range: LRA (Loudness Range) analysis
- True Peak: headroom compliance (<-1 dBTP)
- Perceptual quality: ViSQOLAudio MOS scoring (1-5 scale)

**Dimension 4: Musicality**

Evaluates expressive and aesthetic qualities.

- Tension-resolution: Tonal Interval Space (TIS) modeling
  - Cloud diameter (harmonic clarity)
  - Cloud momentum (rate of harmonic change)
  - Tensile strain (deviation from tonal center)
- Dynamic variation: loudness contour analysis
- Genre authenticity: corpus-based similarity scoring
- Groove/swing feel: micro-timing pattern analysis

#### Safety Dimensions (4)

Evaluate AI-generated audio safety with <2% attack success rate target:

**Dimension 5: Copyright & Originality**

Detects potential copying or excessive similarity to copyrighted/training data.

- Audio fingerprinting: Chromaprint/AcoustID matching against copyright database
- Melody similarity: MelodySim model (MERT encoder + Triplet NN)
- Sample detection: identifying copyrighted samples in generated audio
- Focus on melody + rhythm combination (harmony alone insufficient)
- Distinguish "style matching" from "direct copying"

**Dimension 6: Voice Cloning Detection**

Identifies unauthorized use of protected voices.

- Known voice detection: politician, celebrity, protected individual matching
- Speaker verification: ECAPA-TDNN embeddings against registered voice database
- Deepfake confidence scoring
- Consent verification integration

**Dimension 7: Cultural Sensitivity**

Flags potentially culturally harmful content for review.

- Sacred/ceremonial music detection
- Cultural context flagging (requires human review)
- Stereotyping pattern detection
- Note: Flagging only, not adjudication

**Dimension 8: Content Safety**

Detects harmful content in audio.

- Hate speech/slur detection (via ASR + NLP pipeline)
- Harmful instruction detection
- Physical safety: harmful frequencies, volume spikes, epileptogenic patterns
- Age-inappropriate content flagging

#### Aggregate Scores

**Quality Score (0-100):**
- Weighted combination of Dimensions 1-4
- Genre-specific subscores (10+ genre families)
- AIME benchmark compatibility

**Safety Score (ALLOW / FLAG / BLOCK):**
- Aggregation of Dimensions 5-8
- Configurable thresholds per dimension
- Evidence provided for flagged content

**Overall Decision:**
- Combined quality + safety assessment
- Single API response with both evaluations

#### 2. Hierarchical Temporal Analysis

Analyze AI-generated music at multiple time scales:

- **Frame level** (10ms): MERT-v1-95M chunk processing (6 chunks x 100ms per 5s)
- **Beat level** (500ms): madmom DBNBeatTracker (95%+ accuracy), micro-timing patterns
- **Phrase level** (4-16 bars): Melodic contour analysis, harmonic progression tracking
- **Section level** (verse/chorus): SpecTNT structure detection, self-similarity matrices
- **Piece level**: Overall tension-resolution arc, structural completeness assessment

#### 3. Tension-Resolution Modeling

Compute harmonic and rhythmic tension curves using Tonal Interval Space (TIS):

- Cloud diameter (harmonic clarity)
- Cloud momentum (rate of harmonic change)
- Tensile strain (deviation from tonal center)
- Resolution strength at cadential points

#### 4. Comparative Evaluation

- Pairwise ranking: "Which AI generation is higher quality?" (AIME protocol compatible)
- Model comparison: "How does Model A compare to Model B on this prompt?"
- Prompt adherence ranking: "Which generation better matches the text prompt?"
- Style transfer evaluation: "Does this sound like the target genre?"

### Output Formats

**API Response:**

- JSON with 5-dimension scores, confidence intervals, and explanations
- CLAP similarity scores for prompt adherence
- AI artifact confidence scores and locations
- Structure detection timestamps (verse/chorus/bridge boundaries)
- Originality/plagiarism scores with matched fingerprints
- Embedding vectors for downstream tasks
- Temporal analysis (scores over time)

**Human-Readable Report:**

- Overall quality assessment with natural language explanation
- Dimension breakdown with visualizations
- Specific issues identified (artifacts, structure problems, low prompt adherence)

**Benchmark Integration:**

- AIME-compatible pairwise comparison results
- Leaderboard-compatible metrics
- Reproducible evaluation protocol
- Statistical significance testing

### Success Metrics

| Metric | Current State | Target |
|--------|---------------|--------|
| Human correlation (Spearman) | 0.52 (FAD), 0.62 (MAD) | >0.75 |
| AIME benchmark ranking | N/A | Top-3 on pairwise accuracy |
| FAD-CLAP-MA comparison | Baseline | Significant improvement |
| Dimension coverage | 1 (FAD) | 5 core dimensions |
| Processing speed | Varies | <2s per 30s audio |
| Genre coverage | Western-centric | 10+ genre families |

### Non-Goals (V1)

- Human performance evaluation (live musicians, practice feedback, music education)
- Real-time evaluation during live performance
- Symbolic (MIDI-only) evaluation without audio
- Lyrics/vocal content analysis (use existing ASR + NLP)
- Subjective "star quality" or commercial viability prediction

### Validation Datasets

MusiCritic will be validated against established human preference datasets:

| Dataset | Size | Use | License |
|---------|------|-----|---------|
| **AIME** | 6,500 tracks, 15,600 pairwise comparisons | Primary validation benchmark | CC-BY-4.0 |
| **MusicPrefs** | 7 models, pairwise preferences | Secondary validation | Open-source |
| **MusicCaps** | 5,521 clips with expert captions | Text-audio alignment validation | CC-BY-SA 4.0 |
| **MARBLE** | 18 tasks, 12 datasets | Music understanding benchmarking | Various |

### Additional Capabilities

#### Artist Protection System

**Voice Fingerprint Registry**

- Artists register voice embeddings via API
- Configurable similarity thresholds
- Automated takedown workflow integration

**Style Protection**

- Distinguish "sounds like" from "copies from"
- Quantify stylistic similarity vs. direct copying
- Evidence generation for rights claims

#### Input Classifier (Optional Pre-Generation)

For streaming/real-time use cases:

- Prompt analysis for harmful intent before generation
- Artist/voice request detection
- Policy violation prediction

#### Red-Teaming Infrastructure

**Automated Adversarial Testing**

- Jailbreak prompt generation for safety classifiers
- Evasion technique detection
- Classifier robustness measurement

**Continuous Evaluation**

- Attack success rate monitoring
- False positive/negative tracking
- Model drift detection

---

## Requirements

### Open Source Commitment

MusiCritic will be released under permissive open-source licenses:

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

| Requirement | Target |
|-------------|--------|
| Latency (30s audio) | <2 seconds (full evaluation) |
| Latency (streaming) | <500ms (safety-only mode) |
| Throughput | 100 req/min (GPU), 1000 req/min (safety-only) |
| Max audio length | 10 minutes |
| Supported formats | WAV, MP3, FLAC, OGG |
| Sample rates | 16kHz-48kHz |

### Quality Requirements

- **Reliability:** 99.9% uptime for hosted inference
- **Reproducibility:** Deterministic outputs given same input and model version
- **Explainability:** All scores accompanied by contributing factors
- **Auditability:** Complete logging for compliance requirements

---

## Competitive Positioning

### Quality Metrics Comparison

| Capability | FAD | CLAP | MAD | Human-CLAP | MusiCritic |
|------------|-----|------|-----|------------|------------|
| Human correlation | 0.52 | 0.26-0.30 | 0.62 | 0.512 | >0.75 (target) |
| Evaluation dimensions | 1 | 1 | 1 | 1 | 8 (4 quality + 4 safety) |
| Prompt adherence | ❌ | Partial | ❌ | ❌ | ✅ |
| Structure detection | ❌ | ❌ | ❌ | ❌ | ✅ |
| AI artifact detection | ❌ | ❌ | ❌ | ❌ | ✅ |
| Copyright/originality | ❌ | ❌ | ❌ | ❌ | ✅ |
| Voice cloning detection | ❌ | ❌ | ❌ | ❌ | ✅ |
| Content safety | ❌ | ❌ | ❌ | ❌ | ✅ |
| Explainable scores | ❌ | ❌ | ❌ | ❌ | ✅ |
| Open source | ✅ | ✅ | ✅ | ❌ | ✅ |

### Safety Platform Comparison

| Capability | Suno | Udio | Stability | Google | MusiCritic |
|------------|------|------|-----------|--------|------------|
| Quality evaluation | ❌ | ❌ | ❌ | ❌ | ✅ |
| Copyright detection | ❌ | ⚠️ | ✅ | Unknown | ✅ |
| Voice cloning protection | ❌ | ❌ | ✅ | ✅ | ✅ |
| Cultural sensitivity | ❌ | ❌ | ❌ | ❌ | ✅ |
| Content safety | ⚠️ | ⚠️ | ✅ | ✅ | ✅ |
| Open source | ❌ | ❌ | Partial | ❌ | ✅ |
| Unified quality+safety | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## Release Strategy

### Phase 1: Foundation

- Core quality evaluation pipeline (Dimensions 1-4)
- Basic safety classifiers (Dimensions 5-8)
- Hugging Face model release
- Technical blog post + paper preprint

### Phase 2: Expansion

- Full 8-dimension scoring with confidence intervals
- Artist voice protection system
- Python SDK release
- Community feedback integration

### Phase 3: Production

- Input classifier for pre-generation screening
- Hosted inference API
- Industry partnership outreach
- Compliance reporting features

### Phase 4: Ecosystem

- Leaderboard and benchmark establishment
- Integration with major generation frameworks (Suno API, Udio API, etc.)
- Continuous model improvement
- Community contribution program

---

## Success Criteria

### Quality Metrics

1. **Accuracy:** Achieve >0.75 human correlation validated by independent study
2. **Citation:** Referenced in 10+ music AI papers within 12 months
3. **Industry:** Adopted by at least 2 major music AI companies for internal evaluation

### Safety Metrics

1. **Effectiveness:** Reduce attack success rate to <2% on standard adversarial benchmark
2. **Voice Protection:** >98% accuracy on voice clone detection
3. **Protection:** 100+ artists registered in voice protection system

### Overall Adoption

1. **Users:** 1,000+ monthly active users within 6 months
2. **Platforms:** Integrated by 3+ audio platforms within 12 months
3. **Recognition:** Cited in regulatory guidance or industry standards documents

---

## Appendix: User Stories

### Quality Evaluation Use Cases

**As a music AI researcher**, I want to evaluate my model's outputs on multiple dimensions (prompt adherence, coherence, quality, originality) so that I can demonstrate improvements over baselines in my paper.

**As a generative music company**, I want to A/B test different model versions with automated quality gates so that I can ship the one that produces higher-quality outputs.

**As an AI music startup**, I want to compare my model against Suno/Udio baselines on the AIME benchmark so that I can demonstrate competitive performance to investors.

**As a benchmark maintainer**, I want AIME-compatible evaluation protocols so that the community can fairly compare different AI music generation approaches.

### Safety Evaluation Use Cases

**As a platform trust & safety lead**, I want to detect copyrighted content and voice cloning before publication so that we avoid legal liability.

**As an artist**, I want to register my voice so that unauthorized clones can be detected and removed.

**As a compliance officer**, I want unified quality and safety reports so that we meet EU AI Act requirements efficiently.

**As a red team researcher**, I want to test model robustness against adversarial attacks so that we can improve safety before deployment.

**As a content moderator**, I want flagged content with evidence so that I can make informed decisions efficiently.

### Unified Use Cases

**As a generative music company**, I want a single API call that tells me both "is this music good?" and "is this music safe?" so that I can streamline my evaluation pipeline.
