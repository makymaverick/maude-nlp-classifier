# MAUDE NLP Classifier — Future Scope Implementation Plan

**Version:** 1.0
**Date:** 2026-03-29
**Current Model:** TF-IDF + Logistic Regression / Linear SVM
**Target Architecture:** ClinicalBERT · Multi-label · Multi-task · NER · FHIR

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Limitations](#2-current-architecture-limitations)
3. [Assumptions](#3-assumptions)
4. [Why BERT Beats TF-IDF on This Task](#4-why-bert-beats-tf-idf-on-this-task)
5. [Phase 1 — ClinicalBERT Upgrade](#5-phase-1--clinicalbert-upgrade)
6. [Phase 2 — Multi-label Classification](#6-phase-2--multi-label-classification)
7. [Phase 3 — Multi-task Learning](#7-phase-3--multi-task-learning)
8. [Phase 4 — Named Entity Recognition (NER)](#8-phase-4--named-entity-recognition-ner)
9. [Phase 5 — FHIR Integration](#9-phase-5--fhir-integration)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Deployment Strategy](#11-deployment-strategy)
12. [Risk & Mitigation](#12-risk--mitigation)
13. [Dependency & Infrastructure Changes](#13-dependency--infrastructure-changes)

---

## 1. Executive Summary

The current system classifies MAUDE adverse event narratives into five severity buckets
(Death, Injury, Malfunction, Other, Unknown) using a TF-IDF bag-of-words representation
fed into a logistic regression or SVM classifier. This works as a strong baseline but
misses clinical semantics, cannot handle overlapping labels, and produces no structured
output that downstream healthcare systems can consume.

This plan describes five sequential phases that evolve the system from a lexical
classifier into a clinical-grade NLP pipeline:

| Phase | Scope | Key Gain |
|-------|-------|----------|
| 1 | Replace TF-IDF with ClinicalBERT | +5–12 F1 pts on severity |
| 2 | Multi-label classification | Handle reports with multiple concurrent outcomes |
| 3 | Multi-task learning | Jointly predict severity + device class + event type |
| 4 | NER | Extract structured entities (device, body part, adverse event) |
| 5 | FHIR integration | Emit HL7 FHIR R4 AdverseEvent resources |

Each phase is independently shippable and backward-compatible with the existing
Streamlit UI and openFDA pipeline.

---

## 2. Current Architecture Limitations

| Limitation | Impact |
|------------|--------|
| TF-IDF loses word order and context | "patient death was ruled out" ≠ "patient death" |
| Vocabulary fixed at training time | New medical terminology requires full retrain |
| Single-label only | ~12% of real MAUDE reports involve both injury AND malfunction |
| No entity extraction | Device names, body parts, and adverse event terms are buried in free text |
| No interoperability standard | Output cannot feed EHR systems, FDA databases, or FHIR-capable registries |
| Logistic regression on sparse features | Limited ability to generalize to unseen phrasings |
| Text preprocessing strips digits | Loses dosage and device model numbers |

---

## 3. Assumptions

### Data Assumptions
- **Label quality:** openFDA `event_type` field is used as ground truth. It contains
  self-reported labels, which are noisy (~8–15% mislabeled based on narrative content).
  BERT's contextual embeddings help compensate but do not eliminate this noise.
- **Dataset size:** ClinicalBERT fine-tuning needs a minimum of ~5,000 labeled records
  to generalize; 20,000+ is preferred for multi-task and NER heads.
- **Multi-label ground truth:** openFDA does not natively provide multi-label annotations.
  Multi-label training labels will be derived heuristically from `event_type` combinations
  across linked report numbers (same MDR report filed under multiple event types).
- **Class imbalance:** Malfunction is dominant (~60–65% of records). Death is rare (~3%).
  All phases must account for imbalance via class weighting or focal loss.
- **Text length:** Narrative lengths range from 20 to 2,000+ tokens. ClinicalBERT has a
  512-token limit; longer narratives will be handled by truncation (first 512) or
  hierarchical pooling.

### Infrastructure Assumptions
- **GPU availability:** Phase 1 fine-tuning requires a GPU (minimum 16 GB VRAM for
  bert-base-sized models). Training will run locally or on a cloud GPU instance (e.g.,
  Google Colab Pro, AWS g4dn.xlarge, or Kaggle).
- **HF Spaces CPU inference:** The deployed Streamlit app on HF Spaces runs on CPU.
  BERT inference at CPU is ~500–1000 ms per narrative. Acceptable for demo; production
  would use GPU or model quantization (INT8/ONNX).
- **Model storage:** Fine-tuned BERT checkpoint (~440 MB) cannot be stored in the git
  repo. It will be hosted on Hugging Face Hub (model repository, separate from the
  Spaces app) and loaded at startup.
- **FHIR server:** Phase 5 assumes a FHIR R4-compliant endpoint is available (HAPI FHIR,
  Azure Health Data Services, or AWS HealthLake). Integration is mocked in dev mode.

### Modeling Assumptions
- **Pretrained model:** `emilyalsentzer/Bio_ClinicalBERT` is the primary candidate. It was
  pretrained on MIMIC-III clinical notes, which share vocabulary with MAUDE narratives.
  Alternative: `allenai/biomed_roberta_base` (biomedical abstracts + clinical notes).
- **Fine-tuning strategy:** Full fine-tuning of all layers (not just the classification
  head) because MAUDE narratives differ from general biomedical text in style and domain.
- **Tokenization:** ClinicalBERT WordPiece tokenizer; no custom vocabulary extension
  needed for Phase 1. NER phase may add domain-specific tokens.
- **Multi-task learning:** Task-specific heads share the BERT encoder but have independent
  classification layers. Gradient balancing uses uncertainty weighting (Kendall et al.).

---

## 4. Why BERT Beats TF-IDF on This Task

### Conceptual Differences

| Property | TF-IDF + LR | ClinicalBERT |
|----------|-------------|--------------|
| Representation | Sparse bag-of-words | Dense contextual embeddings |
| Context window | None (unigrams/bigrams) | 512 tokens (bidirectional) |
| Negation handling | None ("no death" ≈ "death") | Learned from context |
| Medical synonymy | Misses unless both in vocabulary | Encoded in pretrained weights |
| New terminology | Unseen → zero weight | Sub-word tokenization handles OOV |
| Domain knowledge | None (learns from scratch) | MIMIC-III pretrained |
| Model size | ~4 MB (TF-IDF + weights) | ~440 MB (bert-base) |
| Training time | < 2 min on CPU | 20–60 min on GPU |
| Inference speed | < 1 ms | 500–1000 ms on CPU |

### Expected Performance Gains

Based on comparable clinical text classification studies (n2c2 challenges, MIMIC
benchmarks), switching from TF-IDF to ClinicalBERT on short clinical narratives yields:

- **Weighted F1:** +5 to +12 percentage points
- **Death class recall:** +15 to +25 points (rare class benefits most from contextual
  embeddings — "the patient expired" → "Death" correctly)
- **Malfunction precision:** +3 to +8 points (reduced false positives from boilerplate text)
- **Out-of-vocabulary robustness:** Complete elimination of zero-weight unseen terms

### Concrete Example

```
Narrative: "No fatality was observed but the infusion pump delivered an
            incorrect dosage resulting in hospitalization."

TF-IDF: Sees "fatality" → Death (FP), misses "hospitalization" signal in context
BERT:   Understands "No fatality" + "hospitalization" → Serious Injury (correct)
```

---

## 5. Phase 1 — ClinicalBERT Upgrade

### Goal
Replace the TF-IDF + LR/SVM pipeline with a fine-tuned ClinicalBERT model for
severity classification. Maintain the same 5-class output and the existing Streamlit
interface.

### Model Architecture

```
Input Narrative (string)
       ↓
ClinicalBERT Tokenizer (WordPiece, max_length=512)
       ↓
Bio_ClinicalBERT Encoder (12 layers, 768 hidden, 12 heads)
       ↓
[CLS] token embedding (768-dim)
       ↓
Dropout(0.1)
       ↓
Linear(768 → 5)  ← Severity classification head
       ↓
Softmax → {D, I, M, O, UNKNOWN}
```

### Implementation Steps

1. **Add dependencies:** `transformers>=4.40.0`, `torch>=2.2.0`, `accelerate>=0.29.0`
2. **New module:** `src/model/bert_classifier.py`
   - `BertSeverityClassifier` (nn.Module wrapping Bio_ClinicalBERT)
   - `train_bert()` — fine-tuning loop with AdamW + linear warmup scheduler
   - `predict_bert()` — inference wrapper returning probabilities
   - `save_bert_checkpoint()` / `load_bert_checkpoint()` — HF Hub integration
3. **New training script:** `src/model/train_bert.py`
   - CLI flags: `--epochs`, `--lr`, `--batch-size`, `--max-length`, `--hub-repo`
   - Same MLflow tracking as `train.py`
   - Same 3-gate model promotion logic (using CV F1)
4. **Preprocessing update:** Preserve digit tokens (remove digit-stripping step from
   `text_cleaner.py`) — BERT tokenizer handles them natively.
5. **Model serialization:** Save fine-tuned weights to HF Hub; `models/` stores only
   a pointer JSON (`bert_model_ref.json`) with the Hub repo path and commit SHA.
6. **Streamlit integration:** Add a model selector in the sidebar (TF-IDF or ClinicalBERT).
   BERT model loads once at startup via `@st.cache_resource`.

### Hyperparameters (starting point)

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 2e-5 (BERT fine-tuning standard) |
| Batch size | 16 (GPU) / 8 (gradient accumulation on small GPU) |
| Epochs | 3–5 |
| Warmup steps | 10% of total steps |
| Weight decay | 0.01 |
| Max token length | 256 (covers 95th percentile of MAUDE narratives) |
| Dropout | 0.1 |
| Class weights | Inverse frequency (same as TF-IDF baseline) |

---

## 6. Phase 2 — Multi-label Classification

### Goal
Allow a single MAUDE report to be assigned multiple severity labels simultaneously.
A report describing a device malfunction that caused patient injury should output
both `M` and `I`, not force a single choice.

### Problem Reframing

Current (single-label):
```
Input → [D, I, M, O, UNKNOWN] → argmax → one label
```

Multi-label:
```
Input → [D, I, M, O, UNKNOWN] → sigmoid per label → threshold → label set
```

### Label Construction Strategy

Since openFDA does not provide native multi-label annotations:

1. **Linked report aggregation:** MAUDE reports share `report_number` prefixes when
   the same event is filed multiple times by different reporters (manufacturer, user
   facility, consumer). Aggregate all `event_type` values per MDR number → multi-hot
   label vector.
2. **NLI-based pseudo-labeling:** Use a zero-shot NLI model
   (`facebook/bart-large-mnli`) to validate candidate labels against narrative text.
   Keep labels with entailment score > 0.7.
3. **Manual annotation sample:** Annotate 500 records via Label Studio to validate
   the automated labeling pipeline before training.

### Architecture Change

Replace the softmax head with independent sigmoid heads:

```
[CLS] embedding (768-dim)
       ↓
Dropout(0.1)
       ↓
Linear(768 → 5)  ← one logit per label
       ↓
Sigmoid(per label) → probabilities ∈ [0,1]
       ↓
Threshold(0.5) → multi-hot vector
```

**Loss:** Binary Cross-Entropy with Logits (BCEWithLogitsLoss), per-label class weights.

### Threshold Tuning

The default 0.5 threshold is suboptimal. Use F1-maximizing threshold search per label
on the validation set:
```python
thresholds = {label: find_best_threshold(val_probs[:, i], val_labels[:, i])
              for i, label in enumerate(LABEL_NAMES)}
```

### Streamlit Update

- Inference tab shows a probability bar for each label (not just the top-1 prediction).
- Reports where multiple labels exceed threshold display a "multi-outcome" badge.

---

## 7. Phase 3 — Multi-task Learning

### Goal
Train a single BERT encoder to simultaneously predict:
- **Task A:** Severity label (D / I / M / O / UNKNOWN) — existing task
- **Task B:** Device problem code (FDA device problem code taxonomy, ~50 classes)
- **Task C:** Patient problem code (patient outcome taxonomy, ~30 classes)

Joint training improves all three tasks by sharing representations of clinical
language across related prediction objectives.

### Architecture

```
Input Narrative
       ↓
ClinicalBERT Shared Encoder
       ↓
[CLS] embedding (768-dim)
    /     |      \
   /      |       \
Head A   Head B   Head C
Severity  Device   Patient
(5 cls)  Problem  Problem
         (~50 cls) (~30 cls)
```

Each head is a `Linear(768 → num_classes)` layer with independent loss functions.

### Loss Balancing

Use learned uncertainty weighting (Kendall et al. 2018):

```python
loss = (1/(2*σ_A²)) * L_A + log(σ_A) +
       (1/(2*σ_B²)) * L_B + log(σ_B) +
       (1/(2*σ_C²)) * L_C + log(σ_C)
```

Where σ_A, σ_B, σ_C are learnable log-variance parameters. This prevents any single
task from dominating the gradient signal.

### Data Requirements

- Task B & C labels come from the openFDA `device_problem_codes` and
  `patient_problem_codes` fields, which are present in a subset of MAUDE records
  (~40% coverage). Records without Task B/C labels use masked loss (set weight = 0
  for those samples on those heads).

---

## 8. Phase 4 — Named Entity Recognition (NER)

### Goal
Extract structured named entities from MAUDE narratives:

| Entity Type | Example |
|-------------|---------|
| `DEVICE` | "insulin pump", "pacemaker lead", "catheter" |
| `BODY_PART` | "left ventricle", "femoral artery", "skin" |
| `ADVERSE_EVENT` | "electrical failure", "dislodgement", "perforation" |
| `PATIENT_OUTCOME` | "hospitalization", "surgery required", "death" |
| `MANUFACTURER` | "Medtronic", "Abbott", "Boston Scientific" |

### Architecture

Add a token classification head to the BERT encoder:

```
Input Tokens
       ↓
ClinicalBERT Encoder
       ↓
Token embeddings (seq_len × 768)
       ↓
Linear(768 → num_entity_types × 3)   ← BIO tagging (Begin, Inside, Outside)
       ↓
CRF layer (optional, improves boundary detection)
       ↓
BIO tag sequence → entity spans
```

### Training Data

1. **Existing annotated corpora:**
   - i2b2/n2c2 2010 NER dataset (medical concepts in clinical notes)
   - NCBI Disease corpus (disease mentions)
   - BC5CDR corpus (chemical/disease mentions)
2. **MAUDE-specific annotation:** Use the 500 manually annotated records from Phase 2
   plus additional annotation of ~2,000 records for device and adverse event entities.
3. **Weak supervision:** Use dictionary lookups (FDA device taxonomy, SNOMED CT body
   parts, MedDRA adverse event terms) to generate noisy BIO labels for the full dataset.
   Then fine-tune on the clean manually annotated subset.

### Streamlit Integration

- Inference tab renders narrative with inline entity highlights (spaCy displacy-style).
- Extracted entities appear in a structured table below the prediction.

---

## 9. Phase 5 — FHIR Integration

### Goal
Transform classifier outputs into HL7 FHIR R4 `AdverseEvent` resources that can be
consumed by any FHIR-capable EHR, regulatory database, or analytics platform.

### FHIR AdverseEvent Resource Mapping

| FHIR Field | Source |
|------------|--------|
| `resourceType` | "AdverseEvent" (static) |
| `status` | "completed" |
| `actuality` | "actual" (MAUDE = confirmed events) |
| `category` | Mapped from severity label (D→"serious", I→"serious", M→"non-serious") |
| `event.coding` | MedDRA code looked up from NER `ADVERSE_EVENT` entity |
| `subject` | Anonymous patient reference |
| `date` | `date_received` from openFDA record |
| `seriousness` | Boolean (True for D, I) |
| `suspectEntity.instance` | NER `DEVICE` entity → Device resource reference |
| `outcome.coding` | Mapped from patient problem code (Task C) |
| `extension[severity_score]` | BERT softmax probability of predicted class |
| `extension[classifier_version]` | Model checkpoint SHA from HF Hub |

### Implementation

1. **New module:** `src/fhir/adverse_event_builder.py`
   - `build_adverse_event(record, prediction, entities) → dict` (FHIR JSON)
   - `validate_fhir_resource(resource) → bool` using `fhir.resources` library
2. **New module:** `src/fhir/fhir_client.py`
   - `post_to_fhir_server(resource, fhir_base_url, auth_token)` — POST to FHIR endpoint
   - `batch_post(resources, bundle_size=50)` — FHIR Bundle transaction
3. **Streamlit integration:**
   - "Export as FHIR" button on the Inference tab — downloads FHIR JSON
   - Pipeline Dashboard shows FHIR export status per ingestion batch
4. **FHIR server options:**
   - **Dev/demo:** HAPI FHIR public test server (`https://hapi.fhir.org/baseR4`)
   - **Production:** Azure Health Data Services / AWS HealthLake (requires auth setup)

---

## 10. Evaluation Metrics

### Phase 1 — ClinicalBERT Severity Classification

| Metric | Formula | Why |
|--------|---------|-----|
| Weighted F1 | ∑ class_weight × F1_class | Primary metric; handles class imbalance |
| Macro F1 | Mean of per-class F1 (unweighted) | Ensures rare classes (Death) are not ignored |
| Per-class Recall (Death) | TP_D / (TP_D + FN_D) | Missing a Death report is the worst error |
| Per-class Precision (Malfunction) | TP_M / (TP_M + FP_M) | Dominant class; precision matters for noise |
| Cohen's Kappa | (p_o - p_e) / (1 - p_e) | Agreement beyond chance, standard for clinical NLP |
| Calibration (ECE) | Avg. gap between confidence and accuracy | Reliable confidence scores for FHIR export |
| Inference latency (p95) | 95th pct of single-record inference time | Production readiness gate |

**Promotion gate (Phase 1):** New BERT checkpoint replaces champion only if
CV F1 (weighted) improves by ≥ 0.5% over current champion (TF-IDF or previous BERT).

### Phase 2 — Multi-label

| Metric | Notes |
|--------|-------|
| Micro F1 | Aggregate TP/FP/FN across all labels × samples |
| Macro F1 | Average F1 per label (treats rare combos equally) |
| Hamming Loss | Fraction of labels incorrectly predicted per sample |
| Jaccard Similarity (per sample) | `\|pred ∩ true\| / \|pred ∪ true\|` |
| Coverage Error | How far down ranked labels to include all true labels |
| Label Ranking Avg Precision | Area under precision-recall for ranked labels |

### Phase 3 — Multi-task

Report each task's metric independently (Weighted F1 for A, Macro F1 for B & C)
plus a composite score:

```
Composite = 0.5 × F1_severity + 0.3 × F1_device_problem + 0.2 × F1_patient_problem
```

Track gradient norm per task head per epoch to detect task dominance.

### Phase 4 — NER

| Metric | Notes |
|--------|-------|
| Entity-level F1 (strict) | Exact span match required (both boundaries + type correct) |
| Entity-level F1 (partial) | Token overlap counted proportionally |
| Per-entity-type F1 | DEVICE, BODY_PART, ADVERSE_EVENT reported separately |
| Span boundary accuracy | Correct boundary detection regardless of entity type |

CoNLL-2003 evaluation script (exact match) is the standard.

### Phase 5 — FHIR

| Metric | Notes |
|--------|-------|
| FHIR validation pass rate | % of resources passing `fhir.resources` schema validator |
| MedDRA code lookup precision | % of NER adverse events mapped to valid MedDRA codes |
| Round-trip fidelity | % of fields recovered correctly after POST and GET from FHIR server |

---

## 11. Deployment Strategy

### Current Deployment (Baseline)

```
HF Spaces (Streamlit SDK)
  └─ TF-IDF model (~4 MB, in repo)
  └─ CPU inference, < 5 ms/record
  └─ No GPU required
```

### Phase 1 Deployment (ClinicalBERT on HF Spaces)

```
HF Spaces (Streamlit SDK, CPU)
  └─ Fine-tuned ClinicalBERT (~440 MB) → loaded from HF Hub at startup
  └─ CPU inference: ~500–1000 ms/record (acceptable for demo)
  └─ st.cache_resource ensures model loads once per container lifecycle
  └─ Quantized INT8 ONNX export reduces latency to ~150 ms on CPU
```

**Model hosting:**
- Fine-tuned weights live at `mukundisb/maude-clinicalbert` on HF Hub (model repo,
  separate from the Spaces app repo).
- `bert_model_ref.json` in the Spaces repo pins the exact commit SHA for reproducibility.

**Memory constraint:** HF Spaces free tier = 16 GB RAM. BERT-base in FP32 = ~1.7 GB
peak during inference. Safe with quantization (< 500 MB).

### Phase 2–3 Deployment (Multi-label / Multi-task)

Same as Phase 1. The additional heads add < 10 MB to the checkpoint. No infrastructure
change needed.

### Phase 4 Deployment (NER)

Add the NER head to the same BERT checkpoint. Inference returns both the classification
result and the entity spans in one forward pass. No additional service needed.

### Phase 5 Deployment (FHIR)

```
HF Spaces (Streamlit)
  └─ FHIR export runs as a post-processing step (no server needed for JSON download)
  └─ Optional: POST to external FHIR endpoint configured via HF Spaces secret
     (FHIR_BASE_URL + FHIR_AUTH_TOKEN environment variables)
```

**Production path (beyond HF Spaces demo):**

```
AWS Architecture
  ├── ECR — Docker image with BERT + FHIR modules
  ├── ECS Fargate (GPU task definition) — async batch inference
  ├── API Gateway + Lambda — synchronous single-record inference endpoint
  ├── AWS HealthLake — FHIR R4 managed store
  └── S3 — model checkpoints, raw FHIR bundles
```

Existing Terraform in `deploy/terraform/` provides the VPC and ECS scaffolding;
add a GPU task definition and HealthLake resource for full production deployment.

### CI/CD Changes

Extend `.github/workflows/ci-cd.yml`:
- Add BERT fine-tuning job (runs on GPU runner or dispatched to Colab/SageMaker)
- Add FHIR validation step (`fhir-validator` CLI on generated test resources)
- Model promotion gate: only push new checkpoint to HF Hub if CV F1 improves

---

## 12. Risk & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| HF Spaces OOM with BERT FP32 | Medium | App crash | INT8 quantization / ONNX export before deploying |
| Label noise in multi-label construction | High | Lower multi-label F1 | Manual validation of 500 records; use NLI pseudo-labeling as filter |
| Multi-task gradient imbalance | Medium | One task dominates | Uncertainty weighting + per-task gradient monitoring |
| BERT 512 token limit on long narratives | Medium | Truncation loses tail information | Test first-512 vs. sliding-window pooling; most MAUDE narratives < 256 tokens |
| MedDRA licensing for FHIR codes | Low | Cannot distribute MedDRA codes | Use SNOMED CT (open) as fallback; MedDRA lookup via API only |
| openFDA API schema changes | Low | Ingestion pipeline breaks | Add schema version check at ingestion; pin tested API version |
| Fine-tuning on small dataset (< 10k) | High | Overfitting | Use early stopping + learning rate warmup; freeze lower BERT layers if < 5k records |

---

## 13. Dependency & Infrastructure Changes

### New Python Dependencies (phased)

```text
# Phase 1 — ClinicalBERT
transformers>=4.40.0
torch>=2.2.0
accelerate>=0.29.0
huggingface_hub>=0.22.0
onnxruntime>=1.18.0        # CPU-optimized inference

# Phase 2–3 — Multi-label / Multi-task
(no new deps beyond Phase 1)

# Phase 4 — NER
seqeval>=1.2.2              # NER evaluation metrics
spacy>=3.7.0                # Entity visualization in Streamlit

# Phase 5 — FHIR
fhir.resources>=7.1.0       # FHIR R4 resource models + validation
```

### Model Storage Strategy

| Artifact | Location | Size |
|----------|----------|------|
| TF-IDF + LR checkpoint | `models/maude_classifier.joblib` (in repo) | ~4 MB |
| ClinicalBERT fine-tuned | HF Hub: `mukundisb/maude-clinicalbert` | ~440 MB |
| BERT ONNX quantized | HF Hub (same repo, `onnx/` subfolder) | ~110 MB |
| bert_model_ref.json | `models/bert_model_ref.json` (in repo, pins commit SHA) | < 1 KB |

### Summary Roadmap

```
Phase 1  ████████░░░░░░░░░░░░  ClinicalBERT fine-tuning + HF Hub hosting
Phase 2  ░░░░████████░░░░░░░░  Multi-label heads + label construction
Phase 3  ░░░░░░░░████████░░░░  Multi-task (device + patient problem codes)
Phase 4  ░░░░░░░░░░░░████████  NER entity extraction
Phase 5  ░░░░░░░░░░░░░░░░████  FHIR R4 AdverseEvent export
```

Each phase gate: **CV F1 improvement ≥ 0.5% over previous champion before merging.**
