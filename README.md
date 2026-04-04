---
title: MAUDE NLP Severity Classifier
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.34.0
app_file: streamlit_app/app.py
pinned: false
license: mit
---

# 🏥 MAUDE NLP Severity Classifier

[![CI/CD Pipeline](https://github.com/makymaverick/maude-nlp-classifier/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/makymaverick/maude-nlp-classifier/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end NLP pipeline that ingests medical device adverse event reports from the [openFDA MAUDE database](https://open.fda.gov/apis/device/event/), classifies them by **severity** (Death · Injury · Malfunction · Other · Unknown), and serves predictions via a Streamlit web app.

Two classifier backends are available:
- **TF-IDF + Logistic Regression/SVM** — fast baseline, CPU-only, trains in seconds
- **ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`) — fine-tuned transformer, CV F1 **0.941**, hosted on [HuggingFace Hub](https://huggingface.co/mukundisb/maude-clinicalbert)

---

## Architecture

```
openFDA MAUDE API
       │
       ▼
┌──────────────────────────┐
│  Data Ingestion           │  src/ingestion/openfda_client.py
│  (paginated REST, incr.)  │  src/ingestion/incremental.py
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Text Preprocessing       │  src/preprocessing/text_cleaner.py
│  clean · normalize ·      │  • Boilerplate removal
│  preserve digits (BERT)   │  • Abbreviation expansion
└────────────┬─────────────┘  • Digit preservation for BERT
             │
             ├─────────────────────────────────────┐
             ▼                                     ▼
┌─────────────────────────┐          ┌──────────────────────────────┐
│  TF-IDF + LogReg / SVM  │          │  ClinicalBERT Fine-tuned      │
│  src/model/classifier.py│          │  src/model/bert_classifier.py │
│  CV F1: baseline         │          │  CV F1: 0.941 (10k records)  │
│  ~4 MB · CPU · <2 min   │          │  ~440 MB · GPU train · HF Hub│
└────────────┬────────────┘          └──────────────┬───────────────┘
             └──────────────┬──────────────────────┘
                            │
                            ▼
             ┌──────────────────────────┐
             │  MLflow Experiment        │
             │  Tracking + Promotion     │
             │  Gate (CV F1 ≥ +0.5%)    │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐      ┌──────────────────────────┐
             │  Streamlit App            │ ───▶ │  AWS ECS/Fargate          │
             │  Train · Infer · Explore  │      │  ALB → Container → 8501  │
             │  Pipeline Dashboard       │      │  GPU task for BERT batch  │
             └──────────────────────────┘      └──────────────────────────┘
                                                          │
                                               ┌──────────┴──────────┐
                                               │  S3 Checkpoints      │
                                               │  HF Hub Model Repo   │
                                               │  ECR Training Image  │
                                               └─────────────────────┘
```

---

## Model Performance (Phase 1)

| Model | CV F1 (weighted) | CV Std | Training Records | Inference Speed |
|-------|-----------------|--------|-----------------|-----------------|
| TF-IDF + LogReg (baseline) | — | — | — | < 1 ms (CPU) |
| **ClinicalBERT (current champion)** | **0.941** | **±0.003** | **10,000** | ~500 ms (CPU) |

Checkpoint: [`mukundisb/maude-clinicalbert`](https://huggingface.co/mukundisb/maude-clinicalbert)

---

## Quickstart (Local)

### Prerequisites
- Python 3.11+
- Docker + Docker Compose (optional)

### 1. Clone & install

```bash
git clone https://github.com/makymaverick/maude-nlp-classifier.git
cd maude-nlp-classifier

# Core dependencies (TF-IDF pipeline, Streamlit, MLflow)
pip install -r requirements.txt

# BERT dependencies (required to use ClinicalBERT inference)
pip install -r requirements-bert.txt
```

### 2. (Optional) Set your openFDA API key

```bash
export OPENFDA_API_KEY=your_key_here
```

Without a key you get the public rate limit (~1,000 req/min). Register free at https://open.fda.gov/apis/authentication/.

### 3a. Train the TF-IDF baseline

```bash
python -m src.model.train \
  --records 5000 \
  --model logreg \
  --drop-unknown
```

### 3b. Fine-tune ClinicalBERT (requires GPU)

```bash
# Local GPU
HF_TOKEN=your_hf_token python -m src.model.train_bert \
  --records 10000 \
  --epochs 3 \
  --hub-repo mukundisb/maude-clinicalbert \
  --drop-unknown

# Or use the Kaggle notebook (free T4 GPU — see notebooks/kaggle_bert_training.ipynb)
```

### 4. Run the Streamlit app

```bash
streamlit run streamlit_app/app.py
```

Open http://localhost:8501. Use the **Inference Model** selector in the sidebar to switch between TF-IDF and ClinicalBERT.

### 5. Run with Docker Compose

```bash
docker compose up -d
docker compose --profile train up train
```

---

## Project Structure

```
maude-nlp-classifier/
├── src/
│   ├── ingestion/
│   │   ├── openfda_client.py        # openFDA MAUDE API client
│   │   └── incremental.py           # incremental ingestion + auto-retrain scheduler
│   ├── preprocessing/
│   │   └── text_cleaner.py          # text normalisation (digit-preserve mode for BERT)
│   └── model/
│       ├── classifier.py            # TF-IDF + LogReg/SVM pipeline
│       ├── train.py                 # TF-IDF training entrypoint (CLI)
│       ├── bert_classifier.py       # ClinicalBERT model, train, CV, predict, HF Hub I/O
│       └── train_bert.py            # BERT fine-tuning entrypoint (CLI)
├── streamlit_app/
│   └── app.py                       # Streamlit app (model selector, train, infer, explore)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   └── kaggle_bert_training.ipynb   # Ready-to-run Kaggle notebook (free T4 GPU)
├── models/
│   ├── maude_classifier.joblib      # TF-IDF champion checkpoint
│   ├── champion_metrics.json        # Current champion metrics + model type
│   └── bert_model_ref.json          # HF Hub pointer (repo + commit SHA)
├── tests/
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   └── test_classifier.py
├── deploy/
│   ├── ecs/
│   │   └── task-definition.json     # ECS Fargate task definition
│   └── terraform/
│       ├── main.tf                  # AWS infra: VPC, ECR, ECS, ALB, S3, SageMaker role
│       ├── variables.tf
│       └── outputs.tf
├── .github/workflows/
│   ├── ci-cd.yml                    # CI: test → build → push ECR → deploy ECS
│   ├── bert-train.yml               # ClinicalBERT fine-tuning via AWS SageMaker
│   └── scheduled-ingestion.yml      # Scheduled incremental data ingestion
├── Dockerfile                       # Multi-stage: TF-IDF (~400 MB) or BERT (~3 GB) variant
├── docker-compose.yml
├── requirements.txt                 # Core dependencies
├── requirements-bert.txt            # BERT dependencies (torch, transformers, accelerate)
└── IMPLEMENTATION_PLAN.md           # Phase 2–5 roadmap
```

---

## ClinicalBERT Training

### Option A — GitHub Actions → AWS SageMaker

Trigger the [`bert-train.yml`](.github/workflows/bert-train.yml) workflow manually from the Actions tab.

Required GitHub Secrets:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM access key |
| `AWS_SECRET_ACCESS_KEY` | IAM secret key |
| `AWS_REGION` | e.g. `us-east-1` |
| `AWS_ACCOUNT_ID` | 12-digit AWS account ID |
| `HF_TOKEN` | HuggingFace write token |
| `OPENFDA_API_KEY` | openFDA API key (optional) |

Infrastructure must be provisioned first via Terraform (see AWS Deployment below).

### Option B — Kaggle (free T4 GPU)

1. Upload `notebooks/kaggle_bert_training.ipynb` to [kaggle.com/code](https://kaggle.com/code)
2. Enable GPU: Notebook Settings → Accelerator → GPU T4 x1
3. Add `HF_TOKEN` via Add-ons → Secrets
4. Run all cells (~60–90 min)
5. Commit the updated `models/bert_model_ref.json` back to the repo

---

## TF-IDF Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--records` | 5000 | Number of MAUDE records to fetch |
| `--model` | `logreg` | Classifier: `logreg` or `svm` |
| `--tune` | off | Enable GridSearchCV hyperparameter tuning |
| `--use-cached` | off | Load from `data/raw/maude_raw.csv` instead of re-fetching |
| `--drop-unknown` | off | Exclude UNKNOWN-labeled records |
| `--cross-validate` | off | Run 5-fold cross-validation |

## ClinicalBERT Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--records` | 5000 | Number of MAUDE records to fetch |
| `--epochs` | 3 | Fine-tuning epochs |
| `--lr` | 2e-5 | AdamW learning rate |
| `--batch-size` | 16 | Per-device batch size |
| `--max-length` | 256 | Max token length (256 = 95th pct of MAUDE narratives) |
| `--hub-repo` | None | HuggingFace Hub repo to push checkpoint |
| `--use-cached` | off | Load from cached CSV |
| `--drop-unknown` | off | Exclude UNKNOWN-labeled records |

---

## Severity Label Mapping

| openFDA `event_type` | Severity Label |
|----------------------|---------------|
| Death | `D` |
| Serious Injury / Injury | `I` |
| Malfunction | `M` |
| Other | `O` |
| No Answer Provided / * | `UNKNOWN` |

---

## AWS Deployment

### Step 1 — Provision infrastructure

```bash
cd deploy/terraform
terraform init
terraform apply -var="openfda_api_key=$OPENFDA_API_KEY"
```

Creates: VPC, ECR, ECS Cluster + Fargate Service, ALB, S3 checkpoints bucket, SageMaker IAM role, CloudWatch logs.

### Step 2 — Build and push Docker image

```bash
ECR_URI=$(terraform output -raw ecr_repository_url)
AWS_REGION=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")

aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ECR_URI

# TF-IDF only image
docker build -t $ECR_URI:latest .
docker push $ECR_URI:latest

# BERT-enabled image (GPU runtime)
docker build --build-arg INCLUDE_BERT=true \
             --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 \
             -t $ECR_URI:bert-latest .
docker push $ECR_URI:bert-latest
```

### Step 3 — The app is live

```bash
terraform output app_url   # → http://<alb-dns-name>
```

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | ClinicalBERT fine-tuning · HF Hub hosting · CV F1 0.941 |
| 2 | Planned | Multi-label classification (sigmoid heads, BCEWithLogitsLoss) |
| 3 | Planned | Multi-task learning (severity + device problem + patient problem) |
| 4 | Planned | Named Entity Recognition (device, body part, adverse event) |
| 5 | Planned | FHIR R4 AdverseEvent export |

See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for full technical design of Phases 2–5.

---

## License

MIT — see [LICENSE](LICENSE)

---

## Disclaimer

This tool is intended for research and informational purposes only. It is not a substitute for professional medical or regulatory review. Always consult qualified professionals for regulatory submissions and safety decisions.
