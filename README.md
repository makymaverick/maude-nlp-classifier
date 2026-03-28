# 🏥 MAUDE NLP Severity Classifier

[![CI/CD Pipeline](https://github.com/makymaverick/maude-nlp-classifier/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/makymaverick/maude-nlp-classifier/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end NLP pipeline that ingests medical device adverse event reports from the [openFDA MAUDE database](https://open.fda.gov/apis/device/event/), classifies them by **severity** (Death · Serious Injury · Injury · Malfunction · Unknown) using a TF-IDF + Logistic Regression/SVM classifier, and serves predictions via a Streamlit web app deployed on AWS ECS/Fargate.

---

## Architecture

```
openFDA MAUDE API
       │
       ▼
┌─────────────────────┐
│  Data Ingestion      │  src/ingestion/openfda_client.py
│  (paginated REST)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Text Preprocessing  │  src/preprocessing/text_cleaner.py
│  (clean + normalize) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  TF-IDF + Classifier │  src/model/classifier.py
│  (LogReg or SVM)     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐      ┌──────────────────────────┐
│  Streamlit App       │ ───▶ │  AWS ECS/Fargate          │
│  (train/infer/explore│      │  (ALB → Container → 8501) │
└─────────────────────┘      └──────────────────────────┘
```

---

## Quickstart (Local)

### Prerequisites
- Python 3.11+
- Docker + Docker Compose

### 1. Clone & install
```bash
git clone https://github.com/YOUR_ORG/maude-nlp-classifier.git
cd maude-nlp-classifier
pip install -r requirements.txt
```

### 2. (Optional) Set your openFDA API key
```bash
export OPENFDA_API_KEY=your_key_here
```
Without a key you'll use the public rate limit (~40 req/min). Register free at https://open.fda.gov/apis/authentication/.

### 3. Fetch data & train from CLI
```bash
python -m src.model.train \
  --records 5000 \
  --model logreg \
  --drop-unknown
```

### 4. Run the Streamlit app locally
```bash
streamlit run streamlit_app/app.py
```
Open http://localhost:8501 in your browser.

### 5. Run with Docker Compose
```bash
# Start the Streamlit app
docker compose up -d

# Optionally run the training job as a one-off container
docker compose --profile train up train
```

---

## Project Structure

```
maude-nlp-classifier/
├── src/
│   ├── ingestion/
│   │   └── openfda_client.py      # openFDA MAUDE API client
│   ├── preprocessing/
│   │   └── text_cleaner.py        # text normalization pipeline
│   └── model/
│       ├── classifier.py          # TF-IDF + LogReg/SVM pipeline
│       └── train.py               # training entrypoint (CLI)
├── streamlit_app/
│   └── app.py                     # Streamlit demo app
├── tests/
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   └── test_classifier.py
├── deploy/
│   ├── ecs/
│   │   └── task-definition.json   # ECS Fargate task definition
│   └── terraform/
│       ├── main.tf                # Full AWS infra (VPC, ECR, ECS, ALB)
│       ├── variables.tf
│       └── outputs.tf
├── .github/workflows/
│   └── ci-cd.yml                  # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## AWS Deployment

### Prerequisites
- AWS CLI configured with appropriate permissions
- Terraform >= 1.5
- Docker

### Step 1 — Provision infrastructure
```bash
cd deploy/terraform
terraform init
terraform plan -var="openfda_api_key=$OPENFDA_API_KEY"
terraform apply -var="openfda_api_key=$OPENFDA_API_KEY"
```

### Step 2 — Build and push Docker image to ECR
```bash
# Get ECR URI from Terraform output
ECR_URI=$(terraform output -raw ecr_repository_url)
AWS_REGION=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")

aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ECR_URI

docker build -t $ECR_URI:latest .
docker push $ECR_URI:latest
```

### Step 3 — The app is live
```bash
terraform output app_url
# → http://<alb-dns-name>
```

### GitHub Actions (CI/CD)
Add the following secrets to your GitHub repository:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM access key |
| `AWS_SECRET_ACCESS_KEY` | IAM secret key |
| `AWS_REGION` | e.g. `us-east-1` |
| `AWS_ACCOUNT_ID` | 12-digit AWS account ID |
| `OPENFDA_API_KEY` | openFDA API key (optional) |

Every push to `main` will automatically test, build, push to ECR, and deploy to ECS.

---

## Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--records` | 5000 | Number of MAUDE records to fetch |
| `--model` | `logreg` | Classifier: `logreg` or `svm` |
| `--tune` | off | Enable GridSearchCV hyperparameter tuning |
| `--use-cached` | off | Load from `data/raw/maude_raw.csv` instead of re-fetching |
| `--drop-unknown` | off | Exclude UNKNOWN-labeled records |

---

## Severity Label Mapping

| openFDA `event_type` | Severity Label |
|----------------------|---------------|
| Death | `DEATH` |
| Serious Injury | `SERIOUS_INJURY` |
| Injury | `INJURY` |
| Malfunction | `MALFUNCTION` |
| Other / No Answer | `UNKNOWN` |

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Future Scope of Work

### 1. Upgrade to Biomedical Transformers
Replace TF-IDF + LogReg with fine-tuned **BioBERT** or **ClinicalBERT** (HuggingFace `transformers`). These models are pre-trained on PubMed/clinical notes and can dramatically improve accuracy on domain-specific medical text. Host via SageMaker Endpoints for scalable inference.

### 2. Multi-label & Multi-task Classification
MAUDE reports often have multiple concurrent issues. Extend to predict:
- **Severity** (current)
- **Root cause** (user error, software bug, manufacturing defect)
- **Device product code / category** (from openFDA product taxonomy)
- **Reporter type** (manufacturer, hospital, patient)

### 3. Named Entity Recognition (NER) for Device & Symptom Extraction
Train a NER model (spaCy or HuggingFace) to extract structured entities: device names, body sites, clinical events, and symptoms from free-text — enabling downstream graph/knowledge base construction.

### 4. Real-time Streaming Ingestion
Replace batch API polling with an **AWS Kinesis Data Streams** or **Kafka** pipeline connected to the openFDA bulk download feed. Enables near-real-time monitoring of new adverse events.

### 5. Automated Signal Detection & Alerting
Implement statistical disproportionality analysis (Reporting Odds Ratio, PRR) on classified data to automatically surface device-specific safety signals. Trigger **SNS/email alerts** when signal thresholds are crossed — similar to FDA's own FAERS signal detection.

### 6. RAG-based Q&A over MAUDE Reports
Build a **Retrieval-Augmented Generation (RAG)** system using MAUDE narratives as a knowledge base (vector store via Pinecone or OpenSearch). Allow safety analysts to ask natural language questions like "What are the most common failure modes for insulin pumps?"

### 7. FHIR / HL7 Integration
Map MAUDE records to **FHIR R4 AdverseEvent** resources for interoperability with hospital EHR systems (Epic, Cerner). This would allow the classifier to be embedded directly in clinical workflows.

### 8. Explainability Dashboard (XAI)
Integrate **SHAP** or **LIME** to show which words/phrases drove each severity prediction. Critical for FDA regulatory use — predictions need to be auditable and explainable to safety reviewers.

### 9. Active Learning Loop
Implement an **active learning** pipeline where low-confidence predictions are flagged for human review via a labeling interface (Label Studio), and reviewed labels are fed back to retrain the model automatically.

### 10. Multi-database Coverage
Extend ingestion beyond MAUDE to:
- **openFDA Drug Events (FAERS)** — drug adverse events
- **EU EUDAMED** — European medical device registry
- **WHO VigiBase** — global pharmacovigilance database

---

## License

MIT — see [LICENSE](LICENSE)

---

## Disclaimer

This tool is intended for research and informational purposes only. It is not a substitute for professional medical or regulatory review. Always consult qualified professionals for regulatory submissions and safety decisions.
