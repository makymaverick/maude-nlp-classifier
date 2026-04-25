"""
Build phase2_bioclinbert.ipynb from cell sources.

Run once: python _build_notebook.py
Outputs: phase2_bioclinbert.ipynb (in same dir).

This builder exists so the notebook source is reviewable as plain Python
(easier to diff and edit) but ships as a proper Kaggle .ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "phase2_bioclinbert.ipynb"


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS: list[dict] = [
    # ─── Cell 1: Title + framing ─────────────────────────────────────────────
    md_cell("""# MAUDE Phase 2 — Bio_ClinicalBERT Fine-tuning

**Goal:** Beat the hardened TF-IDF + Logistic Regression baseline on the same locked CV folds.

| Metric | Phase 1 (hardened) | Phase 2 target |
|--------|--------------------|----------------|
| Weighted F1 | 0.8484 ± 0.0012 | **> 0.853** (baseline + 0.005) |
| Macro F1 | 0.7379 ± 0.0032 | > 0.745 |
| **Death-class F1** | **0.7701 ± 0.0111** | **> 0.77** (clinical gate) |

**Inputs (attach as Kaggle dataset):**
- `maude_raw.csv` — same 154,776 records used for Phase 1
- `cv_splits.json` — locked 5-fold StratifiedKFold indices (data integrity guarded by SHA1 fingerprint)

**Compute:** Single T4 GPU, ~1.5 h for one fold validation run, ~6–8 h for full 5-fold sweep.

**Strategy:**
1. Run a single fold first to validate the pipeline and get a real signal.
2. Only commit the full 5-fold sweep if the single-fold result looks competitive.
3. Save the best per-fold checkpoint to `/kaggle/working/` and optionally push to HF Hub.
"""),

    # ─── Cell 2: Setup ───────────────────────────────────────────────────────
    md_cell("""## 1. Setup — install dependencies and check GPU"""),
    code_cell("""# Kaggle T4 has torch + transformers preinstalled, but versions can lag.
# Pin the versions known to work for this notebook.
%pip install -q -U \\
    transformers==4.44.2 \\
    accelerate==0.34.2 \\
    huggingface_hub==0.25.1 \\
    > /tmp/pip.log
print('pip install ok')
"""),
    code_cell("""import os, sys, json, time, math, random, hashlib, gc, warnings
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import transformers
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print('torch:        ', torch.__version__)
print('transformers: ', transformers.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:       ', torch.cuda.get_device_name(0))
    print('vram:         ', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"""),

    # ─── Cell 3: Configuration ──────────────────────────────────────────────
    md_cell("""## 2. Configuration

All knobs in one place. The defaults are tuned for Kaggle T4 (16 GB VRAM)."""),
    code_cell("""class CFG:
    # ── Model ──
    PRETRAINED_MODEL = 'emilyalsentzer/Bio_ClinicalBERT'
    MAX_LENGTH = 512                       # full BERT context (vs Phase 1 attempt at 256)
    POOLING = 'cls_mean_concat'            # alternatives: 'cls', 'mean'
    DROPOUT = 0.1

    # ── Optimisation ──
    EPOCHS = 5
    BATCH_SIZE = 8                         # T4 fits 16 at L=512 fp16, but 8 leaves headroom
    GRAD_ACCUM = 4                         # effective batch 32
    LR = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    USE_FP16 = True
    MAX_GRAD_NORM = 1.0
    EARLY_STOP_PATIENCE = 1                # epochs without val F1 improvement

    # ── Loss ──
    USE_CLASS_WEIGHTS = True               # inverse-frequency weighted CrossEntropy
    USE_FOCAL_LOSS = False                 # set True for a focal loss ablation

    # ── Data ──
    TEXT_COL = 'narrative_text'            # raw text — let BERT's tokenizer handle cleaning
    LABEL_COL = 'severity_label'
    DROP_UNKNOWN = True                    # match Phase 1 hardened evaluation

    # ── CV ──
    N_FOLDS_TO_RUN = 1                     # CHANGE TO 5 FOR FULL SWEEP (after single fold validates)
    SEED = 42

    # ── Phase 1 reference (for comparison printouts) ──
    PHASE1_CV_F1_WEIGHTED = 0.8484
    PHASE1_CV_F1_MACRO = 0.7379
    PHASE1_DEATH_F1 = 0.7701
    PROMOTION_GATE_F1 = 0.853

    # ── Paths ──
    DATA_PATH = '/kaggle/input/maude-phase2/maude_raw.csv'
    SPLITS_PATH = '/kaggle/input/maude-phase2/cv_splits.json'
    OUTPUT_DIR = Path('/kaggle/working')
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'

CFG.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # keep speed; perfect determinism not critical here

set_seed(CFG.SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE, '| pooling:', CFG.POOLING, '| folds:', CFG.N_FOLDS_TO_RUN)
"""),

    # ─── Cell 4: Data loading + fingerprint verification ────────────────────
    md_cell("""## 3. Load data + verify CV splits fingerprint

The locked splits include a SHA1 fingerprint of the dataset. If the fingerprint doesn't match the cleaned data here, we abort — it would mean we're training on different data than Phase 1, invalidating the comparison."""),
    code_cell("""# ── Replicate Phase 1 cleaning so fingerprint matches ─────────────────────
import re

ABBREVIATION_MAP = {
    r'\\bpt\\b': 'patient', r'\\bpts\\b': 'patients', r'\\bmd\\b': 'physician',
    r'\\bdr\\b': 'doctor', r'\\bhosp\\b': 'hospital', r'\\badm\\b': 'admitted',
    r'\\bdx\\b': 'diagnosis', r'\\btx\\b': 'treatment', r'\\brx\\b': 'prescription',
    r'\\bs/p\\b': 'status post', r'\\bw/\\b': 'with', r'\\bh/o\\b': 'history of',
    r'\\bc/o\\b': 'complaint of', r'\\bn/v\\b': 'nausea vomiting',
    r'\\bSOB\\b': 'shortness of breath', r'\\bUNK\\b': 'unknown',
}
BOILERPLATE = [
    r'it was reported that', r'the reporter stated', r'according to the report',
    r'this is a report', r'per the report',
    r'the following information was received', r'information has been received',
    r'no further information (is|was) available', r'follow.?up (is|will be) requested',
]

def _clean_for_fingerprint(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.strip()
    for pat in BOILERPLATE:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE)
    for pat, rep in ABBREVIATION_MAP.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-zA-Z\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip().lower()
    return text


# ── Load raw and apply same cleaning Phase 1 used to lock splits ──────────
print('Loading', CFG.DATA_PATH)
df_raw = pd.read_csv(CFG.DATA_PATH)
print(f'  raw rows: {len(df_raw):,}')

df = df_raw.copy()
df['clean_text'] = df[CFG.TEXT_COL].apply(_clean_for_fingerprint)
df = df[df['clean_text'].str.len() > 10].reset_index(drop=True)
print(f'  after clean (drop <10 chars): {len(df):,}')

if CFG.DROP_UNKNOWN:
    before = len(df)
    df = df[df[CFG.LABEL_COL] != 'UNKNOWN'].reset_index(drop=True)
    print(f'  after drop UNKNOWN: {len(df):,} (-{before - len(df)})')

print(f'  label distribution:')
for lbl, c in df[CFG.LABEL_COL].value_counts().items():
    print(f'    {lbl}: {c:>7,}')


# ── Fingerprint verify against locked CV splits ───────────────────────────
def sha1_first_n_labels(labels, n=100):
    sample = '|'.join(str(x) for x in list(labels)[:n])
    return hashlib.sha1(sample.encode()).hexdigest()

with open(CFG.SPLITS_PATH) as f:
    splits = json.load(f)

locked_fp = splits['dataset_fingerprint']
current_n = len(df)
current_sha = sha1_first_n_labels(df[CFG.LABEL_COL])

print('\\n── Fingerprint check ──')
print(f'  locked  records: {locked_fp["n_records"]:,}')
print(f'  current records: {current_n:,}')
print(f'  locked  SHA1:    {locked_fp["sha1_first_100_labels"][:16]}...')
print(f'  current SHA1:    {current_sha[:16]}...')

if locked_fp['n_records'] != current_n:
    raise RuntimeError(
        f'Record count mismatch: locked={locked_fp["n_records"]} vs current={current_n}. '
        f'Different cleaning pipeline? Check DROP_UNKNOWN flag and clean_text logic.'
    )
if locked_fp['sha1_first_100_labels'] != current_sha:
    raise RuntimeError(
        'SHA1 fingerprint mismatch — labels differ from when splits were locked. '
        'Did the source CSV change?'
    )
print('  ✓ Fingerprint matches. Safe to use locked CV splits.')

LABELS = sorted(df[CFG.LABEL_COL].unique())
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}
print(f'\\n  labels: {LABELS}')
"""),

    # ─── Cell 5: Token length profiling ─────────────────────────────────────
    md_cell("""## 4. Token length profiling

How much narrative gets truncated at `max_length=512`? If a high fraction is being cut, that's signal lost — and a reason Phase 2 might underperform if we don't address it later."""),
    code_cell("""tokenizer = AutoTokenizer.from_pretrained(CFG.PRETRAINED_MODEL, do_lower_case=True)
print('tokenizer:', type(tokenizer).__name__, '| vocab:', tokenizer.vocab_size)

# Sample 5000 narratives for length profiling (full set is too slow to tokenize)
sample = df[CFG.TEXT_COL].sample(min(5000, len(df)), random_state=42)
lengths = []
for txt in sample:
    lengths.append(len(tokenizer.encode(str(txt), add_special_tokens=True)))

lengths = np.array(lengths)
percentiles = [50, 75, 90, 95, 99]
print('\\nToken length percentiles (n=5000 sample):')
for p in percentiles:
    print(f'  p{p:>2}: {int(np.percentile(lengths, p)):>5}')
print(f'  max: {lengths.max():>5}')
print(f'  mean: {lengths.mean():>5.1f}')
truncated = (lengths > CFG.MAX_LENGTH).mean()
print(f'\\n  fraction truncated at MAX_LENGTH={CFG.MAX_LENGTH}: {truncated:.1%}')
if truncated > 0.30:
    print('  ⚠ >30% truncation — consider Clinical-Longformer (4096-token context) for follow-up')
elif truncated > 0.10:
    print('  ⚠ moderate truncation — monitor Death-class F1; long narratives often contain death cues')
else:
    print('  ✓ truncation level acceptable for Bio_ClinicalBERT')
"""),

    # ─── Cell 6: Dataset and Model ──────────────────────────────────────────
    md_cell("""## 5. Dataset and model

**Pooling:** `[CLS] + mean-pooled` concatenation feeds a 2H → num_labels head. Mean pooling captures information from the full sequence; [CLS] alone often loses long-context signal. Concatenating both gives the classifier head access to both."""),
    code_cell("""class MaudeBertDataset(Dataset):
    \"\"\"Tokenizes on-the-fly so we don't blow memory on 154k pre-tokenized tensors.\"\"\"

    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc.get('token_type_ids', torch.zeros_like(enc['input_ids'])).squeeze(0),
            'labels': self.labels[idx],
        }


class BertSeverityClassifier(nn.Module):
    def __init__(self, num_labels: int, pretrained: str, dropout: float, pooling: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained)
        H = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        head_in = H * 2 if pooling == 'cls_mean_concat' else H
        self.classifier = nn.Linear(head_in, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last = out.last_hidden_state                 # (B, L, H)
        cls = last[:, 0, :]                          # (B, H)
        if self.pooling == 'cls':
            pooled = cls
        elif self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:  # cls_mean_concat
            mask = attention_mask.unsqueeze(-1).float()
            mean = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.cat([cls, mean], dim=-1)
        return self.classifier(self.dropout(pooled))


# Optional focal loss (γ=2). Off by default; flip CFG.USE_FOCAL_LOSS to True.
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()
"""),

    # ─── Cell 7: Training function ──────────────────────────────────────────
    md_cell("""## 6. Training function

Per-fold training with:
- Inverse-frequency class weights (computed on train fold only — no leakage)
- fp16 mixed precision
- Gradient accumulation
- Linear warmup → linear decay scheduler
- Early stopping on validation **weighted F1**
- Per-class F1 reported for the best epoch"""),
    code_cell("""def evaluate_model(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attn = batch['attention_mask'].to(device, non_blocking=True)
            ttype = batch['token_type_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            with autocast(enabled=CFG.USE_FP16):
                logits = model(input_ids, attn, ttype)
            all_pred.extend(logits.argmax(-1).cpu().numpy().tolist())
            all_true.extend(labels.cpu().numpy().tolist())
    return np.array(all_true), np.array(all_pred)


def per_class_f1(y_true, y_pred, label_ids: list[int], label_names: list[str]) -> dict:
    f1_per = f1_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    out = {f'f1_{name}': float(s) for name, s in zip(label_names, f1_per)}
    out['f1_macro'] = float(f1_score(y_true, y_pred, labels=label_ids, average='macro', zero_division=0))
    out['f1_weighted'] = float(f1_score(y_true, y_pred, labels=label_ids, average='weighted', zero_division=0))
    return out


def train_one_fold(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold_num: int,
) -> dict:
    print(f'\\n{"=" * 72}\\n  FOLD {fold_num}: train={len(train_idx):,}  val={len(val_idx):,}\\n{"=" * 72}')

    # Split
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    train_labels = [LABEL2ID[lbl] for lbl in train_df[CFG.LABEL_COL]]
    val_labels   = [LABEL2ID[lbl] for lbl in val_df[CFG.LABEL_COL]]

    # Class weights from train fold
    if CFG.USE_CLASS_WEIGHTS:
        classes = np.array(sorted(set(train_labels)))
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=np.array(train_labels))
        class_weights_t = torch.tensor(weights, dtype=torch.float, device=DEVICE)
        print(f'  class weights: {dict(zip([ID2LABEL[c] for c in classes], np.round(weights, 3)))}')
    else:
        class_weights_t = None

    # Datasets / loaders
    train_ds = MaudeBertDataset(train_df[CFG.TEXT_COL].astype(str), train_labels, tokenizer, CFG.MAX_LENGTH)
    val_ds   = MaudeBertDataset(val_df[CFG.TEXT_COL].astype(str),   val_labels,   tokenizer, CFG.MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

    # Model + loss + optimiser + scheduler
    model = BertSeverityClassifier(
        num_labels=len(LABELS),
        pretrained=CFG.PRETRAINED_MODEL,
        dropout=CFG.DROPOUT,
        pooling=CFG.POOLING,
    ).to(DEVICE)

    if CFG.USE_FOCAL_LOSS:
        criterion = FocalLoss(gamma=2.0, alpha=class_weights_t)
        loss_name = 'focal(γ=2)'
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        loss_name = 'weighted_ce' if CFG.USE_CLASS_WEIGHTS else 'ce'

    no_decay = ('bias', 'LayerNorm.weight')
    optim_groups = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': CFG.WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=CFG.LR)

    steps_per_epoch = math.ceil(len(train_loader) / CFG.GRAD_ACCUM)
    total_steps = steps_per_epoch * CFG.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * CFG.WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=CFG.USE_FP16)

    print(f'  loss: {loss_name}  |  effective batch: {CFG.BATCH_SIZE * CFG.GRAD_ACCUM}  '
          f'|  total steps: {total_steps}')

    # Train
    best_val_f1, best_metrics, best_epoch = -1.0, None, -1
    patience = CFG.EARLY_STOP_PATIENCE
    best_state_path = CFG.CHECKPOINT_DIR / f'fold{fold_num}_best.pt'
    bad_epochs = 0

    for epoch in range(1, CFG.EPOCHS + 1):
        model.train()
        t0 = time.time()
        running_loss, optim_step = 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attn = batch['attention_mask'].to(DEVICE, non_blocking=True)
            ttype = batch['token_type_ids'].to(DEVICE, non_blocking=True)
            labels = batch['labels'].to(DEVICE, non_blocking=True)

            with autocast(enabled=CFG.USE_FP16):
                logits = model(input_ids, attn, ttype)
                loss = criterion(logits, labels) / CFG.GRAD_ACCUM
            scaler.scale(loss).backward()
            running_loss += loss.item() * CFG.GRAD_ACCUM

            if step % CFG.GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

        avg_loss = running_loss / max(1, len(train_loader))
        val_true, val_pred = evaluate_model(model, val_loader, DEVICE)
        m = per_class_f1(val_true, val_pred, list(range(len(LABELS))), LABELS)
        elapsed = time.time() - t0
        improved = m['f1_weighted'] > best_val_f1

        print(
            f'  epoch {epoch}/{CFG.EPOCHS}  loss={avg_loss:.4f}  '
            f'val_w_f1={m["f1_weighted"]:.4f}  val_macro_f1={m["f1_macro"]:.4f}  '
            + ' '.join(f'{k}={v:.3f}' for k, v in m.items() if k.startswith("f1_") and k not in ("f1_weighted", "f1_macro"))
            + f'  ({elapsed:.0f}s){"  ★" if improved else ""}'
        )

        if improved:
            best_val_f1, best_metrics, best_epoch = m['f1_weighted'], m, epoch
            torch.save(model.state_dict(), best_state_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > patience:
                print(f'  early stop at epoch {epoch} (best epoch {best_epoch}, val w_f1={best_val_f1:.4f})')
                break

    # Cleanup VRAM
    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()

    best_metrics['fold'] = fold_num
    best_metrics['best_epoch'] = best_epoch
    best_metrics['checkpoint_path'] = str(best_state_path)
    return best_metrics
"""),

    # ─── Cell 8: Run CV ─────────────────────────────────────────────────────
    md_cell("""## 7. Run cross-validation

`CFG.N_FOLDS_TO_RUN = 1` runs only fold 1 — single-fold validation. Once that looks healthy, set to **5** and re-run for the full sweep."""),
    code_cell("""fold_metrics: list[dict] = []
folds = splits['folds'][:CFG.N_FOLDS_TO_RUN]
print(f'Running {len(folds)} of {len(splits["folds"])} folds.')

t_start = time.time()
for f in folds:
    train_idx = np.asarray(f['train_idx'])
    val_idx   = np.asarray(f['val_idx'])
    m = train_one_fold(df, train_idx, val_idx, fold_num=f['fold'])
    fold_metrics.append(m)
    elapsed_h = (time.time() - t_start) / 3600
    print(f'\\n  cumulative wall time: {elapsed_h:.2f} h')

print(f'\\nFinished {len(fold_metrics)} fold(s) in {(time.time() - t_start)/3600:.2f} h')
"""),

    # ─── Cell 9: Aggregate + compare ────────────────────────────────────────
    md_cell("""## 8. Aggregate results and compare against Phase 1"""),
    code_cell("""def agg_metrics(fold_metrics: list[dict]) -> dict:
    if not fold_metrics:
        return {}
    keys = [k for k in fold_metrics[0] if k.startswith('f1_')]
    out = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        out[f'{k}_mean'] = float(np.mean(vals))
        out[f'{k}_std']  = float(np.std(vals))
    out['per_fold'] = fold_metrics
    out['n_folds'] = len(fold_metrics)
    return out


agg = agg_metrics(fold_metrics)

print('\\n' + '=' * 72)
print('  PHASE 2 — BIO_CLINICALBERT RESULTS')
print('=' * 72)
print(f'  Folds run:                 {agg["n_folds"]}')
print(f'  Phase 1 hardened weighted: {CFG.PHASE1_CV_F1_WEIGHTED:.4f}  (target to beat)')
print(f'  Phase 2 BERT weighted F1:  {agg["f1_weighted_mean"]:.4f} ± {agg["f1_weighted_std"]:.4f}')
print(f'  Phase 1 hardened macro:    {CFG.PHASE1_CV_F1_MACRO:.4f}')
print(f'  Phase 2 BERT macro F1:     {agg["f1_macro_mean"]:.4f} ± {agg["f1_macro_std"]:.4f}')
print('  Per-class F1 (mean ± std across folds):')
for lbl in LABELS:
    key_m = f'f1_{lbl}_mean'
    key_s = f'f1_{lbl}_std'
    if key_m in agg:
        print(f'    {lbl:<10} {agg[key_m]:.4f} ± {agg[key_s]:.4f}')
print('-' * 72)
delta = agg['f1_weighted_mean'] - CFG.PHASE1_CV_F1_WEIGHTED
sign = '+' if delta >= 0 else ''
print(f'  Δ vs Phase 1 hardened (weighted F1): {sign}{delta:.4f}')
death_delta = agg.get('f1_D_mean', 0) - CFG.PHASE1_DEATH_F1
sign_d = '+' if death_delta >= 0 else ''
print(f'  Δ Death-class F1:                   {sign_d}{death_delta:.4f}')

if delta >= (CFG.PROMOTION_GATE_F1 - CFG.PHASE1_CV_F1_WEIGHTED):
    verdict = '✓ MEETS PROMOTION GATE'
elif delta > 0:
    verdict = '◐ POSITIVE BUT BELOW GATE — consider longer training or focal loss ablation'
else:
    verdict = '✗ DOES NOT BEAT BASELINE — see Death-class F1 and consider Clinical-Longformer for length ablation'
print(f'  Verdict: {verdict}')
print('=' * 72)

# Save aggregated metrics
metrics_path = CFG.OUTPUT_DIR / 'phase2_bert_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump({
        'phase': 'phase2-bioclinbert',
        'config': {k: v for k, v in vars(CFG).items()
                   if not k.startswith('_') and isinstance(v, (str, int, float, bool, list, dict))},
        'aggregated': {k: v for k, v in agg.items() if k != 'per_fold'},
        'per_fold': agg.get('per_fold', []),
        'phase1_reference': {
            'cv_f1_weighted': CFG.PHASE1_CV_F1_WEIGHTED,
            'cv_f1_macro': CFG.PHASE1_CV_F1_MACRO,
            'death_f1': CFG.PHASE1_DEATH_F1,
        },
        'delta_vs_phase1_weighted': delta,
        'delta_vs_phase1_death': death_delta,
    }, f, indent=2, default=str)
print(f'\\nMetrics saved to {metrics_path}')
"""),

    # ─── Cell 10: Confusion matrix and inspection ──────────────────────────
    md_cell("""## 9. Inspect predictions on the last fold

A confusion matrix tells you where the model fails — especially what it confuses Death cases with."""),
    code_cell("""# Re-load the best checkpoint of the last fold and run val predictions for inspection
last = fold_metrics[-1]
last_fold_idx = last['fold'] - 1
val_idx = np.asarray(splits['folds'][last_fold_idx]['val_idx'])
val_df = df.iloc[val_idx].reset_index(drop=True)
val_labels = [LABEL2ID[lbl] for lbl in val_df[CFG.LABEL_COL]]

model = BertSeverityClassifier(
    num_labels=len(LABELS),
    pretrained=CFG.PRETRAINED_MODEL,
    dropout=CFG.DROPOUT,
    pooling=CFG.POOLING,
).to(DEVICE)
model.load_state_dict(torch.load(last['checkpoint_path'], map_location=DEVICE))

val_ds = MaudeBertDataset(val_df[CFG.TEXT_COL].astype(str), val_labels, tokenizer, CFG.MAX_LENGTH)
val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

y_true, y_pred = evaluate_model(model, val_loader, DEVICE)
print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0, digits=4))

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
print('\\nConfusion matrix (rows=true, cols=pred):')
print('         ' + '  '.join(f'{l:>5}' for l in LABELS))
for i, lbl in enumerate(LABELS):
    print(f'  {lbl:>5}  ' + '  '.join(f'{cm[i, j]:>5d}' for j in range(len(LABELS))))

del model
gc.collect(); torch.cuda.empty_cache()
"""),

    # ─── Cell 11: Optional HF Hub push ──────────────────────────────────────
    md_cell("""## 10. (Optional) Push best fold-1 model to Hugging Face Hub

Uses the `HF_TOKEN` Kaggle Secret. Comment out if you don't want to push yet."""),
    code_cell("""# from huggingface_hub import HfApi, login
# from kaggle_secrets import UserSecretsClient
#
# token = UserSecretsClient().get_secret('HF_TOKEN')
# login(token=token)
#
# REPO = 'mukundisb/maude-clinicalbert'
# api = HfApi()
# api.create_repo(REPO, exist_ok=True, private=False)
#
# # Save model + tokenizer to a hub-friendly format
# best = sorted(fold_metrics, key=lambda m: -m['f1_weighted'])[0]
# print(f'Pushing fold {best["fold"]} (val_w_f1={best["f1_weighted"]:.4f}) to {REPO}')
#
# best_model = BertSeverityClassifier(num_labels=len(LABELS), pretrained=CFG.PRETRAINED_MODEL,
#                                     dropout=CFG.DROPOUT, pooling=CFG.POOLING).to(DEVICE)
# best_model.load_state_dict(torch.load(best['checkpoint_path'], map_location=DEVICE))
#
# save_dir = CFG.OUTPUT_DIR / 'hf_repo'
# save_dir.mkdir(exist_ok=True)
# torch.save(best_model.state_dict(), save_dir / 'pytorch_model.bin')
# tokenizer.save_pretrained(save_dir)
# # Save metadata
# import json as _json
# (save_dir / 'config_phase2.json').write_text(_json.dumps({
#     'pretrained': CFG.PRETRAINED_MODEL,
#     'pooling': CFG.POOLING,
#     'max_length': CFG.MAX_LENGTH,
#     'labels': LABELS,
#     'val_metrics': {k: best[k] for k in best if k.startswith('f1_')},
# }, indent=2))
#
# api.upload_folder(folder_path=str(save_dir), repo_id=REPO, repo_type='model')
# print('  ✓ pushed')
"""),

    # ─── Cell 12: Footer ─────────────────────────────────────────────────────
    md_cell("""## What to do with the results

- **Single-fold success** (weighted F1 ≥ 0.853, Death F1 ≥ 0.77): change `CFG.N_FOLDS_TO_RUN = 5`, save & run again. Plan ~6–8 h on T4.
- **Single-fold close miss** (within 0.005 of target): try `CFG.USE_FOCAL_LOSS = True`, re-run single fold. Focal loss often picks up the Death class.
- **Clear underperformance** (weighted F1 < 0.84 or Death F1 < 0.70): probable causes are truncation (check § 4 percentiles) or under-training. Either bump epochs or accept that Bio_ClinicalBERT @ 512 ctx isn't enough — that's a finding, not a failure, and frames the Clinical-Longformer follow-up.

Download `phase2_bert_metrics.json` from the Output panel for the LinkedIn write-up."""),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [],
                "isInternetEnabled": True,
                "isGpuEnabled": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {NB_PATH}  ({NB_PATH.stat().st_size / 1024:.1f} KB, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
