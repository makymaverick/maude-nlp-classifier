"""
ClinicalBERT-based Severity Classifier for MAUDE Adverse Event Narratives.

Architecture:
    Input Narrative → ClinicalBERT Tokenizer → Bio_ClinicalBERT Encoder
    → [CLS] embedding (768-dim) → Dropout(0.1) → Linear(768→5) → Softmax

Supports:
  - Full fine-tuning of all BERT layers
  - AdamW optimizer with linear warmup scheduler
  - Inverse-frequency class weighting
  - StratifiedKFold cross-validation (for promotion gate)
  - Local checkpoint save/load
  - HuggingFace Hub push (optional, requires HF_TOKEN env var)
  - bert_model_ref.json pointer file for reproducibility
"""

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

PRETRAINED_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
LABEL_ORDER = ["D", "I", "M", "O", "UNKNOWN"]
BERT_MODEL_REF_PATH = "models/bert_model_ref.json"


# ── Dataset ──────────────────────────────────────────────────────────────────

class MaudeDataset(Dataset):
    """PyTorch Dataset wrapping tokenized MAUDE narratives."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ── Model ─────────────────────────────────────────────────────────────────────

class BertSeverityClassifier(nn.Module):
    """
    Bio_ClinicalBERT encoder with a 5-class severity classification head.

    Input:  tokenized narrative (input_ids, attention_mask, token_type_ids)
    Output: logits of shape (batch_size, 5)  — one per LABEL_ORDER class
    """

    def __init__(self, num_labels: int = 5, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(PRETRAINED_MODEL)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        return logits


# ── Label encoding helpers ────────────────────────────────────────────────────

def encode_labels(y: pd.Series) -> tuple[list[int], dict]:
    """Map string labels to integers using LABEL_ORDER."""
    label2id = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
    # Any label not in LABEL_ORDER maps to UNKNOWN (index 4)
    encoded = [label2id.get(lbl, label2id["UNKNOWN"]) for lbl in y]
    return encoded, label2id


def decode_labels(indices: list[int]) -> list[str]:
    return [LABEL_ORDER[i] for i in indices]


# ── Training ──────────────────────────────────────────────────────────────────

def train_bert(
    df: pd.DataFrame,
    text_col: str = "narrative_text",
    label_col: str = "severity_label",
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_length: int = 256,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    device: Optional[str] = None,
) -> tuple["BertSeverityClassifier", "AutoTokenizer", dict]:
    """
    Fine-tune Bio_ClinicalBERT on the provided DataFrame.

    Args:
        df:          DataFrame with text_col and label_col.
        text_col:    Column name for narrative text.
        label_col:   Column name for string severity labels.
        epochs:      Number of full training passes.
        lr:          Peak learning rate for AdamW.
        batch_size:  Per-device batch size.
        max_length:  Max token length (256 covers 95th pct of MAUDE narratives).
        warmup_ratio: Fraction of total steps used for linear warmup.
        weight_decay: L2 regularization on AdamW.
        device:      'cuda', 'cpu', or None (auto-detect).

    Returns:
        Tuple of (trained model, tokenizer, metrics dict).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    texts = df[text_col].tolist()
    encoded_labels, label2id = encode_labels(df[label_col])

    # Class weights — inverse frequency
    classes = np.array(sorted(set(encoded_labels)))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.array(encoded_labels),
    )
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    # 80/20 split for training progress monitoring
    split_idx = int(len(texts) * 0.8)
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = encoded_labels[:split_idx], encoded_labels[split_idx:]

    train_dataset = MaudeDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = MaudeDataset(val_texts, val_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2)

    model = BertSeverityClassifier(num_labels=len(LABEL_ORDER)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation F1
        val_f1 = _evaluate_f1(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch}/{epochs} — loss: {avg_loss:.4f} | val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best checkpoint from training
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    metrics = {
        "val_f1_weighted": best_val_f1,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_length": max_length,
        "training_records": len(df),
    }
    logger.info(f"Training complete. Best val F1: {best_val_f1:.4f}")
    return model, tokenizer, metrics


def _evaluate_f1(model, dataloader, device) -> float:
    """Run model over dataloader and return weighted F1."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            logits = model(input_ids, attention_mask, token_type_ids)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())
    return f1_score(all_labels, all_preds, average="weighted", zero_division=0)


# ── Cross-validation (for promotion gate) ────────────────────────────────────

def cross_validate_bert(
    df: pd.DataFrame,
    text_col: str = "narrative_text",
    label_col: str = "severity_label",
    n_splits: int = 3,
    epochs: int = 2,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_length: int = 256,
    device: Optional[str] = None,
) -> dict:
    """
    StratifiedKFold cross-validation for BERT.

    Uses fewer folds (3) and epochs (2) than TF-IDF CV to keep runtime reasonable.
    Still provides a stable CV F1 estimate for the promotion gate.

    Returns:
        Dict with cv_f1_per_fold, cv_f1_mean, cv_f1_std.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    texts = df[text_col].tolist()
    encoded_labels, _ = encode_labels(df[label_col])
    y = np.array(encoded_labels)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(texts, y), 1):
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        train_labels = y[train_idx].tolist()
        val_labels = y[val_idx].tolist()

        classes = np.array(sorted(set(train_labels)))
        weights = compute_class_weight("balanced", classes=classes, y=np.array(train_labels))
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)

        train_ds = MaudeDataset(train_texts, train_labels, tokenizer, max_length)
        val_ds = MaudeDataset(val_texts, val_labels, tokenizer, max_length)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2)

        model = BertSeverityClassifier(num_labels=len(LABEL_ORDER)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model.train()
        for _ in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, batch["labels"].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        fold_f1 = _evaluate_f1(model, val_loader, device)
        fold_scores.append(fold_f1)
        logger.info(f"CV fold {fold}/{n_splits} — F1: {fold_f1:.4f}")

        # Free GPU memory between folds
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores = np.array(fold_scores)
    result = {
        "cv_f1_per_fold": scores.tolist(),
        "cv_f1_mean": float(scores.mean()),
        "cv_f1_std": float(scores.std()),
        "n_splits": n_splits,
    }
    logger.info(
        f"BERT CV ({n_splits}-fold) F1: {scores.mean():.4f} ± {scores.std():.4f}"
    )
    return result


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_bert(
    model: BertSeverityClassifier,
    tokenizer,
    text: str,
    max_length: int = 256,
    device: Optional[str] = None,
) -> dict:
    """
    Run inference on a single narrative string.

    Returns:
        Dict with predicted_label and probabilities per class.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    predicted_idx = int(np.argmax(probs))
    return {
        "predicted_label": LABEL_ORDER[predicted_idx],
        "probabilities": {lbl: float(probs[i]) for i, lbl in enumerate(LABEL_ORDER)},
    }


# ── Checkpoint save / load ────────────────────────────────────────────────────

def save_bert_checkpoint(
    model: BertSeverityClassifier,
    tokenizer,
    output_dir: str,
    hub_repo: Optional[str] = None,
) -> None:
    """
    Save fine-tuned model and tokenizer to output_dir.
    Optionally push to HuggingFace Hub if hub_repo is provided
    and HF_TOKEN environment variable is set.

    Also writes/updates models/bert_model_ref.json with the local path
    and (if pushed) the Hub repo name.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save BERT encoder weights embedded in our nn.Module
    model.bert.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save classification head separately
    head_path = os.path.join(output_dir, "classifier_head.pt")
    torch.save(
        {
            "classifier_state_dict": model.classifier.state_dict(),
            "dropout_p": model.dropout.p,
            "num_labels": model.classifier.out_features,
            "label_order": LABEL_ORDER,
        },
        head_path,
    )
    logger.info(f"Checkpoint saved to {output_dir}")

    # Update bert_model_ref.json
    ref = {"local_path": output_dir, "label_order": LABEL_ORDER}

    if hub_repo:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_token)
                api.create_repo(repo_id=hub_repo, exist_ok=True)
                model.bert.push_to_hub(hub_repo, token=hf_token)
                tokenizer.push_to_hub(hub_repo, token=hf_token)
                api.upload_file(
                    path_or_fileobj=head_path,
                    path_in_repo="classifier_head.pt",
                    repo_id=hub_repo,
                    token=hf_token,
                )
                # Get the latest commit SHA for pinning
                info = api.repo_info(repo_id=hub_repo)
                ref["hub_repo"] = hub_repo
                ref["hub_sha"] = info.sha
                logger.info(f"Pushed to HF Hub: {hub_repo} @ {info.sha}")
            except Exception as e:
                logger.warning(f"HF Hub push failed (model still saved locally): {e}")
        else:
            logger.warning("hub_repo specified but HF_TOKEN not set — skipping Hub push.")
            ref["hub_repo"] = hub_repo

    os.makedirs(os.path.dirname(BERT_MODEL_REF_PATH), exist_ok=True)
    with open(BERT_MODEL_REF_PATH, "w") as f:
        json.dump(ref, f, indent=2)
    logger.info(f"bert_model_ref.json updated: {ref}")


def load_bert_checkpoint(
    path_or_repo: str,
    device: Optional[str] = None,
) -> tuple["BertSeverityClassifier", "AutoTokenizer"]:
    """
    Load a fine-tuned BertSeverityClassifier from a local directory
    or a HuggingFace Hub repo ID.

    Args:
        path_or_repo: Local checkpoint directory or HF Hub repo ID
                      (e.g. 'mukundisb/maude-clinicalbert').
        device:       'cuda', 'cpu', or None (auto-detect).

    Returns:
        Tuple of (model, tokenizer).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(path_or_repo)

    model = BertSeverityClassifier()
    model.bert = AutoModel.from_pretrained(path_or_repo)

    # Load classification head
    if os.path.isdir(path_or_repo):
        head_path = os.path.join(path_or_repo, "classifier_head.pt")
    else:
        # Download from HF Hub
        from huggingface_hub import hf_hub_download
        head_path = hf_hub_download(repo_id=path_or_repo, filename="classifier_head.pt")

    head_data = torch.load(head_path, map_location=device)
    model.classifier = nn.Linear(
        model.bert.config.hidden_size,
        head_data["num_labels"],
    )
    model.classifier.load_state_dict(head_data["classifier_state_dict"])
    model.dropout = nn.Dropout(head_data.get("dropout_p", 0.1))

    model = model.to(device)
    model.eval()
    logger.info(f"BERT checkpoint loaded from {path_or_repo}")
    return model, tokenizer


def load_bert_from_ref(device: Optional[str] = None) -> tuple["BertSeverityClassifier", "AutoTokenizer"]:
    """
    Load the checkpoint referenced by bert_model_ref.json.
    Tries local_path first; falls back to hub_repo.
    """
    if not os.path.exists(BERT_MODEL_REF_PATH):
        raise FileNotFoundError(
            f"{BERT_MODEL_REF_PATH} not found. "
            "Train and save a BERT checkpoint first with train_bert.py."
        )
    with open(BERT_MODEL_REF_PATH) as f:
        ref = json.load(f)

    local_path = ref.get("local_path")
    hub_repo = ref.get("hub_repo")

    if local_path and os.path.isdir(local_path):
        return load_bert_checkpoint(local_path, device=device)
    elif hub_repo:
        logger.info(f"Local path not found; loading from HF Hub: {hub_repo}")
        return load_bert_checkpoint(hub_repo, device=device)
    else:
        raise FileNotFoundError(
            f"Neither local_path '{local_path}' nor hub_repo '{hub_repo}' is accessible."
        )
