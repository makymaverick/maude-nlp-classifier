# Phase 2 — Kaggle Runbook: Bio_ClinicalBERT Fine-tuning

**Goal:** Fine-tune `emilyalsentzer/Bio_ClinicalBERT` on MAUDE narratives using the 5-fold CV splits locked in Step 0, and decide whether the BERT model legitimately beats the hardened TF-IDF + LR baseline.

**Target to beat (Phase 1 hardened):**
- Weighted CV F1: **0.8484 ± 0.0012**
- Macro CV F1: **0.7379 ± 0.0032**
- Death-class F1: **0.7701 ± 0.0111**  ← the clinically important one
- Promotion gate: weighted F1 ≥ **0.853** AND f1_D does not regress

**What this notebook produces:**
- `/kaggle/working/phase2_bert_metrics.json` — per-fold and aggregated metrics
- `/kaggle/working/checkpoints/fold_<N>_best.pt` — best checkpoint per fold
- `/kaggle/working/confusion_matrix_fold_<N>.png` — error inspection
- (Optional) push best fold-1 model to `mukundisb/maude-clinicalbert` on HF Hub

**What this notebook does NOT do:**
- Touch the deployed Phase 1 HF Space
- Modify any file in the GitHub repo (Kaggle is sandboxed)
- Promote a champion automatically — promotion is a separate manual decision

---

## Prerequisites

You should already have on the `experimental` branch:
- `data/raw/maude_raw.csv` (154,776 rows, the cached MAUDE dataset)
- `data/cv_splits.json` (locked 5-fold splits, fingerprint SHA1 matches the cleaned dataset)
- `models/hardened_metrics.json` (the Phase 1 hardened numbers — you'll reference these)

---

## One-time Kaggle setup

### 1. Create a Kaggle dataset for the input files

The notebook expects two files at `/kaggle/input/maude-phase2/`. Easiest is to bundle them into a single private Kaggle dataset.

**On your Windows machine:**
```powershell
cd "C:\path\to\maude-nlp-classifier"

# Create a fresh staging folder so we don't accidentally upload extra files
mkdir kaggle_upload
copy data\raw\maude_raw.csv kaggle_upload\
copy data\cv_splits.json kaggle_upload\
```

**On the Kaggle web UI:**
1. Go to https://www.kaggle.com/datasets → **+ New Dataset**.
2. Drag `maude_raw.csv` and `cv_splits.json` into the upload zone.
3. Title: `maude-phase2`  → slug should auto-generate as `maude-phase2`.
4. Visibility: **Private** (this contains FDA narratives — keep private).
5. Click **Create**.

Final dataset path on Kaggle becomes `/kaggle/input/maude-phase2/maude_raw.csv` and `/kaggle/input/maude-phase2/cv_splits.json`. The notebook's `CFG.DATA_PATH` and `CFG.SPLITS_PATH` already point at these.

### 2. Add HF_TOKEN to Kaggle Secrets (only if you want to push to HF Hub)

Skip this if you only want to evaluate locally on Kaggle.

1. https://www.kaggle.com/settings → **Account** tab → scroll to **API**.
2. Or in the notebook editor: **Add-ons → Secrets → Add a new secret**.
3. Name: `HF_TOKEN`, value: a HF write token from https://huggingface.co/settings/tokens.

### 3. Upload the notebook

1. https://www.kaggle.com/code → **+ New Notebook**.
2. **File → Upload Notebook** → select `phase2_kaggle/phase2_bioclinbert.ipynb`.
3. Title it `MAUDE Phase 2 — Bio_ClinicalBERT`.

### 4. Configure the notebook session

Right side panel:
- **Accelerator:** `GPU T4 x1` (free tier; ~30 hours/week quota — plenty for this).
- **Internet:** **ON** (needed to download Bio_ClinicalBERT weights from HF Hub).
- **Persistence:** Files only is fine.
- **Add Data:** Click → search `maude-phase2` → add the dataset you uploaded.
- **Environment:** Latest is fine. Pin to "Always use latest environment" if you want consistency.

Verify the dataset attached at the right path:
```python
!ls /kaggle/input/maude-phase2/
# expect:  cv_splits.json  maude_raw.csv
```

---

## Run order

### Step 1 — Single fold sanity pass (~25–35 min on T4)

The notebook's `CFG.N_FOLDS_TO_RUN = 1` by default. Just **Run All**.

Watch for:
- "Fingerprint matches locked CV splits ✓" → cleaning is identical to Phase 1.
- Token length profile shows `>=512 tokens: <X>%` → sanity-check truncation rate. Expect ≤2% if `MAX_LENGTH=512`.
- Per-epoch validation F1 climbing for 2–3 epochs then plateauing.
- Final fold-1 weighted F1 printed at the end.

**Decision point after fold 1:**

| Fold-1 weighted F1 | Action |
|---|---|
| ≥ 0.852 | **Run all 5 folds.** Promising. Set `CFG.N_FOLDS_TO_RUN = 5`. |
| 0.840–0.851 | **Run all 5 folds anyway** to get tight CV bands. Likely underperforming hardened, but worth confirming with full CV. |
| < 0.840 | **Stop and diagnose.** Likely a config bug — check class weights, learning rate, label encoding, fingerprint match. Do not burn 4 more T4 hours on a broken config. |

### Step 2 — Full 5-fold sweep (~2.5–3 hours on T4)

1. Edit cell 5 (CFG): `N_FOLDS_TO_RUN = 5`.
2. **Run All** again. Each fold writes its own checkpoint and clears GPU memory between folds.
3. Final aggregation cell prints the comparison block:

```
PHASE 2 — BIO_CLINICALBERT vs PHASE 1 HARDENED
==================================================
Phase 1 hardened weighted F1:   0.8484 ± 0.0012
Phase 2 BERT       weighted F1: 0.XXXX ± 0.XXXX
Δ weighted: ±0.XXXX

Phase 1 hardened macro F1:      0.7379 ± 0.0032
Phase 2 BERT       macro F1:    0.XXXX ± 0.XXXX
Δ macro: ±0.XXXX

Per-class F1 (Phase 2, mean ± std):
  D     0.XXXX ± 0.XXXX   (Phase 1: 0.7701)  Δ ±0.XXXX
  I     0.XXXX ± 0.XXXX   (Phase 1: 0.8474)  Δ ±0.XXXX
  M     0.XXXX ± 0.XXXX   (Phase 1: 0.8929)  Δ ±0.XXXX
  O     0.XXXX ± 0.XXXX   (Phase 1: 0.4413)  Δ ±0.XXXX

Promotion gate (weighted ≥ 0.853 AND f1_D not regressed):
  PASS / FAIL
==================================================
```

This block is what you send back. It determines whether BERT replaces the champion.

### Step 3 — Inspect errors

Cell 19 produces a confusion matrix and the top-N misclassified narratives for the last fold. Look specifically at:
- Death misclassified as Injury (most clinically dangerous direction).
- Other misclassified — the small "O" class is fragile; expect this.

### Step 4 — Download artefacts

Right panel → **Output** → download the entire `/kaggle/working/` folder (or pick individual files):
- `phase2_bert_metrics.json` → commit to `experimental` branch under `models/phase2_bert_metrics.json`.
- `confusion_matrix_fold_*.png` → for the LinkedIn followup if Phase 2 wins.
- `fold_*_best.pt` → only keep if you intend to push to HF Hub.

### Step 5 — (Optional) Push the best fold to HF Hub

If you want to deploy a Phase 2 demo Space later, uncomment the cell labeled "10. (Optional) Push best fold-1 model to Hugging Face Hub" and re-run just that cell. It uses the `HF_TOKEN` secret.

Result: a new repo at `mukundisb/maude-clinicalbert` containing the tokenizer + best checkpoint. You can wire this into a new Streamlit Space later — separate from the existing v1 Space.

---

## Decision tree after the full sweep

### Case A — Weighted F1 ≥ 0.860 AND f1_D ≥ 0.78  (clear win)
BERT meaningfully beats the hardened baseline on both weighted F1 and the clinically important Death class.

**Action:** Promote to champion. Build a new HF Space pointing at `mukundisb/maude-clinicalbert`. Write the LinkedIn followup: "Phase 2: BERT actually beat the baseline — and here's why it matters for the Death class."

### Case B — 0.853 ≤ Weighted F1 < 0.860 AND f1_D ≥ 0.77 (gate met, modest)
Promotion gate met but lift is small. The lift is real but expensive (T4 GPU vs CPU inference, 110M params vs ~150KB linear model).

**Action:** Promote *conditionally*. Keep both models served. Document the cost tradeoff in the LinkedIn followup. Mention inference latency comparison.

### Case C — Weighted F1 < 0.853 OR f1_D regresses ≥ 0.02 (gate failed)
Bio_ClinicalBERT does not beat hardened TF-IDF on this task at this scale.

**Action:** This is the most defensible writeup. The narrative becomes: "Bio_ClinicalBERT, the most-cited clinical BERT, did not beat a properly tuned TF-IDF + LR baseline on 154k MAUDE records. The lecture about 'don't reach for ML when stats works' was right. Here's the methodology, here's why, and here's what would change my mind."

This is a stronger LinkedIn post than a marginal win — it's the post that shows scientific rigor.

### Case D — F1 plateau but training loss collapses (overfitting)
Validation F1 stops climbing at epoch 2–3 while training loss keeps falling.

**Action:** Try `WEIGHT_DECAY=0.05`, `EPOCHS=3`, or a higher learning rate to a single fold first. Don't burn another full sweep on a guess.

---

## Cost & quota

- Single fold: ~25–35 min T4 = ~0.5 hours of weekly quota.
- Full 5-fold: ~2.5–3 hours T4.
- Free Kaggle quota: 30 GPU-hours/week → you can afford 8–10 full sweeps. Plenty of room for hyperparameter exploration if needed.

---

## Common failures and fixes

**"Fingerprint mismatch on 'sha1_first_100_labels'"**
The cleaning logic in the notebook diverged from `clean_dataframe()` in the GitHub repo. Either regenerate `cv_splits.json` from the current cleaning, OR fix the notebook's inline cleaning to match. The notebook copies the cleaning rules inline specifically to avoid needing to clone the repo.

**OOM during training**
Reduce `CFG.BATCH_SIZE` from 8 → 4 and increase `CFG.GRAD_ACCUM` from 4 → 8 to keep the effective batch at 32.

**HF Hub `401 Unauthorized` when pushing**
The `HF_TOKEN` secret is missing or expired. Re-add it via Add-ons → Secrets.

**`OSError: emilyalsentzer/Bio_ClinicalBERT requires authorization`**
Internet is OFF in the notebook session, OR the model became gated. Toggle Internet ON in the right panel. (As of April 2026, Bio_ClinicalBERT is publicly accessible without auth.)

**Kaggle session disconnects mid-training**
Kaggle interactive sessions time out after 12 hours of inactivity. For the full 5-fold sweep, click "Save & Run All (Commit)" instead of just running interactively — that runs the notebook headless and saves outputs even if your browser disconnects.

---

## When you report back

Send me the comparison block from Step 2 (the one starting `PHASE 2 — BIO_CLINICALBERT vs PHASE 1 HARDENED`). That single block determines:
1. Which case (A/B/C/D) we're in.
2. Whether we write the LinkedIn followup as "BERT won" or "BERT didn't beat the baseline — and that's the point."
3. Whether we build a Phase 2 HF Space or stop here.
