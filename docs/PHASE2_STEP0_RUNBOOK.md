# Phase 2 — Step 0 Runbook: Harden the TF-IDF + LR Baseline

**Goal:** Establish the real ceiling of the Phase 1 pipeline before declaring Phase 2 (Bio_ClinicalBERT) a winner. Every Phase 2 BERT experiment will be compared against this hardened baseline on identical CV folds.

**Champion to beat:** `cv_f1_mean = 0.8481` (weighted), `training_records = 154,776`

**What this step produces:**

- `data/cv_splits.json` — locked 5-fold StratifiedKFold indices (source of truth for Phase 2)
- `models/hardened_pipeline.joblib` — hardened TF-IDF+LR model
- `models/hardened_metrics.json` — per-class F1, CV F1 mean/std, training records
- One new MLflow experiment: `maude-nlp-severity-phase2`

**What this step does NOT do:**

- Modify `src/model/classifier.py`, `src/model/train.py`, `src/preprocessing/text_cleaner.py`
- Overwrite `models/champion_metrics.json` or `models/maude_classifier.joblib`
- Touch the deployed HF Space

---

## Prerequisites

1. On your Windows machine, switch to `experimental`:
   ```powershell
   cd "C:\path\to\maude-nlp-classifier"
   git checkout experimental
   git pull origin experimental
   ```

2. Copy the 5 new files from Cowork into your repo (same relative paths):
   ```
   src/preprocessing/clinical_abbreviations.py      (new)
   src/model/classifier_hardened.py                 (new)
   src/evaluation/__init__.py                       (new)
   src/evaluation/cv_splits.py                      (new)
   src/model/train_hardened.py                      (new)
   docs/PHASE2_STEP0_RUNBOOK.md                     (this file)
   ```

3. Confirm cached data exists:
   ```powershell
   dir data\raw\maude_raw.csv
   ```
   If missing, fetch it: `python -m src.model.train --records 154000` (will cache along the way).

---

## Run order

### 1. Generate locked CV splits (one-time — DO NOT re-run casually)

```powershell
python -m src.evaluation.cv_splits generate --data-path data\raw\maude_raw.csv --out data\cv_splits.json
```

Expected output:

```
  CV splits locked to: data/cv_splits.json
  Records:            154,776
  Folds:              5 (random_state=42)
  Label distribution: {'M': ..., 'I': ..., 'D': ..., 'O': ..., 'UNKNOWN': ...}
  Fingerprint SHA1:   <16 hex chars>...
```

**Important:** commit `data/cv_splits.json` to the `experimental` branch after this. Phase 2 BERT runs will reference the committed version.

**File size:** ~5 MB for 154k records (5 folds × train+val index arrays).

### 2. Run hardened training — fixed config (fast sanity pass)

```powershell
python -m src.model.train_hardened --use-cached --run-name "p2s0-fixed-baseline"
```

**Runtime:** ~8–12 min on a modern laptop (4-core CPU). Watch the console for per-fold per-class F1.

### 3. Run hardened training — with extended abbreviations

```powershell
python -m src.model.train_hardened --use-cached --enrich --run-name "p2s0-enriched"
```

If this beats run #2 by ≥0.002 weighted F1, keep `--enrich` for all subsequent runs.

### 4. Run hardened training — with grid search (the real hardened run)

```powershell
python -m src.model.train_hardened --use-cached --enrich --tune --run-name "p2s0-tuned"
```

**Runtime:** ~45–90 min depending on grid size. `DEFAULT_HARDENED_GRID` covers:

- `word ngram`: `(1,2)` vs `(1,3)`
- `word max_features`: 50k vs 100k
- `char ngram`: `(3,5)` (fixed)
- `C`: `{0.3, 1.0, 3.0, 10.0}`

→ 16 combinations × 5 folds = 80 fits. Each fit ~40s on 154k records. Plan for ~90 min.

### 5. Inspect results

```powershell
# Show the summary from the last run
type models\hardened_metrics.json

# Compare per-class F1 in MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000 — filter experiment "maude-nlp-severity-phase2"
```

---

## Decision tree after Step 0

### Case A — Hardened weighted F1 ≥ **0.860** (strong lift)

The real ceiling of TF-IDF+LR is substantially above 0.848. Phase 2 BERT must beat **this new number** (not 0.848). Kaggle BERT fine-tuning begins against a harder baseline — defensible for the LinkedIn follow-up.

**Action:** proceed to Phase 2 BERT. Note the hardened number as "Phase 1 (hardened)" in the LinkedIn post.

### Case B — Hardened weighted F1 in **0.852–0.860** (modest lift)

Meaningful but not huge. Promote hardened to champion if Death-class F1 also improves.

**Action:**
```powershell
python -m src.model.train_hardened --use-cached --enrich --tune --promote-champion
```

Then proceed to Phase 2 BERT.

### Case C — Hardened weighted F1 ≤ **0.852** (no meaningful lift)

The v1 champion is already at the TF-IDF+LR ceiling. That's a useful finding — reinforces the "BERT needs to do real work to win" framing.

**Action:** keep v1 champion. Log the hardened config as an MLflow run tagged `no_promotion`. Proceed to Phase 2 BERT.

### Case D — Death-class F1 degrades

If weighted F1 improves but `f1_D` drops by >0.02, reject the config regardless of weighted F1. Death-class recall is the clinical metric.

---

## Verification checklist

Before moving to Phase 2 BERT, confirm:

- [ ] `data/cv_splits.json` exists and is committed to `experimental`
- [ ] `models/hardened_metrics.json` shows per-class F1 (especially `f1_D`)
- [ ] MLflow UI shows `maude-nlp-severity-phase2` experiment with ≥1 run
- [ ] `models/champion_metrics.json` still shows v1 numbers (unless you explicitly promoted)
- [ ] Phase 1 HF Space still works (no files on `experimental` affect deployed main)

---

## Commit + push

After you've selected the best hardened run and updated `models/hardened_metrics.json`:

```powershell
git add src\preprocessing\clinical_abbreviations.py
git add src\model\classifier_hardened.py
git add src\model\train_hardened.py
git add src\evaluation\__init__.py
git add src\evaluation\cv_splits.py
git add docs\PHASE2_STEP0_RUNBOOK.md
git add data\cv_splits.json
git add models\hardened_metrics.json

git commit -m "phase2/step0: harden TF-IDF baseline + lock CV splits

- Add 115-entry clinical abbreviation map (cardiac, respiratory,
  neuro, ICU, MAUDE-specific)
- New classifier_hardened.py with FeatureUnion (word + char_wb),
  wider hyperparam grid, per-class F1 tracking
- Lock 5-fold StratifiedKFold splits to data/cv_splits.json for
  Phase 2 BERT parity
- New standalone train_hardened.py entrypoint (does NOT modify
  train.py or overwrite champion artefacts)
- Runbook at docs/PHASE2_STEP0_RUNBOOK.md"

git push origin experimental
```

`models/hardened_pipeline.joblib` is LFS-tracked (matches `*.joblib` rule on experimental) — ensure `git lfs push origin experimental` runs automatically as part of push, or invoke it explicitly.

---

## When you report back

Send me the summary block from the final hardened run:

```
  PHASE 2 STEP 0 — HARDENED TF-IDF + LR RESULTS
  =======================================
  Champion v1 CV F1 (weighted):   0.8481
  Hardened CV F1 (weighted):      0.XXXX ± 0.XXXX
  Hardened CV F1 (macro):         0.XXXX ± 0.XXXX
  Per-class F1 (hardened, mean across folds):
    D          0.XXXX ± 0.XXXX
    I          0.XXXX ± 0.XXXX
    M          0.XXXX ± 0.XXXX
    O          0.XXXX ± 0.XXXX
    UNKNOWN    0.XXXX ± 0.XXXX
  Δ vs champion: ±0.XXXX  (reason)
```

That one block determines the Phase 2 BERT target. From there we move to Kaggle.
