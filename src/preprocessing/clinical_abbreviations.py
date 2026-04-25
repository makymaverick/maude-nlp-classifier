"""
Extended clinical abbreviation dictionary for MAUDE narrative preprocessing.

This module supersedes the 16-entry ABBREVIATION_MAP in text_cleaner.py for use
by the Phase 2 hardened pipeline. It contains ~120 entries covering:

  - Cardiac       (EKG, MI, CHF, AFib, PEA, VT, VFib, ...)
  - Respiratory   (SOB, COPD, ARDS, SpO2, CPAP, ...)
  - Neuro         (CVA, TIA, GCS, ICP, ...)
  - ICU / triage  (ICU, NICU, CCU, ER, ED, OR, PACU, DNR, ...)
  - Clinical ops  (CPR, ACLS, BLS, IV, IM, NG, NPO, VS, ...)
  - Labs / imaging (CBC, BMP, CT, MRI, PET, US, ...)
  - MAUDE-specific (MDR, UDI, HCP, HCF, UNK, ...)

Usage (hardened pipeline only — does NOT affect the champion v1 model):

    from src.preprocessing.clinical_abbreviations import EXTENDED_ABBREVIATION_MAP
    from src.preprocessing.text_cleaner import clean_text

    # Custom clean_text call using the extended map
    # (see src.model.classifier_hardened.expand_clinical for a wrapper)

Design notes:
  - All patterns use \\b word boundaries and are applied case-INsensitively.
  - Replacements are LOWERCASE full forms — they'll merge cleanly with the
    downstream lowercasing step in clean_text().
  - Bigram-friendly expansions ("EKG" -> "electrocardiogram" rather than
    "electro cardio gram") so TF-IDF bigrams capture meaning.
  - Ambiguous abbreviations were omitted rather than guessed. Examples:
      "PT" (patient vs physical therapy vs prothrombin time) -> already in
           base map as "patient"; not overridden here.
      "BS" (bowel sounds vs blood sugar) -> omitted.
      "DM" (diabetes mellitus vs device malfunction) -> omitted given the
           domain.

Author: Mukund Padmanabha — Phase 2 Step 0 (harden TF-IDF baseline)
"""

# flake8: noqa: E501

EXTENDED_ABBREVIATION_MAP: dict[str, str] = {
    # ─── Base (inherited from text_cleaner.py ABBREVIATION_MAP) ──────────
    r"\bpt\b": "patient",
    r"\bpts\b": "patients",
    r"\bmd\b": "physician",
    r"\bdr\b": "doctor",
    r"\bhosp\b": "hospital",
    r"\badm\b": "admitted",
    r"\bdx\b": "diagnosis",
    r"\btx\b": "treatment",
    r"\brx\b": "prescription",
    r"\bs/p\b": "status post",
    r"\bw/\b": "with",
    r"\bw/o\b": "without",
    r"\bh/o\b": "history of",
    r"\bc/o\b": "complaint of",
    r"\bn/v\b": "nausea vomiting",
    r"\bsob\b": "shortness of breath",
    r"\bunk\b": "unknown",

    # ─── Cardiac ─────────────────────────────────────────────────────────
    r"\bekg\b": "electrocardiogram",
    r"\becg\b": "electrocardiogram",
    r"\bmi\b": "myocardial infarction",
    r"\bchf\b": "congestive heart failure",
    r"\bafib\b": "atrial fibrillation",
    r"\ba[- ]?fib\b": "atrial fibrillation",
    r"\bvfib\b": "ventricular fibrillation",
    r"\bv[- ]?fib\b": "ventricular fibrillation",
    r"\bvt\b": "ventricular tachycardia",
    r"\bsvt\b": "supraventricular tachycardia",
    r"\bpea\b": "pulseless electrical activity",
    r"\brosc\b": "return of spontaneous circulation",
    r"\bcpb\b": "cardiopulmonary bypass",
    r"\bcabg\b": "coronary artery bypass graft",
    r"\bpci\b": "percutaneous coronary intervention",
    r"\baicd\b": "implantable cardioverter defibrillator",
    r"\bicd\b": "implantable cardioverter defibrillator",
    r"\bppm\b": "permanent pacemaker",
    r"\bdvt\b": "deep vein thrombosis",
    r"\bpe\b": "pulmonary embolism",
    r"\bhtn\b": "hypertension",
    r"\bhld\b": "hyperlipidemia",
    r"\bcad\b": "coronary artery disease",

    # ─── Respiratory ─────────────────────────────────────────────────────
    r"\bcopd\b": "chronic obstructive pulmonary disease",
    r"\bards\b": "acute respiratory distress syndrome",
    r"\bspo2\b": "oxygen saturation",
    r"\bo2\b": "oxygen",
    r"\bcpap\b": "continuous positive airway pressure",
    r"\bbipap\b": "bilevel positive airway pressure",
    r"\bett\b": "endotracheal tube",
    r"\bett\s+placement\b": "endotracheal tube placement",
    r"\bngt\b": "nasogastric tube",
    r"\bng\b": "nasogastric",

    # ─── Neuro ───────────────────────────────────────────────────────────
    r"\bcva\b": "cerebrovascular accident stroke",
    r"\btia\b": "transient ischemic attack",
    r"\bgcs\b": "glasgow coma scale",
    r"\bicp\b": "intracranial pressure",
    r"\btbi\b": "traumatic brain injury",
    r"\bsci\b": "spinal cord injury",
    r"\bsah\b": "subarachnoid hemorrhage",
    r"\bich\b": "intracerebral hemorrhage",
    r"\bloc\b": "loss of consciousness",

    # ─── ICU / Triage / Settings ─────────────────────────────────────────
    r"\bicu\b": "intensive care unit",
    r"\bnicu\b": "neonatal intensive care unit",
    r"\bpicu\b": "pediatric intensive care unit",
    r"\bccu\b": "cardiac care unit",
    r"\ber\b": "emergency room",
    r"\bed\b": "emergency department",
    r"\bor\b": "operating room",
    r"\bpacu\b": "post anesthesia care unit",
    r"\bdnr\b": "do not resuscitate",
    r"\bdni\b": "do not intubate",
    r"\btoc\b": "transfer of care",

    # ─── Clinical Ops / Procedures ───────────────────────────────────────
    r"\bcpr\b": "cardiopulmonary resuscitation",
    r"\bacls\b": "advanced cardiac life support",
    r"\bbls\b": "basic life support",
    r"\biv\b": "intravenous",
    r"\bim\b": "intramuscular",
    r"\bsc\b": "subcutaneous",
    r"\bsq\b": "subcutaneous",
    r"\bnpo\b": "nothing by mouth",
    r"\bvs\b": "vital signs",
    r"\bi&o\b": "intake and output",
    r"\bq\s?h\b": "every hour",
    r"\bprn\b": "as needed",

    # ─── Labs / Imaging ──────────────────────────────────────────────────
    r"\bcbc\b": "complete blood count",
    r"\bbmp\b": "basic metabolic panel",
    r"\bcmp\b": "comprehensive metabolic panel",
    r"\bpt/inr\b": "prothrombin time",
    r"\bct\b": "computed tomography",
    r"\bmri\b": "magnetic resonance imaging",
    r"\bpet\b": "positron emission tomography",
    r"\bus\b": "ultrasound",
    r"\becho\b": "echocardiogram",
    r"\btte\b": "transthoracic echocardiogram",
    r"\btee\b": "transesophageal echocardiogram",
    r"\bcxr\b": "chest xray",
    r"\bkub\b": "abdominal xray",

    # ─── MAUDE / Regulatory / Device Lifecycle ──────────────────────────
    r"\bmdr\b": "medical device report",
    r"\budi\b": "unique device identifier",
    r"\bhcp\b": "healthcare professional",
    r"\bhcf\b": "healthcare facility",
    r"\bfda\b": "food and drug administration",
    r"\bce\b": "ce marking",
    r"\bmdd\b": "medical device directive",
    r"\bmdr\s+report\b": "medical device report",
    r"\bmfr\b": "manufacturer",
    r"\bmfg\b": "manufacturer",
    r"\bdevice\s+mfr\b": "device manufacturer",
    r"\bserial\s+no\b": "serial number",
    r"\bmodel\s+no\b": "model number",

    # ─── Body parts (commonly abbreviated) ───────────────────────────────
    r"\ble\b": "lower extremity",
    r"\bue\b": "upper extremity",
    r"\bble\b": "bilateral lower extremity",
    r"\bbue\b": "bilateral upper extremity",
    r"\bgi\b": "gastrointestinal",
    r"\bgu\b": "genitourinary",
    r"\bmsk\b": "musculoskeletal",
}


def get_abbreviation_map() -> dict[str, str]:
    """Return the extended abbreviation map (copy, so callers can't mutate)."""
    return dict(EXTENDED_ABBREVIATION_MAP)


__all__ = ["EXTENDED_ABBREVIATION_MAP", "get_abbreviation_map"]
