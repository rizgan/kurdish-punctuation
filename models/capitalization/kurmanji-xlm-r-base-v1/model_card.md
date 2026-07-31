# kurmanji-capitalization-xlm-r-base-v1.0

```text
Model ID:
kurmanji-capitalization-xlm-r-base-v1.0

Model status:
Frozen

Evaluation status:
PASS (long-text gate)
```

Frozen path: `models/capitalization/kurmanji-xlm-r-base-v1/`  
Train run: `outputs/capitalization-xlm-r-base-v1/`

---

## Role in pipeline

```text
raw text
→ punctuation (v2)
→ sentence-start rule (. ? !)
→ this model (TITLE / UPPER / KEEP)
```

Sentence starts are **not** learned. Deterministic `capitalize_sentence_starts` handles them; those positions are `IGNORE` in training.

---

## Labels

| Label | Meaning |
|-------|---------|
| KEEP | leave casing after sentence-start rule |
| TITLE | first letter upper (Kurmanji-safe: `i→I`, not Turkish `İ`) |
| UPPER | all letters upper (acronyms) |

Label on **first** SentencePiece subtoken. Primary metric: macro F1 over **TITLE + UPPER only**.

---

## Long-text gate (400 held-out test articles)

Protocol: original → lower → sentence-start rule → windowed restore → compare to gold.

| Metric | Value | Gate |
|--------|------:|------|
| TITLE F1 | **0.885** | ≥ 0.85 |
| TITLE precision | **0.916** | ≥ 0.88 |
| TITLE recall | **0.857** | — |
| UPPER F1 | **0.945** | ≥ 0.85 |
| KEEP F1 | **0.984** | ≥ 0.97 |
| macro F1 (TITLE+UPPER) | **0.915** | — |
| case-only preservation | **1.0** | = 1.0 |
| sentence-start accuracy | **0.997** | — |

Inference thresholds used: `TITLE=0.80`, `UPPER=0.90`.

Window-level argmax test (no thresholds) had lower TITLE precision (~0.845) and more KEEP→TITLE errors; production thresholds correct this.

---

## Threshold note

Validation sweep (`threshold_tuning.json`): `TITLE=0.80` / `UPPER=0.85` maximizes user target (P≥0.90, R≥0.85). Frozen defaults keep `UPPER=0.90` as in the long-text gate run. Raising TITLE further trades recall for precision (e.g. 0.90 → P≈0.94, R≈0.81).

---

## Data / leakage

- Split by `article_id` only; zero ID overlap across train/val/test
- Zero exact normalized-text hash overlap across splits
- Model input for non-starts is fully lowercased (`input_casing_ok`)
- ~85% of test TITLE token mass shares surface forms with train (common geo names) — expected; see unseen-name probe for rare names

Train size ≈ 5.5M words / 46 897 windows (kuwiki).

---

## Known limits (v1)

- Over-capitalization of common nouns still possible (wiki editorial caps in training)
- Rare personal names / rare acronyms: high precision, weaker recall on tiny hand probes
- Unseen-name probe is small (expand before claiming NER-level coverage)
- No MixedCase restoration

---

## Usage

```powershell
python predict_capitalization.py `
  --model models/capitalization/kurmanji-xlm-r-base-v1 `
  --text "ez li amedê dijîm. navê min azad e."
```
