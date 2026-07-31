# Capitalization v1

Second stage of the Kurmanji text pipeline:

```text
raw text → punctuation v2 → sentence-start rule → XLM-R capitalization
```

## Labels

| Label | Meaning |
|-------|---------|
| `KEEP` | leave token as after sentence-start rule |
| `TITLE` | first letter upper (Kurmanji-safe) |
| `UPPER` | all letters upper (acronyms) |

Sentence starts are **not** learned: deterministic `capitalize_sentence_starts` handles them; those positions are `IGNORE` (`-100`) in training.

Primary metric: **macro F1 over TITLE + UPPER only** (`capitalization_macro_f1`).

## Commands

```powershell
# 1) Inspect auto-labels before full training
python scripts/diagnose_capitalization_labels.py --max-articles 500

# 2) Diagnostic dataset (~800 articles)
python prepare_capitalization_dataset.py `
  --input data/raw/wikipedia.jsonl `
  --config configs/capitalization-diagnostic.yaml

# 3) Diagnostic train
python train_capitalization.py --config configs/capitalization-diagnostic.yaml

# 4) Full v1 dataset (5–10M words target via max_train_words)
python prepare_capitalization_dataset.py `
  --input data/raw/wikipedia.jsonl `
  --config configs/capitalization-v1.yaml

python train_capitalization.py --config configs/capitalization-v1.yaml

# 5) Predict (rule + model)
python predict_capitalization.py `
  --model outputs/capitalization-xlm-r-base-diagnostic/best `
  --text "ez li amedê dijîm. navê min azad e."
```

## Known wiki risks

Wikipedia casing can teach headers, link titles, and editorial caps. Always review `label_diagnosis.json` (`frequent_title_tokens`, `sample_TITLE`, `sample_UPPER`) before a full run. Filters already drop single-letter TITLE and non-Latin tokens.

Frozen v1.0 (after long-text gate): `models/capitalization/kurmanji-xlm-r-base-v1/` — see `model_card.md`.

## Human audit (punctuation v2) — parallel

When the filled CSV is ready:

```powershell
python scripts/score_v2_human_audit.py `
  --csv data/v2_question_processed/human_audit_filled.csv
```
