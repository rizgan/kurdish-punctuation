# v2 independent human audit protocol

Post-freeze corpus audit. **Do not retrain** by default until this audit finishes.
Retrain only if a systematic source failure is found.

Until this audit completes, **v2.0** remains a frozen research release with:

```text
Corpus review status:
Assisted review completed.
A full independent human audit of the sampled records is pending.
```

## Goal

Confirm quality/provenance of the v2 question training addition. Model long-text metrics already passed on the frozen test; this audit does not change those numbers.

## Blind review rules

1. Start from `data/v2_question_processed/human_audit_template.csv` (or regenerate via script).
2. Review **400–500** stratified records (min 50 per used source).
3. Prefer a higher share of rows that contain `QUESTION` / `?` (already true for this corpus).
4. Reviewer must **not** see assisted-review decisions (`manual_review_assisted.csv`).
5. After scoring, compute agreement with assisted labels offline.

## CSV fields

```text
record_id
source
kurmanji_ok
question_is_real
punctuation_ok
context_ok
encoding_ok
duplicate_or_template
accept
comment
reviewer
reviewed_at
```

Values for boolean columns: `true` / `false` (or `1` / `0`).

## Metrics in the report

```text
overall_accept_rate
language_accuracy
question_validity_rate
punctuation_accuracy
context_accuracy
accept_rate_by_source
agreement_with_assisted_review
reviewed_records
invalid_or_missing_rows
```

Plus one final status:

```text
PASS
PASS_WITH_NOTES
FAIL
```

## Recommended gate

```text
PASS:
  overall_accept_rate >= 0.90
  language_accuracy >= 0.95
  question_validity_rate >= 0.95

PASS_WITH_NOTES:
  overall_accept_rate >= 0.85
  and errors localized in one–two sources
  (or mild shortfalls vs PASS thresholds)

FAIL:
  overall_accept_rate < 0.85
  or systematic language mixing
```

Force language-mix FAIL when confirmed by the auditor:

```powershell
python scripts/score_v2_human_audit.py `
  --csv data/v2_question_processed/human_audit_filled.csv `
  --flag-language-mix
```

## Outcomes

| Status | Action |
|--------|--------|
| `PASS` | Mark corpus review complete; keep **v2.0**; set `manually_verified: true` only then |
| `PASS_WITH_NOTES` | Keep frozen release; document weak source(s); consider cleaned **v2.0.1** |
| `FAIL` | Do not claim corpus confirmation; plan cleaned rebuild / v2.0.1 |

## Commands

```powershell
# Regenerate blind template from stratified sample
python scripts/prepare_v2_human_audit.py

# After filling human_audit_filled.csv:
python scripts/score_v2_human_audit.py `
  --csv data/v2_question_processed/human_audit_filled.csv
```

Exit codes: `0` = PASS, `1` = PASS_WITH_NOTES, `2` = FAIL.
