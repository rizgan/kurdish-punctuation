# kurmanji-punctuation-xlm-r-base-v2.0-question

```text
Model ID:
kurmanji-punctuation-xlm-r-base-v2.0-question

Model status:
Frozen

Evaluation status:
STRONG_PASS

Corpus review status:
Assisted review completed.
A full independent human audit of the sampled records is pending.
```

Do **not** describe the v2 question corpus as `manually verified` until the independent human audit CSV is completed and gates pass.

Frozen path: `models/punctuation/kurmanji-xlm-r-base-v2/`  
Selected run: `v2-exp-01` (`kurmanji-punctuation-v2-exp-01-real-q-weight-1`)  
Rejected for release: `v2-exp-02` (same corpus, weight 2.0 — no recall gain, lower precision)

---

## What changed vs v1

Improvement came from a **new real question corpus**, not from oversampling:

| Metric (long-text 400, article) | v1 | v2.0 | Δ |
|---------------------------------|-----|------|---|
| QUESTION precision | 0.560 | **0.625** | +0.065 |
| QUESTION recall | 0.690 | **0.781** | +0.091 |
| QUESTION F1 | 0.620 | **0.694** | +0.074 |
| Q→O | 9 | **6** | −3 |
| O→QUESTION | 6 | 7 | +1 |
| PERIOD F1 | 0.935 | 0.933 | −0.002 |
| COMMA F1 | 0.662 | 0.669 | +0.007 |
| sentence_boundary F1 | 0.931 | 0.930 | −0.001 |
| text_preservation | 1.0 | **1.0** | 0 |

Specialized question-test (800 held-out contexts): QUESTION F1 ≈ 0.694 (P 0.757 / R 0.640, support 1743); preservation = 1.0.

`v2-exp-02` (weight 2.0): QUESTION F1 0.685, P 0.610, R 0.781 — STRONG_PASS but weaker than exp-01; discarded for freeze.

---

## Frozen technical surface (unchanged from v1)

- Architecture: `FacebookAI/xlm-roberta-base`
- Labels: `O`, `COMMA`, `PERIOD`, `QUESTION`, `EXCLAMATION`
- Continuous word windows; label on last subtoken
- Seed `42`; core training HPs unchanged
- Same frozen long-text 400-article test protocol and text-preservation check

Allowed change in this release: additional real question train contexts + train dataset hash / output paths.

---

## Train corpus (v2 question addition)

| Item | Value |
|------|-------|
| Dataset hash | `0eea499e0a4102c4fbedffba40634c18365f1e0c269c8b9fad41428a99093fd7` |
| First-batch train questions | 3000 contexts |
| Train QUESTION label support | 5298 (v1: 907) |
| New question windows | ~5339 |
| Question-window share | ≈ 9.1% |
| Constructed-context share | ≈ 7.1% (≤ 20% cap) |
| Sources used | Tatoeba kmr, kurmanji_news, kurdish-ai `kmr` |
| OpenSubtitles | Rejected (`ku` only, not separable `kmr`) |
| Sampling weight (selected) | 1.0 |

Validation / test JSONL hashes match frozen v1. Leakage check: `test_contamination = false`.

---

## Corpus review status (detail)

### Completed: assisted review

- Stratified sample: 450 records (`manual_review_sample.jsonl`)
- Method: heuristic assisted judge (`scripts/assisted_v2_manual_review.py`)
- Report: `data/v2_question_processed/manual_review_report.json`
- This is a **bootstrap quality filter**, not an independent human audit.

### Pending: independent human audit

Use the blind template (assisted labels hidden from the reviewer):

```text
data/v2_question_processed/human_audit_template.csv
```

Protocol: `docs/v2_human_audit_protocol.md`

Minimum practical gates after human audit:

```text
PASS:
  overall_accept_rate >= 0.90
  language_accuracy >= 0.95
  question_validity_rate >= 0.95

PASS_WITH_NOTES:
  overall_accept_rate >= 0.85
  and errors localized in one–two sources

FAIL:
  overall_accept_rate < 0.85
  or systematic language mixing
```

If `PASS` → document corpus as fully confirmed for **v2.0** (`manually_verified: true` only then).  
If `PASS_WITH_NOTES` / 5–10% bad rows clustered in one source → consider cleaned **v2.0.1** (same architecture/HPs), not an immediate revoke of technical STRONG_PASS.  
If `FAIL` → do not claim corpus confirmation.

---

## Known limitations

- Independent human audit of the train question sample is still pending.
- Long-text gold QUESTION support is only 32 labels — F1 variance remains high.
- Question-window share in train is ~9% (below the 15–30% preference band); weight 2.0 did not help.
- Assisted accept rates must not be cited as human verification.

---

## Artifacts

```text
models/punctuation/kurmanji-xlm-r-base-v2/
outputs/punctuation-xlm-r-base-v2-exp-01/
docs/v2_experiments.csv
docs/v2_experiments.md
data/processed_v2_question/
data/v2_question_processed/
data/test_question_specialized/
```
