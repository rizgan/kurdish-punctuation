# Kurmanji punctuation — XLM-RoBERTa base v1.0

**Model ID:** `kurmanji-punctuation-xlm-r-base-v1.0`  
**Path:** `models/punctuation/kurmanji-xlm-r-base-v1`  
**Status:** Frozen production baseline (do not overwrite)

## Task

Restore `, . ? !` in Kurmanji Latin (`kmr_Latn`) text **without** changing words, case, diacritics, numbers, URLs, or emails.

## Architecture

- Base: [`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base)
- Head: token classification (label on **last** subtoken of each word)
- Labels: `O`, `COMMA`, `PERIOD`, `QUESTION`, `EXCLAMATION`

## Training unit (critical)

```text
Training unit: continuous word windows
Train samples: 40,714
Fraction PERIOD at final word: 4.6%
```

Examples are **not** single sentences. Windows are ~110–180 words, sized dynamically so XLM-R encoding fits `max_length=256`. Random target length avoids learning “end of sequence ⇒ PERIOD”.

The old sentence-split regime put nearly **100%** of `PERIOD` labels on the last word of each example and inflated sentence-test PERIOD F1 (~0.99). **Sentence-level test is not the primary metric for this model.**

## Data

- Source: Kurmanji Wikipedia (`ku.wikipedia.org` / kuwiki dump)
- Split by `article_id` (90 / 5 / 5, seed 42)
- Train QUESTION support ≈ 907 (main remaining weakness)

## Primary evaluation (honest long-text)

400 held-out test articles → strip `, . ? !` → sliding-window inference → compare to gold.

| Metric | Value |
|--------|------:|
| Long-text test articles | 400 |
| PERIOD F1 | **0.93** |
| PERIOD recall | **0.95** |
| COMMA F1 | **0.66** |
| QUESTION F1 | **0.62** |
| EXCLAMATION F1 | **0.75** |
| Sentence-boundary F1 | **0.93** |
| Macro F1 | **0.74** |
| Text preservation | **1.0** |

Source report: `evaluation_report.json` / `long_text_eval.json`.

## Intended use

| Domain | Status |
|--------|--------|
| Written articles | Ready |
| Long continuous text | Ready |
| Sentence boundaries | Strong |
| Commas | Good baseline |
| Questions | Works; needs more in-context questions for v2 |
| ASR / interviews | Needs a separate benchmark before claiming readiness |

## Freeze integrity (v1.0)

| Artifact | Path |
|----------|------|
| File hashes | `SHA256SUMS.txt` |
| Dataset hashes | `DATASET_SHA256SUMS.txt` |
| Env lock | repo root `requirements-lock.txt` (`pip freeze`) |
| Run metadata | `run_info.json` (`git_commit`, torch/transformers/cuda, `dataset_hash`, seed) |
| Smoke test | `python scripts/smoke_test_v1.py` → `smoke_test_report.json` |

```powershell
python scripts/smoke_test_v1.py --model models/punctuation/kurmanji-xlm-r-base-v1
```

All smoke cases require `text_preservation == 1.0`.

## Inference

```powershell
python predict.py --model models/punctuation/kurmanji-xlm-r-base-v1 `
  --text "ez îro çûm bajarê lê baran dibariya"
```

```python
from kurmanji_punctuation import PunctuationRestorer
r = PunctuationRestorer("models/punctuation/kurmanji-xlm-r-base-v1")
print(r.restore("ez îro çûm bajarê lê baran dibariya"))
```

## Versioning / next steps (v2)

Do **not** replace this folder. Train future runs into a new path, e.g. `kurmanji-xlm-r-base-v2-question`.

**Frozen for v2:** architecture, continuous windows, tokenizer, labels, 400-article long-text eval, preservation algorithm, seed, base HPs.  
**Allowed to change:** question corpus (in-context only), sampling weights, train `dataset_hash`, `output_dir`, model ID.

Target ID: `kurmanji-punctuation-xlm-r-base-v2.0-question`  
Gate: QUESTION F1 ↑ while PERIOD / COMMA / sentence_boundary drop ≤ 0.01–0.02 and `text_preservation = 1.0`.

Full checklist: [`docs/V2_PLAN.md`](../../docs/V2_PLAN.md).

## License notes

- Wikipedia text: CC BY-SA (share-alike if redistributing derived corpora).
- Base model: see FacebookAI / Hugging Face XLM-RoBERTa terms.
