# Kurmanji punctuation + capitalization restoration

Production pipeline for **Kurmanji Latin** (`kmr_Latn`):

```text
raw / ASR text
→ punctuation v2
→ sentence-start rule
→ capitalization v1
→ preservation checks
→ final text
```

Models do **not** change words, spelling, or Kurmanji letters — only insert `, . ? !` and adjust case.

## Production CLI

```powershell
python restore_text.py `
  --punctuation-model models/punctuation/kurmanji-xlm-r-base-v2 `
  --capitalization-model models/capitalization/kurmanji-xlm-r-base-v1 `
  --input input.txt `
  --output output.txt

python restore_text.py --mode full --text "ez li amedê dijîm navê min azad e"
python restore_text.py --mode punctuation --text "ez li amedê dijîm"
python restore_text.py --mode capitalization --text "ez li amedê dijîm. navê min azad e."
python restore_text.py --mode full --json-output --text "ez li amedê dijîm navê min azad e"
```

Example (`--mode full`):

```text
Ez li Amedê dijîm, navê min Azad e.
```

| Flag | Meaning |
|------|---------|
| `--mode full` | punctuation → sentence-start → capitalization (default) |
| `--mode punctuation` | punctuation only |
| `--mode capitalization` | sentence-start rule + capitalization (expects punctuated input) |
| `--json-output` | stages + `preservation.punctuation_only` / `case_only` |

Production capitalization thresholds: `TITLE=0.80`, `UPPER=0.85`.

Pre-release smoke (edge cases, 400 articles, 10k+ word blob, near-idempotency):

```powershell
python scripts/smoke_test_production_pipeline.py `
  --max-articles 400 `
  --long-text-words 10000
```

Last run: **PASS** — preservation 1.0, near-idempotency 1.0 (`outputs/production_pipeline_smoke.json`). Exact `pipeline(pipeline(x)) == pipeline(x)` is rare (~1%); a second pass may still move `,` / `.`, but word identity after lower+strip stays stable.

| Component | Status | Path |
|-----------|--------|------|
| Punctuation v2 | Frozen, STRONG_PASS | `models/punctuation/kurmanji-xlm-r-base-v2` |
| Capitalization v1 | Frozen, long-text gate PASS | `models/capitalization/kurmanji-xlm-r-base-v1` |
| Unified pipeline | Smoke PASS | `restore_text.py` · `src/kurmanji_pipeline/` |
| ASR benchmark | Scaffold (fill clips next) | `data/asr_benchmark_v1/` |

Model cards: [punctuation v2](models/punctuation/kurmanji-xlm-r-base-v2/model_card.md) · [capitalization v1](models/capitalization/kurmanji-xlm-r-base-v1/model_card.md) · [capitalization docs](docs/capitalization_v1.md)

## Results — punctuation v2.0 (`kurmanji-punctuation-xlm-r-base-v2.0-question`)

Frozen path: [`models/punctuation/kurmanji-xlm-r-base-v2`](models/punctuation/kurmanji-xlm-r-base-v2)

Selected run: `v2-exp-01`. Same architecture/windows/preservation protocol as v1; improvement from a real question corpus.

| Metric (long-text 400, article) | v1 | v2.0 |
|---------------------------------|-----|------|
| QUESTION F1 | 0.620 | **0.694** |
| QUESTION precision | 0.560 | **0.625** |
| QUESTION recall | 0.690 | **0.781** |
| PERIOD F1 | 0.935 | 0.933 |
| COMMA F1 | 0.662 | 0.669 |
| Sentence-boundary F1 | 0.931 | 0.930 |
| Text preservation | 1.0 | **1.0** |

See [`docs/V2_PLAN.md`](docs/V2_PLAN.md) and [`docs/v2_experiments.md`](docs/v2_experiments.md). Independent human audit of the question sample is still pending ([protocol](docs/v2_human_audit_protocol.md)).

## Results — punctuation v1.0 (`kurmanji-punctuation-xlm-r-base-v1.0`)

Frozen path: [`models/punctuation/kurmanji-xlm-r-base-v1`](models/punctuation/kurmanji-xlm-r-base-v1) · [`model_card.md`](models/punctuation/kurmanji-xlm-r-base-v1/model_card.md)

**Primary metric:** honest long-text (400 held-out Wikipedia articles → strip `, . ? !` → windowed restore). Sentence-level F1 is *not* the headline score (it can leak “end of sequence ⇒ PERIOD”).

| Metric | Long-text (400 articles) |
|--------|-------------------------:|
| PERIOD F1 | **0.93** |
| PERIOD recall | **0.95** |
| COMMA F1 | **0.66** |
| QUESTION F1 | **0.62** |
| EXCLAMATION F1 | **0.75** |
| Sentence-boundary F1 | **0.93** |
| Macro F1 (excl. `O`) | **0.74** |
| Text preservation | **1.0** |

Training unit: continuous word windows (~110–180 words), 40 714 train samples. PERIOD @ last word of window: **4.6%** (was ~100% with sentence-split). Earlier sentence-split model on the same protocol: PERIOD F1 ≈ **0.30**.

> Legacy FullStop fine-tune scripts live under `scripts/` + `src/kurdish_punctuation/` and `data/processed_fullstop/`. The active stack is XLM-R punctuation + capitalization.

## Results — capitalization v1.0

Frozen path: [`models/capitalization/kurmanji-xlm-r-base-v1`](models/capitalization/kurmanji-xlm-r-base-v1)

| Metric (long-text 400) | Value |
|------------------------|------:|
| TITLE F1 | **0.885** |
| TITLE precision | **0.916** |
| UPPER F1 | **0.945** |
| case-only preservation | **1.0** |

Sentence starts are deterministic (`capitalize_sentence_starts`); the model predicts TITLE / UPPER / KEEP only.

## Project layout

```text
restore_text.py                 # production CLI (full / punctuation / capitalization)
config.yaml                     # punctuation train/infer defaults
configs/capitalization-v1.yaml
prepare_dataset.py
prepare_capitalization_dataset.py
train.py / train_capitalization.py
evaluate.py / evaluate_long_text.py
evaluate_capitalization.py / evaluate_capitalization_long_text.py
predict.py / predict_capitalization.py
scripts/
  smoke_test_production_pipeline.py
  download_wiki.py
  wiki_dump_to_jsonl.py
  eval_asr_benchmark.py
src/
  kurmanji_pipeline/            # unified restore orchestration
  kurmanji_punctuation/
  kurmanji_capitalization/
models/
  punctuation/kurmanji-xlm-r-base-v{1,2}/
  capitalization/kurmanji-xlm-r-base-v1/
tests/
data/raw|processed|processed_capitalization/
outputs/
```

## Labels — punctuation

| Label | Mark |
|-------|------|
| O | (none) |
| COMMA | `,` |
| PERIOD | `.` |
| QUESTION | `?` |
| EXCLAMATION | `!` |

Label sits on the **last** SentencePiece subtoken of each word.

## Labels — capitalization

| Label | Meaning |
|-------|---------|
| KEEP | leave casing after sentence-start rule |
| TITLE | first letter upper (Kurmanji-safe) |
| UPPER | all letters upper |

Label on the **first** SentencePiece subtoken.

## Setup (Windows PowerShell)

```powershell
cd C:\Users\gerd\Documents\kurdish-punctuation
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
pip install -r requirements.txt
pip install -e .
```

## Setup (Linux bash)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
pip install -r requirements.txt
pip install -e .
```

## Wikipedia data

```powershell
python scripts/download_wiki.py
python scripts/wiki_dump_to_jsonl.py --dump data/raw/kuwiki-latest-pages-articles.xml.bz2 --out data/raw/wikipedia.jsonl
python prepare_dataset.py --input data/raw/wikipedia.jsonl --output-dir data/processed --config config.yaml
```

Splits are **by `article_id`** (90/5/5, seed 42) with an overlap assertion.

Integrity for frozen punctuation v1: `SHA256SUMS.txt`, `requirements-lock.txt`, `run_info.json`.

```powershell
python scripts/smoke_test_v1.py --model models/punctuation/kurmanji-xlm-r-base-v1
```

```powershell
# Punctuation — train into a NEW output_dir only; never overwrite frozen models/
python train.py --config config.yaml
python evaluate_long_text.py --model models/punctuation/kurmanji-xlm-r-base-v2 --max-articles 400 --tag check

# Capitalization
python prepare_capitalization_dataset.py --input data/raw/wikipedia.jsonl --config configs/capitalization-v1.yaml
python train_capitalization.py --config configs/capitalization-v1.yaml
python evaluate_capitalization_long_text.py --max-articles 400
```

Train examples for punctuation are **continuous word windows** (not single sentences). Check `data/processed/statistics.json` → `fraction_of_period_labels_at_final_word` (should be ≪ 10%).

## Evaluate

```powershell
python evaluate.py `
  --model models/punctuation/kurmanji-xlm-r-base-v2 `
  --data data/processed/test.jsonl `
  --config config.yaml

python evaluate_long_text.py `
  --model models/punctuation/kurmanji-xlm-r-base-v2 `
  --max-articles 400 `
  --tag v2_check

python evaluate_capitalization_long_text.py --max-articles 400
```

Use **long-text** metrics for model selection. Window/sentence eval is secondary.

## Single-stage predict

```powershell
python predict.py `
  --model models/punctuation/kurmanji-xlm-r-base-v2 `
  --text "ez îro çûm bajarê lê baran dibariya"

python predict_capitalization.py `
  --model models/capitalization/kurmanji-xlm-r-base-v1 `
  --text "ez li amedê dijîm. navê min azad e."
```

Prefer `restore_text.py` for the full production path. Long texts use overlapping word windows; logits are averaged. `restore()` raises if preservation fails.

## ONNX (optional)

```powershell
python export_onnx.py --model models/punctuation/kurmanji-xlm-r-base-v2 --out-dir onnx
```

## Tests

```powershell
pytest -q
```

## Limits / domain gap

- Train corpus is Wikipedia → expect weaker TV/YouTube ASR punctuation until you add news/subtitles.
- Does not insert `: ; — …` quotes/brackets.
- Does not fix ASR word errors.
- Second pipeline pass can still move commas/periods (near-idempotent on words, not always exact).

## Add a new punctuation mark later

1. Extend `LABELS` / maps in `src/kurmanji_punctuation/constants.py`.
2. Keep it in corpus prep (stop stripping it).
3. Bump `model.num_labels` and re-train (or continue training from `best`).

## Continue training

Point `model.name` in a copied config at a checkpoint under `outputs/` and run `train.py` again on new JSONL. Never overwrite frozen paths under `models/`.

## Omnilingual (later)

Post-process only for `ku` / `kmr_Latn` after ASR — do not fine-tune the ASR itself for this.
