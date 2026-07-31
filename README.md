# Kurmanji punctuation restoration

Token-classification model that inserts `, . ? !` into **Kurmanji Latin** (`kmr_Latn`) text **without changing words, case, or spelling**.

Base model: [`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base).

```text
ez îro çûm bajarê lê baran dibariya
→ ez îro çûm bajarê, lê baran dibariya.
```

## Results — v1.0 (`kurmanji-punctuation-xlm-r-base-v1.0`)

Frozen path: [`models/punctuation/kurmanji-xlm-r-base-v1`](models/punctuation/kurmanji-xlm-r-base-v1) · details in [`model_card.md`](models/punctuation/kurmanji-xlm-r-base-v1/model_card.md)

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

Training setup that made this honest:

| | |
|--|--:|
| Training unit | continuous word windows (~110–180 words) |
| Train samples | 40 714 |
| PERIOD @ last word of window | **4.6%** (was ~100% with sentence-split) |

Earlier sentence-split model on the same long-text protocol: PERIOD F1 ≈ **0.30**, recall ≈ **0.18**.

**v2** will keep architecture, windows, tokenizer, labels, the same 400-article long-text protocol, preservation, seed, and base HPs fixed — change only the question corpus / sampling / dataset hash / output path. Target ID: `kurmanji-punctuation-xlm-r-base-v2.0-question`. See [`docs/V2_PLAN.md`](docs/V2_PLAN.md) and the run table in [`docs/v2_experiments.md`](docs/v2_experiments.md).

> Legacy FullStop fine-tune scripts live under `scripts/` + `src/kurdish_punctuation/` and `data/processed_fullstop/`. The active pipeline is this XLM-R project.

## Project layout

```text
config.yaml
prepare_dataset.py
train.py
evaluate.py
predict.py
export_onnx.py
scripts/
  download_wikipedia.md
  wiki_dump_to_jsonl.py
  download_wiki.py          # still used to fetch kuwiki dump
src/kurmanji_punctuation/
tests/
data/raw|processed/
outputs/
```

## Labels

| Label | Mark |
|-------|------|
| O | (none) |
| COMMA | `,` |
| PERIOD | `.` |
| QUESTION | `?` |
| EXCLAMATION | `!` |

Label sits on the **last SentencePiece subtoken** of each word.

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
# Dump already may exist at data/raw/kuwiki-latest-pages-articles.xml.bz2
python scripts/download_wiki.py
python scripts/wiki_dump_to_jsonl.py --dump data/raw/kuwiki-latest-pages-articles.xml.bz2 --out data/raw/wikipedia.jsonl
python prepare_dataset.py --input data/raw/wikipedia.jsonl --output-dir data/processed --config config.yaml
```

Splits are **by `article_id`** (90/5/5, seed 42) with an overlap assertion.

Best **frozen** checkpoint: `models/punctuation/kurmanji-xlm-r-base-v1` (`kurmanji-punctuation-xlm-r-base-v1.0`). See `model_card.md` there.

Integrity: `SHA256SUMS.txt`, `requirements-lock.txt`, `run_info.json` (git/torch/cuda/`dataset_hash`).

```powershell
python scripts/smoke_test_v1.py --model models/punctuation/kurmanji-xlm-r-base-v1
```

Training run artifacts (may be overwritten by experiments): `outputs/punctuation-xlm-r-base-windows/`.  
Sentence-level legacy data: `data/processed_sentence/`.

```powershell
python prepare_dataset.py --input data/raw/wikipedia.jsonl --output-dir data/processed --config config.yaml
# Train into a NEW output_dir only — never overwrite models/punctuation/kurmanji-xlm-r-base-v1
python train.py --config config.yaml
python evaluate_long_text.py --model models/punctuation/kurmanji-xlm-r-base-v1 --max-articles 400 --tag check
python predict.py --model models/punctuation/kurmanji-xlm-r-base-v1 --text "ez îro çûm bajarê lê baran dibariya"
```

Train examples are **continuous word windows** (not single sentences). Check `data/processed/statistics.json` → `fraction_of_period_labels_at_final_word` (should be ≪ 10%).

## Evaluate

```powershell
python evaluate.py `
  --model models/punctuation/kurmanji-xlm-r-base-v1 `
  --data data/processed/test.jsonl `
  --config config.yaml

python evaluate_long_text.py `
  --model models/punctuation/kurmanji-xlm-r-base-v1 `
  --max-articles 400 `
  --tag v1_check
```

Use **long-text** `punctuation_macro_f1` / `sentence_boundary_f1` for model selection. Window/sentence eval is secondary.

## Predict

```powershell
python predict.py `
  --model outputs/punctuation-xlm-r-base/best `
  --text "ez îro çûm bajarê lê baran dibariya"

python predict.py `
  --model outputs/punctuation-xlm-r-base/best `
  --input-file input.txt `
  --output-file output.txt
```

Long texts use overlapping word windows; logits are averaged. `restore()` raises if word sequence is not preserved.

## ONNX (optional)

```powershell
python export_onnx.py --model outputs/punctuation-xlm-r-base/best --out-dir onnx
```

## Tests

```powershell
pytest -q
```

## Limits / domain gap

- Train corpus is Wikipedia → expect weaker TV/YouTube ASR punctuation until you add news/subtitles.
- Does not insert `: ; — …` quotes/brackets.
- Does not fix ASR word errors.

## Add a new punctuation mark later

1. Extend `LABELS` / maps in `src/kurmanji_punctuation/constants.py`.
2. Keep it in corpus prep (stop stripping it).
3. Bump `model.num_labels` and re-train (or continue training from `best`).

## Continue training

Point `model.name` in a copied config at `outputs/punctuation-xlm-r-base/best` and run `train.py` again on new JSONL.

## Omnilingual (later)

Post-process only for `ku` / `kmr_Latn` after ASR — do not fine-tune the ASR itself for this.
