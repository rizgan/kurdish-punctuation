# Kurmanji punctuation restoration

Token-classification model that inserts `, . ? !` into **Kurmanji Latin** (`kmr_Latn`) text **without changing words, case, or spelling**.

Base model: [`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base).

```text
ez îro çûm bajarê lê baran dibariya
→ ez îro çûm bajarê, lê baran dibariya.
```

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

## Train on RTX 4090

```powershell
python train.py --config config.yaml
```

Uses BF16 on Ampere+ (4090); otherwise FP16. Class weights: `1/sqrt(freq)` normalized so `O=1`, clipped at 8. Checkpoint: `outputs/punctuation-xlm-r-base/best`.

## Evaluate

```powershell
python evaluate.py `
  --model outputs/punctuation-xlm-r-base/best `
  --data data/processed/test.jsonl `
  --config config.yaml
```

Primary metric: **`punctuation_macro_f1`** (mean F1 of COMMA/PERIOD/QUESTION/EXCLAMATION only).

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
