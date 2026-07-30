# Preparing Kurmanji Wikipedia data

## Official dump

- Wiki: https://ku.wikipedia.org (Kurmanji Latin / Hawar)
- Dump index: https://dumps.wikimedia.org/kuwiki/latest/
- File: `kuwiki-latest-pages-articles.xml.bz2` (~43 MB)

Sorani (Arabic script) lives on `ckb.wikipedia.org` — do **not** mix it into this corpus.

## Download (PowerShell)

```powershell
python scripts/download_wiki.py
# or:
# Invoke-WebRequest -Uri "https://dumps.wikimedia.org/kuwiki/latest/kuwiki-latest-pages-articles.xml.bz2" `
#   -OutFile "data/raw/kuwiki-latest-pages-articles.xml.bz2" -UserAgent "kurmanji-punctuation/0.1"
```

## Convert dump → raw JSONL

```powershell
python scripts/wiki_dump_to_jsonl.py `
  --dump data/raw/kuwiki-latest-pages-articles.xml.bz2 `
  --out data/raw/wikipedia.jsonl
```

Each line:

```json
{"article_id":"wiki_000001","text":"Lê belê, em ê sibê biçin?"}
```

## Build train/validation/test

```powershell
python prepare_dataset.py `
  --input data/raw/wikipedia.jsonl `
  --output-dir data/processed `
  --config config.yaml
```

Splits are by `article_id` (90/5/5, seed 42) with a programmatic no-overlap check.
