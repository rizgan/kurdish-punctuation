# Data layout

```text
data/
  raw/
    kuwiki-latest-pages-articles.xml.bz2   # Wikimedia dump (gitignored)
    wikipedia.jsonl                        # article_id + cleaned text
  processed/
    train.jsonl
    validation.jsonl
    test.jsonl
    statistics.json
  processed_fullstop/                      # legacy FullStop-format corpus (kept)
```

See `scripts/download_wikipedia.md` for dump → JSONL → splits.
