# ASR benchmark v1 — Kurmanji speech → punctuation (+ optional capitalization)

## Goal

Evaluate the **production pipeline** on real ASR output, not Wikipedia text.

Gold must keep ASR word forms unchanged (no orthography fixes):

```text
ASR:  ez iro hatim tu li ku yi
Gold: ez iro hatim. tu li ku yi?
```

Do **not** correct `iro → îro` or `yi → yî` in gold. That would mix punctuation evaluation with spelling correction.

## Layout

```text
data/asr_benchmark_v1/
├── audio/                  # optional wav/mp3 clips (gitignored binaries)
├── asr_raw.jsonl           # ASR hypotheses (words as produced)
├── punctuation_gold.jsonl  # gold punctuation on the same word sequence
├── metadata.csv            # speaker / style / quality metadata
└── README.md
```

## Target size (v1)

| Item | Target |
|------|--------|
| Clips | 50–100 |
| Speakers | 10–20 |
| Words | 20–50k |
| Mix | male/female, interview/monologue, Q&A, fast/slow, regional variants, varied quality |

## JSONL schemas

`asr_raw.jsonl`:

```json
{"id": "asr_0001", "text": "ez iro hatim tu li ku yi", "audio": "audio/asr_0001.wav"}
```

`punctuation_gold.jsonl`:

```json
{"id": "asr_0001", "text": "ez iro hatim. tu li ku yi?"}
```

`metadata.csv` columns:

```text
id,speaker_id,gender,region,style,speech_rate,audio_quality,duration_sec,n_words,notes
```

Styles: `interview`, `monologue`, `qa`, `other`.  
Speech rate: `slow`, `normal`, `fast`.  
Audio quality: `clean`, `noisy`, `telephone`, `other`.

## Metrics (punctuation)

```text
PERIOD F1
COMMA F1
QUESTION F1
sentence_boundary F1
text_preservation
```

Optional later: capitalization on ASR after predicted punctuation (report separately from gold-punct capitalization).

## Collecting clips

Prefer licensed / permission-cleared Kurmanji speech. Keep speaker IDs anonymized. Store originals under `audio/` (not committed if large).

## Eval command (when filled)

```powershell
python scripts/eval_asr_benchmark.py `
  --asr data/asr_benchmark_v1/asr_raw.jsonl `
  --gold data/asr_benchmark_v1/punctuation_gold.jsonl `
  --punctuation-model models/punctuation/kurmanji-xlm-r-base-v2
```
