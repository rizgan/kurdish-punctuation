# v2 release plan — question corpus only

Parent baseline: **`kurmanji-punctuation-xlm-r-base-v1.0`**  
(`models/punctuation/kurmanji-xlm-r-base-v1`)

Recommended v2 ID:

```text
kurmanji-punctuation-xlm-r-base-v2.0-question
```

Frozen path for v2 (new folder, never overwrite v1):

```text
models/punctuation/kurmanji-xlm-r-base-v2-question/
```

Train `output_dir` example:

```text
outputs/punctuation-xlm-r-base-v2-question/
```

---

## Frozen from v1 (do not change)

| Item | v1 value / rule |
|------|-----------------|
| Architecture | `FacebookAI/xlm-roberta-base` |
| Example format | continuous word windows |
| Tokenizer | same XLM-R tokenizer (do not retrain SPM) |
| Label schema | `O`, `COMMA`, `PERIOD`, `QUESTION`, `EXCLAMATION` |
| Long-text test | **same 400 test articles** protocol (`evaluate_long_text.py`) |
| Metrics | per-label P/R/F1, macro F1 (excl. O), sentence_boundary F1 |
| Text preservation | algorithm + requirement **= 1.0** |
| Seed | `42` |
| Core training HPs | lr `2e-5`, epochs `3`, bf16, batch/accum as in v1 `training_config.yaml`, edge mask 2/4, window 110–180, `max_length` 256 |

Windowing rules stay the same: random target length, **no** snap to `.?!`, PERIOD@last-word fraction should stay ≪ 10%.

---

## Allowed to change in v2

```text
question corpus          # in-context: statement. question? answer.
sampling weights         # e.g. question_window_weight 2–4×
train dataset hash       # will change; record in run_info.json
output_dir               # new path only
model ID                 # kurmanji-punctuation-xlm-r-base-v2.0-question
```

Do **not**:

- duplicate the 869 wiki questions alone as isolated samples;
- put questions only at end-of-sequence;
- swap to `xlm-roberta-large` or FullStop in this release;
- retune the whole HP grid unless long-text clearly regresses.

---

## Acceptance criteria (same long-text 400-article eval)

| Metric | Requirement vs v1.0 |
|--------|---------------------|
| QUESTION F1 | **increases** (primary goal) |
| PERIOD F1 | drop ≤ **0.02** (v1 ≈ 0.93 → ≥ 0.91) |
| COMMA F1 | drop ≤ **0.02** (v1 ≈ 0.66 → ≥ 0.64) |
| sentence_boundary F1 | drop ≤ **0.02** (v1 ≈ 0.93 → ≥ 0.91) |
| text_preservation | **1.0** |
| Macro F1 | should not collapse; prefer ≥ v1 − 0.02 |

Also log for QUESTION:

```text
precision, recall, F1, support
QUESTION → PERIOD
QUESTION → O
O → QUESTION
PERIOD → QUESTION
```

Optional extra: dedicated question-test (500–1000 in-context items, held out from train).

---

## Checklist before tagging v2.0-question

1. [ ] New corpus merged into continuous windows (questions inside multi-sentence context)
2. [ ] `data/processed` rebuilt → new `dataset_hash` in `run_info.json`
3. [ ] Train to `outputs/...-v2-question/` (not v1 path)
4. [ ] `evaluate_long_text.py` on **same** 400-article protocol
5. [ ] Acceptance table above passes
6. [ ] Copy to `models/punctuation/kurmanji-xlm-r-base-v2-question/`
7. [ ] `SHA256SUMS.txt`, smoke test, `requirements-lock.txt` / env versions, `model_card.md`
8. [ ] README results section: add v2 row/table without removing v1.0
