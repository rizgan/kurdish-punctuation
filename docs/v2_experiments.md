# v2 experiment log

Primary long-text protocol: **400 held-out articles** (`evaluate_long_text.py`, article mode).

**Overall gate:** QUESTION F1 ↑; PERIOD / COMMA / Boundary drop ≤ 0.02; preservation = 1.0.

**QUESTION practical gate (v1 reading: P 0.56 < R 0.69 → over-fires `?`):**

```text
F1 > 0.62
precision > 0.56
recall ≥ 0.67
O→QUESTION does not grow disproportionately
```

Watch especially on **v2-exp-02**: weight 2.0–3.0 may lift recall while inflating `O→QUESTION`.

## Summary (gate metrics)

| Run | Новый question corpus | Sampling weight | Train Q support | QUESTION F1 | PERIOD F1 | COMMA F1 | Boundary F1 | Preservation | Gate |
| --- | --------------------: | --------------: | --------------: | ----------: | --------: | -------: | ----------: | -----------: | ---- |
| v1 baseline | 0 (wiki only) | 1.0 | 907 | 0.62 | 0.93 | 0.66 | 0.93 | 1.0 | baseline |
| v2-exp-01 | real Q in multi-sentence context | **1.0** | 5298 | **0.694** | 0.933 | 0.669 | 0.930 | 1.0 | **STRONG_PASS** ✓ selected |
| v2-exp-02 | same corpus as exp-01 | **2.0** | 5298 | 0.685 | 0.933 | 0.659 | 0.930 | 1.0 | STRONG_PASS |

## QUESTION diagnostics (why F1 moved)

| Run | Q P | Q R | Q F1 | Q→PERIOD | Q→O | O→QUESTION | Train Q support | Notes |
| --- | --: | --: | ---: | -------: | --: | ---------: | --------------: | ----- |
| v1 baseline | 0.56 | 0.69 | 0.62 | 1 | 9 | 6 | 907 | long-text support=32 gold Q |
| v2-exp-01 | 0.625 | 0.781 | **0.694** | 1 | 6 | 7 | 5298 | selected; frozen as v2; human audit pending |
| v2-exp-02 | 0.610 | 0.781 | 0.685 | 1 | 6 | 7 | 5298 | weight 2.0; lower P |

How to read:

* **Q→PERIOD / Q→O** — missed or weakened questions (hurts recall). v1: Q→O dominates (9), Q→PERIOD rare (1).
* **O→QUESTION** — false alarms (hurts precision). v1: 6 — already material vs P=0.56.
* Prefer **precision↑ with recall≥0.67** over recall-only gains from oversampling.
* If P high & R low → more / more diverse in-context questions.
* If R high & P low → harder negatives near questions; do **not** raise sampling weight further.

Machine-readable: [`v2_experiments.csv`](v2_experiments.csv).
