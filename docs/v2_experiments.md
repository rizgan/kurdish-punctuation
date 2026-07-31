# v2 experiment log

Primary long-text protocol: **400 held-out articles** (`evaluate_long_text.py`, article mode).  
Gate: QUESTION F1 ↑ vs v1; PERIOD / COMMA / Boundary drop ≤ 0.02; preservation = 1.0.

## Summary (gate metrics)

| Run | Новый question corpus | Sampling weight | Train Q support | QUESTION F1 | PERIOD F1 | COMMA F1 | Boundary F1 | Preservation | Gate |
| --- | --------------------: | --------------: | --------------: | ----------: | --------: | -------: | ----------: | -----------: | ---- |
| v1 baseline | 0 (wiki only) | 1.0 | 907 | 0.62 | 0.93 | 0.66 | 0.93 | 1.0 | baseline |
| v2-exp-01 | real Q in multi-sentence context | **1.0** | | | | | | | |
| v2-exp-02 | same corpus as exp-01 | **2.0–3.0** | | | | | | | |

## QUESTION diagnostics (why F1 moved)

| Run | Q P | Q R | Q F1 | Q→PERIOD | Q→O | O→QUESTION | Train Q support | Notes |
| --- | --: | --: | ---: | -------: | --: | ---------: | --------------: | ----- |
| v1 baseline | 0.56 | 0.69 | 0.62 | 1 | 9 | 6 | 907 | long-text support=32 gold Q |
| v2-exp-01 | | | | | | | | corpus only |
| v2-exp-02 | | | | | | | | + sampling |

How to read:

* **Q→PERIOD / Q→O** — missed or weakened questions (hurts recall).
* **O→QUESTION** — false alarms (hurts precision).
* If P high & R low → need more / more diverse questions (or lower Q threshold later).
* If R high & P low → need harder negatives (assertions near questions), not more oversampling.

Machine-readable: [`v2_experiments.csv`](v2_experiments.csv).
