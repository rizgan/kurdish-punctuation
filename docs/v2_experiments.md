# v2 experiment log

Primary long-text protocol: **400 held-out articles** (`evaluate_long_text.py`, article mode).  
Gate: QUESTION F1 ↑ vs v1; PERIOD / COMMA / Boundary drop ≤ 0.02; preservation = 1.0.

| Run | Новый question corpus | Sampling weight | QUESTION F1 | PERIOD F1 | COMMA F1 | Boundary F1 | Preservation | Gate |
| --- | --------------------: | --------------: | ----------: | --------: | -------: | ----------: | -----------: | ---- |
| v1 baseline | 0 (wiki only, ~907 Q labels) | 1.0 | 0.62 | 0.93 | 0.66 | 0.93 | 1.0 | baseline |
| v2-exp-01 | TBD — real Q in multi-sentence context | **1.0** (no oversample) | | | | | | |
| v2-exp-02 | same corpus as exp-01 | **2.0–3.0** question windows | | | | | | |

Fill F1 cells from `long_text_eval_*.json` → `modes.article` (macro labels + `sentence_boundary_f1` + `text_preservation_rate`).

Machine-readable twin: [`v2_experiments.csv`](v2_experiments.csv).
