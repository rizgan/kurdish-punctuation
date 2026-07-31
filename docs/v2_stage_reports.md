# Stage reports — Kurmanji Punctuation v2

## Этап 1 — Источники

```text
Этап: 1 Источники
Выполнено: Tatoeba скачан; OPUS OpenSubtitles проверен и отклонён; kurmanji-news скачан; kurdish-ai kmr извлечён (soft-cap 15k); KurCorpus отложен
Созданные файлы:
  data/v2_question_raw/**
  data/v2_question_raw/manifests/*.json
  scripts/download_v2_question_sources.py
Количество принятых записей (raw candidates):
  tatoeba: 1107
  kurmanji_news: 35666
  kurdish_ai_corpus: 15000 (cap)
  opensubtitles: 0 (rejected)
Количество отклонённых записей: OpenSubtitles целиком
Обнаруженные проблемы:
  - OpenSubtitles только `ku`, не `kmr`
  - kurmanji_news: лицензия на карточке HF неясна
  - Tatoeba: короткие предложения, нельзя как standalone windows
Gate этапа: PASS_WITH_NOTES
Следующий шаг: Этап 2 фильтрация
```

## Этап 2 — Фильтрация

```text
Этап: 2 Фильтрация
Выполнено: script/arabic/question/?/length + exact/nopunct/near dedup; исправлен классификатор типов (убран голый \bku\b); первая партия 3000
Созданные файлы:
  data/v2_question_processed/accepted.jsonl
  data/v2_question_processed/rejected.jsonl
  data/v2_question_processed/train_questions.jsonl   # first batch 3000
  data/v2_question_processed/manual_review_*.csv/jsonl
  data/v2_question_processed/*statistics*.json
  data/v2_question_processed/deduplication_report.json
  scripts/filter_v2_question_corpus.py
  scripts/select_v2_first_batch.py
Количество принятых записей: 49277 (auto-filter pool) / 3000 (first-batch train)
Количество отклонённых записей: 2496
Обнаруженные проблемы:
  - auto-pool слишком большой для первого эксперимента → ограничен first batch 3000
  - MULTIPLE_QUESTION был раздут из-за маркера ku (исправлено)
Gate этапа: PASS (first batch 3000, type share ≤ 0.40)
Следующий шаг: Этап 3 ручная проверка CSV
```

## Этап 3 — Ручная проверка (ожидает человека)

```text
Этап: 3 Ручная проверка
Выполнено: стратифицированная выборка 450 + CSV-шаблон
Созданные файлы:
  data/v2_question_processed/manual_review_sample.jsonl
  data/v2_question_processed/manual_review_template.csv
  data/v2_question_processed/manual_review_preview.json
Количество принятых записей: (заполняет человек)
Количество отклонённых записей: (заполняет человек)
Обнаруженные проблемы: нужна человеческая разметка accept rate ≥ 90% по источникам
Gate этапа: PENDING_HUMAN
Следующий шаг: заполнить manual_review_template.csv
```

## Этап 4 — Leakage (первая партия)

```text
Этап: 4 Дедупликация и leakage
Выполнено: exact/lower/nopunct/MinHash/long-sequence vs validation+test (4370 refs)
Созданные файлы: data/v2_question_processed/leakage_report.json
Количество принятых записей: 3000 clean
Количество отклонённых записей: 0 leaks
Обнаруженные проблемы: нет
Gate этапа: PASS (test_contamination=false)
Следующий шаг: Этап 3 human review (блокирует train), затем specialized question-test и Stage 6 merge
```
