#!/usr/bin/env python3
"""Download raw v2 question-source corpora (Stage 1).

Sources:
  A — Tatoeba Northern Kurdish (kmr)
  B — OPUS OpenSubtitles (availability check only; may skip)
  C — muzaffercky/kurdish-kurmanji-news
  D — kurdish-ai/kurdish-corpus (kmr / Kurmanji only)

Does NOT download KurCorpus 2B by default (Stage 1 fallback only).
"""

from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "v2_question_raw"
MANIFESTS = RAW / "manifests"
RETRIEVED_AT = date.today().isoformat()

TATOEBA_DETAILED_URL = (
    "https://downloads.tatoeba.org/exports/per_language/kmr/kmr_sentences_detailed.tsv.bz2"
)
TATOEBA_SENTENCES_URL = (
    "https://downloads.tatoeba.org/exports/per_language/kmr/kmr_sentences.tsv.bz2"
)
OPUS_LANGS_URL = "https://opus.nlpl.eu/opusapi/?languages=True&corpus=OpenSubtitles"
OPUS_KU_URL = "https://opus.nlpl.eu/opusapi/?corpus=OpenSubtitles&source=ku&preprocessing=raw&version=latest"
OPUS_KMR_URL = "https://opus.nlpl.eu/opusapi/?corpus=OpenSubtitles&source=kmr&preprocessing=raw&version=latest"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest} already exists ({dest.stat().st_size} bytes)")
        return dest
    print(f"[download] {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "kurmanji-punctuation-v2/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    print(f"[ok] {dest} ({dest.stat().st_size} bytes) sha256={sha256_file(dest)}")
    return dest


def write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_manifest(name: str, payload: dict[str, Any]) -> Path:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    path = MANIFESTS / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Source A — Tatoeba
# ---------------------------------------------------------------------------

def download_tatoeba() -> dict[str, Any]:
    out_dir = RAW / "tatoeba"
    detailed = download(TATOEBA_DETAILED_URL, out_dir / "kmr_sentences_detailed.tsv.bz2")
    sentences = download(TATOEBA_SENTENCES_URL, out_dir / "kmr_sentences.tsv.bz2")

    # sentences_detailed: id, lang, text, username, date_added, date_last_modified
    questions: list[dict[str, Any]] = []
    all_count = 0
    with bz2.open(detailed, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            sid, lang, text = row[0], row[1], row[2]
            username = row[3] if len(row) > 3 else None
            all_count += 1
            if "?" not in text:
                continue
            questions.append(
                {
                    "record_id": f"tatoeba_{sid}",
                    "source": "tatoeba",
                    "source_document_id": sid,
                    "source_url": f"https://tatoeba.org/en/sentences/show/{sid}",
                    "license": "see sentence page / contributor license (often CC BY)",
                    "retrieved_at": RETRIEVED_AT,
                    "text": text,
                    "language_code": lang or "kmr",
                    "dialect": "kurmanji",
                    "script": "Latn",
                    "author": username,
                    "url_or_reference": f"https://tatoeba.org/en/sentences/show/{sid}",
                }
            )

    jsonl = out_dir / "questions_raw.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    manifest = {
        "source": "tatoeba",
        "language": "Northern Kurdish (kmr)",
        "retrieved_at": RETRIEVED_AT,
        "urls": {
            "sentences_detailed": TATOEBA_DETAILED_URL,
            "sentences": TATOEBA_SENTENCES_URL,
        },
        "files": {
            "kmr_sentences_detailed.tsv.bz2": {
                "path": str(detailed.relative_to(ROOT)),
                "sha256": sha256_file(detailed),
                "bytes": detailed.stat().st_size,
            },
            "kmr_sentences.tsv.bz2": {
                "path": str(sentences.relative_to(ROOT)),
                "sha256": sha256_file(sentences),
                "bytes": sentences.stat().st_size,
            },
            "questions_raw.jsonl": {
                "path": str(jsonl.relative_to(ROOT)),
                "sha256": sha256_file(jsonl),
                "bytes": jsonl.stat().st_size,
            },
        },
        "total_sentences": all_count,
        "question_candidate_sentences": len(questions),
        "note": (
            "Tatoeba sentences are short. Do not use as standalone train windows; "
            "group by author/links or attach real context. Preserve sentence ID, author, license."
        ),
        "status": "downloaded",
    }
    write_manifest("tatoeba", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Source B — OpenSubtitles via OPUS
# ---------------------------------------------------------------------------

def check_opensubtitles(timeout: int = 45) -> dict[str, Any]:
    out_dir = RAW / "opensubtitles"
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "source": "opensubtitles",
        "retrieved_at": RETRIEVED_AT,
        "opus_corpus": "OpenSubtitles",
        "status": "rejected",
        "reason": None,
        "api_checks": {},
    }

    def _get(url: str) -> Any:
        req = urllib.request.Request(url, headers={"User-Agent": "kurmanji-punctuation-v2/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    langs: list[str] = []
    try:
        langs_payload = _get(OPUS_LANGS_URL)
        langs = langs_payload.get("languages") or langs_payload.get("source_languages") or []
        if isinstance(langs, dict):
            langs = list(langs.keys())
        result["api_checks"]["languages_sample"] = sorted(str(x) for x in langs)[:50]
        result["api_checks"]["has_kmr"] = "kmr" in {str(x).lower() for x in langs}
        result["api_checks"]["has_ku"] = "ku" in {str(x).lower() for x in langs}
        result["api_checks"]["has_ku_latn"] = any(
            str(x).lower() in {"ku_latn", "ku-latn"} for x in langs
        )
    except Exception as e:
        result["api_checks"]["languages_error"] = f"{type(e).__name__}: {e}"
        # Offline / timeout fallback from documented OPUS OpenSubtitles language table:
        # OpenSubtitles lists generic `ku` (Kurdish), not `kmr`.
        result["api_checks"]["fallback"] = "documented OpenSubtitles table lists ku, not kmr"
        result["api_checks"]["has_kmr"] = False
        result["api_checks"]["has_ku"] = True

    try:
        kmr_payload = _get(OPUS_KMR_URL)
        result["api_checks"]["kmr_corpora"] = kmr_payload
    except Exception as e:
        result["api_checks"]["kmr_error"] = f"{type(e).__name__}: {e}"
        kmr_payload = {"corpora": []}

    try:
        ku_payload = _get(OPUS_KU_URL)
        result["api_checks"]["ku_corpora_count"] = len(ku_payload.get("corpora") or [])
    except Exception as e:
        result["api_checks"]["ku_error"] = f"{type(e).__name__}: {e}"

    has_kmr = bool(result["api_checks"].get("has_kmr"))
    kmr_hits = kmr_payload.get("corpora") if isinstance(kmr_payload, dict) else None
    if has_kmr and kmr_hits:
        result["status"] = "available_kmr"
        result["reason"] = "OPUS reports kmr for OpenSubtitles; manual QA still required before train use."
    else:
        result["status"] = "rejected"
        result["reason"] = (
            "OpenSubtitles on OPUS exposes generic Kurdish code `ku` (and/or no reliable `kmr`). "
            "Plan forbids using `ku` when Kurmanji cannot be separated from other varieties."
        )

    write_manifest("opensubtitles", result)
    (out_dir / "STATUS.md").write_text(
        f"# OpenSubtitles / OPUS\n\nStatus: **{result['status']}**\n\n{result['reason']}\n",
        encoding="utf-8",
    )
    return result


# ---------------------------------------------------------------------------
# Source C — Kurmanji news
# ---------------------------------------------------------------------------

def download_kurmanji_news(max_rows: int | None = None) -> dict[str, Any]:
    from datasets import load_dataset

    out_dir = RAW / "kurmanji_news"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[hf] loading muzaffercky/kurdish-kurmanji-news …")
    ds = load_dataset("muzaffercky/kurdish-kurmanji-news")

    def iter_rows() -> Iterator[dict[str, Any]]:
        n = 0
        for split_name, split in ds.items():
            for i, row in enumerate(split):
                text = (row.get("content") or row.get("text") or "").strip()
                title = (row.get("title") or "").strip()
                url = row.get("url")
                if not text:
                    continue
                if "?" not in text and "?" not in title:
                    continue
                # Prefer body; keep title only as metadata.
                record = {
                    "record_id": f"kurmanji_news_{split_name}_{i}",
                    "source": "kurmanji_news",
                    "source_document_id": url or f"{split_name}_{i}",
                    "source_url": url,
                    "license": "check dataset card / original news site licenses before redistribution",
                    "retrieved_at": RETRIEVED_AT,
                    "text": text,
                    "title": title,
                    "language_code": "kmr",
                    "dialect": "kurmanji",
                    "script": "Latn",
                    "hf_split": split_name,
                }
                yield record
                n += 1
                if max_rows is not None and n >= max_rows:
                    return

    jsonl = out_dir / "questions_raw.jsonl"
    count = write_jsonl(jsonl, iter_rows())
    manifest = {
        "source": "kurmanji_news",
        "hf_id": "muzaffercky/kurdish-kurmanji-news",
        "retrieved_at": RETRIEVED_AT,
        "license_note": (
            "Dataset card does not list a clear redistribution license; "
            "retain URL provenance; do not treat HF hosting as license clearance."
        ),
        "question_candidate_articles": count,
        "files": {
            "questions_raw.jsonl": {
                "path": str(jsonl.relative_to(ROOT)),
                "sha256": sha256_file(jsonl),
                "bytes": jsonl.stat().st_size,
            }
        },
        "status": "downloaded",
        "note": "Only articles containing '?' were kept. Prefer interviews/FAQ/quotes in filtering.",
    }
    write_manifest("kurmanji_news", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Source D — Kurdish AI corpus (kmr)
# ---------------------------------------------------------------------------

def download_kurdish_ai(max_rows: int | None = None) -> dict[str, Any]:
    """Extract kmr rows with '?' from kurdish-ai parquet shards (skip ckb-only shards)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out_dir = RAW / "kurdish_ai_corpus"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Empirically: shards 0–1 are ckb-only; kmr starts in shard_0002 and fills shard_0003.
    shard_files = ["shard_0002.parquet", "shard_0003.parquet"]
    jsonl = out_dir / "questions_raw.jsonl"
    kept = 0
    scanned_kmr = 0
    per_shard: dict[str, int] = {}

    with jsonl.open("w", encoding="utf-8") as f:
        for fn in shard_files:
            if max_rows is not None and kept >= max_rows:
                break
            if max_rows is None and kept >= 15000:
                break
            print(f"[hf] downloading/reading {fn} …")
            path = hf_hub_download("kurdish-ai/kurdish-corpus", fn, repo_type="dataset")
            table = pq.read_table(path)
            mask = pc.equal(table["language"], "kmr")
            kmr = table.filter(mask)
            scanned_kmr += kmr.num_rows
            shard_kept = 0
            cols = {name: kmr.column(name) for name in kmr.column_names}
            for i in range(kmr.num_rows):
                text = str(cols["text"][i].as_py() or "").strip()
                if not text or "?" not in text:
                    continue
                url = cols["url"][i].as_py() if "url" in cols else None
                record = {
                    "record_id": f"kurdish_ai_{fn}_{i}",
                    "source": "kurdish_ai_corpus",
                    "source_document_id": str(url or f"{fn}_{i}"),
                    "source_url": url,
                    "license": "CC BY 4.0 (dataset card)",
                    "retrieved_at": RETRIEVED_AT,
                    "text": text,
                    "language_code": "kmr",
                    "dialect": "kurmanji",
                    "script": "Latn",
                    "hf_config": "default",
                    "hf_split": "train",
                    "hf_shard": fn,
                    "source_type": cols["source_type"][i].as_py() if "source_type" in cols else None,
                    "original_source": cols["source"][i].as_py() if "source" in cols else None,
                    "quality_score": cols["quality_score"][i].as_py() if "quality_score" in cols else None,
                    "word_count": cols["word_count"][i].as_py() if "word_count" in cols else None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                shard_kept += 1
                if max_rows is not None and kept >= max_rows:
                    break
                if max_rows is None and kept >= 15000:
                    print("[hf] soft-cap 15000 question candidates reached; stopping first batch")
                    break
            per_shard[fn] = shard_kept
            print(f"[hf] {fn}: kmr_rows={kmr.num_rows} question_kept={shard_kept} total_kept={kept}")

    manifest = {
        "source": "kurdish_ai_corpus",
        "hf_id": "kurdish-ai/kurdish-corpus",
        "hf_config": "parquet_kmr_shards_0002_0003",
        "available_configs": ["default"],
        "retrieved_at": RETRIEVED_AT,
        "license": "CC BY 4.0",
        "kmr_rows_scanned": scanned_kmr,
        "question_kept_per_shard": per_shard,
        "question_candidate_records": kept,
        "files": {
            "questions_raw.jsonl": {
                "path": str(jsonl.relative_to(ROOT)),
                "sha256": sha256_file(jsonl),
                "bytes": jsonl.stat().st_size,
            }
        },
        "status": "downloaded",
        "note": (
            "Shards 0–1 are ckb-only and were skipped. "
            "Language labels are not trusted automatically; "
            "manual review of >=300 contexts required. First batch soft-capped at 15000."
        ),
    }
    write_manifest("kurdish_ai_corpus", manifest)
    return manifest


def write_stage1_summary(parts: list[dict[str, Any]]) -> Path:
    summary = {
        "stage": 1,
        "retrieved_at": RETRIEVED_AT,
        "sources": parts,
        "kurcorpus_status": "deferred — use only if A–D insufficient after filtering",
    }
    path = write_manifest("stage1_sources", summary)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sources",
        nargs="+",
        default=["tatoeba", "opensubtitles", "kurmanji_news", "kurdish_ai"],
        choices=["tatoeba", "opensubtitles", "kurmanji_news", "kurdish_ai", "all"],
    )
    p.add_argument("--max-rows", type=int, default=None, help="Cap per HF source (debug)")
    args = p.parse_args()
    sources = args.sources
    if "all" in sources:
        sources = ["tatoeba", "opensubtitles", "kurmanji_news", "kurdish_ai"]

    RAW.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    if "tatoeba" in sources:
        results.append(download_tatoeba())
    if "opensubtitles" in sources:
        results.append(check_opensubtitles())
    if "kurmanji_news" in sources:
        results.append(download_kurmanji_news(max_rows=args.max_rows))
    if "kurdish_ai" in sources:
        results.append(download_kurdish_ai(max_rows=args.max_rows))

    summary_path = write_stage1_summary(results)
    print(json.dumps({"summary": str(summary_path), "sources": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.URLError as e:
        print(f"Network error: {e}", file=sys.stderr)
        raise SystemExit(2)
