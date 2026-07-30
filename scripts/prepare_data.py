#!/usr/bin/env python3
"""
Parse kuwiki XML dump → Kurmanji Latin sentences → FullStop word/label JSONL.

Schema per line:
  {"words": ["ez", "dixwazim", ...], "labels": ["0", ".", ...], "text": "original sentence"}
"""

from __future__ import annotations

import argparse
import bz2
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import mwparserfromhell

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kurdish_punctuation.text_utils import (  # noqa: E402
    is_kurmanji_latin,
    split_paragraphs,
    split_sentences,
    strip_wiki_markup,
    words_and_labels,
)

NS = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}
# MediaWiki dumps vary slightly in export schema version; match localname.
PAGE_TAG_END = "page"
TEXT_TAG_END = "text"
TITLE_TAG_END = "title"
NS_TAG_END = "ns"


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def wikitext_to_plain(wikitext: str) -> str:
    try:
        code = mwparserfromhell.parse(wikitext)
        # Drop templates / files heavily; keep link text.
        for node in code.filter_templates(recursive=True):
            try:
                code.remove(node)
            except ValueError:
                pass
        plain = code.strip_code(normalize=True, collapse=True)
    except Exception:  # noqa: BLE001
        plain = strip_wiki_markup(wikitext)
    plain = strip_wiki_markup(plain)
    # Drop table leftovers / list markers.
    plain = re.sub(r"(?m)^\s*[\|*#:]+", " ", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain


def iter_article_texts(dump_path: Path):
    opener = bz2.open if str(dump_path).endswith(".bz2") else open
    with opener(dump_path, "rb") as fh:
        context = ET.iterparse(fh, events=("end",))
        title = None
        ns = None
        for _event, elem in context:
            tag = _local(elem.tag)
            if tag == "title":
                title = elem.text or ""
            elif tag == "ns":
                ns = elem.text or ""
            elif tag == "text":
                # Only main namespace articles.
                if ns == "0" and title and elem.text and not title.startswith("MediaWiki:"):
                    yield title, elem.text
                # Clear text node; page clear happens below.
            elif tag == "page":
                title, ns = None, None
                elem.clear()


def extract_examples(dump_path: Path, max_articles: int | None = None):
    examples = []
    label_counts: Counter[str] = Counter()
    n_articles = 0
    n_skipped_script = 0

    for title, wikitext in iter_article_texts(dump_path):
        n_articles += 1
        if max_articles and n_articles > max_articles:
            break
        if n_articles % 2000 == 0:
            print(f"  articles={n_articles} examples={len(examples)} skipped_script={n_skipped_script}")

        if wikitext.lstrip().lower().startswith(("#redirect", "#beralîkirin")):
            continue

        plain = wikitext_to_plain(wikitext)
        if not plain or not is_kurmanji_latin(plain):
            n_skipped_script += 1
            continue

        for para in split_paragraphs(plain):
            if not is_kurmanji_latin(para):
                continue
            for sent in split_sentences(para):
                if '"' in sent or "«" in sent:
                    # Same filter as FullStop Catalan prep — quotes confuse word/punct alignment.
                    continue
                if not is_kurmanji_latin(sent):
                    continue
                pair = words_and_labels(sent, lowercase=True)
                if not pair:
                    continue
                words, labels = pair
                if len(words) > 128:
                    continue
                examples.append({"words": words, "labels": labels, "text": sent})
                label_counts.update(labels)

    return examples, label_counts, n_articles, n_skipped_script


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: list[dict]) -> None:
    """SEPP-NLG-ish TSV: word\\tbinary_sentence_end\\tpunct_label"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            for w, lab in zip(row["words"], row["labels"]):
                t1 = 1 if lab in {".", "?", "!"} else 0
                fh.write(f"{w}\t{t1}\t{lab}\n")
            fh.write("\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dump",
        type=Path,
        default=Path("data/raw/kuwiki-latest-pages-articles.xml.bz2"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-ratio", type=float, default=0.08)
    p.add_argument("--test-ratio", type=float, default=0.08)
    p.add_argument("--max-articles", type=int, default=None, help="Debug cap on articles")
    args = p.parse_args()

    if not args.dump.exists():
        print(f"Missing dump: {args.dump}\nRun: python scripts/download_wiki.py", file=sys.stderr)
        return 1

    print(f"Parsing {args.dump} …")
    examples, label_counts, n_articles, n_skipped = extract_examples(args.dump, args.max_articles)
    print(f"Articles scanned: {n_articles} (script-skipped≈{n_skipped})")
    print(f"Sentence examples: {len(examples)}")
    print("Label counts:", dict(label_counts.most_common()))

    if len(examples) < 100:
        print("ERROR: too few examples — check dump / filters", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n = len(examples)
    n_test = max(1, int(n * args.test_ratio))
    n_dev = max(1, int(n * args.dev_ratio))
    test = examples[:n_test]
    dev = examples[n_test : n_test + n_dev]
    train = examples[n_test + n_dev :]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "dev.jsonl", dev)
    write_jsonl(args.out_dir / "test.jsonl", test)
    write_tsv(args.out_dir / "train.tsv", train)
    write_tsv(args.out_dir / "dev.tsv", dev)
    write_tsv(args.out_dir / "test.tsv", test)

    meta = {
        "n_articles": n_articles,
        "n_examples": n,
        "n_train": len(train),
        "n_dev": len(dev),
        "n_test": len(test),
        "label_counts": dict(label_counts),
        "dump": str(args.dump),
        "note": "Train domain is Wikipedia; expect domain gap on TV/YouTube ASR.",
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote train/dev/test under {args.out_dir}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
