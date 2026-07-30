#!/usr/bin/env python3
"""Convert kuwiki XML dump to data/raw/wikipedia.jsonl (article_id + cleaned text)."""

from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mwparserfromhell

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kurmanji_punctuation.normalization import normalize_for_dataset, to_nfc  # noqa: E402

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
LATIN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ſÇçÊêÎîŞşÛû]")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def is_kurmanji_latin(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return False
    arabic = sum(1 for c in letters if ARABIC_RE.match(c))
    if arabic / len(letters) > 0.05:
        return False
    latin = sum(1 for c in letters if LATIN_RE.match(c))
    return (latin / len(letters)) >= 0.85


def wikitext_to_plain(wikitext: str) -> str:
    # Drop headings / list-heavy lines early.
    lines = []
    for line in wikitext.splitlines():
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if s.startswith("=") and s.endswith("="):
            continue
        if s.startswith("{|") or s.startswith("|}") or s.startswith("|-") or s.startswith("|"):
            continue
        if s.startswith("*") or s.startswith("#") or s.startswith(":"):
            continue
        if s.lower().startswith("[[file:") or s.lower().startswith("[[dosya:"):
            continue
        if s.lower().startswith("[[category:") or s.lower().startswith("[[kategorî:"):
            continue
        lines.append(line)
    text = "\n".join(lines)
    try:
        code = mwparserfromhell.parse(text)
        for node in list(code.filter_templates(recursive=True)):
            try:
                code.remove(node)
            except ValueError:
                pass
        plain = code.strip_code(normalize=True, collapse=True)
    except Exception:
        plain = text
    plain = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", plain)
    plain = re.sub(r"\{\{[^{}]*\}\}", " ", plain)
    plain = re.sub(r"<[^>]+>", " ", plain)
    plain = re.sub(r"'{2,}", "", plain)
    plain = re.sub(r"https?://\S+", " ", plain)
    return plain


def iter_articles(dump_path: Path):
    opener = bz2.open if str(dump_path).endswith(".bz2") else open
    with opener(dump_path, "rb") as fh:
        title = None
        ns = None
        for _event, elem in ET.iterparse(fh, events=("end",)):
            tag = _local(elem.tag)
            if tag == "title":
                title = elem.text or ""
            elif tag == "ns":
                ns = elem.text or ""
            elif tag == "text":
                if ns == "0" and title and elem.text:
                    yield title, elem.text
            elif tag == "page":
                title, ns = None, None
                elem.clear()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dump",
        type=Path,
        default=Path("data/raw/kuwiki-latest-pages-articles.xml.bz2"),
    )
    p.add_argument("--out", type=Path, default=Path("data/raw/wikipedia.jsonl"))
    p.add_argument("--max-articles", type=int, default=None)
    args = p.parse_args()

    if not args.dump.exists():
        print(f"Missing dump: {args.dump}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with args.out.open("w", encoding="utf-8") as out:
        for title, wikitext in iter_articles(args.dump):
            n_in += 1
            if args.max_articles and n_in > args.max_articles:
                break
            if wikitext.lstrip().lower().startswith(("#redirect", "#beralîkirin")):
                continue
            plain = wikitext_to_plain(wikitext)
            plain = to_nfc(plain)
            if not is_kurmanji_latin(plain):
                continue
            plain = normalize_for_dataset(plain, map_ellipsis_to_period=True)
            if len(plain.split()) < 5:
                continue
            n_out += 1
            article_id = f"wiki_{n_out:06d}"
            out.write(
                json.dumps({"article_id": article_id, "text": plain}, ensure_ascii=False) + "\n"
            )
            if n_out % 2000 == 0:
                print(f"wrote {n_out} articles (scanned {n_in})")
    print(f"Done. scanned={n_in} wrote={n_out} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
