# app/data/ingest_kaggle.py
"""
Convert the Kaggle News Category Dataset (JSON/JSONL) into Markdown files
that our existing indexer can read from the ./data folder.

Input schema (v2 commonly has):
{
  "category": "...",
  "headline": "...",
  "short_description": "...",
  "authors": "...",
  "link": "...",
  "date": "YYYY-MM-DD HH:MM:SS"
}

Usage:
  python -m app.data.ingest_kaggle \
      --input /path/to/News_Category_Dataset_v2.json \
      --outdir data \
      --limit 0
"""
import argparse, json, os, re, uuid
from pathlib import Path

def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\-_]+", "_", s.strip())
    return s[:120] or str(uuid.uuid4())  # cap file name length

def to_markdown(rec: dict) -> str:
    cat = rec.get("category", "").strip()
    headline = rec.get("headline", "").strip()
    desc = rec.get("short_description", "").strip()
    authors = rec.get("authors", "").strip()
    link = rec.get("link", "").strip()
    date = rec.get("date", "").strip()

    # Keep article content up top; store rich metadata as front matter-like block
    md = []
    md.append(f"# {headline}".strip())
    md.append("")
    if desc:
        md.append(desc)
        md.append("")
    md.append("---")
    md.append(f"category: {cat}")
    if authors: md.append(f"authors: {authors}")
    if date:    md.append(f"date: {date}")
    if link:    md.append(f"link: {link}")
    md.append("---")
    return "\n".join(md).strip() + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to Kaggle JSON/JSONL file")
    ap.add_argument("--outdir", default="data", help="Output folder (indexed by prepare_indexes.py)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of items (0 = all)")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            cat = rec.get("category", "uncategorized")
            headline = rec.get("headline", "") or "untitled"
            prefix = sanitize_filename(cat.lower())
            name = sanitize_filename(headline.lower()) or str(uuid.uuid4())
            # Optional: keep categories in subfolders
            cat_dir = outdir / prefix
            cat_dir.mkdir(parents=True, exist_ok=True)

            md = to_markdown(rec)
            (cat_dir / f"{name}.md").write_text(md, encoding="utf-8")

            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"✅ Wrote {n} files into {outdir}/<category>/")

if __name__ == "__main__":
    main()
