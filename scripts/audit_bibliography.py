#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


CITE_PATTERN = re.compile(r"\\cite\w*\{([^}]*)\}")
ENTRY_PATTERN = re.compile(r"@\w+\{\s*([^,\s]+)")


def _collect_citations(tex_paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in tex_paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in CITE_PATTERN.finditer(text):
            for part in match.group(1).split(","):
                key = part.strip()
                if key:
                    keys.add(key)
    return keys


def _load_bib_keys(bib_path: Path) -> tuple[set[str], str]:
    text = bib_path.read_text(encoding="utf-8", errors="ignore")
    keys = set(ENTRY_PATTERN.findall(text))
    return keys, text


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit bibliography quality for thesis chapters.")
    parser.add_argument("--thesis-dir", type=str, default="thesis")
    args = parser.parse_args()

    thesis_dir = Path(args.thesis_dir)
    bib_path = thesis_dir / "references.bib"
    if not bib_path.exists():
        raise SystemExit(f"Missing bibliography file: {bib_path}")

    chapter_paths = sorted((thesis_dir / "chapters").glob("*.tex"))
    if not chapter_paths:
        raise SystemExit("No chapter .tex files found for citation audit.")

    cited = _collect_citations(chapter_paths)
    bib_keys, bib_text = _load_bib_keys(bib_path)

    errors: list[str] = []

    if "author={Unknown}" in bib_text or "Local source file in repository papers folder" in bib_text:
        errors.append("references.bib still contains placeholder metadata ('Unknown' or local-source note).")

    local_cites = sorted(k for k in cited if k.startswith("local_"))
    if local_cites:
        errors.append(f"Non-appendix chapters cite unresolved local_* keys: {', '.join(local_cites[:12])}")

    missing = sorted(cited - bib_keys)
    if missing:
        errors.append(f"Cited keys missing from references.bib: {', '.join(missing[:12])}")

    if errors:
        print("[bib-audit] FAILED")
        for e in errors:
            print(f" - {e}")
        raise SystemExit(1)

    print(f"[bib-audit] OK: {len(cited)} chapter citations, {len(bib_keys)} bib entries")


if __name__ == "__main__":
    main()
