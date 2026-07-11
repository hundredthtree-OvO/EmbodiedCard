#!/usr/bin/env python3
"""Validate embodied-learning Markdown cards without third-party dependencies."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


REQUIRED_FRONTMATTER = {
    "title",
    "aliases",
    "category",
    "level",
    "status",
    "prerequisites",
    "related",
    "embodied_roles",
    "created",
    "updated",
}

REQUIRED_MAIN_HEADINGS = {
    "## 自测",
    "## 学习导航",
    "## 参考资料",
}

LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
PYTHON_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def discover(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.md") if p.name not in {"README.md", "INDEX.md", "KNOWLEDGE_GRAPH.md"})


def frontmatter_keys(text: str) -> set[str]:
    if not text.startswith("---\n"):
        return set()
    end = text.find("\n---\n", 4)
    if end < 0:
        return set()
    keys = set()
    for line in text[4:end].splitlines():
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):", line)
        if match:
            keys.add(match.group(1))
    return keys


def validate(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []

    keys = frontmatter_keys(text)
    if not keys:
        errors.append("missing or malformed YAML frontmatter")
    else:
        missing = sorted(REQUIRED_FRONTMATTER - keys)
        if missing:
            errors.append(f"missing frontmatter keys: {', '.join(missing)}")

    for heading in sorted(REQUIRED_MAIN_HEADINGS):
        if heading not in text:
            errors.append(f"missing heading: {heading}")
    for level in ("L0", "L1", "L2"):
        if not re.search(rf"^## {level}[：:]", text, re.MULTILINE):
            errors.append(f"missing {level} heading")

    if re.search(r"\[\[[^\]]+\]\]", text):
        errors.append("contains Obsidian-style wiki links")

    if re.search(r"\\\(|\\\)|\\\[|\\\]", text):
        errors.append(r"contains incompatible math delimiters; use $...$ and $$...$$")

    if "\t" in text:
        errors.append("contains tab characters; possible escaped-math corruption")

    prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    prose = re.sub(r"^\$\$\s*$.*?^\$\$\s*$", "", prose, flags=re.DOTALL | re.MULTILINE)
    outside_math = "".join(prose.split("$")[::2])
    if re.search(
        r"\\(?:mathcal|mathbb|mathrm|operatorname|phi|theta|sigma|mu|epsilon|mid|hat)\b",
        outside_math,
    ):
        errors.append("contains LaTeX commands outside $ math delimiters")

    display_delimiters = sum(1 for line in text.splitlines() if line.strip() == "$$")
    if display_delimiters % 2:
        errors.append("has an unmatched display-math $$ delimiter")
    for block in re.findall(r"^\$\$\s*$\n(.*?)^\$\$\s*$", text, re.DOTALL | re.MULTILINE):
        if "$" in block:
            errors.append("contains $ delimiters inside a $$ display-math block")
            break

    if text.count("```") % 2:
        errors.append("has an unmatched fenced-code delimiter")

    for target in LINK_RE.findall(text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        clean_target = target.split("#", 1)[0]
        if clean_target and not (path.parent / clean_target).resolve().exists():
            errors.append(f"broken local link: {target}")

    for index, block in enumerate(PYTHON_BLOCK_RE.findall(text), start=1):
        try:
            ast.parse(block)
        except SyntaxError as exc:
            errors.append(f"Python block {index} has invalid syntax at line {exc.lineno}: {exc.msg}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Markdown card or directory")
    args = parser.parse_args()

    files = discover(args.path)
    if not files:
        print(f"ERROR: no Markdown cards found under {args.path}", file=sys.stderr)
        return 2

    failed = False
    for file in files:
        errors = validate(file)
        if errors:
            failed = True
            print(f"FAIL {file}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {file}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
