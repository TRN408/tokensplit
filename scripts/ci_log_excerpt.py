#!/usr/bin/env python3
from __future__ import annotations
import pathlib
import re
import sys

PATTERN = re.compile(r"error|failed|failure|traceback|exception|assert", re.I)

def main() -> int:
    if len(sys.argv) != 2:
        return 2
    lines = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines()
    hits = [index for index, line in enumerate(lines) if PATTERN.search(line)]
    if not hits:
        print("no error marker found; inspect the protected full log")
        return 0
    selected = set()
    for index in hits[:8]:
        selected.update(range(max(0, index - 2), min(len(lines), index + 3)))
    for index in sorted(selected)[:80]:
        print(lines[index])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
