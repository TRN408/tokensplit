#!/usr/bin/env python3
"""Product CI stages for the dependency-free token-split policy package."""
from __future__ import annotations

import compileall
import pathlib
import subprocess
import sys
import venv
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
DIST = ROOT / "dist"
PACKAGE_FILES = ("README.md", "LICENSE", "policy.json", "scripts", "tokensplit", "benchmarks", "tests")


def install() -> None:
    if not (VENV / "bin" / "python").exists():
        venv.EnvBuilder(with_pip=False, clear=False).create(VENV)
    print(f"runtime ready: {VENV}")


def test() -> None:
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run([sys.executable, "scripts/validate_policy.py"], cwd=ROOT, check=True)


def typecheck() -> None:
    if not all(compileall.compile_dir(str(ROOT / path), quiet=1) for path in ("scripts", "tokensplit", "benchmarks", "tests")):
        raise SystemExit("typecheck failed")
    print("typecheck passed")


def build() -> None:
    DIST.mkdir(parents=True, exist_ok=True)
    output = DIST / "agent-token-split-policy.zip"
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in PACKAGE_FILES:
            path = ROOT / entry
            if path.is_file():
                archive.write(path, path.relative_to(ROOT))
            else:
                for child in sorted(path.rglob("*")):
                    if child.is_file() and "__pycache__" not in child.parts:
                        archive.write(child, child.relative_to(ROOT))
    print(f"built: {output}")


def full_ci() -> None:
    test()
    typecheck()
    build()


STAGES = {"install": install, "test": test, "typecheck": typecheck, "build": build, "full_ci": full_ci}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: {pathlib.Path(sys.argv[0]).name} <{'|'.join(STAGES)}>")
    STAGES[sys.argv[1]]()
