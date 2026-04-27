#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
import sys

import nbformat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create cleaned copies of Jupyter notebooks without outputs."
    )
    parser.add_argument(
        "--glob",
        required=True,
        help='Glob pattern for source notebooks, for example "**/solution.ipynb".',
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help='Имя выходного файла рядом с исходным notebook, например "solution_clean.ipynb".',
    )
    parser.add_argument(
        "--output",
        default=None,
        help='Полный путь к выходному notebook, например "review/solution_clean.ipynb".',
    )
    return parser.parse_args()


def find_notebooks(pattern: str) -> list[Path]:
    matches: list[Path] = []
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        if fnmatch.fnmatch(path.as_posix(), pattern):
            matches.append(path)
    return sorted(matches)


def clean_notebook(src: Path, dst: Path) -> bool:
    nb = nbformat.read(src, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell["metadata"] = {}

    nb["metadata"] = {}

    dst.parent.mkdir(parents=True, exist_ok=True)

    old_content = dst.read_text(encoding="utf-8") if dst.exists() else None
    new_content = nbformat.writes(nb)

    if old_content == new_content:
        return False

    dst.write_text(new_content, encoding="utf-8")
    return True


def main() -> int:
    args = parse_args()

    if not args.output_name and not args.output:
        print("ERROR: you must provide either --output-name or --output", file=sys.stderr)
        return 1

    if args.output_name and args.output:
        print("ERROR: use only one of --output-name or --output", file=sys.stderr)
        return 1

    notebooks = find_notebooks(args.glob)
    if not notebooks:
        print(f"ERROR: no notebooks found for pattern: {args.glob}", file=sys.stderr)
        return 1

    changed_outputs: list[str] = []

    if args.output:
        if len(notebooks) != 1:
            print(
                f"ERROR: pattern {args.glob!r} matched {len(notebooks)} notebooks, "
                f"but --output expects exactly one source file.",
                file=sys.stderr,
            )
            for path in notebooks:
                print(f" - {path.as_posix()}", file=sys.stderr)
            return 1

        src = notebooks[0]
        dst = Path(args.output)
        changed = clean_notebook(src, dst)
        if changed:
            changed_outputs.append(dst.as_posix())
        print(f"Processed: {src.as_posix()} -> {dst.as_posix()}")

    else:
        for src in notebooks:
            dst = src.with_name(args.output_name)
            changed = clean_notebook(src, dst)
            if changed:
                changed_outputs.append(dst.as_posix())
            print(f"Processed: {src.as_posix()} -> {dst.as_posix()}")

    Path(".nb_clean_changed.txt").write_text(
        "\n".join(changed_outputs) + ("\n" if changed_outputs else ""),
        encoding="utf-8",
    )

    if changed_outputs:
        print("Changed files:")
        for path in changed_outputs:
            print(f" - {path}")
    else:
        print("No cleaned notebooks changed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())