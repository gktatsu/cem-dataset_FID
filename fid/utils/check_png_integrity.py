#!/usr/bin/env python3
"""Utility to detect unreadable PNG files inside a directory tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

try:
    from PIL import Image, UnidentifiedImageError  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("Pillow (PIL) is required to run this script.") from exc

try:
    from tqdm import tqdm  # type: ignore[import]
except ImportError:  # pragma: no cover - progress fallback
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a directory (recursively) for PNG files that "
            "cannot be opened and report their paths."
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Path to the directory containing PNG files",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help=(
            "Disable the progress bar (enabled by default if tqdm "
            "is available)"
        ),
    )
    return parser.parse_args()


def iter_pngs(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if root.is_file():
        raise ValueError(f"Expected a directory, but got a file: {root}")
    yield from root.rglob("*.png")


def is_corrupted(png_path: Path) -> bool:
    try:
        with Image.open(png_path) as img:
            img.verify()
        # Re-open the file to forcefully load the data after verify()
        with Image.open(png_path) as img:
            img.load()
        return False
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        print(f"[CORRUPTED] {png_path} -> {exc}")
        return True


def main() -> int:
    args = parse_args()
    png_root: Path = args.directory.resolve()

    try:
        png_files: List[Path] = list(iter_pngs(png_root))
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    if not png_files:
        print(f"No PNG files found under {png_root}")
        return 0

    iterator: Iterable[Path]
    if not args.no_progress and tqdm is not None:
        iterator = tqdm(png_files, desc="Checking PNGs", unit="file")
    else:
        if not args.no_progress and tqdm is None:
            print("tqdm not installed; progress disabled.", file=sys.stderr)
        iterator = png_files

    corrupted: List[Path] = []
    for path in iterator:
        if is_corrupted(path):
            corrupted.append(path)

    if corrupted:
        print(f"\nDetected {len(corrupted)} corrupted PNG file(s).")
        return 2

    print(f"All {len(png_files)} PNG file(s) opened successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
