"""Verify that the frozen A131 source import matches its pinned source-member manifest.

Deliberately excludes derived repository scaffolding and large research archives.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "SOURCE_IMPORT_SHA256.csv"
WINDOWS_CMD = "psp/release_shell/START_CROSS.cmd"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify() -> None:
    errors: list[str] = []
    count = 0
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        required = {"repo_path", "source_component", "original_member_path", "size_bytes", "sha256"}
        if set(rows.fieldnames or ()) != required:
            raise ValueError(f"Unexpected manifest columns: {rows.fieldnames!r}")
        seen: set[str] = set()
        for row in rows:
            count += 1
            name = row["repo_path"]
            relative = Path(name)
            if name in seen or relative.is_absolute() or ".." in relative.parts:
                errors.append(f"invalid or duplicated path: {name}")
                continue
            seen.add(name)
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
                errors.append(f"missing or outside repository: {name}")
                continue
            actual = path.read_bytes()
            if name == WINDOWS_CMD:
                # Git stores text with LF; .gitattributes checks it out as CRLF
                # on Windows. Reconstruct the exact frozen 333-byte input.
                lf = actual.replace(b"\r\n", b"\n")
                if b"\r" in lf:
                    errors.append(f"unexpected CR in {name}")
                    continue
                actual = lf.replace(b"\n", b"\r\n")
            expected_size = int(row["size_bytes"])
            if len(actual) != expected_size or sha256(actual) != row["sha256"]:
                errors.append(f"source-byte SHA/size mismatch: {name}")
    if count != 390:
        errors.append(f"source manifest count expected 390, got {count}")
    if errors:
        raise SystemExit("PINNED SOURCE VERIFICATION FAILED:\n" + "\n".join(errors[:30]))
    print(f"PINNED SOURCE PASS: {count} exact source members; Windows .cmd CRLF normalized by rule")


if __name__ == "__main__":
    verify()
