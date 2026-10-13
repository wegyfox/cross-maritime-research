"""Integrity and environment-provenance helpers for archived run packages."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
from typing import Mapping

import numpy
import scipy

from .version import __version__


@dataclass(frozen=True, slots=True)
class FileIntegrity:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class PackageVerification:
    passed: bool
    checked_files: tuple[FileIntegrity, ...]
    errors: tuple[str, ...]


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_manifest() -> dict[str, str]:
    return {
        "package_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
    }


def file_integrity(path: str | Path, *, relative_to: str | Path | None = None) -> FileIntegrity:
    path = Path(path)
    display = str(path if relative_to is None else path.relative_to(relative_to))
    return FileIntegrity(display, sha256_file(path), path.stat().st_size)


def write_package_index(
    directory: str | Path,
    files: tuple[str, ...] | list[str],
    *,
    run_id: str,
    schema_version: str = "1",
) -> Path:
    directory = Path(directory)
    records = [file_integrity(directory / name, relative_to=directory) for name in files]
    payload = {
        "schema_version": schema_version,
        "run_id": run_id,
        "files": {
            item.path: {"sha256": item.sha256, "size_bytes": item.size_bytes}
            for item in records
        },
    }
    path = directory / "package_index.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_sha256sums(directory: str | Path, files: tuple[str, ...] | list[str]) -> Path:
    directory = Path(directory)
    path = directory / "SHA256SUMS"
    lines = [f"{sha256_file(directory / name)}  {name}" for name in sorted(files)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def verify_run_package(directory: str | Path) -> PackageVerification:
    directory = Path(directory)
    errors: list[str] = []
    checked: list[FileIntegrity] = []
    index_path = directory / "package_index.json"
    sums_path = directory / "SHA256SUMS"
    if not index_path.exists():
        return PackageVerification(False, (), ("package_index.json is missing",))
    if not sums_path.exists():
        return PackageVerification(False, (), ("SHA256SUMS is missing",))

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        indexed_files: Mapping[str, Mapping[str, object]] = index["files"]
    except Exception as exc:
        return PackageVerification(False, (), (f"invalid package_index.json: {exc}",))

    for name, expected in sorted(indexed_files.items()):
        file_path = directory / name
        if not file_path.exists():
            errors.append(f"missing indexed file: {name}")
            continue
        integrity = file_integrity(file_path, relative_to=directory)
        checked.append(integrity)
        if integrity.sha256 != expected.get("sha256"):
            errors.append(f"SHA-256 mismatch: {name}")
        if integrity.size_bytes != expected.get("size_bytes"):
            errors.append(f"size mismatch: {name}")

    sums: dict[str, str] = {}
    try:
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            digest, name = line.split(maxsplit=1)
            sums[name.strip()] = digest
    except Exception as exc:
        errors.append(f"invalid SHA256SUMS: {exc}")

    for name, expected_digest in sorted(sums.items()):
        file_path = directory / name
        if not file_path.exists():
            errors.append(f"SHA256SUMS references missing file: {name}")
            continue
        if sha256_file(file_path) != expected_digest:
            errors.append(f"SHA256SUMS mismatch: {name}")

    required = {
        "run_manifest.json",
        "summary_results.json",
        "study_protocol.json",
        "execution_protocol.json",
        "environment.json",
    }
    missing_from_index = required - set(indexed_files)
    if missing_from_index:
        errors.append(f"required files missing from package index: {sorted(missing_from_index)}")

    return PackageVerification(not errors, tuple(checked), tuple(errors))
