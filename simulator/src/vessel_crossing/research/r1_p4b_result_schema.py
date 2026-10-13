"""R1 P4B result artifact naming and durable result-pair writes."""
from __future__ import annotations

import os
from pathlib import Path

from vessel_crossing.research.r1_p4b_experiment_spec import _sha_bytes
from vessel_crossing.research.r1_p4b_plan_materialization import P4BPhysicalPoint

def result_file_name(point: P4BPhysicalPoint) -> str:
    return f"{point.execution_order:04d}_{point.point_id}.json"

def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)

def write_result_atomically(results_dir: Path, point: P4BPhysicalPoint, payload: bytes) -> tuple[Path, str]:
    result = results_dir / result_file_name(point)
    digest = _sha_bytes(payload)
    _atomic_write(result, payload)
    _atomic_write(Path(str(result) + ".sha256"), f"{digest}  {result.name}\n".encode("utf-8"))
    if _sha_bytes(result.read_bytes()) != digest:
        raise RuntimeError("result raw readback SHA mismatch")
    return result, digest
