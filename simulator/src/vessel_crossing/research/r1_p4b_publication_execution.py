"""Compatibility facade for the refactored R1 P4B publication pipeline.

M0-08 moved responsibilities into focused modules while preserving the historical import
surface.  No scientific or numerical behavior is defined here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from vessel_crossing.research.r1_p4b_experiment_spec import *  # noqa: F401,F403
from vessel_crossing.research.r1_p4b_experiment_spec import (
    _canonical_json, _sha_bytes, _valid_sha256, _room_tag, _demand_tag,
    _penetration_tag, _plant_profiles, _timing_endpoint, _network_for_room,
)
from vessel_crossing.research.r1_p4b_seed_schedule import P4BSeedRecord, _seed_record
from vessel_crossing.research.r1_p4b_plan_materialization import (
    P4BInputBasis,
    P4BPhysicalPoint,
    P4BConceptualRow,
    P4BPublicationExecutionPlan,
    _basis_key,
    _materialize_input_basis,
    _bindings_and_backend_id,
    _physical_point,
    build_publication_execution_plan,
    validate_publication_execution_plan,
    plan_from_json_bytes,
    summarize_plan,
)
from vessel_crossing.research.r1_p4b_physical_runner import (
    _materialize_runtime_bindings,
    _source_window_metrics,
    _route_completion_metrics,
    run_publication_physical_point,
)
from vessel_crossing.research import r1_p4b_result_schema as _result_schema
from vessel_crossing.research.r1_p4b_result_schema import result_file_name
from vessel_crossing.research.r1_p4b_campaign_executor import (
    initial_checkpoint,
    load_checkpoint,
    verify_execution_authorization,
)


def _atomic_write(path: Path, payload: bytes) -> None:
    """Compatibility hook retained for historical chaos tests and private callers."""
    _result_schema._atomic_write(path, payload)


def write_result_atomically(results_dir: Path, point: P4BPhysicalPoint, payload: bytes) -> tuple[Path, str]:
    result = results_dir / result_file_name(point)
    digest = _sha_bytes(payload)
    _atomic_write(result, payload)
    _atomic_write(Path(str(result) + ".sha256"), f"{digest}  {result.name}\n".encode("utf-8"))
    if _sha_bytes(result.read_bytes()) != digest:
        raise RuntimeError("result raw readback SHA mismatch")
    return result, digest


def write_checkpoint_atomic(path: Path, checkpoint: Mapping[str, object]) -> str:
    payload = _canonical_json(dict(checkpoint))
    digest = _sha_bytes(payload)
    _atomic_write(path, payload)
    _atomic_write(Path(str(path) + ".sha256"), f"{digest}  {path.name}\n".encode("utf-8"))
    if _sha_bytes(path.read_bytes()) != digest:
        raise RuntimeError("checkpoint raw readback SHA mismatch")
    return digest
