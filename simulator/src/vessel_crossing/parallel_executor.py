"""Process-parallel, resumable execution for frozen replication protocols.

Parallelism changes only *where* independent seeded replications are computed.
The frozen protocol, seed schedule, batch boundaries, precision rule and final
statistics remain identical to :mod:`vessel_crossing.executor`.

Checkpoints are internal trusted computation artifacts.  They use pickle so a
partially completed run can retain compact per-replication aggregate metrics.
Never load a checkpoint obtained from an untrusted source.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import time
from typing import Mapping

from .executor import (
    ProtocolRunResult,
    RunStoppingReason,
    build_execution_components,
    finalize_protocol_run,
    precision_status,
    _point_result,
)
from .experiments import (
    DemandSweepPoint,
    DemandSweepPointResult,
    demand_point_seed_map,
    replication_seeds,
    run_paired_replications,
)
from .metrics import AggregateMetrics
from .protocol import ExecutableReplicationProtocol
from .version import __version__


@dataclass(frozen=True, slots=True)
class CompactExperimentResult:
    """Only the fields needed by replication summaries and paired differences."""

    policy_name: str
    aggregate: AggregateMetrics


@dataclass(frozen=True, slots=True)
class CompactReplicationResult:
    """Checkpoint-friendly equivalent of one heavy ``ReplicationResult``."""

    replication_index: int
    seed: int
    experiments: tuple[CompactExperimentResult, ...]


@dataclass(slots=True)
class ParallelCheckpointState:
    schema_version: str
    execution_protocol_fingerprint_sha256: str
    study_protocol_fingerprint_sha256: str
    software_version: str
    point_seeds: tuple[tuple[str, int], ...]
    batch_end_counts: tuple[int, ...]
    replications_by_point: dict[str, dict[int, CompactReplicationResult]]


@dataclass(frozen=True, slots=True)
class ParallelExecutionProgress:
    completed: bool
    result: ProtocolRunResult | None
    checkpoint_path: str | None
    completed_replications_by_point: tuple[tuple[str, int], ...]
    fully_completed_replications: int
    next_target_replications: int | None
    batch_end_counts: tuple[int, ...]
    tasks_completed_this_call: int


_WORKER_COMPONENT_CACHE: dict[str, tuple[object, object, tuple[object, ...]]] = {}


def _compact_replication(replication) -> CompactReplicationResult:
    return CompactReplicationResult(
        replication_index=replication.replication_index,
        seed=replication.seed,
        experiments=tuple(
            CompactExperimentResult(exp.policy_name, exp.aggregate)
            for exp in replication.experiments
        ),
    )


def _worker_replication(payload):
    """Top-level picklable process worker for exactly one seeded replication."""

    protocol, point, point_seed, replication_index = payload
    fingerprint = protocol.fingerprint_sha256()
    components = _WORKER_COMPONENT_CACHE.get(fingerprint)
    if components is None:
        geometry, simulation_config, controllers = build_execution_components(protocol)
        components = (geometry, simulation_config, controllers)
        _WORKER_COMPONENT_CACHE[fingerprint] = components
    geometry, simulation_config, controllers = components
    replication = run_paired_replications(
        geometry=geometry,
        scenario_config=point.scenario_config,
        base_config=simulation_config,
        controllers=controllers,
        scenario_name=f"{protocol.study.protocol_id}:{point.name}",
        base_seed=point_seed,
        replications=1,
        replication_start_index=replication_index,
    )[0]
    return point.name, _compact_replication(replication)


def checkpoint_file(checkpoint_directory: str | Path) -> Path:
    return Path(checkpoint_directory) / "parallel_checkpoint.pkl"


def checkpoint_checksum_file(checkpoint_directory: str | Path) -> Path:
    return Path(checkpoint_directory) / "parallel_checkpoint.pkl.sha256"


def checkpoint_progress_file(checkpoint_directory: str | Path) -> Path:
    return Path(checkpoint_directory) / "parallel_progress.json"


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _point_counts(
    state: ParallelCheckpointState,
    point_names: tuple[str, ...],
) -> tuple[tuple[str, int], ...]:
    return tuple((name, len(state.replications_by_point[name])) for name in point_names)


def _contiguous_count(indexed: Mapping[int, CompactReplicationResult]) -> int:
    count = 0
    while count in indexed:
        count += 1
    return count


def _fully_completed_count(
    state: ParallelCheckpointState,
    point_names: tuple[str, ...],
) -> int:
    if not point_names:
        return 0
    return min(_contiguous_count(state.replications_by_point[name]) for name in point_names)


def _schedule_boundaries(protocol: ExecutableReplicationProtocol) -> tuple[int, ...]:
    study = protocol.study
    values = [study.initial_replications]
    while values[-1] < study.max_replications:
        values.append(min(study.max_replications, values[-1] + study.replication_batch_size))
    return tuple(values)


def _new_checkpoint_state(
    protocol: ExecutableReplicationProtocol,
    points: tuple[DemandSweepPoint, ...],
    point_seed_map: Mapping[str, int],
) -> ParallelCheckpointState:
    return ParallelCheckpointState(
        schema_version="1",
        execution_protocol_fingerprint_sha256=protocol.fingerprint_sha256(),
        study_protocol_fingerprint_sha256=protocol.study.fingerprint_sha256(),
        software_version=__version__,
        point_seeds=tuple((point.name, int(point_seed_map[point.name])) for point in points),
        batch_end_counts=(),
        replications_by_point={point.name: {} for point in points},
    )


def _validate_checkpoint_state(
    state: ParallelCheckpointState,
    protocol: ExecutableReplicationProtocol,
    points: tuple[DemandSweepPoint, ...],
    point_seed_map: Mapping[str, int],
) -> None:
    if state.schema_version != "1":
        raise ValueError(f"unsupported checkpoint schema: {state.schema_version!r}")
    if state.execution_protocol_fingerprint_sha256 != protocol.fingerprint_sha256():
        raise ValueError("checkpoint execution-protocol fingerprint mismatch")
    if state.study_protocol_fingerprint_sha256 != protocol.study.fingerprint_sha256():
        raise ValueError("checkpoint study-protocol fingerprint mismatch")
    if state.software_version != __version__:
        raise ValueError(
            f"checkpoint software version {state.software_version!r} does not match {__version__!r}"
        )

    expected_point_seeds = tuple(
        (point.name, int(point_seed_map[point.name])) for point in points
    )
    if state.point_seeds != expected_point_seeds:
        raise ValueError("checkpoint demand-point seed map mismatch")
    if tuple(state.replications_by_point) != tuple(point.name for point in points):
        raise ValueError("checkpoint demand-point names/order mismatch")

    policy_names = (protocol.study.baseline_policy, protocol.study.comparison_policy)
    for point in points:
        point_seed = int(point_seed_map[point.name])
        indexed = state.replications_by_point[point.name]
        for index, replication in indexed.items():
            if index < 0 or index >= protocol.study.max_replications:
                raise ValueError(f"checkpoint replication index out of range: {index}")
            if replication.replication_index != index:
                raise ValueError("checkpoint replication index/key mismatch")
            expected_seed = replication_seeds(
                base_seed=point_seed,
                count=1,
                start_index=index,
            )[0]
            if replication.seed != expected_seed:
                raise ValueError(
                    f"checkpoint seed mismatch at {point.name} replication {index}"
                )
            if tuple(exp.policy_name for exp in replication.experiments) != policy_names:
                raise ValueError(
                    f"checkpoint policy order mismatch at {point.name} replication {index}"
                )

    boundaries = set(_schedule_boundaries(protocol))
    if any(value not in boundaries for value in state.batch_end_counts):
        raise ValueError("checkpoint contains invalid batch boundary")
    if tuple(sorted(set(state.batch_end_counts))) != state.batch_end_counts:
        raise ValueError("checkpoint batch boundaries must be strictly increasing")


def save_parallel_checkpoint(
    state: ParallelCheckpointState,
    checkpoint_directory: str | Path,
    *,
    point_names: tuple[str, ...],
) -> Path:
    """Atomically save trusted internal checkpoint + checksum + readable progress."""

    directory = Path(checkpoint_directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    path = checkpoint_file(directory)
    _atomic_write_bytes(path, data)
    _atomic_write_text(checkpoint_checksum_file(directory), _sha256_bytes(data) + "\n")
    progress = {
        "schema_version": state.schema_version,
        "execution_protocol_fingerprint_sha256": state.execution_protocol_fingerprint_sha256,
        "software_version": state.software_version,
        "batch_end_counts": list(state.batch_end_counts),
        "completed_replications_by_point": dict(_point_counts(state, point_names)),
        "checkpoint_sha256": _sha256_bytes(data),
    }
    _atomic_write_text(
        checkpoint_progress_file(directory),
        json.dumps(progress, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    return path


def load_parallel_checkpoint(
    checkpoint_directory: str | Path,
    protocol: ExecutableReplicationProtocol,
    *,
    points: tuple[DemandSweepPoint, ...],
    point_seed_map: Mapping[str, int],
) -> ParallelCheckpointState:
    """Load and validate a checkpoint created locally by this project."""

    path = checkpoint_file(checkpoint_directory)
    checksum_path = checkpoint_checksum_file(checkpoint_directory)
    if not path.exists() or not checksum_path.exists():
        raise FileNotFoundError("parallel checkpoint or checksum is missing")
    data = path.read_bytes()
    expected = checksum_path.read_text(encoding="utf-8").strip().split()[0]
    actual = _sha256_bytes(data)
    if actual != expected:
        raise ValueError("parallel checkpoint SHA-256 mismatch")
    state = pickle.loads(data)  # trusted local artifact only; see module docstring
    if not isinstance(state, ParallelCheckpointState):
        raise ValueError("parallel checkpoint has unexpected object type")
    _validate_checkpoint_state(state, protocol, points, point_seed_map)
    return state


def _latest_results_from_state(
    state: ParallelCheckpointState,
    points: tuple[DemandSweepPoint, ...],
    point_seed_map: Mapping[str, int],
    count: int,
) -> tuple[DemandSweepPointResult, ...]:
    return tuple(
        _point_result(
            point,
            int(point_seed_map[point.name]),
            [state.replications_by_point[point.name][index] for index in range(count)],
        )
        for point in points
    )


def _progress(
    *,
    state: ParallelCheckpointState,
    point_names: tuple[str, ...],
    next_target: int | None,
    tasks_completed: int,
    checkpoint_directory: str | Path | None,
    result: ProtocolRunResult | None = None,
) -> ParallelExecutionProgress:
    return ParallelExecutionProgress(
        completed=result is not None,
        result=result,
        checkpoint_path=(
            str(checkpoint_file(checkpoint_directory))
            if checkpoint_directory is not None
            else None
        ),
        completed_replications_by_point=_point_counts(state, point_names),
        fully_completed_replications=_fully_completed_count(state, point_names),
        next_target_replications=next_target,
        batch_end_counts=state.batch_end_counts,
        tasks_completed_this_call=tasks_completed,
    )


def execute_replication_protocol_parallel(
    protocol: ExecutableReplicationProtocol,
    *,
    workers: int | None = None,
    checkpoint_directory: str | Path | None = None,
    max_wall_seconds: float | None = None,
    max_task_groups: int | None = None,
    run_timestamp_utc: str | None = None,
) -> ParallelExecutionProgress:
    """Advance or complete the frozen protocol with process-level parallelism.

    Re-running with the same ``checkpoint_directory`` resumes only missing
    seeded replications.  ``max_wall_seconds`` and ``max_task_groups`` are
    *compute chunking controls*, not statistical stopping rules; they never
    alter the frozen replication schedule or precision decision.
    """

    if workers is None:
        workers = max(1, min(4, os.cpu_count() or 1))
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if max_wall_seconds is not None and max_wall_seconds <= 0:
        raise ValueError("max_wall_seconds must be positive")
    if max_task_groups is not None and max_task_groups < 1:
        raise ValueError("max_task_groups must be >= 1")
    if (max_wall_seconds is not None or max_task_groups is not None) and checkpoint_directory is None:
        raise ValueError("chunked execution requires checkpoint_directory")

    study = protocol.study
    points = study.demand_points()
    point_names = tuple(point.name for point in points)
    point_by_name = {point.name: point for point in points}
    point_seed_map = demand_point_seed_map(points, base_seed=study.base_seed)

    if checkpoint_directory is not None and checkpoint_file(checkpoint_directory).exists():
        state = load_parallel_checkpoint(
            checkpoint_directory,
            protocol,
            points=points,
            point_seed_map=point_seed_map,
        )
    else:
        state = _new_checkpoint_state(protocol, points, point_seed_map)
        if checkpoint_directory is not None:
            save_parallel_checkpoint(state, checkpoint_directory, point_names=point_names)

    boundaries = _schedule_boundaries(protocol)
    tasks_completed = 0
    task_groups_completed = 0
    started = time.monotonic()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        while True:
            current_count = _fully_completed_count(state, point_names)

            # A batch boundary may have completed immediately before an external
            # interruption.  Reconstruct and evaluate it before scheduling more.
            if current_count in boundaries and current_count not in state.batch_end_counts:
                state.batch_end_counts = tuple((*state.batch_end_counts, current_count))
                if checkpoint_directory is not None:
                    save_parallel_checkpoint(state, checkpoint_directory, point_names=point_names)
                latest = _latest_results_from_state(
                    state, points, point_seed_map, current_count
                )
                status = precision_status(latest, protocol, replication_count=current_count)
                if status.all_points_meet_target:
                    result = finalize_protocol_run(
                        protocol,
                        latest_results=latest,
                        point_seed_map=point_seed_map,
                        current_count=current_count,
                        batch_end_counts=state.batch_end_counts,
                        stopping_reason=RunStoppingReason.PRECISION_TARGET_MET,
                        run_timestamp_utc=run_timestamp_utc,
                    )
                    return _progress(
                        state=state,
                        point_names=point_names,
                        next_target=None,
                        tasks_completed=tasks_completed,
                        checkpoint_directory=checkpoint_directory,
                        result=result,
                    )
                if current_count >= study.max_replications:
                    result = finalize_protocol_run(
                        protocol,
                        latest_results=latest,
                        point_seed_map=point_seed_map,
                        current_count=current_count,
                        batch_end_counts=state.batch_end_counts,
                        stopping_reason=RunStoppingReason.MAX_REPLICATIONS_REACHED,
                        run_timestamp_utc=run_timestamp_utc,
                    )
                    return _progress(
                        state=state,
                        point_names=point_names,
                        next_target=None,
                        tasks_completed=tasks_completed,
                        checkpoint_directory=checkpoint_directory,
                        result=result,
                    )

            next_target = next((value for value in boundaries if value > current_count), None)
            if next_target is None:
                # Defensive: max boundary should have been finalized above.
                latest = _latest_results_from_state(state, points, point_seed_map, current_count)
                result = finalize_protocol_run(
                    protocol,
                    latest_results=latest,
                    point_seed_map=point_seed_map,
                    current_count=current_count,
                    batch_end_counts=state.batch_end_counts,
                    stopping_reason=RunStoppingReason.MAX_REPLICATIONS_REACHED,
                    run_timestamp_utc=run_timestamp_utc,
                )
                return _progress(
                    state=state,
                    point_names=point_names,
                    next_target=None,
                    tasks_completed=tasks_completed,
                    checkpoint_directory=checkpoint_directory,
                    result=result,
                )

            missing: list[tuple[int, int, str]] = []
            point_order = {name: position for position, name in enumerate(point_names)}
            for point_name in point_names:
                indexed = state.replications_by_point[point_name]
                for index in range(next_target):
                    if index not in indexed:
                        missing.append((index, point_order[point_name], point_name))
            missing.sort()

            if not missing:
                # All tasks for the boundary exist; loop to boundary evaluation.
                continue

            for offset in range(0, len(missing), workers):
                group = missing[offset : offset + workers]
                payloads = [
                    (
                        protocol,
                        point_by_name[point_name],
                        int(point_seed_map[point_name]),
                        index,
                    )
                    for index, _, point_name in group
                ]
                for point_name, replication in pool.map(_worker_replication, payloads):
                    state.replications_by_point[point_name][replication.replication_index] = replication
                    tasks_completed += 1
                task_groups_completed += 1
                if checkpoint_directory is not None:
                    save_parallel_checkpoint(state, checkpoint_directory, point_names=point_names)

                wall_limit_hit = (
                    max_wall_seconds is not None
                    and time.monotonic() - started >= max_wall_seconds
                )
                group_limit_hit = (
                    max_task_groups is not None
                    and task_groups_completed >= max_task_groups
                )
                if wall_limit_hit or group_limit_hit:
                    return _progress(
                        state=state,
                        point_names=point_names,
                        next_target=next_target,
                        tasks_completed=tasks_completed,
                        checkpoint_directory=checkpoint_directory,
                    )

            # The full missing list for this target was processed; loop back to
            # compute contiguous count, evaluate precision, and possibly advance.
