"""Durable same-version checkpoint/resume for the persistent environmental fleet runtime.

A129 serializes the complete A117–A128 object graph rather than reconstructing a second
approximate controller from public snapshots.  This matters because command authority is
stateful below the fleet surface: pair/joint lifecycle hysteresis, route follower cursors,
review calendars, incarnation sequence numbers and same-time event ordering all affect the
next writer.

The checkpoint format therefore has two layers:

* a small JSON header with schema/runtime/Python/cloudpickle compatibility markers, a
  SHA-256 digest of the opaque payload and a deterministic audit of the committed runtime
  boundary;
* a ``cloudpickle`` payload containing the exact Python object graph, including the local
  command-guard/planning callables installed by the existing orchestration layers.

Checkpoint files are trusted local artifacts, not an untrusted interchange format.
Restoring arbitrary checkpoint bytes has the same security model as Python pickle.
A129 deliberately requires the exact simulator version, Python major/minor and cloudpickle
version that produced the artifact.  Cross-version migration must be an explicit future
contract rather than accidental pickle compatibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import struct
import sys
import warnings
from typing import Any

import cloudpickle

from ..version import __version__
from .environmental_fleet_guarded_joint_control import EnvironmentalFleetGuardedJointControlSession
from .environmental_fleet_guarded_pair_control import EnvironmentalFleetGuardedPairControlSession
from .environmental_fleet_multi_owner_recapture import (
    EnvironmentalStartedJointOwner,
    EnvironmentalStartedPairOwner,
)
from .environmental_idle_fleet_control import EnvironmentalPersistentFleetControlSession
from .environmental_joint_control_cycle import EnvironmentalJointControlStage
from .environmental_mixed_owner_control import EnvironmentalMixedRouteStage
from .environmental_pair_control_cycle import EnvironmentalPairControlStage

_EPS = 1e-10
_SCHEMA_VERSION = 1
_MAGIC = b"VCS-A129-CHECKPOINT\x00"
_HEADER_LENGTH = struct.Struct(">Q")


class EnvironmentalFleetRuntimeCheckpointError(RuntimeError):
    """Base error for A129 checkpoint capture/load/restore failures."""


class EnvironmentalFleetRuntimeCheckpointUnsafeBoundary(EnvironmentalFleetRuntimeCheckpointError):
    """The live runtime is between committed boundaries and must not be captured."""


class EnvironmentalFleetRuntimeCheckpointIntegrityError(EnvironmentalFleetRuntimeCheckpointError):
    """Checkpoint bytes were truncated, corrupted or internally inconsistent."""


class EnvironmentalFleetRuntimeCheckpointCompatibilityError(EnvironmentalFleetRuntimeCheckpointError):
    """Checkpoint was produced by an incompatible simulator/runtime stack."""


class EnvironmentalFleetRuntimeCheckpointConsistencyError(EnvironmentalFleetRuntimeCheckpointError):
    """The object graph disagrees with its own committed-world authority state."""


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeCheckpointAudit:
    """Deterministic, non-executable summary of state that must survive resume exactly."""

    world_time: float
    vessel_ids: tuple[str, ...]
    commands: tuple[tuple[str, float, float], ...]
    queued_command_events: tuple[tuple[float, int, str, float, float], ...]
    registry_last_update_time: float | None
    registry_pairs: tuple[tuple[Any, ...], ...]
    fleet_observation_times: tuple[float, ...]
    avoidance_owners: tuple[tuple[Any, ...], ...]
    route_owners: tuple[tuple[Any, ...], ...]
    idle_active: bool
    next_idle_review_time: float | None
    membership_events: tuple[tuple[Any, ...], ...]
    active_incarnations: tuple[tuple[str, int | None], ...]
    pending_incarnations: tuple[tuple[Any, ...], ...]
    next_incarnation_by_vessel: tuple[tuple[str, int], ...]
    route_targets: tuple[tuple[str, str], ...]
    profiles: tuple[tuple[str, str], ...]
    audit_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeCheckpoint:
    """Opaque durable checkpoint plus its independently readable committed-state audit."""

    schema_version: int
    runtime_version: str
    python_version: str
    cloudpickle_version: str
    payload_sha256: str
    audit: EnvironmentalFleetRuntimeCheckpointAudit
    payload: bytes

    def to_bytes(self) -> bytes:
        actual = sha256(self.payload).hexdigest()
        if actual != self.payload_sha256:
            raise EnvironmentalFleetRuntimeCheckpointIntegrityError(
                "checkpoint payload digest no longer matches payload bytes"
            )
        header = {
            "schema_version": self.schema_version,
            "runtime_version": self.runtime_version,
            "python_version": self.python_version,
            "cloudpickle_version": self.cloudpickle_version,
            "payload_sha256": self.payload_sha256,
            "audit": asdict(self.audit),
        }
        header_bytes = json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return b"".join((_MAGIC, _HEADER_LENGTH.pack(len(header_bytes)), header_bytes, self.payload))

    @classmethod
    def from_bytes(cls, data: bytes) -> "EnvironmentalFleetRuntimeCheckpoint":
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("checkpoint data must be bytes-like")
        raw = bytes(data)
        prefix = len(_MAGIC) + _HEADER_LENGTH.size
        if len(raw) < prefix or not raw.startswith(_MAGIC):
            raise EnvironmentalFleetRuntimeCheckpointIntegrityError(
                "checkpoint magic/header is missing or invalid"
            )
        header_len = _HEADER_LENGTH.unpack(raw[len(_MAGIC) : prefix])[0]
        header_end = prefix + header_len
        if header_end > len(raw):
            raise EnvironmentalFleetRuntimeCheckpointIntegrityError("checkpoint header is truncated")
        try:
            header = json.loads(raw[prefix:header_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise EnvironmentalFleetRuntimeCheckpointIntegrityError(
                "checkpoint JSON header is invalid"
            ) from exc
        payload = raw[header_end:]
        required = {
            "schema_version",
            "runtime_version",
            "python_version",
            "cloudpickle_version",
            "payload_sha256",
            "audit",
        }
        if not isinstance(header, dict) or set(header) != required:
            raise EnvironmentalFleetRuntimeCheckpointIntegrityError(
                "checkpoint header fields do not match the A129 schema"
            )
        expected = str(header["payload_sha256"])
        actual = sha256(payload).hexdigest()
        if actual != expected:
            raise EnvironmentalFleetRuntimeCheckpointIntegrityError(
                "checkpoint payload SHA-256 mismatch"
            )
        audit = _audit_from_json_dict(header["audit"])
        return cls(
            schema_version=int(header["schema_version"]),
            runtime_version=str(header["runtime_version"]),
            python_version=str(header["python_version"]),
            cloudpickle_version=str(header["cloudpickle_version"]),
            payload_sha256=expected,
            audit=audit,
            payload=payload,
        )


def _python_compatibility_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _command_tuple(command) -> tuple[float, float]:
    return (
        float(command.target_through_water_heading_deg),
        float(command.target_through_water_speed),
    )


def _stable(value: Any) -> str:
    """Same-version audit representation for immutable model/config evidence."""

    return repr(value)


def _registry_pair_audit(session: EnvironmentalPersistentFleetControlSession) -> tuple[tuple[Any, ...], ...]:
    result: list[tuple[Any, ...]] = []
    for key in sorted(session.registry._tracked):  # noqa: SLF001 - checkpoint authority snapshot
        tracked = session.registry._tracked[key]  # noqa: SLF001
        lifecycle = tracked.lifecycle
        result.append(
            (
                key[0],
                key[1],
                float(tracked.first_seen_min),
                float(tracked.last_seen_min),
                int(tracked.update_count),
                _stable(tracked.entry),
                _stable(lifecycle.observations),
                _stable(lifecycle._last_time),  # noqa: SLF001
                _stable(lifecycle._risk_active),  # noqa: SLF001
                _stable(lifecycle._risk_clear_streak),  # noqa: SLF001
                _stable(lifecycle._risk_scaffold),  # noqa: SLF001
                _stable(lifecycle._last_risk_scaffold),  # noqa: SLF001
                _stable(lifecycle._rule13_scaffold),  # noqa: SLF001
                _stable(lifecycle._rule13_overtaker_id),  # noqa: SLF001
                _stable(lifecycle._rule13_overtaken_id),  # noqa: SLF001
                _stable(lifecycle._rule13_clear_streak),  # noqa: SLF001
            )
        )
    return tuple(result)


def _avoidance_owner_audit(session: EnvironmentalPersistentFleetControlSession) -> tuple[tuple[Any, ...], ...]:
    result: list[tuple[Any, ...]] = []
    for started in session.mixed_session._avoidance_started():  # noqa: SLF001
        guard = started.guarded_session
        if isinstance(started, EnvironmentalStartedPairOwner):
            cycle = started.pair_cycle
            execution = cycle._execution_session  # noqa: SLF001
            result.append(
                (
                    "pair",
                    started.owner.component_id,
                    started.owner.vessel_ids,
                    cycle.stage.value,
                    guard.next_review_time,
                    execution.next_boundary_time(),
                    _stable(cycle._pending_policy),  # noqa: SLF001
                    _stable(cycle.latest_decision),
                    len(cycle.policy_decisions),
                    len(cycle.replans),
                )
            )
        elif isinstance(started, EnvironmentalStartedJointOwner):
            cycle = started.joint_cycle
            execution = cycle._execution_session  # noqa: SLF001
            result.append(
                (
                    "joint",
                    started.owner.component_id,
                    started.owner.vessel_ids,
                    cycle.stage.value,
                    guard.next_review_time,
                    execution.next_boundary_time(),
                    _stable(cycle._pending_policy),  # noqa: SLF001
                    _stable(cycle.latest_pair_decisions),
                    len(cycle.policy_decisions),
                    len(cycle.replans),
                )
            )
        else:  # pragma: no cover - closed runtime owner union
            raise AssertionError(type(started))
    return tuple(sorted(result, key=lambda item: item[1]))


def _route_owner_audit(session: EnvironmentalPersistentFleetControlSession) -> tuple[tuple[Any, ...], ...]:
    result: list[tuple[Any, ...]] = []
    for owner_id, owner in sorted(session.mixed_session._route_owners.items()):  # noqa: SLF001
        result.append(
            (
                owner_id,
                owner.vessel_ids,
                owner.stage.value,
                owner.next_fleet_review_time,
                _stable(owner.binding),
                _stable(owner.handoff_session._cursors),  # noqa: SLF001
                bool(owner.handoff_session._started),  # noqa: SLF001
                _stable(owner.handoff_session._terminal_reason),  # noqa: SLF001
                _stable(None if owner.nominal_session is None else owner.nominal_session._cursors),  # noqa: SLF001
                False if owner.nominal_session is None else bool(owner.nominal_session._started),  # noqa: SLF001
                _stable(None if owner.nominal_session is None else owner.nominal_session._terminal_reason),  # noqa: SLF001
            )
        )
    return tuple(result)


def _membership_event_audit(session: EnvironmentalPersistentFleetControlSession) -> tuple[tuple[Any, ...], ...]:
    result: list[tuple[Any, ...]] = []
    for event in sorted(session.coordinator._membership_events):  # noqa: SLF001
        vessel_id = event.vessel.vessel_id if event.vessel is not None else event.vessel_id
        result.append(
            (
                float(event.at),
                int(event.sequence),
                event.kind.value,
                vessel_id,
                event.incarnation,
                _stable(event.vessel),
                _stable(event.limits),
                _stable(event.drift),
            )
        )
    return tuple(result)


def _pending_incarnation_audit(session: EnvironmentalPersistentFleetControlSession) -> tuple[tuple[Any, ...], ...]:
    result: list[tuple[Any, ...]] = []
    for vessel_id in sorted(session._pending_incarnations):  # noqa: SLF001
        for admission in session._pending_incarnations[vessel_id]:  # noqa: SLF001
            result.append(
                (
                    vessel_id,
                    admission.incarnation,
                    float(admission.ingress_at),
                    admission.egress_at,
                    bool(admission.metadata_activated),
                    _stable(admission.episode.ingress.vessel),
                    _stable(admission.episode.ingress.route_target),
                    _stable(admission.episode.ingress.effective_limits),
                    _stable(admission.episode.ingress.environmental_drift_velocity),
                    _stable(admission.episode.ingress.profile),
                )
            )
    return tuple(result)


def _profile_audit(session: EnvironmentalPersistentFleetControlSession) -> tuple[tuple[str, str], ...]:
    catalog = session.profile_catalog
    if catalog is None:
        return ()
    return tuple((vessel_id, _stable(catalog.profile_for(vessel_id))) for vessel_id in catalog.vessel_ids)


def _audit_runtime(session: EnvironmentalPersistentFleetControlSession) -> EnvironmentalFleetRuntimeCheckpointAudit:
    world = session.world
    vessels = tuple(v.vessel_id for v in world.snapshot().vessels)
    commands = tuple((vessel_id, *_command_tuple(world.command_for(vessel_id))) for vessel_id in vessels)
    queued_commands = tuple(
        (
            float(event.at),
            int(event.sequence),
            event.vessel_id,
            *_command_tuple(event.command),
        )
        for event in sorted(world._events)  # noqa: SLF001
    )
    route_targets = tuple((vessel_id, _stable(session.route_targets[vessel_id])) for vessel_id in sorted(session.route_targets))
    counts = (
        ("idle_boundaries", len(session.idle_boundaries)),
        ("bootstrap_results", len(session.bootstrap_results)),
        ("ingress_admissions", len(session.ingress_admissions)),
        ("egress_admissions", len(session.egress_admissions)),
        ("egress_completions", len(session.egress_completions)),
        ("incarnation_admissions", len(session.incarnation_admissions)),
        ("incarnation_cancellations", len(session.incarnation_cancellations)),
        ("incarnation_amendments", len(session.incarnation_amendments)),
        ("membership_applications", len(session.membership_applications)),
        ("rebootstrap_results", len(session.persistent.rebootstrap_results)),
        ("disposition_results", len(session.persistent.disposition_results)),
    )
    return EnvironmentalFleetRuntimeCheckpointAudit(
        world_time=float(world.time),
        vessel_ids=vessels,
        commands=commands,
        queued_command_events=queued_commands,
        registry_last_update_time=session.registry._last_update_time,  # noqa: SLF001
        registry_pairs=_registry_pair_audit(session),
        fleet_observation_times=tuple(float(item.time_min) for item in session.coordinator.fleet_observations),
        avoidance_owners=_avoidance_owner_audit(session),
        route_owners=_route_owner_audit(session),
        idle_active=bool(session._idle_active),  # noqa: SLF001
        next_idle_review_time=session._next_idle_review_time,  # noqa: SLF001
        membership_events=_membership_event_audit(session),
        active_incarnations=tuple(sorted(session.coordinator._active_incarnations.items())),  # noqa: SLF001
        pending_incarnations=_pending_incarnation_audit(session),
        next_incarnation_by_vessel=tuple(sorted(session._next_incarnation_by_vessel.items())),  # noqa: SLF001
        route_targets=route_targets,
        profiles=_profile_audit(session),
        audit_counts=counts,
    )


def _validate_runtime_consistency(session: EnvironmentalPersistentFleetControlSession) -> None:
    if not isinstance(session, EnvironmentalPersistentFleetControlSession):
        raise TypeError("session must be EnvironmentalPersistentFleetControlSession")
    if not isfinite(session.world.time) or session.world.time < 0.0:
        raise EnvironmentalFleetRuntimeCheckpointConsistencyError("world time is invalid")

    # The persistent stack must be one object graph, not several clocks/registries/catalogues.
    if not (
        session.world is session.persistent.world
        and session.world is session.mixed_session.world
        and session.world is session.coordinator.world
    ):
        raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
            "persistent runtime does not share exactly one EnvironmentalWorld2D"
        )
    if not (
        session.registry is session.coordinator.registry
        and session.registry is session.mixed_session.registry
    ):
        raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
            "persistent runtime does not share exactly one EncounterRegistry"
        )
    if not (
        session.route_targets is session.persistent.route_targets
        and session.route_targets is session.mixed_session.route_targets
    ):
        raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
            "runtime route-target catalogues are not one shared authority"
        )
    if session.profile_catalog is not None and not (
        session.profile_catalog is session.persistent.profile_catalog
        and session.profile_catalog is session.mixed_session.profile_catalog
    ):
        raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
            "runtime profile catalogues are not one shared authority"
        )

    active_ids = {item.vessel_id for item in session.world.snapshot().vessels}
    if set(session.world._commands) != active_ids:  # noqa: SLF001
        raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
            "active command channels do not equal active physical vessel ids"
        )
    for event in session.world._events:  # noqa: SLF001
        if event.at < session.world.time - _EPS or event.vessel_id not in active_ids:
            raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
                "queued command event refers to past time or inactive vessel"
            )
    pending_command = session.world.next_command_time()
    if pending_command is not None and pending_command <= session.world.time + _EPS:
        raise EnvironmentalFleetRuntimeCheckpointUnsafeBoundary(
            "checkpoint requires all current-time command events to be consumed"
        )

    last_update = session.registry._last_update_time  # noqa: SLF001
    if last_update is not None and last_update > session.world.time + _EPS:
        raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
            "registry update time lies ahead of the physical world"
        )
    if session.coordinator.fleet_observations:
        observation_times = [item.time_min for item in session.coordinator.fleet_observations]
        if any(b <= a + _EPS for a, b in zip(observation_times, observation_times[1:])):
            raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
                "canonical fleet observation history is not strictly increasing"
            )
        if observation_times[-1] > session.world.time + _EPS:
            raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
                "latest fleet observation lies ahead of world time"
            )
        if last_update is None or abs(last_update - observation_times[-1]) > _EPS:
            raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
                "registry last-update time disagrees with latest canonical fleet observation"
            )

    session.mixed_session._validate_disjoint_channels()  # noqa: SLF001
    avoidance_ids: set[str] = set()
    for started in session.mixed_session._avoidance_started():  # noqa: SLF001
        if set(started.owner.vessel_ids) - active_ids:
            raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
                f"avoidance owner {started.owner.component_id!r} contains inactive vessel"
            )
        guard = started.guarded_session
        if guard.world is not session.world or guard.registry is not session.registry:
            raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
                "avoidance guard is detached from shared world/registry"
            )
        avoidance_ids.update(started.owner.vessel_ids)
        if isinstance(started, EnvironmentalStartedPairOwner):
            if started.pair_cycle.stage is not EnvironmentalPairControlStage.AVOIDANCE:
                raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
                    "mixed avoidance set contains non-avoidance PAIR stage"
                )
            if not isinstance(guard, EnvironmentalFleetGuardedPairControlSession):
                raise EnvironmentalFleetRuntimeCheckpointConsistencyError("PAIR owner has wrong guard type")
        else:
            if started.joint_cycle.stage is not EnvironmentalJointControlStage.AVOIDANCE:
                raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
                    "mixed avoidance set contains non-avoidance JOINT stage"
                )
            if not isinstance(guard, EnvironmentalFleetGuardedJointControlSession):
                raise EnvironmentalFleetRuntimeCheckpointConsistencyError("JOINT owner has wrong guard type")

    route_ids: set[str] = set()
    for owner in session.mixed_session._route_owners.values():  # noqa: SLF001
        if owner.stage is EnvironmentalMixedRouteStage.COMPLETE:
            continue
        if set(owner.vessel_ids) - active_ids:
            raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
                f"route owner {owner.owner_id!r} contains inactive vessel"
            )
        route_ids.update(owner.vessel_ids)
    if avoidance_ids.intersection(route_ids):
        raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
            "avoidance and route ownership overlap command channels"
        )

    if session.mixed_session._current_write_observation is not None:  # noqa: SLF001
        raise EnvironmentalFleetRuntimeCheckpointUnsafeBoundary(
            "checkpoint cannot be taken inside a transient route command-write boundary"
        )

    next_membership = session.coordinator._next_membership_time()  # noqa: SLF001
    if next_membership is not None and next_membership <= session.world.time + _EPS:
        raise EnvironmentalFleetRuntimeCheckpointUnsafeBoundary(
            "checkpoint requires all current-time membership events to be consumed"
        )
    sequences = [event.sequence for event in session.coordinator._membership_events]  # noqa: SLF001
    if len(sequences) != len(set(sequences)):
        raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
            "membership event sequence numbers are not unique"
        )
    for event in session.coordinator._membership_events:  # noqa: SLF001
        if event.at < session.world.time - _EPS:
            raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
                "membership queue contains a past event"
            )

    if set(session.coordinator._active_incarnations) != active_ids:  # noqa: SLF001
        raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
            "active incarnation ledger does not match active physical vessel ids"
        )
    for vessel_id, pending in session._pending_incarnations.items():  # noqa: SLF001
        previous_end = None
        active_incarnation = session.coordinator._active_incarnations.get(vessel_id)  # noqa: SLF001
        seen_active = False
        for index, admission in enumerate(pending):
            if admission.vessel_id != vessel_id:
                raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
                    "pending incarnation queue contains a mismatched vessel id"
                )
            arrived = admission.ingress_at <= session.world.time + _EPS
            if arrived:
                if (
                    index != 0
                    or seen_active
                    or active_incarnation != admission.incarnation
                    or admission.egress_at is None
                    or admission.egress_at <= session.world.time + _EPS
                ):
                    raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
                        "incarnation queue contains an arrived generation that is not the current active generation"
                    )
                seen_active = True
            elif previous_end is not None and admission.ingress_at < previous_end - _EPS:
                raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
                    "pending incarnation generations overlap"
                )
            if previous_end is not None and admission.ingress_at < previous_end - _EPS:
                raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
                    "pending incarnation generations overlap"
                )
            previous_end = admission.egress_at

    if session._processed_membership_application_count > len(session.membership_applications):  # noqa: SLF001
        raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
            "processed membership evidence count exceeds committed history"
        )
    if session._processed_disposition_count > len(session.persistent.disposition_results):  # noqa: SLF001
        raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
            "processed route disposition count exceeds committed history"
        )
    if session._idle_active and (avoidance_ids or route_ids):  # noqa: SLF001
        raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
            "idle runtime still has active command owners"
        )
    if (
        session._idle_active  # noqa: SLF001
        and session._next_idle_review_time is not None  # noqa: SLF001
        and session._next_idle_review_time <= session.world.time + _EPS  # noqa: SLF001
    ):
        raise EnvironmentalFleetRuntimeCheckpointUnsafeBoundary(
            "checkpoint cannot leave an idle review due at the current timestamp"
        )


def capture_environmental_fleet_runtime_checkpoint(
    session: EnvironmentalPersistentFleetControlSession,
) -> EnvironmentalFleetRuntimeCheckpoint:
    """Capture one quiescent committed runtime boundary into a durable opaque payload."""

    _validate_runtime_consistency(session)
    audit = _audit_runtime(session)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Pickle, copy, and deepcopy support will be removed from itertools.*",
                category=DeprecationWarning,
            )
            payload = cloudpickle.dumps(session, protocol=5)
    except Exception as exc:  # pragma: no cover - cloudpickle failure is environment-specific
        raise EnvironmentalFleetRuntimeCheckpointError("failed to serialize persistent runtime") from exc
    return EnvironmentalFleetRuntimeCheckpoint(
        schema_version=_SCHEMA_VERSION,
        runtime_version=__version__,
        python_version=_python_compatibility_version(),
        cloudpickle_version=cloudpickle.__version__,
        payload_sha256=sha256(payload).hexdigest(),
        audit=audit,
        payload=payload,
    )


def _validate_checkpoint_compatibility(checkpoint: EnvironmentalFleetRuntimeCheckpoint) -> None:
    if checkpoint.schema_version != _SCHEMA_VERSION:
        raise EnvironmentalFleetRuntimeCheckpointCompatibilityError(
            f"unsupported checkpoint schema {checkpoint.schema_version}; expected {_SCHEMA_VERSION}"
        )
    if checkpoint.runtime_version != __version__:
        raise EnvironmentalFleetRuntimeCheckpointCompatibilityError(
            f"checkpoint runtime version {checkpoint.runtime_version!r} != current {__version__!r}"
        )
    current_python = _python_compatibility_version()
    if checkpoint.python_version != current_python:
        raise EnvironmentalFleetRuntimeCheckpointCompatibilityError(
            f"checkpoint Python {checkpoint.python_version!r} != current {current_python!r}"
        )
    if checkpoint.cloudpickle_version != cloudpickle.__version__:
        raise EnvironmentalFleetRuntimeCheckpointCompatibilityError(
            "checkpoint cloudpickle version differs from current runtime"
        )
    if sha256(checkpoint.payload).hexdigest() != checkpoint.payload_sha256:
        raise EnvironmentalFleetRuntimeCheckpointIntegrityError("checkpoint payload SHA-256 mismatch")


def restore_environmental_fleet_runtime_checkpoint(
    checkpoint: EnvironmentalFleetRuntimeCheckpoint,
) -> EnvironmentalPersistentFleetControlSession:
    """Restore an exact same-version runtime and re-prove its committed-state audit."""

    if not isinstance(checkpoint, EnvironmentalFleetRuntimeCheckpoint):
        raise TypeError("checkpoint must be EnvironmentalFleetRuntimeCheckpoint")
    _validate_checkpoint_compatibility(checkpoint)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Pickle, copy, and deepcopy support will be removed from itertools.*",
                category=DeprecationWarning,
            )
            restored = cloudpickle.loads(checkpoint.payload)
    except Exception as exc:
        raise EnvironmentalFleetRuntimeCheckpointIntegrityError(
            "checkpoint payload cannot be deserialized"
        ) from exc
    if not isinstance(restored, EnvironmentalPersistentFleetControlSession):
        raise EnvironmentalFleetRuntimeCheckpointIntegrityError(
            "checkpoint payload does not contain an EnvironmentalPersistentFleetControlSession"
        )
    _validate_runtime_consistency(restored)
    audit = _audit_runtime(restored)
    if audit != checkpoint.audit:
        raise EnvironmentalFleetRuntimeCheckpointConsistencyError(
            "restored runtime does not reproduce the captured committed-state audit"
        )
    return restored


def write_environmental_fleet_runtime_checkpoint(
    path: str | Path,
    session: EnvironmentalPersistentFleetControlSession,
) -> EnvironmentalFleetRuntimeCheckpoint:
    """Capture and atomically replace one local checkpoint file."""

    checkpoint = capture_environmental_fleet_runtime_checkpoint(session)
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(checkpoint.to_bytes())
    temporary.replace(destination)
    return checkpoint


def read_environmental_fleet_runtime_checkpoint(
    path: str | Path,
) -> EnvironmentalFleetRuntimeCheckpoint:
    """Read and verify the envelope/payload digest without executing the payload."""

    return EnvironmentalFleetRuntimeCheckpoint.from_bytes(Path(path).read_bytes())


def resume_environmental_fleet_runtime_checkpoint(
    path: str | Path,
) -> EnvironmentalPersistentFleetControlSession:
    """Read, compatibility-check, deserialize and audit one trusted checkpoint file."""

    return restore_environmental_fleet_runtime_checkpoint(
        read_environmental_fleet_runtime_checkpoint(path)
    )


def _audit_from_json_dict(value: Any) -> EnvironmentalFleetRuntimeCheckpointAudit:
    if not isinstance(value, dict):
        raise EnvironmentalFleetRuntimeCheckpointIntegrityError("checkpoint audit must be a JSON object")
    required = {field for field in EnvironmentalFleetRuntimeCheckpointAudit.__dataclass_fields__}
    if set(value) != required:
        raise EnvironmentalFleetRuntimeCheckpointIntegrityError(
            "checkpoint audit fields do not match the A129 schema"
        )

    def tuples(name: str) -> tuple:
        raw = value[name]
        if not isinstance(raw, list):
            raise EnvironmentalFleetRuntimeCheckpointIntegrityError(
                f"checkpoint audit field {name!r} must be an array"
            )
        return tuple(_deep_tuple(item) for item in raw)

    try:
        return EnvironmentalFleetRuntimeCheckpointAudit(
            world_time=float(value["world_time"]),
            vessel_ids=tuple(str(item) for item in value["vessel_ids"]),
            commands=tuples("commands"),
            queued_command_events=tuples("queued_command_events"),
            registry_last_update_time=(
                None if value["registry_last_update_time"] is None else float(value["registry_last_update_time"])
            ),
            registry_pairs=tuples("registry_pairs"),
            fleet_observation_times=tuple(float(item) for item in value["fleet_observation_times"]),
            avoidance_owners=tuples("avoidance_owners"),
            route_owners=tuples("route_owners"),
            idle_active=bool(value["idle_active"]),
            next_idle_review_time=(
                None if value["next_idle_review_time"] is None else float(value["next_idle_review_time"])
            ),
            membership_events=tuples("membership_events"),
            active_incarnations=tuples("active_incarnations"),
            pending_incarnations=tuples("pending_incarnations"),
            next_incarnation_by_vessel=tuples("next_incarnation_by_vessel"),
            route_targets=tuples("route_targets"),
            profiles=tuples("profiles"),
            audit_counts=tuples("audit_counts"),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise EnvironmentalFleetRuntimeCheckpointIntegrityError("checkpoint audit values are invalid") from exc


def _deep_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_deep_tuple(item) for item in value)
    return value
