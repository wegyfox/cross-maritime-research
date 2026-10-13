"""Generic causal telemetry adapter for a live/captured FIFO network runtime.

This module is an observation-only companion to ``environmental_network_fifo_causal_diagnostics``.
It exists for callers, such as interaction-active research worlds, that use the canonical
A151 FIFO runtime directly rather than the A152 experiment wrapper.  It records the same
causal evidence classes and never changes simulation state or control decisions.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from hashlib import sha256
import math
from typing import Any, Mapping

from .causal_telemetry import CausalEventKind, CausalTelemetryLedger, CausalTelemetrySnapshot
from .environmental_network_experiment import _canonical_json, _json_value
from .environmental_network_source_fifo_runtime import (
    EnvironmentalNetworkSourceFIFORuntimeStartResult,
    EnvironmentalNetworkSourceFIFOTerminalOutcome,
)
from .environmental_network_source_reservoir import source_queue_metrics
from .pair_context_following import PairContextFollowingSupervisor


def _diagnostic_json_value(value: Any) -> Any:
    """Stable JSON-safe projection for diagnostic hashing, including sentinel infinities.

    The physical runtime may carry +/-inf as internal no-bound/no-limit sentinels.  T1
    portable evidence intentionally forbids non-finite JSON numbers, so diagnostic-only
    hashing represents those sentinels as explicit strings rather than mutating runtime.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        if math.isnan(value):
            return "NaN"
        return "+Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Enum):
        return _diagnostic_json_value(value.value)
    if is_dataclass(value):
        return {field.name: _diagnostic_json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _diagnostic_json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_diagnostic_json_value(v) for v in value]
    method = getattr(value, "to_json_object", None)
    if callable(method):
        return _diagnostic_json_value(method())
    return repr(value)


def _sha_json(value: Any) -> str:
    return sha256(_canonical_json(_diagnostic_json_value(value))).hexdigest()


def _pair_metadata(
    supervisor: PairContextFollowingSupervisor | None,
    follower_id: str,
    leader_id: str,
) -> dict[str, Any]:
    if supervisor is None:
        return {}
    return {"pair_context_response": supervisor.evaluate_pair(follower_id, leader_id).to_json_object()}


def causal_telemetry_from_fifo_runtime(
    started: EnvironmentalNetworkSourceFIFORuntimeStartResult,
    outcome: EnvironmentalNetworkSourceFIFOTerminalOutcome,
    *,
    pair_context_supervisor: PairContextFollowingSupervisor | None = None,
) -> CausalTelemetrySnapshot:
    """Freeze deterministic T1-style evidence from the canonical FIFO runtime handles."""
    if not isinstance(started, EnvironmentalNetworkSourceFIFORuntimeStartResult):
        raise TypeError("started must be EnvironmentalNetworkSourceFIFORuntimeStartResult")
    if not isinstance(outcome, EnvironmentalNetworkSourceFIFOTerminalOutcome):
        raise TypeError("outcome must be EnvironmentalNetworkSourceFIFOTerminalOutcome")
    if pair_context_supervisor is not None and not isinstance(
        pair_context_supervisor, PairContextFollowingSupervisor
    ):
        raise TypeError("pair_context_supervisor must be PairContextFollowingSupervisor or None")

    pending: list[tuple[float, int, str, dict[str, Any]]] = []
    demand_stream_by_vessel_id = {
        demand.vessel_id: demand.stream_id for demand in outcome.reservoir.demands
    }

    def add(time_min: float, priority: int, key: str, **kwargs: Any) -> None:
        pending.append((float(time_min), priority, key, kwargs))

    for demand in outcome.reservoir.demands:
        add(
            demand.demand_time,
            0,
            f"demand:{demand.stream_id}:{demand.vessel_id}",
            event_kind=CausalEventKind.SOURCE_DEMAND,
            source_component="environmental_network_source_reservoir",
            entity_ids=(demand.vessel_id,),
            owner="source",
            reason="exogenous_demand",
            metadata={"stream_id": demand.stream_id, "demand_time_min": demand.demand_time},
        )

    resolutions = [("INITIAL", started.initial_resolution)] + [
        (f"BOUNDARY_{index}", boundary.resolution)
        for index, boundary in enumerate(outcome.source_boundaries)
    ]
    for resolution_id, resolution in resolutions:
        for decision_index, decision in enumerate(resolution.decisions):
            metadata: dict[str, Any] = {
                "resolution_id": resolution_id,
                "decision_index": decision_index,
                "stream_id": decision.stream_id,
                "demand_time_min": decision.demand_time,
                "due_queue_depth": decision.due_queue_depth,
                "status": decision.status.value,
                "waiting_time_min": decision.waiting_time_min,
                "dynamic_launch_speed_ceiling_nm_per_min": decision.dynamic_launch_speed_ceiling_nm_per_min,
                "dynamic_limiting_leader_id": decision.dynamic_limiting_leader_id,
                "clearance_count": len(decision.clearances),
                "future_clearance_count": len(decision.future_clearances),
            }
            if decision.dynamic_limiting_leader_id is not None:
                metadata.update(
                    _pair_metadata(
                        pair_context_supervisor,
                        decision.vessel_id,
                        decision.dynamic_limiting_leader_id,
                    )
                )
                metadata["pair_context_evidence_role"] = "dynamic_limiting_leader"
            elif pair_context_supervisor is not None:
                candidates: list[dict[str, Any]] = []
                for clearance in decision.clearances:
                    if demand_stream_by_vessel_id.get(clearance.other_vessel_id) != decision.stream_id:
                        continue
                    candidates.append(
                        {
                            "other_vessel_id": clearance.other_vessel_id,
                            "actual_distance_nm": clearance.clearance.actual_distance_nm,
                            "required_distance_nm": clearance.clearance.required_distance_nm,
                            "intrudes": clearance.clearance.intrudes,
                            "response": pair_context_supervisor.evaluate_pair(
                                decision.vessel_id, clearance.other_vessel_id
                            ).to_json_object(),
                        }
                    )
                if candidates:
                    metadata["pair_context_candidate_responses"] = candidates
                    metadata["pair_context_evidence_role"] = "same_stream_source_clearance"
            add(
                decision.evaluated_at,
                1,
                f"source-decision:{resolution_id}:{decision_index}:{decision.vessel_id}",
                event_kind=CausalEventKind.SAME_ROUTE_EVALUATION,
                source_component="environmental_network_source_fifo",
                entity_ids=(
                    (decision.vessel_id,)
                    if decision.dynamic_limiting_leader_id is None
                    else (decision.vessel_id, decision.dynamic_limiting_leader_id)
                ),
                owner="source_admission",
                reason=decision.status.value,
                action_sha256=_sha_json(decision),
                metadata=metadata,
            )

    for admission in outcome.reservoir.admissions:
        add(
            admission.admission_time,
            2,
            f"admission:{admission.stream_id}:{admission.vessel_id}",
            event_kind=CausalEventKind.SOURCE_ADMISSION,
            source_component="environmental_network_source_reservoir",
            entity_ids=(admission.vessel_id,),
            owner="source",
            reason="physical_admission",
            action_sha256=_sha_json(admission.record),
            metadata={
                "stream_id": admission.stream_id,
                "demand_time_min": admission.demand_time,
                "admission_time_min": admission.admission_time,
                "waiting_time_min": admission.waiting_time_min,
            },
        )

    runtime = started.runtime_start.session
    for index, observation in enumerate(runtime.coordinator.fleet_observations):
        owners = [
            {
                "component_id": owner.component_id,
                "backend": owner.backend.value,
                "vessel_ids": list(owner.vessel_ids),
                "pair_keys": [list(pair) for pair in owner.pair_keys],
            }
            for owner in observation.dispatch.owners
        ]
        add(
            observation.time_min,
            3,
            f"ownership:{index}",
            event_kind=CausalEventKind.CONTROL_TRANSITION,
            source_component="environmental_fleet_ownership",
            entity_ids=tuple(observation.dispatch.owned_vessel_ids),
            owner="fleet_dispatch",
            reason="canonical_fleet_ownership",
            truth_state_sha256=_sha_json(observation.registry_update),
            action_sha256=_sha_json(observation.dispatch),
            metadata={
                "owner_count": len(owners),
                "pair_owner_count": len(observation.dispatch.pair_owners),
                "joint_owner_count": len(observation.dispatch.joint_owners),
                "owners": owners,
            },
        )

    for index, record in enumerate(runtime.persistent.mixed_session.same_route_supervisions):
        metadata = {
            "owner_id": record.owner_id,
            "max_follower_speed_nm_per_min": record.max_follower_speed_nm_per_min,
            "previous_command": _diagnostic_json_value(record.previous_command),
            "issued_command": _diagnostic_json_value(record.issued_command),
        }
        metadata.update(_pair_metadata(pair_context_supervisor, record.follower_id, record.leader_id))
        add(
            record.time_min,
            4,
            f"same-route-supervision:{index}:{record.follower_id}:{record.leader_id}",
            event_kind=CausalEventKind.CONTROL_DECISION,
            source_component="environmental_mixed_owner_control",
            entity_ids=(record.follower_id, record.leader_id),
            owner=record.owner_id,
            reason="persistent_same_route_speed_ceiling_applied",
            action_sha256=_sha_json(record.issued_command),
            metadata=metadata,
        )

    queue = source_queue_metrics(outcome.reservoir, at=outcome.terminal_time_min)
    snapshot = outcome.snapshot
    add(
        outcome.terminal_time_min,
        9,
        "terminal",
        event_kind=CausalEventKind.TERMINAL,
        source_component="environmental_network_source_fifo_runtime",
        owner="runtime",
        reason=outcome.status.value,
        truth_state_sha256=_sha_json(snapshot),
        metadata={
            "final_vessel_count": len(snapshot.vessels),
            "physical_admission_count": queue.admission_count,
            "pending_source_count": queue.pending_count,
            "due_pending_source_count": queue.due_pending_count,
            "infeasibility_type": outcome.infeasibility_type,
            "resource_unavailability_type": outcome.resource_unavailability_type,
        },
    )

    ledger = CausalTelemetryLedger()
    for time_min, _, _, kwargs in sorted(pending, key=lambda item: (item[0], item[1], item[2])):
        if "metadata" in kwargs:
            kwargs["metadata"] = _diagnostic_json_value(kwargs["metadata"])
        ledger.append(time_min=time_min, **kwargs)
    return ledger.freeze()
