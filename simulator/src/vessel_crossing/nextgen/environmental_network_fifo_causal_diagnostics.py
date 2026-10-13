"""Read-only causal diagnostics derived from existing FIFO/full-network audit evidence.

No simulation callback or command path is modified here.  The adapter converts evidence
that the runtime already freezes (source decisions, admissions, fleet ownership and
persistent same-route supervision) into the generic causal telemetry contract.  An
identity-bearing pair-context supervisor may be supplied only to annotate already
recorded follower/leader pairs with their accepted response decomposition.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import Any

from .causal_telemetry import (
    CausalEventKind,
    CausalTelemetryLedger,
    CausalTelemetrySnapshot,
)
from .environmental_network_experiment import _canonical_json, _json_value
from .environmental_network_fifo_experiment import EnvironmentalNetworkFIFOExperimentResult
from .pair_context_following import PairContextFollowingSupervisor


def _sha_json(value: Any) -> str:
    return sha256(_canonical_json(_json_value(value))).hexdigest()


def _pair_metadata(
    supervisor: PairContextFollowingSupervisor | None,
    follower_id: str,
    leader_id: str,
) -> dict[str, Any]:
    if supervisor is None:
        return {}
    value = supervisor.evaluate_pair(follower_id, leader_id)
    return {"pair_context_response": value.to_json_object()}


def _all_source_resolutions(result: EnvironmentalNetworkFIFOExperimentResult):
    yield "INITIAL", result.runtime_start.initial_resolution
    for index, boundary in enumerate(result.run.source_boundaries):
        yield f"BOUNDARY_{index}", boundary.resolution


def causal_telemetry_from_fifo_result(
    result: EnvironmentalNetworkFIFOExperimentResult,
    *,
    pair_context_supervisor: PairContextFollowingSupervisor | None = None,
) -> CausalTelemetrySnapshot:
    """Build one deterministic causal stream without changing runtime semantics."""

    if not isinstance(result, EnvironmentalNetworkFIFOExperimentResult):
        raise TypeError("result must be EnvironmentalNetworkFIFOExperimentResult")
    if pair_context_supervisor is not None and not isinstance(
        pair_context_supervisor, PairContextFollowingSupervisor
    ):
        raise TypeError("pair_context_supervisor must be PairContextFollowingSupervisor or None")

    pending: list[tuple[float, int, str, dict[str, Any]]] = []
    demand_stream_by_vessel_id = {
        demand.vessel_id: demand.stream_id for demand in result.run.reservoir.demands
    }

    def add(time_min: float, priority: int, key: str, **kwargs: Any) -> None:
        pending.append((float(time_min), priority, key, kwargs))

    # Exogenous demand exists independently of arm-local control.
    for demand in result.run.reservoir.demands:
        add(
            demand.demand_time,
            0,
            f"demand:{demand.stream_id}:{demand.vessel_id}",
            event_kind=CausalEventKind.SOURCE_DEMAND,
            source_component="environmental_network_source_reservoir",
            entity_ids=(demand.vessel_id,),
            owner="source",
            reason="exogenous_demand",
            metadata={
                "stream_id": demand.stream_id,
                "demand_time_min": demand.demand_time,
            },
        )

    # Every source-head decision is already immutable runtime evidence.  Keep repeated
    # evaluations: they explain queueing and the first moment an arm admits or blocks.
    for resolution_id, resolution in _all_source_resolutions(result):
        for decision_index, decision in enumerate(resolution.decisions):
            metadata: dict[str, Any] = {
                "resolution_id": resolution_id,
                "decision_index": decision_index,
                "stream_id": decision.stream_id,
                "demand_time_min": decision.demand_time,
                "due_queue_depth": decision.due_queue_depth,
                "status": decision.status.value,
                "waiting_time_min": decision.waiting_time_min,
                "dynamic_launch_speed_ceiling_nm_per_min": (
                    decision.dynamic_launch_speed_ceiling_nm_per_min
                ),
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
                # A source-head clearance against a vessel from the same traffic stream
                # is useful *exposure* evidence even when the pair-context mechanism was
                # not binding.  Keep it separate from an actual intervention so a null
                # full-network effect can later be classified as exposed-but-nonbinding
                # instead of incorrectly labelled NO_TREATMENT_EXPOSURE.
                candidate_responses: list[dict[str, Any]] = []
                for clearance in decision.clearances:
                    if (
                        demand_stream_by_vessel_id.get(clearance.other_vessel_id)
                        != decision.stream_id
                    ):
                        continue
                    candidate_responses.append(
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
                if candidate_responses:
                    metadata["pair_context_candidate_responses"] = candidate_responses
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

    for admission in result.run.reservoir.admissions:
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

    # Canonical encounter ownership is the generic competing-constraint marker.
    observations = tuple(
        result.runtime_start.runtime_start.session.coordinator.fleet_observations
    )
    for index, observation in enumerate(observations):
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

    # Persistent route supervision already records only interventions that actually
    # lowered a commanded speed.  This is stronger than mere mechanism eligibility.
    mixed = result.runtime_start.session.runtime_session.mixed_session
    for index, record in enumerate(mixed.same_route_supervisions):
        metadata = {
            "owner_id": record.owner_id,
            "max_follower_speed_nm_per_min": record.max_follower_speed_nm_per_min,
            "previous_command": _json_value(record.previous_command),
            "issued_command": _json_value(record.issued_command),
        }
        metadata.update(
            _pair_metadata(
                pair_context_supervisor,
                record.follower_id,
                record.leader_id,
            )
        )
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

    add(
        result.summary.final_time,
        9,
        "terminal",
        event_kind=CausalEventKind.TERMINAL,
        source_component="environmental_network_fifo_experiment",
        owner="runtime",
        reason=result.summary.stop_reason,
        truth_state_sha256=result.summary.final_snapshot_sha256,
        metadata={
            "final_vessel_count": result.summary.final_vessel_count,
            "physical_admission_count": result.summary.physical_admission_count,
            "pending_source_count": result.summary.pending_source_count,
            "due_pending_source_count": result.summary.due_pending_source_count,
        },
    )

    ledger = CausalTelemetryLedger()
    for time_min, _, _, kwargs in sorted(pending, key=lambda item: (item[0], item[1], item[2])):
        ledger.append(time_min=time_min, **kwargs)
    return ledger.freeze()


def causal_telemetry_summary(snapshot: CausalTelemetrySnapshot) -> dict[str, Any]:
    """Compact cross-research accounting over one frozen causal stream."""

    if not isinstance(snapshot, CausalTelemetrySnapshot):
        raise TypeError("snapshot must be CausalTelemetrySnapshot")
    by_kind = Counter(event.event_kind.value for event in snapshot.events)
    by_owner = Counter(event.owner for event in snapshot.events if event.owner is not None)
    pair_evaluations = 0
    pair_context_positive = 0
    typed_response_positive = 0
    persistent_speed_interventions = 0
    persistent_pair_context_positive_interventions = 0
    persistent_typed_response_positive_interventions = 0
    for event in snapshot.events:
        responses: list[dict[str, Any]] = []
        response = event.metadata.get("pair_context_response")
        if isinstance(response, dict):
            responses.append(response)
        candidates = event.metadata.get("pair_context_candidate_responses")
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, dict) and isinstance(candidate.get("response"), dict):
                    responses.append(candidate["response"])
        for response_item in responses:
            pair_evaluations += 1
            pair_extra = float(response_item.get("pair_context_extra_min", 0.0))
            total = float(response_item.get("total_reaction_interval_min", 0.0))
            base = float(response_item.get("base_reaction_interval_min", 0.0))
            if pair_extra > 0.0:
                pair_context_positive += 1
            if total > base:
                typed_response_positive += 1
        if event.reason == "persistent_same_route_speed_ceiling_applied":
            persistent_speed_interventions += 1
            direct = event.metadata.get("pair_context_response")
            if isinstance(direct, dict):
                pair_extra = float(direct.get("pair_context_extra_min", 0.0))
                total = float(direct.get("total_reaction_interval_min", 0.0))
                base = float(direct.get("base_reaction_interval_min", 0.0))
                if pair_extra > 0.0:
                    persistent_pair_context_positive_interventions += 1
                if total > base:
                    persistent_typed_response_positive_interventions += 1
    return {
        "event_count": len(snapshot.events),
        "event_kind_counts": dict(sorted(by_kind.items())),
        "owner_counts": dict(sorted(by_owner.items())),
        "pair_response_evaluation_count": pair_evaluations,
        "pair_context_positive_evaluation_count": pair_context_positive,
        "typed_response_positive_evaluation_count": typed_response_positive,
        "persistent_same_route_speed_intervention_count": persistent_speed_interventions,
        "persistent_pair_context_positive_intervention_count": persistent_pair_context_positive_interventions,
        "persistent_typed_response_positive_intervention_count": persistent_typed_response_positive_interventions,
        "telemetry_sha256": snapshot.result_sha256,
    }
