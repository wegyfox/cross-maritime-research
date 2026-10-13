"""R1-P4B engineering-only structural-stress preflight.

This module deliberately does *not* define or execute the publication-scale R1 protocol.
It preserves the deterministic ``d12-room1.2 / rep03`` recovery case used to exercise
CROSS's two-stream crossing, FIFO admission, PAIR/JOINT ownership, Rule-13 persistence,
and route-recovery plumbing before a later research protocol is frozen.

Any candidate-grid expansion exposed here is diagnostic only.  A value that makes this
one seed complete is not evidence that the setting is scientifically justified; it must
be independently frozen before a research campaign can use it.
"""
from __future__ import annotations

from dataclasses import replace
from math import ceil
from typing import Any

from vessel_crossing.nextgen import (
    EncounterRegistry,
    EnvironmentalNetworkFIFOExperimentConfig,
    EnvironmentalNetworkScenarioConfig,
    EnvironmentalNetworkSourceReservoir,
    EnvironmentalWorld2D,
    PoissonNetworkStreamSpec,
    Route2D,
    RouteEndAction,
    RouteNetwork,
    RouteNetworkEdge,
    RouteNetworkNode,
    RouteRequest,
    ShipDomain,
    VesselClassProfile,
    VesselDimensions,
    VesselProfileCatalog,
    VesselSpeedEnvelope,
    generate_poisson_network_traffic,
    start_environmental_network_source_fifo_runtime,
)
from vessel_crossing.nextgen.candidate_generation import IntentDrivenCandidatePolicy
from vessel_crossing.nextgen.encounter_classification import EncounterClassificationConfig
from vessel_crossing.nextgen.encounter_registry import EncounterDiscoveryConfig
from vessel_crossing.nextgen.environmental_network_scenario import _environmental_ingress_records
from vessel_crossing.nextgen.environmental_network_fifo_experiment import EnvironmentalNetworkFIFOAdmissionPolicy
from vessel_crossing.nextgen.environmental_network_paired_experiment import _MetricEnvironmentalWorld
from vessel_crossing.nextgen.lifecycle import AmbiguousCrossingPolicy, EncounterLifecycleConfig
from vessel_crossing.nextgen.maneuver_intent import Rule13LivenessPolicy
from vessel_crossing.nextgen.multi_intent import MultiIntentResolutionPolicy
from vessel_crossing.nextgen.observation import BearingTrendConfig
from vessel_crossing.nextgen.risk import RiskGateConfig
from vessel_crossing.nextgen.safety_metrics import PairSafetyMetricConfig
from vessel_crossing.nextgen.mission_safety import summarize_mission_safety
from vessel_crossing.nextgen.environmental_network_source_reservoir import source_queue_metrics
from vessel_crossing.nextgen.same_route_following import SameRouteFollowingSupervisor

from .r1_headway_screening import _limits, _recapture_config

P4B_RECOVERY_LABEL = "d12-room1.2/rep03"
P4B_SHARED_REPLICATION_SEED = 5063951187418340982
P4B_REPLICATION_INDEX = 3
P4B_TOTAL_DEMAND_PER_HOUR = 12.0
P4B_PER_STREAM_DEMAND_PER_HOUR = 6.0
P4B_ARM_SEA_ROOM_NM = 1.2
P4B_CRUISE_SPEED_NM_PER_MIN = 0.1
P4B_TRAFFIC_HORIZON_MIN = 90.0
P4B_END_TIME_MIN = 150.0
# Reuse the already-established generic A104 bounded Cartesian contract rather than
# choosing a post-hoc ceiling from this one recovery seed.  P4A intentionally used
# 100 because its action space was tiny; that value is not a generic JOINT limit.
P4B_ENGINEERING_JOINT_MAX_COMBINATIONS = 10_000


def _horizon_safe_replan_cap(*, duration_min: float, review_interval_min: float) -> int:
    """Execution guard that cannot bind a positive-time review loop within the horizon."""

    return int(ceil(duration_min / review_interval_min)) + 1

def _policy_with_progress_expansion(
    policy: IntentDrivenCandidatePolicy,
    *,
    speed_fractions: tuple[float, ...],
    course_offsets_from_counterpart_deg: tuple[float, ...],
    include_speed_envelope_ceiling: bool = False,
) -> IntentDrivenCandidatePolicy:
    return replace(
        policy,
        rule13_progress_speed_fractions=tuple(speed_fractions),
        rule13_progress_course_offsets_from_counterpart_deg=tuple(
            course_offsets_from_counterpart_deg
        ),
        rule13_progress_include_speed_envelope_ceiling=include_speed_envelope_ceiling,
    )


def build_preflight_config(
    *,
    rule13_progress_speed_fractions: tuple[float, ...] = (),
    rule13_progress_course_offsets_from_counterpart_deg: tuple[float, ...] = (),
    rule13_progress_include_speed_envelope_ceiling: bool = False,
    maximum_speed_nm_per_min: float | None = None,
    joint_max_combinations: int = 100,
    multi_intent_resolution: MultiIntentResolutionPolicy = MultiIntentResolutionPolicy.STRICT,
    horizon_safe_replan_caps: bool = False,
    dynamic_same_route_ingress: bool = False,
) -> EnvironmentalNetworkFIFOExperimentConfig:
    """Build the deterministic engineering case with recovered generic safeguards.

    All optional arguments are engineering diagnostics and preserve the original
    recovery case when omitted. Rule-13 expansion, a class speed ceiling, JOINT resource
    budget, and multi-intent conflict policy must be independently justified/frozen
    before any publication-scale R1 protocol uses them.
    """

    room = P4B_ARM_SEA_ROOM_NM
    limits = _limits()
    profile = VesselClassProfile(
        class_id="r1-p4b-homogeneous-stress",
        dimensions=VesselDimensions.from_meters(length_m=120.0, beam_m=20.0),
        maneuver_limits=limits,
        safety_domain=ShipDomain.circular(0.10),
        speed_envelope=(
            None
            if maximum_speed_nm_per_min is None
            else VesselSpeedEnvelope(maximum_speed_nm_per_min)
        ),
    )
    network = RouteNetwork(
        (
            RouteNetworkNode("S", (0.0, -room)),
            RouteNetworkNode("N", (0.0, room)),
            RouteNetworkNode("E", (room, 0.0)),
            RouteNetworkNode("W", (-room, 0.0)),
        ),
        (
            RouteNetworkEdge("sn", "S", "N", Route2D(((0.0, -room), (0.0, room)))),
            RouteNetworkEdge("ew", "E", "W", Route2D(((room, 0.0), (-room, 0.0)))),
        ),
    )
    streams = (
        PoissonNetworkStreamSpec(
            "north",
            RouteRequest("S", "N"),
            P4B_PER_STREAM_DEMAND_PER_HOUR / 60.0,
            P4B_CRUISE_SPEED_NM_PER_MIN,
            end_action=RouteEndAction.REMOVE,
            limits=limits,
            vessel_class=profile,
        ),
        PoissonNetworkStreamSpec(
            "west",
            RouteRequest("E", "W"),
            P4B_PER_STREAM_DEMAND_PER_HOUR / 60.0,
            P4B_CRUISE_SPEED_NM_PER_MIN,
            end_action=RouteEndAction.REMOVE,
            limits=limits,
            vessel_class=profile,
        ),
    )

    recovered = _recapture_config()
    duration = P4B_END_TIME_MIN
    pair_cycle = recovered.pair_cycle
    joint_cycle = recovered.joint_cycle
    if horizon_safe_replan_caps:
        pair_cycle = replace(
            pair_cycle,
            max_replans=_horizon_safe_replan_cap(
                duration_min=duration,
                review_interval_min=pair_cycle.execution.review_interval_min,
            ),
        )
        joint_cycle = replace(
            joint_cycle,
            max_replans=_horizon_safe_replan_cap(
                duration_min=duration,
                review_interval_min=joint_cycle.execution.review_interval_min,
            ),
        )
    pair_replanning = replace(
        pair_cycle.replanning,
        candidate_policy=_policy_with_progress_expansion(
            pair_cycle.replanning.candidate_policy,
            speed_fractions=rule13_progress_speed_fractions,
            course_offsets_from_counterpart_deg=rule13_progress_course_offsets_from_counterpart_deg,
            include_speed_envelope_ceiling=rule13_progress_include_speed_envelope_ceiling,
        ),
        rule13_liveness_policy=Rule13LivenessPolicy.KEEP_CLEAR_WHEN_ALREADY_CLEAR,
    )
    joint_replanning = replace(
        joint_cycle.replanning,
        semantic_config=replace(
            joint_cycle.replanning.semantic_config,
            candidate_policy=_policy_with_progress_expansion(
                joint_cycle.replanning.semantic_config.candidate_policy,
                speed_fractions=rule13_progress_speed_fractions,
                course_offsets_from_counterpart_deg=rule13_progress_course_offsets_from_counterpart_deg,
                include_speed_envelope_ceiling=rule13_progress_include_speed_envelope_ceiling,
            ),
            physics=replace(
                joint_cycle.replanning.semantic_config.physics,
                max_combinations=joint_max_combinations,
            ),
        ),
        multi_intent_resolution=multi_intent_resolution,
        rule13_liveness_policy=Rule13LivenessPolicy.KEEP_CLEAR_WHEN_ALREADY_CLEAR,
    )
    recapture = replace(
        recovered,
        pair_cycle=replace(pair_cycle, replanning=pair_replanning),
        joint_cycle=replace(joint_cycle, replanning=joint_replanning),
        same_route_following=SameRouteFollowingSupervisor(),
    )

    lifecycle = EncounterLifecycleConfig(
        risk_gate=RiskGateConfig(0.5, 30.0),
        bearing_trend=BearingTrendConfig(
            min_duration_min=2.0,
            appreciable_change_deg=3.0,
        ),
        clear_observations_required=2,
        overtaking_clear_distance_nm=0.5,
        max_observations=8,
        ambiguous_crossing_policy=AmbiguousCrossingPolicy.MUTUAL_KEEP_CLEAR,
    )
    scenario = EnvironmentalNetworkScenarioConfig(
        seed=P4B_SHARED_REPLICATION_SEED,
        traffic_horizon_min=P4B_TRAFFIC_HORIZON_MIN,
        start_time=0.0,
        end_time=P4B_END_TIME_MIN,
        discovery=EncounterDiscoveryConfig(5.0, 1.0, 60.0),
        lifecycle=lifecycle,
        recapture=recapture,
        default_environmental_drift_velocity=(0.0, 0.0),
    )
    return EnvironmentalNetworkFIFOExperimentConfig(
        network=network,
        streams=streams,
        scenario=scenario,
        admission_policy=(
            EnvironmentalNetworkFIFOAdmissionPolicy.ARM_LOCAL_FIFO_SHIP_DOMAIN_WITH_DYNAMIC_SAME_ROUTE_BRAKING
            if dynamic_same_route_ingress
            else EnvironmentalNetworkFIFOAdmissionPolicy.ARM_LOCAL_FIFO_SHIP_DOMAIN
        ),
        label="r1-p4b-engineering-d12-room1.2-rep03",
    )


def preflight_classification_config() -> EncounterClassificationConfig:
    """Recovered opt-in that prevents same-speed stern-sector false overtaking activation."""

    return EncounterClassificationConfig(require_overtaking_longitudinal_gain=True)


def run_preflight(
    *,
    rule13_progress_speed_fractions: tuple[float, ...] = (),
    rule13_progress_course_offsets_from_counterpart_deg: tuple[float, ...] = (),
    rule13_progress_include_speed_envelope_ceiling: bool = False,
    maximum_speed_nm_per_min: float | None = None,
    joint_max_combinations: int = 100,
    multi_intent_resolution: MultiIntentResolutionPolicy = MultiIntentResolutionPolicy.STRICT,
    horizon_safe_replan_caps: bool = False,
    dynamic_same_route_ingress: bool = False,
    persistent_same_route_supervision: bool = False,
    safety_metric_config: PairSafetyMetricConfig | None = None,
):
    """Execute the engineering case with modeled-infeasibility terminal capture."""

    config = build_preflight_config(
        rule13_progress_speed_fractions=rule13_progress_speed_fractions,
        rule13_progress_course_offsets_from_counterpart_deg=(
            rule13_progress_course_offsets_from_counterpart_deg
        ),
        rule13_progress_include_speed_envelope_ceiling=(
            rule13_progress_include_speed_envelope_ceiling
        ),
        maximum_speed_nm_per_min=maximum_speed_nm_per_min,
        joint_max_combinations=joint_max_combinations,
        multi_intent_resolution=multi_intent_resolution,
        horizon_safe_replan_caps=horizon_safe_replan_caps,
        dynamic_same_route_ingress=dynamic_same_route_ingress,
    )
    scenario = config.scenario
    traffic = generate_poisson_network_traffic(
        config.network,
        config.streams,
        horizon=scenario.traffic_horizon_min,
        seed=scenario.seed,
        start_time=scenario.start_time,
    )
    reservoir = EnvironmentalNetworkSourceReservoir.from_scenario_ingress(
        _environmental_ingress_records(traffic, scenario)
    )
    profiles = VesselProfileCatalog()
    world = (
        EnvironmentalWorld2D(
            (),
            limits={},
            environmental_drift_velocity={},
            start_time=scenario.start_time,
        )
        if safety_metric_config is None
        else _MetricEnvironmentalWorld(
            (),
            limits={},
            environmental_drift_velocity={},
            start_time=scenario.start_time,
            profile_catalog=profiles,
            safety_config=safety_metric_config,
        )
    )
    registry = EncounterRegistry(
        discovery=scenario.discovery,
        lifecycle=scenario.lifecycle,
        profile_catalog=profiles,
    )
    started = start_environmental_network_source_fifo_runtime(
        registry,
        world,
        {},
        reservoir,
        recapture_config=scenario.recapture,
        runtime_config=scenario.runtime,
        recheck_config=config.recheck,
        classification_config=preflight_classification_config(),
        profile_catalog=profiles,
        same_route_dynamic_ingress=(
            SameRouteFollowingSupervisor() if dynamic_same_route_ingress else None
        ),
        persistent_same_route_following=(
            SameRouteFollowingSupervisor() if persistent_same_route_supervision else None
        ),
    )
    outcome = started.session.run_until_capturing_terminal_outcome(
        scenario.end_time
    )
    return config, traffic, started, outcome



def build_bounded_resource_preflight_config(
    *,
    rule13_progress_speed_fractions: tuple[float, ...] = (),
    rule13_progress_course_offsets_from_counterpart_deg: tuple[float, ...] = (),
    rule13_progress_include_speed_envelope_ceiling: bool = False,
    maximum_speed_nm_per_min: float | None = None,
    multi_intent_resolution: MultiIntentResolutionPolicy = MultiIntentResolutionPolicy.STRICT,
    dynamic_same_route_ingress: bool = False,
) -> EnvironmentalNetworkFIFOExperimentConfig:
    """Build P4B with non-post-hoc execution-resource guards.

    The JOINT Cartesian ceiling reuses the simulator's established bounded A104
    contract (10,000 exact tuples).  Pair/JOINT replan caps are derived from the
    scenario horizon and positive review interval, so they cannot bind a cycle that
    advances by at least one review interval on each replan.  Resource exhaustion
    remains a distinct ``RESOURCE_UNAVAILABLE`` outcome; this helper never prunes or
    silently expands the command product.
    """

    return build_preflight_config(
        rule13_progress_speed_fractions=rule13_progress_speed_fractions,
        rule13_progress_course_offsets_from_counterpart_deg=(
            rule13_progress_course_offsets_from_counterpart_deg
        ),
        rule13_progress_include_speed_envelope_ceiling=(
            rule13_progress_include_speed_envelope_ceiling
        ),
        maximum_speed_nm_per_min=maximum_speed_nm_per_min,
        joint_max_combinations=P4B_ENGINEERING_JOINT_MAX_COMBINATIONS,
        multi_intent_resolution=multi_intent_resolution,
        horizon_safe_replan_caps=True,
        dynamic_same_route_ingress=dynamic_same_route_ingress,
    )


def run_bounded_resource_preflight(
    *,
    rule13_progress_speed_fractions: tuple[float, ...] = (),
    rule13_progress_course_offsets_from_counterpart_deg: tuple[float, ...] = (),
    rule13_progress_include_speed_envelope_ceiling: bool = False,
    maximum_speed_nm_per_min: float | None = None,
    multi_intent_resolution: MultiIntentResolutionPolicy = MultiIntentResolutionPolicy.STRICT,
    dynamic_same_route_ingress: bool = False,
    persistent_same_route_supervision: bool = False,
    safety_metric_config: PairSafetyMetricConfig | None = None,
):
    """Execute the same recovery case under the established bounded resource policy."""

    config = build_bounded_resource_preflight_config(
        rule13_progress_speed_fractions=rule13_progress_speed_fractions,
        rule13_progress_course_offsets_from_counterpart_deg=(
            rule13_progress_course_offsets_from_counterpart_deg
        ),
        rule13_progress_include_speed_envelope_ceiling=(
            rule13_progress_include_speed_envelope_ceiling
        ),
        maximum_speed_nm_per_min=maximum_speed_nm_per_min,
        multi_intent_resolution=multi_intent_resolution,
        dynamic_same_route_ingress=dynamic_same_route_ingress,
    )
    scenario = config.scenario
    traffic = generate_poisson_network_traffic(
        config.network,
        config.streams,
        horizon=scenario.traffic_horizon_min,
        seed=scenario.seed,
        start_time=scenario.start_time,
    )
    reservoir = EnvironmentalNetworkSourceReservoir.from_scenario_ingress(
        _environmental_ingress_records(traffic, scenario)
    )
    profiles = VesselProfileCatalog()
    world = (
        EnvironmentalWorld2D(
            (),
            limits={},
            environmental_drift_velocity={},
            start_time=scenario.start_time,
        )
        if safety_metric_config is None
        else _MetricEnvironmentalWorld(
            (),
            limits={},
            environmental_drift_velocity={},
            start_time=scenario.start_time,
            profile_catalog=profiles,
            safety_config=safety_metric_config,
        )
    )
    registry = EncounterRegistry(
        discovery=scenario.discovery,
        lifecycle=scenario.lifecycle,
        profile_catalog=profiles,
    )
    started = start_environmental_network_source_fifo_runtime(
        registry,
        world,
        {},
        reservoir,
        recapture_config=scenario.recapture,
        runtime_config=scenario.runtime,
        recheck_config=config.recheck,
        classification_config=preflight_classification_config(),
        profile_catalog=profiles,
        same_route_dynamic_ingress=(
            SameRouteFollowingSupervisor() if dynamic_same_route_ingress else None
        ),
        persistent_same_route_following=(
            SameRouteFollowingSupervisor() if persistent_same_route_supervision else None
        ),
    )
    outcome = started.session.run_until_capturing_terminal_outcome(scenario.end_time)
    return config, traffic, started, outcome

def diagnostic_summary(config, traffic, started, outcome) -> dict[str, Any]:
    """Small stable engineering summary; deliberately not a research result schema."""

    runtime = started.runtime_start.session
    queue = source_queue_metrics(outcome.reservoir, at=outcome.terminal_time_min)
    safety_intervals = getattr(started.session.world, "safety_intervals", ())
    safety = summarize_mission_safety(safety_intervals) if safety_intervals else None
    return {
        "evidence_role": "ENGINEERING_DIAGNOSTIC_ONLY_NOT_RESEARCH_RESULT",
        "recovery_case": P4B_RECOVERY_LABEL,
        "config_sha256": config.config_sha256,
        "traffic_count": traffic.count,
        "stream_counts": {
            stream.stream_id: len(stream.arrival_times) for stream in traffic.streams
        },
        "terminal_status": outcome.status.value,
        "terminal_time_min": outcome.terminal_time_min,
        "infeasibility_type": outcome.infeasibility_type,
        "infeasibility_message": outcome.infeasibility_message,
        "resource_unavailability_type": outcome.resource_unavailability_type,
        "resource_unavailability_message": outcome.resource_unavailability_message,
        "resource_kind": None if outcome.resource_kind is None else outcome.resource_kind.value,
        "resource_required_count": outcome.resource_required_count,
        "resource_limit_count": outcome.resource_limit_count,
        "resource_option_counts": [list(item) for item in outcome.resource_option_counts],
        "rebootstrap_count": len(runtime.persistent.rebootstrap_results),
        "fleet_observation_count": len(runtime.coordinator.fleet_observations),
        "active_vessel_ids": [v.vessel_id for v in outcome.snapshot.vessels],
        "source": {
            "demand_count": queue.demand_count,
            "admission_count": queue.admission_count,
            "pending_count": queue.pending_count,
            "due_pending_count": queue.due_pending_count,
            "mean_admitted_waiting_time_min": queue.mean_admitted_waiting_time_min,
            "max_admitted_waiting_time_min": queue.max_admitted_waiting_time_min,
            "mean_due_waiting_time_min": queue.mean_due_waiting_time_min,
            "max_due_waiting_time_min": queue.max_due_waiting_time_min,
        },
        "safety": (
            None
            if safety is None
            else {
                "interval_count": safety.interval_summary.interval_count,
                "modeled_hull_overlap_episode_count": safety.modeled_hull_overlap_episode_count,
                "safety_domain_intrusion_episode_count": safety.safety_domain_intrusion_episode_count,
                "near_miss_episode_count": safety.near_miss_episode_count,
                "unresolved_episode_count": safety.unresolved_episode_count,
                "nonclear_pair_exposure_min": safety.nonclear_pair_exposure_min,
                "sampled_min_center_separation_nm": safety.interval_summary.sampled_min_center_separation_nm,
                "sampled_min_hull_margin_nm": safety.interval_summary.sampled_min_hull_margin_nm,
                "sampled_min_safety_domain_margin_nm": safety.interval_summary.sampled_min_safety_domain_margin_nm,
                "worst_observed_evidence": safety.worst_observed_evidence.value,
            }
        ),
    }
