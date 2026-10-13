"""Reproducible one-scenario policy comparison harness."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .control import PolicyController
from .geometry import CrossingGeometry
from .metrics import (
    AggregateMetrics,
    ConflictZoneTiming,
    VesselLifecycleTiming,
    aggregate_metrics,
    conflict_zone_timing,
    lifecycle_timing,
)
from .scenarios import PoissonScenarioConfig
from .simulation import EffectiveTargetChange, Simulation, SimulationConfig
from .vessel import Vessel
from .zones import ConflictZonePassage


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Auditable output from one scenario/policy run."""

    scenario_name: str
    policy_name: str
    seed: int | None
    until: float | None
    final_time: float
    policy_evaluation_count: int
    aggregate: AggregateMetrics
    vessel_timings: tuple[VesselLifecycleTiming, ...]
    final_vessels: tuple[Vessel, ...]
    zone_passages: tuple[ConflictZonePassage, ...] = ()
    zone_timings: tuple[ConflictZoneTiming, ...] = ()
    effective_target_changes: tuple[EffectiveTargetChange, ...] = ()


def run_experiment(
    vessels: tuple[Vessel, ...] | list[Vessel],
    *,
    base_config: SimulationConfig,
    controller: PolicyController | None,
    scenario_name: str,
    seed: int | None = None,
    until: float | None = None,
    observation_end: float | None = None,
) -> ExperimentResult:
    """Run one immutable input scenario under one policy/controller."""

    if not scenario_name.strip():
        raise ValueError("scenario_name must be non-empty")

    # Vessel is frozen and Simulation updates its own dictionary by replacement,
    # so the same tuple can safely be reused for another policy run.
    input_vessels = tuple(vessels)
    config = replace(base_config, policy_controller=controller)
    sim = Simulation(list(input_vessels), config)
    sim.run(until=until)

    final_vessels = tuple(sorted(sim.vessels, key=lambda vessel: vessel.vessel_id))
    metric_end = (
        observation_end
        if observation_end is not None
        else (until if until is not None else sim.time)
    )
    zone_passages = sim.zone_passages
    aggregate = aggregate_metrics(
        final_vessels,
        observation_start=0.0,
        observation_end=metric_end,
        zone_passages=(zone_passages if config.conflict_zone is not None else None),
        conflict_zone=config.conflict_zone,
    )
    by_id = {vessel.vessel_id: vessel for vessel in final_vessels}
    zone_timings = ()
    if config.conflict_zone is not None:
        zone_timings = tuple(
            conflict_zone_timing(by_id[passage.vessel_id], passage, config.conflict_zone)
            for passage in zone_passages
        )
    policy_name = "no_controller" if controller is None else controller.policy.name

    return ExperimentResult(
        scenario_name=scenario_name,
        policy_name=policy_name,
        seed=seed,
        until=until,
        final_time=sim.time,
        policy_evaluation_count=sim.policy_evaluation_count,
        aggregate=aggregate,
        vessel_timings=tuple(lifecycle_timing(vessel) for vessel in final_vessels),
        final_vessels=final_vessels,
        zone_passages=zone_passages,
        zone_timings=zone_timings,
        effective_target_changes=sim.effective_target_changes,
    )


def compare_policies(
    vessels: tuple[Vessel, ...] | list[Vessel],
    *,
    base_config: SimulationConfig,
    controllers: tuple[PolicyController, ...] | list[PolicyController],
    scenario_name: str,
    seed: int | None = None,
    until: float | None = None,
    observation_end: float | None = None,
) -> tuple[ExperimentResult, ...]:
    """Run identical vessel inputs under every supplied policy controller."""

    controllers = tuple(controllers)
    names = [controller.policy.name for controller in controllers]
    if len(names) != len(set(names)):
        raise ValueError("comparison policy names must be unique")

    return tuple(
        run_experiment(
            vessels,
            base_config=base_config,
            controller=controller,
            scenario_name=scenario_name,
            seed=seed,
            until=until,
            observation_end=observation_end,
        )
        for controller in controllers
    )

@dataclass(frozen=True, slots=True)
class ReplicationResult:
    """One traffic realization reused across all policy runs."""

    replication_index: int
    seed: int
    scenario_vessels: tuple[Vessel, ...]
    experiments: tuple[ExperimentResult, ...]


@dataclass(frozen=True, slots=True)
class ScalarSummary:
    """Across-replication mean, sample SD and standard error."""

    n: int
    mean: float | None
    sample_std: float | None
    standard_error: float | None
    confidence_level: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


@dataclass(frozen=True, slots=True)
class PolicyReplicationSummary:
    policy_name: str
    completed_in_window: ScalarSummary
    completion_fraction_in_window: ScalarSummary
    throughput_per_hour: ScalarSummary
    mean_waiting_delay: ScalarSummary
    mean_total_system_time: ScalarSummary
    post_window_clearance_time: ScalarSummary
    crossing_service_fraction_in_window: ScalarSummary
    crossing_throughput_per_hour: ScalarSummary
    mean_conflict_zone_occupancy_time: ScalarSummary
    post_window_conflict_zone_clearance_time: ScalarSummary
    mean_conflict_zone_entry_delay: ScalarSummary
    mean_conflict_zone_exit_delay: ScalarSummary
    mean_active_approach_count: ScalarSummary
    peak_active_approach_backlog: ScalarSummary
    service_fraction_jain_index: ScalarSummary
    starved_route_count_in_window: ScalarSummary
    max_abs_exit_share_error: ScalarSummary
    median_post_window_clearance_time: float | None = None
    p95_post_window_clearance_time: float | None = None
    max_post_window_clearance_time: float | None = None


def replication_seeds(
    *,
    base_seed: int,
    count: int,
    start_index: int = 0,
) -> tuple[int, ...]:
    """Derive deterministic independent child seeds from one base seed.

    ``start_index`` allows an adaptive replication executor to append only the
    newly required replications while preserving the exact seed sequence that a
    single one-shot run of ``start_index + count`` replications would use.
    """

    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if count < 0:
        raise ValueError("count must be non-negative")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    import numpy as np

    root = np.random.SeedSequence(base_seed)
    children = root.spawn(start_index + count)
    return tuple(
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in children[start_index:]
    )


def run_paired_replications(
    *,
    geometry,
    scenario_config,
    base_config: SimulationConfig,
    controllers: tuple[PolicyController, ...] | list[PolicyController],
    scenario_name: str,
    base_seed: int,
    replications: int,
    replication_start_index: int = 0,
) -> tuple[ReplicationResult, ...]:
    """Generate one scenario per replication and reuse it across policies."""

    from .scenarios import generate_poisson_scenario

    seeds = replication_seeds(
        base_seed=base_seed,
        count=replications,
        start_index=replication_start_index,
    )
    results: list[ReplicationResult] = []
    for offset, seed in enumerate(seeds):
        index = replication_start_index + offset
        vessels = generate_poisson_scenario(geometry, scenario_config, seed=seed)
        experiments = compare_policies(
            vessels,
            base_config=base_config,
            controllers=controllers,
            scenario_name=scenario_name,
            seed=seed,
            # Run the generated arrival cohort to clearance so waiting/travel
            # timing is complete, but measure throughput only inside the
            # original fixed demand/observation horizon.
            until=None,
            observation_end=scenario_config.horizon,
        )
        results.append(
            ReplicationResult(
                replication_index=index,
                seed=seed,
                scenario_vessels=vessels,
                experiments=experiments,
            )
        )
    return tuple(results)


def _scalar_summary(values: list[float]) -> ScalarSummary:
    from math import sqrt

    n = len(values)
    if n == 0:
        return ScalarSummary(0, None, None, None)
    mean = sum(values) / n
    if n < 2:
        return ScalarSummary(n, mean, None, None)
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    sample_std = sqrt(variance)
    standard_error = sample_std / sqrt(n)

    # Two-sided Student-t interval for an unknown variance estimated from the
    # same finite replication sample.  Keep the confidence convention explicit
    # in the result instead of presenting an unlabeled +/- value.
    confidence_level = 0.95
    from scipy.stats import t as student_t

    critical = float(student_t.ppf((1.0 + confidence_level) / 2.0, df=n - 1))
    half_width = critical * standard_error
    return ScalarSummary(
        n,
        mean,
        sample_std,
        standard_error,
        confidence_level,
        mean - half_width,
        mean + half_width,
    )


def _sample_median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _sample_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_replications(
    replications: tuple[ReplicationResult, ...] | list[ReplicationResult],
) -> tuple[PolicyReplicationSummary, ...]:
    """Summarize paired replications policy-by-policy without hiding runs."""

    replications = tuple(replications)
    if not replications:
        return ()

    policy_names = tuple(result.policy_name for result in replications[0].experiments)
    for replication in replications[1:]:
        if tuple(result.policy_name for result in replication.experiments) != policy_names:
            raise ValueError("every replication must contain the same policy order")

    summaries: list[PolicyReplicationSummary] = []
    for policy_index, policy_name in enumerate(policy_names):
        experiments = [rep.experiments[policy_index] for rep in replications]
        completed_in_window = [
            float(exp.aggregate.completed_in_window) for exp in experiments
        ]
        completion_fraction = [
            exp.aggregate.completion_fraction_in_window
            for exp in experiments
            if exp.aggregate.completion_fraction_in_window is not None
        ]
        throughput = [
            exp.aggregate.throughput_per_hour
            for exp in experiments
            if exp.aggregate.throughput_per_hour is not None
        ]
        waiting = [
            exp.aggregate.mean_waiting_delay
            for exp in experiments
            if exp.aggregate.mean_waiting_delay is not None
        ]
        total = [
            exp.aggregate.mean_total_system_time
            for exp in experiments
            if exp.aggregate.mean_total_system_time is not None
        ]
        clearance = [
            exp.aggregate.post_window_clearance_time
            for exp in experiments
            if exp.aggregate.post_window_clearance_time is not None
        ]
        crossing_throughput = [
            exp.aggregate.crossing_throughput_per_hour
            for exp in experiments
            if exp.aggregate.crossing_throughput_per_hour is not None
        ]
        crossing_service_fraction = [
            exp.aggregate.crossing_service_fraction_in_window
            for exp in experiments
            if exp.aggregate.crossing_service_fraction_in_window is not None
        ]
        zone_occupancy = [
            exp.aggregate.mean_conflict_zone_occupancy_time
            for exp in experiments
            if exp.aggregate.mean_conflict_zone_occupancy_time is not None
        ]
        zone_clearance = [
            exp.aggregate.post_window_conflict_zone_clearance_time
            for exp in experiments
            if exp.aggregate.post_window_conflict_zone_clearance_time is not None
        ]
        zone_entry_delay = [
            exp.aggregate.mean_conflict_zone_entry_delay
            for exp in experiments
            if exp.aggregate.mean_conflict_zone_entry_delay is not None
        ]
        zone_exit_delay = [
            exp.aggregate.mean_conflict_zone_exit_delay
            for exp in experiments
            if exp.aggregate.mean_conflict_zone_exit_delay is not None
        ]
        approach_mean = [
            exp.aggregate.mean_active_approach_count
            for exp in experiments
            if exp.aggregate.mean_active_approach_count is not None
        ]
        approach_peak = [
            float(exp.aggregate.peak_active_approach_backlog)
            for exp in experiments
            if exp.aggregate.peak_active_approach_backlog is not None
        ]
        service_fairness = [
            exp.aggregate.service_fraction_jain_index
            for exp in experiments
            if exp.aggregate.service_fraction_jain_index is not None
        ]
        starved_routes = [
            float(exp.aggregate.starved_route_count_in_window)
            for exp in experiments
            if exp.aggregate.starved_route_count_in_window is not None
        ]
        share_error = [
            exp.aggregate.max_abs_exit_share_error
            for exp in experiments
            if exp.aggregate.max_abs_exit_share_error is not None
        ]
        summaries.append(
            PolicyReplicationSummary(
                policy_name=policy_name,
                completed_in_window=_scalar_summary(completed_in_window),
                completion_fraction_in_window=_scalar_summary(completion_fraction),
                throughput_per_hour=_scalar_summary(throughput),
                mean_waiting_delay=_scalar_summary(waiting),
                mean_total_system_time=_scalar_summary(total),
                post_window_clearance_time=_scalar_summary(clearance),
                crossing_service_fraction_in_window=_scalar_summary(
                    crossing_service_fraction
                ),
                crossing_throughput_per_hour=_scalar_summary(crossing_throughput),
                mean_conflict_zone_occupancy_time=_scalar_summary(zone_occupancy),
                post_window_conflict_zone_clearance_time=_scalar_summary(zone_clearance),
                mean_conflict_zone_entry_delay=_scalar_summary(zone_entry_delay),
                mean_conflict_zone_exit_delay=_scalar_summary(zone_exit_delay),
                mean_active_approach_count=_scalar_summary(approach_mean),
                peak_active_approach_backlog=_scalar_summary(approach_peak),
                service_fraction_jain_index=_scalar_summary(service_fairness),
                starved_route_count_in_window=_scalar_summary(starved_routes),
                max_abs_exit_share_error=_scalar_summary(share_error),
                median_post_window_clearance_time=_sample_median(clearance),
                p95_post_window_clearance_time=_sample_percentile(clearance, 95.0),
                max_post_window_clearance_time=max(clearance) if clearance else None,
            )
        )
    return tuple(summaries)


@dataclass(frozen=True, slots=True)
class PolicyDifferenceSummary:
    """Paired comparison-minus-baseline difference across replications."""

    baseline_policy: str
    comparison_policy: str
    metric_name: str
    difference: ScalarSummary


PAIRABLE_METRICS = (
    "completed_in_window",
    "completion_fraction_in_window",
    "throughput_per_hour",
    "mean_waiting_delay",
    "mean_total_system_time",
    "post_window_clearance_time",
    "crossing_service_fraction_in_window",
    "crossing_throughput_per_hour",
    "mean_conflict_zone_occupancy_time",
    "post_window_conflict_zone_clearance_time",
    "mean_conflict_zone_entry_delay",
    "mean_conflict_zone_exit_delay",
    "mean_active_approach_count",
    "peak_active_approach_backlog",
    "service_fraction_jain_index",
    "starved_route_count_in_window",
    "max_abs_exit_share_error",
)


def summarize_paired_policy_differences(
    replications: tuple[ReplicationResult, ...] | list[ReplicationResult],
    *,
    baseline_policy: str,
    comparison_policy: str,
) -> tuple[PolicyDifferenceSummary, ...]:
    """Summarize paired ``comparison - baseline`` differences by replication."""

    replications = tuple(replications)
    differences: dict[str, list[float]] = {metric: [] for metric in PAIRABLE_METRICS}

    for replication in replications:
        by_policy = {experiment.policy_name: experiment for experiment in replication.experiments}
        if baseline_policy not in by_policy or comparison_policy not in by_policy:
            raise ValueError("both requested policies must exist in every replication")
        baseline = by_policy[baseline_policy].aggregate
        comparison = by_policy[comparison_policy].aggregate
        for metric in PAIRABLE_METRICS:
            baseline_value = getattr(baseline, metric)
            comparison_value = getattr(comparison, metric)
            if baseline_value is None or comparison_value is None:
                continue
            differences[metric].append(float(comparison_value - baseline_value))

    return tuple(
        PolicyDifferenceSummary(
            baseline_policy=baseline_policy,
            comparison_policy=comparison_policy,
            metric_name=metric,
            difference=_scalar_summary(values),
        )
        for metric, values in differences.items()
        if values
    )

def summarize_all_paired_policy_differences(
    replications: tuple[ReplicationResult, ...] | list[ReplicationResult],
) -> tuple[PolicyDifferenceSummary, ...]:
    """Return all ordered-by-input pairwise policy comparisons.

    For policies ``P0, P1, P2`` this returns ``P1-P0``, ``P2-P0`` and
    ``P2-P1`` for every pairable metric.  The sign convention remains
    comparison minus baseline and is stored in each summary object.
    """

    replications = tuple(replications)
    if not replications:
        return ()
    names = tuple(exp.policy_name for exp in replications[0].experiments)
    output: list[PolicyDifferenceSummary] = []
    for baseline_index, baseline_name in enumerate(names):
        for comparison_name in names[baseline_index + 1 :]:
            output.extend(
                summarize_paired_policy_differences(
                    replications,
                    baseline_policy=baseline_name,
                    comparison_policy=comparison_name,
                )
            )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class DemandSweepPoint:
    """Named demand level with a complete Poisson scenario configuration."""

    name: str
    scenario_config: PoissonScenarioConfig

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("demand sweep point name must be non-empty")


@dataclass(frozen=True, slots=True)
class DemandSweepPointResult:
    point: DemandSweepPoint
    point_seed: int
    replications: tuple[ReplicationResult, ...]
    policy_summaries: tuple[PolicyReplicationSummary, ...]
    paired_differences: tuple[PolicyDifferenceSummary, ...]



def demand_point_seed_map(
    points: tuple[DemandSweepPoint, ...] | list[DemandSweepPoint],
    *,
    base_seed: int,
) -> dict[str, int]:
    """Derive order-independent deterministic child seeds for demand points."""

    import numpy as np

    points = tuple(points)
    names = [point.name for point in points]
    if len(names) != len(set(names)):
        raise ValueError("demand sweep point names must be unique")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")

    sorted_points = sorted(points, key=lambda point: point.name)
    root = np.random.SeedSequence(base_seed)
    children = root.spawn(len(sorted_points))
    return {
        point.name: int(child.generate_state(1, dtype=np.uint64)[0])
        for point, child in zip(sorted_points, children)
    }

def run_demand_sweep(
    *,
    geometry: CrossingGeometry,
    points: tuple[DemandSweepPoint, ...] | list[DemandSweepPoint],
    base_config: SimulationConfig,
    controllers: tuple[PolicyController, ...] | list[PolicyController],
    scenario_name: str,
    base_seed: int,
    replications: int,
) -> tuple[DemandSweepPointResult, ...]:
    """Run paired replications at multiple demand points.

    Each demand point receives a deterministic child seed independent of input
    point ordering.  Policy pairing remains inside each point/replication; the
    function does not force identical arrival counts across different rates.
    """

    points = tuple(points)
    names = [point.name for point in points]
    if len(names) != len(set(names)):
        raise ValueError("demand sweep point names must be unique")
    if points:
        horizons = {point.scenario_config.horizon for point in points}
        if len(horizons) != 1:
            raise ValueError("demand sweep points must use one common horizon")
        desired_speeds = {point.scenario_config.desired_speed for point in points}
        if len(desired_speeds) != 1:
            raise ValueError("demand sweep points must use one common desired speed")

    seed_by_name = demand_point_seed_map(points, base_seed=base_seed)

    output: list[DemandSweepPointResult] = []
    for point in points:
        point_seed = seed_by_name[point.name]
        paired = run_paired_replications(
            geometry=geometry,
            scenario_config=point.scenario_config,
            base_config=base_config,
            controllers=controllers,
            scenario_name=f"{scenario_name}:{point.name}",
            base_seed=point_seed,
            replications=replications,
        )
        output.append(
            DemandSweepPointResult(
                point=point,
                point_seed=point_seed,
                replications=paired,
                policy_summaries=summarize_replications(paired),
                paired_differences=summarize_all_paired_policy_differences(paired),
            )
        )
    return tuple(output)
