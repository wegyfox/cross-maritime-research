"""Execution bridge from D0 into the existing NextGen environmental-network FIFO runtime.

P5c intentionally implements baseline evidence only.  Managed interventions and physical
reference execution stay fail-closed in their concrete binding registries until separate
qualified runners exist.  No simulator or statistical estimator is defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import fmean

from ..experiments import replication_seeds
from ..nextgen.encounter_registry import EncounterRegistry
from ..nextgen.environmental_network_fifo_experiment import (
    EnvironmentalNetworkFIFOAdmissionPolicy,
    EnvironmentalNetworkFIFOExperimentConfig,
    EnvironmentalNetworkFIFOExperimentResult,
    run_environmental_network_fifo_experiment,
)
from ..nextgen.environmental_network_scenario import build_environmental_network_scenario_ingress
from ..nextgen.environmental_network_source_fifo_runtime import (
    start_environmental_network_source_fifo_runtime,
)
from ..nextgen.environmental_network_source_reservoir import (
    EnvironmentalNetworkSourceReservoir,
    source_queue_metrics,
)
from ..nextgen.environmental_world import EnvironmentalWorld2D
from ..nextgen.network_traffic import (
    PoissonNetworkTrafficRealization,
    generate_poisson_network_traffic,
)
from ..nextgen.reference_envelope import ReferenceEnvelopeLevel
from ..nextgen.same_route_following import SameRouteFollowingSupervisor
from ..nextgen.vessel_classes import VesselProfileCatalog
from .evidence import DiagnosticLevelEvidence, DiagnosticReplicationEvidence
from .execution import DiagnosticExecutionContext
from .interventions import InterventionDisposition
from .nextgen_fifo_bindings import NextgenFIFOBindingRegistry


class NextgenFIFOExecutionError(RuntimeError):
    """Raised when canonical FIFO evidence cannot be translated safely into D0 evidence."""


def _completion_ids(result: EnvironmentalNetworkFIFOExperimentResult) -> tuple[str, ...]:
    """Return canonical route-completion identities used by the existing A159 metric.

    A159 defines route completion from persistent runtime disposition evidence.  D0 reads
    the same ledger and does not infer completion from final vessel absence or queue state.
    """

    persistent = result.runtime_start.session.runtime_session.persistent
    ids: list[str] = []
    for disposition_result in persistent.disposition_results:
        ids.extend(item.vessel_id for item in disposition_result.dispositions)
    if len(ids) != len(set(ids)):
        raise NextgenFIFOExecutionError("duplicate route-completion evidence in canonical runtime")
    admitted = {item.vessel_id for item in result.run.reservoir.admissions}
    unknown = set(ids) - admitted
    if unknown:
        raise NextgenFIFOExecutionError(
            f"route completion lacks canonical FIFO admission evidence: {sorted(unknown)}"
        )
    return tuple(ids)


def _baseline_replication_evidence(
    result: EnvironmentalNetworkFIFOExperimentResult,
    *,
    replication_index: int,
) -> DiagnosticReplicationEvidence:
    duration = result.summary.end_time - result.summary.start_time
    if duration <= 0.0:
        raise NextgenFIFOExecutionError("service-yield KPI requires a positive runtime window")
    completed_ids = _completion_ids(result)
    service_yield = 60.0 * len(completed_ids) / duration
    summary = result.summary
    return DiagnosticReplicationEvidence(
        replication_id=f"R{replication_index:03d}",
        primary_kpi=service_yield,
        metrics={
            "seed": summary.seed,
            "canonical_run_id": summary.run_id,
            "canonical_config_sha256": summary.config_sha256,
            "canonical_result_sha256": result.result_sha256,
            "runtime_window_min": duration,
            "realized_demand_count": summary.realized_demand_count,
            "physical_admission_count": summary.physical_admission_count,
            "route_completed_count": len(completed_ids),
            "route_completion_rate_per_hour": service_yield,
            "pending_source_count": summary.pending_source_count,
            "due_pending_source_count": summary.due_pending_source_count,
            "mean_admitted_waiting_time_min": summary.mean_admitted_waiting_time_min,
            "max_admitted_waiting_time_min": summary.max_admitted_waiting_time_min,
            "final_vessel_count": summary.final_vessel_count,
            "source_plan_count": summary.source_plan_count,
            "source_boundary_count": summary.source_boundary_count,
            "fleet_observation_count": summary.fleet_observation_count,
            "stop_reason": summary.stop_reason,
        },
    )


def run_nextgen_fifo_baseline(
    context: DiagnosticExecutionContext,
    *,
    bindings: NextgenFIFOBindingRegistry,
) -> DiagnosticLevelEvidence:
    """Execute deterministic child-seed B0 replications through canonical A152."""

    if context.level is not ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK:
        raise NextgenFIFOExecutionError("P5c baseline runner accepts B0 only")
    if context.plan_entry is not None or context.previous_level is not None or context.previous_runs:
        raise NextgenFIFOExecutionError("B0 execution context must not contain prior ladder evidence")
    config = bindings.config_builder(context.scenario)
    if not isinstance(config, EnvironmentalNetworkFIFOExperimentConfig):
        raise NextgenFIFOExecutionError(
            "NextGen FIFO config builder must return EnvironmentalNetworkFIFOExperimentConfig"
        )
    count = context.scenario.experiment.replications
    seeds = replication_seeds(base_seed=config.scenario.seed, count=count)
    replications: list[DiagnosticReplicationEvidence] = []
    for index, seed in enumerate(seeds):
        child_config = replace(config, scenario=replace(config.scenario, seed=seed))
        result = run_environmental_network_fifo_experiment(child_config)
        replications.append(
            _baseline_replication_evidence(result, replication_index=index)
        )
    mean_yield = fmean(item.primary_kpi for item in replications)
    return DiagnosticLevelEvidence(
        ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK,
        {
            "mean_service_yield_per_hour": mean_yield,
            "replication_count": len(replications),
            "seed_schedule": list(seeds),
        },
        tuple(replications),
        causal=False,
        reason="CANONICAL_NEXTGEN_ENVIRONMENTAL_NETWORK_FIFO_BASELINE",
    )


def build_nextgen_fifo_baseline_runner(bindings: NextgenFIFOBindingRegistry):
    """Bind one immutable runtime registry into the generic DiagnosticExecutionAdapter."""

    if not isinstance(bindings, NextgenFIFOBindingRegistry):
        raise TypeError("bindings must be NextgenFIFOBindingRegistry")

    def runner(context: DiagnosticExecutionContext) -> DiagnosticLevelEvidence:
        return run_nextgen_fifo_baseline(context, bindings=bindings)

    return runner

@dataclass(frozen=True, slots=True)
class NextgenFIFOControllerRuntimeContext:
    """Concrete canonical runtime objects exposed to one registered controller factory."""

    config: EnvironmentalNetworkFIFOExperimentConfig
    traffic: PoissonNetworkTrafficRealization
    reservoir: EnvironmentalNetworkSourceReservoir
    world: EnvironmentalWorld2D
    profile_catalog: VesselProfileCatalog
    same_route_dynamic_ingress: SameRouteFollowingSupervisor | None


def _managed_completion_ids(started, run) -> tuple[str, ...]:
    persistent = started.runtime_start.session.persistent
    ids: list[str] = []
    for disposition_result in persistent.disposition_results:
        ids.extend(item.vessel_id for item in disposition_result.dispositions)
    if len(ids) != len(set(ids)):
        raise NextgenFIFOExecutionError("duplicate managed route-completion evidence")
    admitted = {item.vessel_id for item in run.reservoir.admissions}
    unknown = set(ids) - admitted
    if unknown:
        raise NextgenFIFOExecutionError(
            f"managed route completion lacks FIFO admission evidence: {sorted(unknown)}"
        )
    return tuple(ids)


def _run_managed_replication(
    config: EnvironmentalNetworkFIFOExperimentConfig,
    *,
    controller_factory,
    context: DiagnosticExecutionContext,
    replication_index: int,
) -> DiagnosticReplicationEvidence:
    scenario = config.scenario
    traffic = generate_poisson_network_traffic(
        config.network,
        config.streams,
        horizon=scenario.traffic_horizon_min,
        seed=scenario.seed,
        start_time=scenario.start_time,
    )
    ingress = build_environmental_network_scenario_ingress(traffic, scenario)
    reservoir = EnvironmentalNetworkSourceReservoir.from_scenario_ingress(ingress)
    profiles = VesselProfileCatalog()
    world = EnvironmentalWorld2D(
        (),
        limits={},
        environmental_drift_velocity={},
        start_time=scenario.start_time,
    )
    registry = EncounterRegistry(
        discovery=scenario.discovery,
        lifecycle=scenario.lifecycle,
        profile_catalog=profiles,
    )
    same_route = (
        SameRouteFollowingSupervisor()
        if config.admission_policy
        is EnvironmentalNetworkFIFOAdmissionPolicy.ARM_LOCAL_FIFO_SHIP_DOMAIN_WITH_DYNAMIC_SAME_ROUTE_BRAKING
        else None
    )
    factory_context = NextgenFIFOControllerRuntimeContext(
        config,
        traffic,
        reservoir,
        world,
        profiles,
        same_route,
    )
    controller = controller_factory(context, factory_context)
    if controller is None:
        raise NextgenFIFOExecutionError("registered controller factory returned None")
    started = start_environmental_network_source_fifo_runtime(
        registry,
        world,
        {},
        reservoir,
        recapture_config=scenario.recapture,
        runtime_config=scenario.runtime,
        recheck_config=config.recheck,
        profile_catalog=profiles,
        same_route_dynamic_ingress=same_route,
        source_admission_controller=controller,
    )
    run = started.session.run_until(float(scenario.end_time))
    queue = source_queue_metrics(run.reservoir, at=run.snapshot.time)
    duration = scenario.end_time - scenario.start_time
    if duration <= 0.0:
        raise NextgenFIFOExecutionError("service-yield KPI requires a positive runtime window")
    completed_ids = _managed_completion_ids(started, run)
    service_yield = 60.0 * len(completed_ids) / duration
    return DiagnosticReplicationEvidence(
        replication_id=f"R{replication_index:03d}",
        primary_kpi=service_yield,
        metrics={
            "seed": scenario.seed,
            "canonical_config_sha256": config.config_sha256,
            "runtime_window_min": duration,
            "realized_demand_count": traffic.count,
            "physical_admission_count": len(run.reservoir.admissions),
            "route_completed_count": len(completed_ids),
            "route_completion_rate_per_hour": service_yield,
            "pending_source_count": queue.pending_count,
            "due_pending_source_count": queue.due_pending_count,
            "mean_admitted_waiting_time_min": queue.mean_admitted_waiting_time_min,
            "max_admitted_waiting_time_min": queue.max_admitted_waiting_time_min,
            "final_vessel_count": len(run.snapshot.vessels),
            "source_plan_count": len(started.session.source_plans),
            "source_boundary_count": len(started.session.source_boundaries),
            "execution_path": "canonical_source_fifo_runtime_with_registered_controller",
        },
    )


def run_nextgen_fifo_managed_level(
    context: DiagnosticExecutionContext,
    *,
    bindings: NextgenFIFOBindingRegistry,
) -> DiagnosticLevelEvidence:
    """Execute one concrete controller-backed causal ladder step.

    P5d intentionally supports one concrete applied intervention only.  Composing two
    source-management controllers has ordering/scientific semantics and remains
    fail-closed until an explicit qualified composition binding is introduced.
    """

    entry = context.plan_entry
    if entry is None or entry.disposition is not InterventionDisposition.APPLICABLE:
        raise NextgenFIFOExecutionError("managed runner requires one APPLICABLE plan entry")
    if context.level is not entry.level:
        raise NextgenFIFOExecutionError("managed context level disagrees with plan entry")
    applied = context.applied_intervention_ids
    if not applied or applied[-1] != entry.intervention_id:
        raise NextgenFIFOExecutionError(
            "managed execution context must terminate in the current intervention id"
        )
    if len(applied) == 1:
        controller_factory = bindings.intervention(entry.intervention_id).controller_factory
        reason = "REGISTERED_CANONICAL_SOURCE_ADMISSION_CONTROLLER"
    else:
        controller_factory = bindings.composition(applied).controller_factory
        reason = "REGISTERED_CANONICAL_CUMULATIVE_SOURCE_ADMISSION_CONTROLLER"
    base_config = bindings.config_builder(context.scenario)
    if not isinstance(base_config, EnvironmentalNetworkFIFOExperimentConfig):
        raise NextgenFIFOExecutionError(
            "NextGen FIFO config builder must return EnvironmentalNetworkFIFOExperimentConfig"
        )
    seeds = replication_seeds(
        base_seed=base_config.scenario.seed,
        count=context.scenario.experiment.replications,
    )
    replications: list[DiagnosticReplicationEvidence] = []
    for index, seed in enumerate(seeds):
        child_config = replace(
            base_config,
            scenario=replace(base_config.scenario, seed=seed),
        )
        replications.append(
            _run_managed_replication(
                child_config,
                controller_factory=controller_factory,
                context=context,
                replication_index=index,
            )
        )
    if context.previous_level is not None:
        prior_ids = tuple(item.replication_id for item in context.previous_level.replications)
        current_ids = tuple(item.replication_id for item in replications)
        if prior_ids and prior_ids != current_ids:
            raise NextgenFIFOExecutionError("managed replication IDs do not match prior ladder level")
    return DiagnosticLevelEvidence(
        context.level,
        {
            "mean_service_yield_per_hour": fmean(item.primary_kpi for item in replications),
            "replication_count": len(replications),
            "seed_schedule": list(seeds),
        },
        tuple(replications),
        intervention_id=entry.intervention_id,
        target_factor=entry.target_factor,
        causal=True,
        reason=reason,
    )


def run_nextgen_fifo_reference_level(
    context: DiagnosticExecutionContext,
    *,
    bindings: NextgenFIFOBindingRegistry,
) -> DiagnosticLevelEvidence:
    """Execute one explicitly registered non-causal reference model.

    A reference binding is an execution boundary, not a scientific default.  The
    concrete runner owns its model semantics and must return portable evidence for the
    exact requested ladder entry.  In particular, this wrapper does not reinterpret an
    article/R2 reference implementation as a universal B6 model.
    """

    entry = context.plan_entry
    if entry is None or entry.disposition is not InterventionDisposition.REFERENCE:
        raise NextgenFIFOExecutionError("reference runner requires one REFERENCE plan entry")
    if context.level is not entry.level:
        raise NextgenFIFOExecutionError("reference context level disagrees with plan entry")
    if not context.applied_intervention_ids or context.applied_intervention_ids[-1] != entry.intervention_id:
        raise NextgenFIFOExecutionError(
            "reference execution context must terminate in the requested reference id"
        )

    binding = bindings.reference(entry.intervention_id)
    result = binding.runner(context)
    if not isinstance(result, DiagnosticLevelEvidence):
        raise NextgenFIFOExecutionError("registered reference runner must return DiagnosticLevelEvidence")
    if result.level is not context.level:
        raise NextgenFIFOExecutionError("registered reference runner returned the wrong ladder level")
    if result.intervention_id != entry.intervention_id:
        raise NextgenFIFOExecutionError("registered reference runner returned the wrong reference id")
    if result.target_factor != entry.target_factor:
        raise NextgenFIFOExecutionError("registered reference runner returned the wrong target factor")
    if result.causal:
        raise NextgenFIFOExecutionError("reference evidence must be non-causal")
    if result.identity_of is not None:
        raise NextgenFIFOExecutionError("reference evidence must not masquerade as an identity level")
    return result


def build_nextgen_fifo_runtime_runner(bindings: NextgenFIFOBindingRegistry):
    """Dispatch B0, one-controller causal levels, and explicit reference bindings."""

    if not isinstance(bindings, NextgenFIFOBindingRegistry):
        raise TypeError("bindings must be NextgenFIFOBindingRegistry")

    def runner(context: DiagnosticExecutionContext) -> DiagnosticLevelEvidence:
        if context.level is ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK:
            return run_nextgen_fifo_baseline(context, bindings=bindings)
        if context.plan_entry is not None and context.plan_entry.disposition is InterventionDisposition.APPLICABLE:
            return run_nextgen_fifo_managed_level(context, bindings=bindings)
        if context.plan_entry is not None and context.plan_entry.disposition is InterventionDisposition.REFERENCE:
            return run_nextgen_fifo_reference_level(context, bindings=bindings)
        raise NextgenFIFOExecutionError(
            "no qualified NextGen FIFO runtime path for requested diagnostic level"
        )

    return runner

