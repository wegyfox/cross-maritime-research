"""Q5 diagnostic orchestration for deterministic sampled Custom Physics v2 bindings.

The frozen D0 wheel and existing fixed Q5 integration remain untouched.  This module
reuses their public runtime components while replacing only traffic realization with the
identity-bearing heterogeneous realization produced by ``custom_physics_binding``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
import json

from vessel_crossing.diagnostic.evidence import DiagnosticLevelEvidence, DiagnosticReplicationEvidence
from vessel_crossing.diagnostic.execution import (
    DiagnosticAdapterRegistry,
    DiagnosticExecutionContext,
    DiagnosticExecutionIdentity,
)
from vessel_crossing.experiments import replication_seeds
from vessel_crossing.diagnostic.interventions import InterventionDisposition
from vessel_crossing.diagnostic.nextgen_fifo_bindings import (
    NextgenFIFOBindingRegistry,
    NextgenFIFOCompositionBinding,
    NextgenFIFOInterventionBinding,
    NextgenFIFOReferenceBinding,
    build_nextgen_fifo_execution_adapter,
)
from vessel_crossing.diagnostic.nextgen_fifo_execution import (
    NextgenFIFOControllerRuntimeContext,
    NextgenFIFOExecutionError,
    run_nextgen_fifo_reference_level,
)
from vessel_crossing.nextgen.encounter_registry import EncounterRegistry
from vessel_crossing.nextgen.environmental_network_fifo_experiment import (
    EnvironmentalNetworkFIFOAdmissionPolicy,
    EnvironmentalNetworkFIFOExperimentConfig,
    run_environmental_network_fifo_experiment_with_traffic,
)
from vessel_crossing.nextgen.environmental_network_scenario import (
    build_environmental_network_scenario_ingress,
)
from vessel_crossing.nextgen.environmental_network_source_fifo_runtime import (
    start_environmental_network_source_fifo_runtime,
)
from vessel_crossing.nextgen.environmental_network_source_reservoir import (
    EnvironmentalNetworkSourceReservoir,
    source_queue_metrics,
)
from vessel_crossing.nextgen.environmental_world import EnvironmentalWorld2D
from vessel_crossing.nextgen.reference_envelope import ReferenceEnvelopeLevel
from vessel_crossing.nextgen.same_route_following import SameRouteFollowingSupervisor
from vessel_crossing.nextgen.vessel_classes import VesselProfileCatalog

from .b4 import B4EarliestSafeSlotController
from .b5 import B5BoundedSlotCostController
from .custom_physics_binding import (
    BoundCustomPhysicsScenario,
    build_bound_reference_cache_identity,
    materialize_bound_traffic,
    resolve_bound_dynamic_reference,
)
from .fleet_mixtures import FleetMixturePlan
from .physics_distributions import PhysicsDistributionPlan
from .q5_integration import (
    A70_REFERENCE_EVIDENCE_SHA256,
    B4_D0_ID,
    B5_D0_ID,
    B6_D0_ID,
    Q5IntegrationError,
    _build_execution_identity,
    integration_contract_sha256,
    prepare_q5_scenario,
)

SAMPLED_Q5_EXECUTION_IDENTITY_ID = "cross-psp-two-stream-q5-sampled-physics-v1"


def _completion_ids(result) -> tuple[str, ...]:
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


def _baseline_replication_evidence(result, *, replication_index: int) -> DiagnosticReplicationEvidence:
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
            "sampled_physics": True,
        },
    )


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


class _BoundReferenceProvider:
    def __init__(self, bound: BoundCustomPhysicsScenario, cache_dir: Path | None) -> None:
        self.bound = bound
        self.cache_dir = cache_dir
        self.memory: dict[str, tuple[object, str]] = {}
        self.identity = build_bound_reference_cache_identity(bound)

    def resolve(self):
        cached = self.memory.get(self.identity.sha256)
        if cached is not None:
            result, status = cached
            return result, "MEMORY_HIT_AFTER_" + status
        result, status, identity = resolve_bound_dynamic_reference(
            self.bound, cache_dir=self.cache_dir
        )
        if identity.sha256 != self.identity.sha256:
            raise Q5IntegrationError("sampled B6 cache identity changed during resolution")
        self.memory[self.identity.sha256] = (result, status)
        return result, status

    def level_evidence(self, context: DiagnosticExecutionContext) -> DiagnosticLevelEvidence:
        entry = context.plan_entry
        if entry is None or entry.intervention_id != B6_D0_ID:
            raise Q5IntegrationError("sampled Q5 B6 runner invoked outside the B6 plan entry")
        result, cache_status = self.resolve()
        evidence = result.evidence
        if not result.promotable_reference:
            raise Q5IntegrationError(
                "DYNAMIC_REFERENCE_FAIL_CLOSED: " + str(evidence.get("aggregate_status"))
            )
        r0_rate = float(evidence["runs"]["R0"]["run"]["measurement"]["service_rate_per_min"])
        return DiagnosticLevelEvidence(
            context.level,
            {
                "fixed_window_service_yield_per_hour": 60.0 * r0_rate,
                "reference_mode": "DYNAMIC_SAME_PHYSICS",
                "scope_status": self.bound.scope_status,
                "physics_identity_sha256": self.bound.physics_identity_sha256,
                "custom_physics_mode": self.bound.mode,
                "custom_physics_plan_identity_sha256": self.bound.plan_identity_sha256,
                "dynamic_reference_evidence_sha256": result.evidence_sha256,
                "dynamic_reference_cache_identity_sha256": self.identity.sha256,
                "dynamic_reference_cache_status": cache_status,
                "dynamic_reference_evidence": dict(evidence),
                "a70_reference_evidence_sha256": A70_REFERENCE_EVIDENCE_SHA256,
                "frozen_a70_reference_value_reused": False,
                "promotable_reference": True,
                "robustness_qualified": False,
                "aggregate_status": evidence["aggregate_status"],
                "deployment_b4_executed": False,
                "deployment_b5_executed": False,
                "field_calibrated": False,
                "deployment_recommendation": False,
                "global_optimum_claim": False,
                "reference_causal": False,
            },
            intervention_id=entry.intervention_id,
            target_factor=entry.target_factor,
            causal=False,
            reason="DYNAMIC_SAME_PHYSICS_SAMPLED_A70_REFERENCE_MECHANISMS",
        )


def _sampled_execution_identity(bound: BoundCustomPhysicsScenario) -> DiagnosticExecutionIdentity:
    base = _build_execution_identity(
        bound.prepared.physical,
        bound.prepared.config.config_sha256,
    )
    components = json.loads(json.dumps(dict(base.components)))
    components["psp_profile"]["contract_sha256"] = bound.profile.contract_sha256
    canonical_profile = bound.profile.canonical_dict()
    components["slot_solver_parameters"] = canonical_profile["slot_solver"]
    components["b5_fairness_parameters"] = canonical_profile["b5_fairness"]
    components["b6_reference_parameters"] = canonical_profile["b6_reference"]
    components["dynamic_physics"] = {
        "scope_status": bound.scope_status,
        "physics_identity_sha256": bound.physics_identity_sha256,
        "custom_physics_mode": bound.mode,
        "custom_physics_plan_identity_sha256": bound.plan_identity_sha256,
        "binding_algorithm_id": bound.binding_algorithm_id,
        "reference_runtime_binding_id": bound.prepared.reference_runtime.binding_id,
        "reference_runtime_binding_sha256": bound.prepared.reference_runtime.identity_sha256,
        "frozen_a70_reference_value_reused": False,
        "robustness_qualified": False,
    }
    components["custom_physics_binding"] = dict(bound.identity_material)
    components["q5_integration_contract_sha256"] = integration_contract_sha256(bound.profile)
    return DiagnosticExecutionIdentity(SAMPLED_Q5_EXECUTION_IDENTITY_ID, components)


def _config_builder(bound: BoundCustomPhysicsScenario):
    expected_config_sha = bound.prepared.config.config_sha256
    expected_base_physics_sha = bound.prepared.physical.physics_identity_sha256

    def build(item):
        candidate = prepare_q5_scenario(item)
        if candidate.config.config_sha256 != expected_config_sha:
            raise Q5IntegrationError("sampled Q5 scenario/config identity changed after registry construction")
        if candidate.physical.physics_identity_sha256 != expected_base_physics_sha:
            raise Q5IntegrationError("sampled Q5 base physical identity changed after registry construction")
        return candidate.config

    return build


def _b4_factory(bound: BoundCustomPhysicsScenario):
    def factory(context: DiagnosticExecutionContext, runtime_context: object):
        if not isinstance(runtime_context, NextgenFIFOControllerRuntimeContext):
            raise Q5IntegrationError("sampled Q5 B4 received an unsupported D0 runtime context")
        return B4EarliestSafeSlotController(bound.profile, center_observation_provider=lambda: ())

    return factory


def _b5_factory(bound: BoundCustomPhysicsScenario):
    def factory(context: DiagnosticExecutionContext, runtime_context: object):
        if not isinstance(runtime_context, NextgenFIFOControllerRuntimeContext):
            raise Q5IntegrationError("sampled Q5 B5 received an unsupported D0 runtime context")
        return B5BoundedSlotCostController(bound.profile, center_observation_provider=lambda: ())

    return factory


@dataclass(frozen=True, slots=True)
class _PlanBinding:
    distribution_plan: PhysicsDistributionPlan | None
    fleet_mixture_plan: FleetMixturePlan | None



def _materialize(bound, plans: _PlanBinding, config):
    return materialize_bound_traffic(
        bound,
        config=config,
        distribution_plan=plans.distribution_plan,
        fleet_mixture_plan=plans.fleet_mixture_plan,
    )


def _run_baseline(
    context: DiagnosticExecutionContext,
    *,
    bindings: NextgenFIFOBindingRegistry,
    bound: BoundCustomPhysicsScenario,
    plans: _PlanBinding,
) -> DiagnosticLevelEvidence:
    if context.level is not ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK:
        raise NextgenFIFOExecutionError("sampled baseline runner accepts B0 only")
    if context.plan_entry is not None or context.previous_level is not None or context.previous_runs:
        raise NextgenFIFOExecutionError("B0 execution context must not contain prior ladder evidence")
    config = bindings.config_builder(context.scenario)
    if not isinstance(config, EnvironmentalNetworkFIFOExperimentConfig):
        raise NextgenFIFOExecutionError("sampled Q5 config builder returned unsupported config")
    seeds = replication_seeds(base_seed=config.scenario.seed, count=context.scenario.experiment.replications)
    replications: list[DiagnosticReplicationEvidence] = []
    for index, seed in enumerate(seeds):
        child = replace(config, scenario=replace(config.scenario, seed=seed))
        traffic = _materialize(bound, plans, child)
        result = run_environmental_network_fifo_experiment_with_traffic(child, traffic)
        replications.append(_baseline_replication_evidence(result, replication_index=index))
    return DiagnosticLevelEvidence(
        context.level,
        {
            "mean_service_yield_per_hour": fmean(item.primary_kpi for item in replications),
            "replication_count": len(replications),
            "seed_schedule": list(seeds),
            "physics_identity_sha256": bound.physics_identity_sha256,
            "scope_status": bound.scope_status,
        },
        tuple(replications),
        causal=False,
        reason="CANONICAL_NEXTGEN_FIFO_SAMPLED_PHYSICS_BASELINE",
    )


def _run_managed_replication(
    config: EnvironmentalNetworkFIFOExperimentConfig,
    *,
    bound: BoundCustomPhysicsScenario,
    plans: _PlanBinding,
    controller_factory,
    context: DiagnosticExecutionContext,
    replication_index: int,
) -> DiagnosticReplicationEvidence:
    scenario = config.scenario
    traffic = _materialize(bound, plans, config)
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
        raise NextgenFIFOExecutionError("registered sampled controller factory returned None")
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
            "execution_path": "canonical_source_fifo_runtime_with_sampled_physics_and_registered_controller",
            "physics_identity_sha256": bound.physics_identity_sha256,
        },
    )


def _run_managed(
    context: DiagnosticExecutionContext,
    *,
    bindings: NextgenFIFOBindingRegistry,
    bound: BoundCustomPhysicsScenario,
    plans: _PlanBinding,
) -> DiagnosticLevelEvidence:
    entry = context.plan_entry
    if entry is None or entry.disposition is not InterventionDisposition.APPLICABLE:
        raise NextgenFIFOExecutionError("sampled managed runner requires one APPLICABLE plan entry")
    if context.level is not entry.level:
        raise NextgenFIFOExecutionError("sampled managed context level disagrees with plan entry")
    applied = context.applied_intervention_ids
    if not applied or applied[-1] != entry.intervention_id:
        raise NextgenFIFOExecutionError("sampled managed execution context does not terminate in current intervention")
    if len(applied) == 1:
        controller_factory = bindings.intervention(entry.intervention_id).controller_factory
        reason = "REGISTERED_CANONICAL_SOURCE_ADMISSION_CONTROLLER_SAMPLED_PHYSICS"
    else:
        controller_factory = bindings.composition(applied).controller_factory
        reason = "REGISTERED_CANONICAL_CUMULATIVE_SOURCE_ADMISSION_CONTROLLER_SAMPLED_PHYSICS"
    base_config = bindings.config_builder(context.scenario)
    seeds = replication_seeds(
        base_seed=base_config.scenario.seed,
        count=context.scenario.experiment.replications,
    )
    replications: list[DiagnosticReplicationEvidence] = []
    for index, seed in enumerate(seeds):
        child = replace(base_config, scenario=replace(base_config.scenario, seed=seed))
        replications.append(
            _run_managed_replication(
                child,
                bound=bound,
                plans=plans,
                controller_factory=controller_factory,
                context=context,
                replication_index=index,
            )
        )
    if context.previous_level is not None:
        prior_ids = tuple(item.replication_id for item in context.previous_level.replications)
        current_ids = tuple(item.replication_id for item in replications)
        if prior_ids and prior_ids != current_ids:
            raise NextgenFIFOExecutionError("sampled managed replication IDs do not match prior ladder level")
    return DiagnosticLevelEvidence(
        context.level,
        {
            "mean_service_yield_per_hour": fmean(item.primary_kpi for item in replications),
            "replication_count": len(replications),
            "seed_schedule": list(seeds),
            "physics_identity_sha256": bound.physics_identity_sha256,
            "scope_status": bound.scope_status,
        },
        tuple(replications),
        intervention_id=entry.intervention_id,
        target_factor=entry.target_factor,
        causal=True,
        reason=reason,
    )


def build_sampled_q5_bindings(
    bound: BoundCustomPhysicsScenario,
    *,
    distribution_plan: PhysicsDistributionPlan | None = None,
    fleet_mixture_plan: FleetMixturePlan | None = None,
    reference_cache_dir: Path | None = None,
) -> NextgenFIFOBindingRegistry:
    if bound.mode == "fixed":
        raise Q5IntegrationError("sampled Q5 bindings are for distribution/fleet-mixture modes only")
    # Trigger plan/binding identity validation before constructing an execution adapter.
    plans = _PlanBinding(distribution_plan, fleet_mixture_plan)
    _materialize(bound, plans, bound.prepared.config)
    provider = _BoundReferenceProvider(bound, reference_cache_dir)
    return NextgenFIFOBindingRegistry(
        config_builder=_config_builder(bound),
        interventions=(NextgenFIFOInterventionBinding(B4_D0_ID, _b4_factory(bound)),),
        compositions=(
            NextgenFIFOCompositionBinding((B4_D0_ID, B5_D0_ID), _b5_factory(bound)),
        ),
        references=(NextgenFIFOReferenceBinding(B6_D0_ID, provider.level_evidence),),
        execution_identity=_sampled_execution_identity(bound),
    )


def build_sampled_q5_registry(
    bound: BoundCustomPhysicsScenario,
    *,
    distribution_plan: PhysicsDistributionPlan | None = None,
    fleet_mixture_plan: FleetMixturePlan | None = None,
    reference_cache_dir: Path | None = None,
) -> DiagnosticAdapterRegistry:
    bindings = build_sampled_q5_bindings(
        bound,
        distribution_plan=distribution_plan,
        fleet_mixture_plan=fleet_mixture_plan,
        reference_cache_dir=reference_cache_dir,
    )
    plans = _PlanBinding(distribution_plan, fleet_mixture_plan)

    def runner(context: DiagnosticExecutionContext) -> DiagnosticLevelEvidence:
        if context.level is ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK:
            return _run_baseline(context, bindings=bindings, bound=bound, plans=plans)
        if context.plan_entry is not None and context.plan_entry.disposition is InterventionDisposition.APPLICABLE:
            return _run_managed(context, bindings=bindings, bound=bound, plans=plans)
        if context.plan_entry is not None and context.plan_entry.disposition is InterventionDisposition.REFERENCE:
            return run_nextgen_fifo_reference_level(context, bindings=bindings)
        raise NextgenFIFOExecutionError("no sampled Q5 runtime path for requested diagnostic level")

    adapter = build_nextgen_fifo_execution_adapter(bindings, runner=runner)
    return DiagnosticAdapterRegistry((adapter,))
