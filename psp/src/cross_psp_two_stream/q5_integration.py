"""Profile-owned Q5 integration into the frozen CROSS D0 diagnostic framework.

This module binds the already-qualified PSP mechanisms to D0 extension points.  It does
not modify or register generic D0 defaults.  B4/B5 deployment execution remains causal;
B6 is a separate frozen non-causal A70 reference evidence binding.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib.resources import as_file, files
import json
from math import hypot
from types import MappingProxyType
from pathlib import Path
from typing import Any, Mapping

from vessel_crossing.diagnostic.contract import DiagnosticScenarioSpec, load_diagnostic_scenario
from vessel_crossing.diagnostic.evidence import DiagnosticLevelEvidence
from vessel_crossing.diagnostic.execution import (
    DiagnosticAdapterRegistry,
    DiagnosticExecutionContext,
    DiagnosticExecutionIdentity,
)
from vessel_crossing.diagnostic.nextgen_fifo_bindings import (
    NEXTGEN_ENVIRONMENTAL_NETWORK_FIFO_ADAPTER_ID,
    NextgenFIFOBindingRegistry,
    NextgenFIFOCompositionBinding,
    NextgenFIFOInterventionBinding,
    NextgenFIFOReferenceBinding,
    build_nextgen_fifo_execution_adapter,
)
from vessel_crossing.diagnostic.nextgen_fifo_execution import (
    NextgenFIFOControllerRuntimeContext,
    build_nextgen_fifo_runtime_runner,
)
from vessel_crossing.diagnostic.nextgen_fifo_translation import (
    NextgenFIFOSemanticsProfile,
    NextgenFIFOSemanticsProfileRegistry,
    translate_nextgen_fifo_config,
)
from vessel_crossing.nextgen.avoidance import AvoidanceCandidateConfig
from vessel_crossing.nextgen.candidate_generation import IntentDrivenCandidatePolicy
from vessel_crossing.nextgen.encounter_registry import EncounterDiscoveryConfig
from vessel_crossing.nextgen.environmental_fleet_guarded_joint_control import EnvironmentalFleetGuardedJointConfig
from vessel_crossing.nextgen.environmental_fleet_guarded_pair_control import EnvironmentalFleetGuardedPairConfig
from vessel_crossing.nextgen.environmental_fleet_multi_owner_recapture import EnvironmentalFleetMultiOwnerRecaptureConfig
from vessel_crossing.nextgen.environmental_fleet_runtime import EnvironmentalFleetRuntimeConfig
from vessel_crossing.nextgen.environmental_joint_control_cycle import EnvironmentalJointControlCycleConfig
from vessel_crossing.nextgen.environmental_joint_execution_session import EnvironmentalJointExecutionSessionConfig
from vessel_crossing.nextgen.environmental_joint_physics import EnvironmentalJointPhysicsConfig
from vessel_crossing.nextgen.environmental_joint_replanning import EnvironmentalJointReplanningConfig
from vessel_crossing.nextgen.environmental_joint_semantics import EnvironmentalJointSemanticConfig
from vessel_crossing.nextgen.environmental_nominal_route_session import EnvironmentalNominalRouteSessionConfig
from vessel_crossing.nextgen.environmental_pair_candidate import EnvironmentalPairCandidateConfig
from vessel_crossing.nextgen.environmental_pair_clearance import EnvironmentalPairTransitionConfig
from vessel_crossing.nextgen.environmental_pair_control_cycle import EnvironmentalPairControlCycleConfig
from vessel_crossing.nextgen.environmental_pair_execution_session import EnvironmentalPairExecutionSessionConfig
from vessel_crossing.nextgen.environmental_pair_replanning import EnvironmentalPairReplanningConfig
from vessel_crossing.nextgen.environmental_route_handoff_session import EnvironmentalRouteHandoffSessionConfig
from vessel_crossing.nextgen.environmental_network_fifo_experiment import EnvironmentalNetworkFIFOAdmissionPolicy
from vessel_crossing.nextgen.environmental_network_scenario import EnvironmentalNetworkIngressTimingPolicy
from vessel_crossing.nextgen.environmental_network_source_recheck import EnvironmentalNetworkSourceRecheckConfig
from vessel_crossing.nextgen.joint_arbiter import JointArbitrationConfig
from vessel_crossing.nextgen.lifecycle import EncounterLifecycleConfig
from vessel_crossing.nextgen.observation import BearingTrendConfig
from vessel_crossing.nextgen.pair_arbiter import PairArbitrationConfig
from vessel_crossing.nextgen.risk import RiskGateConfig

from .b4 import B4_POLICY_ALGORITHM_ID, B4EarliestSafeSlotController
from .b5 import B5_POLICY_ALGORITHM_ID, B5BoundedSlotCostController
from .b6 import B6_REFERENCE_ALGORITHM_ID, B6RuntimeBinding, run_balanced_saturated_reference
from .contract import (
    B4_POLICY_CONTRACT_ID,
    B5_POLICY_CONTRACT_ID,
    B6_REFERENCE_CONTRACT_ID,
    PROFILE_VERSION,
    REFERENCE_ARBITER_CONTRACT_ID,
    SLOT_SOLVER_CONTRACT_ID,
    TwoStreamProfileContract,
    load_profile_contract,
)
from .dynamic_physics import (
    NormalizedPhysicalScenario,
    build_normalized_physical_scenario,
    target_shares_from_metadata,
)
from .reference_arbiter import REFERENCE_ARBITER_ALGORITHM_ID
from .reference_runtime import build_a70_reference_runtime_binding
from .reference_cache import (
    DynamicReferenceCache,
    build_reference_cache_identity,
    resolve_dynamic_reference,
)
from .solver import SOLVER_ALGORITHM_ID

Q5_CANDIDATE_VERSION = "0.1.0a8"
Q5_EXECUTION_IDENTITY_ID = "cross-psp-two-stream-q5-integration-v1"
Q5_SEMANTICS_PROFILE_ID = "cross-psp-q5-zero-drift-fifo-v1"
Q5_CENTER_ANCHOR_MODE = "canonical-admission-prediction-only-v1"
D0_BASE_VERSION = "1.1.0a36"
D0_WHEEL_SHA256 = "4d7ad98584dfb574d85597709396ad7b46055202d6817a872b519809370c1876"
A69_SPEC_SHA256 = "03829b884b3533366bc040330db298be5e19865a131ebfef953874b3721a592d"
A70_IMPLEMENTATION_MANIFEST_SHA256 = "e50078c8f5c4bcd0933501b4b23ee455566025d5d79e6d45573e635a5744b9f1"
A70_WHEEL_SHA256 = "e39bbd565530784af39903bb9d14aac3bfb88176ea9afbc8a02a1af7b27d34fd"
A70_REFERENCE_EVIDENCE_SHA256 = "f20aa579eb3144ffe3b2e6f20d56bb7d7982fdd4173f281f8348bd4bf5fa6e17"
A70_REFERENCE_PROFILE_CONTRACT_SHA256 = "e8ddfb717dae0df148bf477bc7e232c8534c5d690ed0b4ac5641d3e7cd149892"
B4_D0_ID = "global_slot_coordination"
B5_D0_ID = "type_order_staging"
B6_D0_ID = "physical_balanced_reference"
_EPS = 1e-9


class Q5IntegrationError(ValueError):
    """Raised when the frozen Q5 integration contract cannot be satisfied."""


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def load_qualified_profile() -> TwoStreamProfileContract:
    path = files("cross_psp_two_stream.data").joinpath("profile_a70.yaml")
    with path.open("rb") as stream:
        payload = stream.read()
    # load_profile_contract owns schema validation; materialize only inside the package path.
    return load_profile_contract(path)


def _recapture_config() -> EnvironmentalFleetMultiOwnerRecaptureConfig:
    pair_transition = EnvironmentalPairTransitionConfig(10.0, 0.0)
    pair_cycle = EnvironmentalPairControlCycleConfig(
        execution=EnvironmentalPairExecutionSessionConfig(review_interval_min=0.75),
        replanning=EnvironmentalPairReplanningConfig(
            candidate_policy=IntentDrivenCandidatePolicy(AvoidanceCandidateConfig((30.0, 60.0), (0.8, 1.0), 10.0, 0.0)),
            candidate_config=EnvironmentalPairCandidateConfig(pair_transition),
            arbitration_config=PairArbitrationConfig.least_intervention(),
        ),
        route_handoff=EnvironmentalRouteHandoffSessionConfig(
            review_interval_min=0.1,
            recovery_lookahead_nm=0.15,
            cross_track_tolerance_nm=0.015,
            ground_velocity_tolerance=1e-5,
        ),
        nominal_route=EnvironmentalNominalRouteSessionConfig(
            review_interval_min=0.1,
            route_lookahead_nm=0.15,
            cross_track_tolerance_nm=0.015,
        ),
        max_replans=16,
        max_route_events_per_run=4096,
    )
    joint_transition = EnvironmentalPairTransitionConfig(12.0, 0.0)
    joint_cycle = EnvironmentalJointControlCycleConfig(
        execution=EnvironmentalJointExecutionSessionConfig(review_interval_min=1.5),
        replanning=EnvironmentalJointReplanningConfig(
            semantic_config=EnvironmentalJointSemanticConfig(
                candidate_policy=IntentDrivenCandidatePolicy(
                    AvoidanceCandidateConfig((30.0, 60.0), (1.0,), 12.0, 0.0, substantial_course_change_deg=10.0, substantial_speed_change_fraction=0.1)
                ),
                physics=EnvironmentalJointPhysicsConfig(joint_transition, max_combinations=100),
            ),
            arbitration_config=JointArbitrationConfig.least_intervention(),
        ),
        max_replans=16,
    )
    return EnvironmentalFleetMultiOwnerRecaptureConfig(
        pair_cycle=pair_cycle,
        pair_fleet_guard=EnvironmentalFleetGuardedPairConfig(0.75),
        joint_cycle=joint_cycle,
        joint_fleet_guard=EnvironmentalFleetGuardedJointConfig(0.75),
    )


_SEMANTICS_MANIFEST = MappingProxyType({
    "semantics_profile_id": Q5_SEMANTICS_PROFILE_ID,
    "discovery": {"current_distance_nm": 0.2, "dcpa_distance_nm": 0.01, "tcpa_horizon_min": 1.0},
    "lifecycle": {
        "risk_gate": {"dcpa_threshold_nm": 0.5, "tcpa_horizon_min": 30.0},
        "bearing_trend": {"min_duration_min": 2.0, "appreciable_change_deg": 3.0},
        "clear_observations_required": 2,
        "overtaking_clear_distance_nm": 0.5,
        "max_observations": 8,
    },
    "recapture_configuration_id": "cross-r1-generic-frozen-controller-config-v1",
    "runtime_configuration": "EnvironmentalFleetRuntimeConfig-default-v1",
    "source_recheck": {"time_tolerance_min": 0.01, "max_subdivisions": 24},
    "admission_policy": "ARM_LOCAL_FIFO_SHIP_DOMAIN",
    "ingress_timing_policy": "DEMAND_TIME_IS_PHYSICAL_INGRESS",
})


def semantics_manifest() -> dict[str, object]:
    return json.loads(json.dumps(dict(_SEMANTICS_MANIFEST), sort_keys=True))


def semantics_manifest_sha256() -> str:
    return _sha(semantics_manifest())


def build_q5_semantics_profiles() -> NextgenFIFOSemanticsProfileRegistry:
    lifecycle = EncounterLifecycleConfig(
        risk_gate=RiskGateConfig(0.5, 30.0),
        bearing_trend=BearingTrendConfig(min_duration_min=2.0, appreciable_change_deg=3.0),
        clear_observations_required=2,
        overtaking_clear_distance_nm=0.5,
        max_observations=8,
    )
    profile = NextgenFIFOSemanticsProfile(
        Q5_SEMANTICS_PROFILE_ID,
        EncounterDiscoveryConfig(0.2, 0.01, 1.0),
        lifecycle,
        _recapture_config(),
        EnvironmentalFleetRuntimeConfig(),
        EnvironmentalNetworkSourceRecheckConfig(),
        EnvironmentalNetworkFIFOAdmissionPolicy.ARM_LOCAL_FIFO_SHIP_DOMAIN,
        EnvironmentalNetworkIngressTimingPolicy.DEMAND_TIME_IS_PHYSICAL_INGRESS,
    )
    return NextgenFIFOSemanticsProfileRegistry((profile,))




def integration_contract_manifest(profile: TwoStreamProfileContract) -> dict[str, object]:
    return {
        "schema": "cross-psp-q5-integration-contract-v1",
        "candidate_version": Q5_CANDIDATE_VERSION,
        "execution_identity_id": Q5_EXECUTION_IDENTITY_ID,
        "profile_contract_sha256": profile.contract_sha256,
        "d0_version": D0_BASE_VERSION,
        "d0_wheel_sha256": D0_WHEEL_SHA256,
        "runtime_adapter_id": NEXTGEN_ENVIRONMENTAL_NETWORK_FIFO_ADAPTER_ID,
        "b4": {"d0_id": B4_D0_ID, "algorithm_id": B4_POLICY_ALGORITHM_ID},
        "b4_b5": {"sequence": [B4_D0_ID, B5_D0_ID], "algorithm_id": B5_POLICY_ALGORITHM_ID},
        "b6": {"d0_id": B6_D0_ID, "algorithm_id": B6_REFERENCE_ALGORITHM_ID, "reference_arbiter_algorithm_id": REFERENCE_ARBITER_ALGORITHM_ID},
        "center_anchor_mode": Q5_CENTER_ANCHOR_MODE,
        "semantics_manifest_sha256": semantics_manifest_sha256(),
        "a69_spec_sha256": A69_SPEC_SHA256,
        "a70_implementation_manifest_sha256": A70_IMPLEMENTATION_MANIFEST_SHA256,
        "a70_reference_evidence_sha256": A70_REFERENCE_EVIDENCE_SHA256,
    }


def integration_contract_sha256(profile: TwoStreamProfileContract) -> str:
    return _sha(integration_contract_manifest(profile))


def _load_a70_reference_evidence() -> dict[str, Any]:
    path = files("cross_psp_two_stream.data").joinpath("a70_q4_symmetric_aggregate_evidence.json")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != A70_REFERENCE_EVIDENCE_SHA256:
        raise Q5IntegrationError("packaged A70 reference evidence hash mismatch")
    payload = json.loads(raw)
    if payload.get("promotable_reference") is not True:
        raise Q5IntegrationError("A70 reference evidence is not promotable")
    if payload.get("aggregate_status") != "PASS_BALANCED_SATURATED_REFERENCE":
        raise Q5IntegrationError("A70 reference aggregate status is not qualified")
    if payload.get("b6_spec_sha256") != A69_SPEC_SHA256:
        raise Q5IntegrationError("A70 reference spec identity mismatch")
    if payload.get("implementation_manifest_sha256") != A70_IMPLEMENTATION_MANIFEST_SHA256:
        raise Q5IntegrationError("A70 reference implementation identity mismatch")
    if payload.get("profile_contract_sha256") != A70_REFERENCE_PROFILE_CONTRACT_SHA256:
        raise Q5IntegrationError("A70 reference profile identity mismatch")
    for key in ("causal", "field_calibrated", "deployment_recommendation", "global_optimum_claim"):
        if payload.get(key) is not False:
            raise Q5IntegrationError(f"A70 reference claim boundary drift: {key}")
    if payload.get("deployment_b4_executed") is not False or payload.get("deployment_b5_executed") is not False:
        raise Q5IntegrationError("A70 reference illegally executes deployment B4/B5")
    return payload


@dataclass(frozen=True, slots=True)
class PreparedQ5Scenario:
    """Validated translation plus one canonical same-physics reference context."""

    config: object
    physical: NormalizedPhysicalScenario
    reference_runtime: B6RuntimeBinding


@lru_cache(maxsize=1)
def qualified_q5_physics_identity_sha256() -> str:
    """Identity of the frozen A72 Q5 physical scenario under the A81 normalization."""
    profile = load_qualified_profile()
    runtime = build_a70_reference_runtime_binding()
    resource = files("cross_psp_two_stream.data").joinpath("q5_scenario.yaml")
    with as_file(resource) as path:
        scenario = load_diagnostic_scenario(path)
    config = translate_nextgen_fifo_config(scenario, profiles=build_q5_semantics_profiles())
    defaults = {item.stream_id: float(item.share) for item in profile.b6_reference.stream_shares}
    shares = target_shares_from_metadata(
        scenario.metadata,
        tuple(item.stream_id for item in profile.streams),
        defaults,
    )
    physical = build_normalized_physical_scenario(
        config,
        profile,
        runtime,
        target_shares=shares,
    )
    return physical.physics_identity_sha256


def prepare_q5_scenario(scenario: DiagnosticScenarioSpec) -> PreparedQ5Scenario:
    """Translate and fail closed before any expensive diagnostic execution begins."""
    if scenario.model_adapters.get("runtime") != NEXTGEN_ENVIRONMENTAL_NETWORK_FIFO_ADAPTER_ID:
        raise Q5IntegrationError("Q5 runtime adapter identity mismatch")
    params = scenario.model_parameters
    if params.get("semantics_profile") != Q5_SEMANTICS_PROFILE_ID:
        raise Q5IntegrationError("Q5 semantics profile identity mismatch")

    profile = load_qualified_profile()
    runtime = build_a70_reference_runtime_binding()
    stream_ids = tuple(item.stream_id for item in profile.streams)
    defaults = {item.stream_id: float(item.share) for item in profile.b6_reference.stream_shares}
    try:
        config = translate_nextgen_fifo_config(scenario, profiles=build_q5_semantics_profiles())
        shares = target_shares_from_metadata(scenario.metadata, stream_ids, defaults)
        physical = build_normalized_physical_scenario(
            config,
            profile,
            runtime,
            target_shares=shares,
            qualified_physics_identity_sha256=qualified_q5_physics_identity_sha256(),
            # A70's portable 85/h evidence is a 0.15-nm q4 qualification fixture,
            # not the A72 Q5 product physics.  RC2 custom analysis therefore never
            # reuses that scalar merely because it is packaged.
            frozen_a70_physics_identity_sha256=None,
        )
    except ValueError as exc:
        raise Q5IntegrationError(str(exc)) from exc
    return PreparedQ5Scenario(config=config, physical=physical, reference_runtime=runtime)


def _build_execution_identity(
    physical: NormalizedPhysicalScenario,
    canonical_config_sha256: str,
) -> DiagnosticExecutionIdentity:
    profile = physical.profile
    evidence = _load_a70_reference_evidence()
    components = {
        "psp_profile": {
            "profile_id": profile.profile_id,
            "schema": profile.schema,
            "profile_version": profile.profile_version,
            "contract_sha256": profile.contract_sha256,
        },
        "psp_package_candidate_version": Q5_CANDIDATE_VERSION,
        "d0": {"version": D0_BASE_VERSION, "wheel_sha256": D0_WHEEL_SHA256},
        "runtime_adapter_id": NEXTGEN_ENVIRONMENTAL_NETWORK_FIFO_ADAPTER_ID,
        "scope": {
            "route_model_id": profile.route_model_id,
            "environment_model_id": profile.environment_model_id,
            "planned_slot_prediction_scope_id": profile.planned_slot_prediction_scope_id,
        },
        "components": {
            "a64": {"contract_id": SLOT_SOLVER_CONTRACT_ID, "algorithm_id": SOLVER_ALGORITHM_ID},
            "a65_b4": {"contract_id": B4_POLICY_CONTRACT_ID, "algorithm_id": B4_POLICY_ALGORITHM_ID},
            "a66_b5": {"contract_id": B5_POLICY_CONTRACT_ID, "algorithm_id": B5_POLICY_ALGORITHM_ID},
            "a70_b6": {"contract_id": B6_REFERENCE_CONTRACT_ID, "algorithm_id": B6_REFERENCE_ALGORITHM_ID},
            "a70_reference_arbiter": {
                "contract_id": REFERENCE_ARBITER_CONTRACT_ID,
                "algorithm_id": REFERENCE_ARBITER_ALGORITHM_ID,
            },
        },
        "d0_bindings": {
            "b4": B4_D0_ID,
            "b4_b5_sequence": [B4_D0_ID, B5_D0_ID],
            "b6_reference": B6_D0_ID,
        },
        "center_anchor_mode": Q5_CENTER_ANCHOR_MODE,
        "slot_solver_parameters": profile.canonical_dict()["slot_solver"],
        "b5_fairness_parameters": profile.canonical_dict()["b5_fairness"],
        "b6_reference_parameters": profile.canonical_dict()["b6_reference"],
        "dynamic_physics": {
            "scope_status": physical.scope_status,
            "physics_identity_sha256": physical.physics_identity_sha256,
            "reference_runtime_binding_id": physical.identity_material["reference_runtime"]["binding_id"],
            "reference_runtime_binding_sha256": physical.identity_material["reference_runtime"]["binding_sha256"],
            "frozen_a70_reference_value_reused": False,
        },
        "semantics_profile": {
            "profile_id": Q5_SEMANTICS_PROFILE_ID,
            "manifest": semantics_manifest(),
            "manifest_sha256": semantics_manifest_sha256(),
        },
        "translated_canonical_config_sha256": canonical_config_sha256,
        "a69_spec_sha256": A69_SPEC_SHA256,
        "a70_implementation_manifest_sha256": A70_IMPLEMENTATION_MANIFEST_SHA256,
        "a70_qualified_wheel_sha256": A70_WHEEL_SHA256,
        # Provenance only: RC2 custom analysis does not use its 85/h scalar.
        "a70_reference_evidence_sha256": A70_REFERENCE_EVIDENCE_SHA256,
        "a70_reference_profile_contract_sha256": evidence["profile_contract_sha256"],
        "a70_reference_execution_identity_sha256": evidence["reference_execution_identity_sha256"],
        "q5_integration_contract_sha256": integration_contract_sha256(profile),
    }
    return DiagnosticExecutionIdentity(Q5_EXECUTION_IDENTITY_ID, components)


class _DynamicReferenceProvider:
    """Resolve one same-physics B6 reference with verified memory/persistent caching."""

    def __init__(self, prepared: PreparedQ5Scenario, cache_dir: Path | None) -> None:
        self.prepared = prepared
        self.profile = prepared.physical.profile
        self.memory: dict[str, tuple[object, str]] = {}
        self.cache = None if cache_dir is None else DynamicReferenceCache(cache_dir)
        self.identity = build_reference_cache_identity(
            physics_identity_sha256=prepared.physical.physics_identity_sha256,
            profile=self.profile,
            runtime=prepared.reference_runtime,
            b6_spec_sha256=A69_SPEC_SHA256,
            implementation_manifest_sha256=A70_IMPLEMENTATION_MANIFEST_SHA256,
        )

    def _compute(self):
        return run_balanced_saturated_reference(
            self.profile,
            self.prepared.physical.profile_classes,
            self.prepared.reference_runtime,
            b6_spec_sha256=A69_SPEC_SHA256,
            implementation_manifest_sha256=A70_IMPLEMENTATION_MANIFEST_SHA256,
        )

    def resolve(self):
        cached = self.memory.get(self.identity.sha256)
        if cached is not None:
            result, prior_status = cached
            return result, "MEMORY_HIT_AFTER_" + prior_status
        result, status = resolve_dynamic_reference(self.cache, self.identity, self._compute)
        self.memory[self.identity.sha256] = (result, status)
        return result, status

    def level_evidence(self, context: DiagnosticExecutionContext) -> DiagnosticLevelEvidence:
        entry = context.plan_entry
        if entry is None or entry.intervention_id != B6_D0_ID:
            raise Q5IntegrationError("Q5 B6 runner invoked outside the B6 plan entry")
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
                "reference_model": B6_REFERENCE_ALGORITHM_ID,
                "reference_arbiter": REFERENCE_ARBITER_ALGORITHM_ID,
                "reference_mode": "DYNAMIC_SAME_PHYSICS",
                "scope_status": self.prepared.physical.scope_status,
                "physics_identity_sha256": self.prepared.physical.physics_identity_sha256,
                "dynamic_reference_evidence_sha256": result.evidence_sha256,
                "dynamic_reference_cache_identity_sha256": self.identity.sha256,
                "dynamic_reference_cache_status": cache_status,
                "dynamic_reference_evidence": dict(evidence),
                "a70_reference_evidence_sha256": A70_REFERENCE_EVIDENCE_SHA256,
                "frozen_a70_reference_value_reused": False,
                "promotable_reference": True,
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
            reason="DYNAMIC_SAME_PHYSICS_A70_REFERENCE_MECHANISMS",
        )


def _checked_config_builder(
    expected_config_sha: str,
    expected_physics_sha: str,
):
    def config_builder(item: DiagnosticScenarioSpec):
        candidate = prepare_q5_scenario(item)
        if candidate.config.config_sha256 != expected_config_sha:
            raise Q5IntegrationError("Q5 scenario/config identity changed after registry construction")
        if candidate.physical.physics_identity_sha256 != expected_physics_sha:
            raise Q5IntegrationError("Q5 physical identity changed after registry construction")
        return candidate.config

    return config_builder


def _b4_factory(profile: TwoStreamProfileContract):
    def factory(context: DiagnosticExecutionContext, runtime_context: object):
        if not isinstance(runtime_context, NextgenFIFOControllerRuntimeContext):
            raise Q5IntegrationError("Q5 B4 received an unsupported D0 runtime context")
        return B4EarliestSafeSlotController(profile, center_observation_provider=lambda: ())

    return factory


def _b5_factory(profile: TwoStreamProfileContract):
    def factory(context: DiagnosticExecutionContext, runtime_context: object):
        if not isinstance(runtime_context, NextgenFIFOControllerRuntimeContext):
            raise Q5IntegrationError("Q5 B5 received an unsupported D0 runtime context")
        return B5BoundedSlotCostController(profile, center_observation_provider=lambda: ())

    return factory


def build_q5_bindings(
    scenario: DiagnosticScenarioSpec,
    *,
    reference_cache_dir: Path | None = None,
) -> NextgenFIFOBindingRegistry:
    prepared = prepare_q5_scenario(scenario)
    profile = prepared.physical.profile
    provider = _DynamicReferenceProvider(prepared, reference_cache_dir)
    identity = _build_execution_identity(prepared.physical, prepared.config.config_sha256)
    config_builder = _checked_config_builder(
        prepared.config.config_sha256, prepared.physical.physics_identity_sha256
    )

    return NextgenFIFOBindingRegistry(
        config_builder=config_builder,
        interventions=(NextgenFIFOInterventionBinding(B4_D0_ID, _b4_factory(profile)),),
        compositions=(
            NextgenFIFOCompositionBinding((B4_D0_ID, B5_D0_ID), _b5_factory(profile)),
        ),
        references=(NextgenFIFOReferenceBinding(B6_D0_ID, provider.level_evidence),),
        execution_identity=identity,
    )

def build_q5_registry(
    scenario: DiagnosticScenarioSpec,
    *,
    reference_cache_dir: Path | None = None,
) -> DiagnosticAdapterRegistry:
    bindings = build_q5_bindings(scenario, reference_cache_dir=reference_cache_dir)
    adapter = build_nextgen_fifo_execution_adapter(bindings, runner=build_nextgen_fifo_runtime_runner(bindings))
    return DiagnosticAdapterRegistry((adapter,))
