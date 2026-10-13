"""Bind Custom Physics v2 declarations to the frozen supported simulator seams.

This module is an orchestration layer.  It does not alter D0 physics, B4/B5 control,
B6 mechanics, or the frozen Q5 fixed-physics identity.  Distribution and fleet-mixture
plans are materialized as deterministic per-arrival vessel profiles through D0's public
pre-realized traffic seam and as deterministic B6 supply cycles.  Their binding identity
is separate from the fixed/default identity so evidence/cache reuse remains fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from vessel_crossing.kinematics import knots_to_nm_per_minute, nm_per_minute_to_knots
from vessel_crossing.nextgen import (
    ManeuverLimits,
    NominalVesselArrival,
    PoissonNetworkTrafficRealization,
    ShipDomain,
    VesselClassProfile,
    VesselDimensions,
    VesselSpeedEnvelope,
    generate_poisson_network_traffic,
    run_environmental_network_fifo_experiment_with_traffic,
)

from .contract import TwoStreamProfileContract, profile_contract_from_mapping
from .dynamic_physics import QUALIFIED_PROFILE, SUPPORTED_EXPLORATORY
from .fleet_mixtures import (
    FleetMixturePlan,
    FleetMixtureSample,
    VesselClassSpec,
    sample_fleet_mixture,
)
from .physics_distributions import (
    DistributionSample,
    PhysicsDistributionPlan,
    sample_distribution_plan,
)
from .b6 import run_balanced_saturated_reference
from .q5_integration import (
    A69_SPEC_SHA256,
    A70_IMPLEMENTATION_MANIFEST_SHA256,
    PreparedQ5Scenario,
)
from .reference_cache import (
    DynamicReferenceCache,
    build_reference_cache_identity,
    resolve_dynamic_reference,
)

CUSTOM_PHYSICS_BINDING_SCHEMA = "cross-psp-custom-physics-binding-v1"
DISTRIBUTION_BINDING_ALGORITHM_ID = "cross-psp-distribution-arrival-binding-v1"
FLEET_MIXTURE_BINDING_ALGORITHM_ID = "cross-psp-fleet-mixture-arrival-binding-v1"
FIXED_BINDING_ALGORITHM_ID = "cross-psp-fixed-physics-passthrough-v1"

_MODE_FIXED = "fixed"
_MODE_DISTRIBUTIONS = "distributions"
_MODE_FLEET_MIXTURE = "fleet_mixture"
_EPS = 1e-10

_PARAMETER_UNITS = MappingProxyType({
    "length_m": "m",
    "beam_m": "m",
    "cruise_speed_knots": "knots",
    "maximum_speed_knots": "knots",
    "max_acceleration_knots_per_min": "knots/min",
    "max_deceleration_knots_per_min": "knots/min",
    "max_turn_rate_deg_per_min": "deg/min",
    "safety_ahead_nm": "nm",
    "safety_astern_nm": "nm",
    "safety_port_nm": "nm",
    "safety_starboard_nm": "nm",
})


class CustomPhysicsBindingError(ValueError):
    """Raised when a sampled declaration cannot be bound safely to the supported path."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _stream_ids(prepared: PreparedQ5Scenario) -> tuple[str, ...]:
    return tuple(stream.stream_id for stream in prepared.config.streams)


def _stream_heading(prepared: PreparedQ5Scenario, stream_id: str) -> float:
    for stream in prepared.config.streams:
        if stream.stream_id == stream_id:
            return float(prepared.config.network.resolve(stream.request).route.segment_heading(0))
    raise CustomPhysicsBindingError(f"unknown stream {stream_id!r}")


def _base_vessel_spec(prepared: PreparedQ5Scenario, stream_id: str) -> VesselClassSpec:
    for stream in prepared.config.streams:
        if stream.stream_id != stream_id:
            continue
        profile = stream.vessel_class
        if profile is None or profile.safety_domain is None or profile.speed_envelope is None:
            raise CustomPhysicsBindingError(
                f"stream {stream_id!r} requires dimensions, speed envelope, maneuver limits, and safety domain"
            )
        domain = profile.safety_domain
        return VesselClassSpec(
            class_id=profile.class_id,
            length_m=profile.dimensions.length_m,
            beam_m=profile.dimensions.beam_m,
            cruise_speed_knots=nm_per_minute_to_knots(stream.cruise_speed),
            maximum_speed_knots=profile.speed_envelope.maximum_speed_knots,
            max_acceleration_knots_per_min=nm_per_minute_to_knots(
                profile.maneuver_limits.max_acceleration
            ),
            max_deceleration_knots_per_min=nm_per_minute_to_knots(
                profile.maneuver_limits.max_deceleration
            ),
            max_turn_rate_deg_per_min=profile.maneuver_limits.max_turn_rate_deg_per_min,
            safety_ahead_nm=domain.ahead_nm,
            safety_astern_nm=domain.astern_nm,
            safety_port_nm=domain.port_nm,
            safety_starboard_nm=domain.starboard_nm,
        )
    raise CustomPhysicsBindingError(f"unknown stream {stream_id!r}")


def _runtime_profile(spec: VesselClassSpec, *, class_id: str) -> VesselClassProfile:
    return VesselClassProfile(
        class_id=class_id,
        dimensions=VesselDimensions.from_meters(length_m=spec.length_m, beam_m=spec.beam_m),
        maneuver_limits=ManeuverLimits.from_knots_per_minute(
            max_acceleration_knots_per_min=spec.max_acceleration_knots_per_min,
            max_deceleration_knots_per_min=spec.max_deceleration_knots_per_min,
            max_turn_rate_deg_per_min=spec.max_turn_rate_deg_per_min,
        ),
        safety_domain=ShipDomain(
            ahead_nm=spec.safety_ahead_nm,
            astern_nm=spec.safety_astern_nm,
            port_nm=spec.safety_port_nm,
            starboard_nm=spec.safety_starboard_nm,
        ),
        speed_envelope=VesselSpeedEnvelope.from_knots(
            maximum_speed_knots=spec.maximum_speed_knots
        ),
    )


def _replace_spec_scalar(
    base: VesselClassSpec,
    sample: DistributionSample,
    *,
    class_id: str,
) -> VesselClassSpec:
    expected_unit = _PARAMETER_UNITS.get(sample.parameter_id)
    if expected_unit is None:
        raise CustomPhysicsBindingError(
            f"unsupported distributed parameter {sample.parameter_id!r}; "
            f"supported={sorted(_PARAMETER_UNITS)}"
        )
    if sample.unit != expected_unit:
        raise CustomPhysicsBindingError(
            f"parameter {sample.parameter_id!r} requires unit {expected_unit!r}, got {sample.unit!r}"
        )
    if not isfinite(sample.value):
        raise CustomPhysicsBindingError("distribution sample must be finite")
    values = {
        "class_id": class_id,
        "length_m": base.length_m,
        "beam_m": base.beam_m,
        "cruise_speed_knots": base.cruise_speed_knots,
        "maximum_speed_knots": base.maximum_speed_knots,
        "max_acceleration_knots_per_min": base.max_acceleration_knots_per_min,
        "max_deceleration_knots_per_min": base.max_deceleration_knots_per_min,
        "max_turn_rate_deg_per_min": base.max_turn_rate_deg_per_min,
        "safety_ahead_nm": base.safety_ahead_nm,
        "safety_astern_nm": base.safety_astern_nm,
        "safety_port_nm": base.safety_port_nm,
        "safety_starboard_nm": base.safety_starboard_nm,
    }
    values[sample.parameter_id] = float(sample.value)
    try:
        return VesselClassSpec(**values)
    except ValueError as exc:
        raise CustomPhysicsBindingError(
            f"distribution sample for {sample.stream_id!r}/{sample.parameter_id!r} "
            f"resolves to an invalid vessel profile: {exc}"
        ) from exc


def _replace_arrival(
    arrival: NominalVesselArrival,
    *,
    spec: VesselClassSpec,
    class_id: str,
) -> NominalVesselArrival:
    profile = _runtime_profile(spec, class_id=class_id)
    speed = knots_to_nm_per_minute(spec.cruise_speed_knots)
    vessel = replace(arrival.vessel, speed=speed)
    follower = replace(arrival.target.navigator.follower, cruise_speed=speed)
    navigator = replace(arrival.target.navigator, follower=follower)
    target = replace(arrival.target, navigator=navigator)
    try:
        return NominalVesselArrival(
            at=arrival.at,
            vessel=vessel,
            target=target,
            limits=profile.maneuver_limits,
            vessel_class=profile,
        )
    except ValueError as exc:
        raise CustomPhysicsBindingError(
            f"resolved arrival {arrival.vessel.vessel_id!r} violates D0 vessel constraints: {exc}"
        ) from exc


@dataclass(frozen=True, slots=True)
class ResolvedVesselSample:
    stream_id: str
    sample_index: int
    source_class_id: str
    runtime_class_id: str
    profile: VesselClassSpec
    sample_identity_sha256: str

    def canonical_dict(self) -> dict[str, object]:
        return {
            "stream_id": self.stream_id,
            "sample_index": self.sample_index,
            "source_class_id": self.source_class_id,
            "runtime_class_id": self.runtime_class_id,
            "profile": self.profile.canonical_dict(),
            "sample_identity_sha256": self.sample_identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class BoundCustomPhysicsScenario:
    """A deterministic binding with a cache/evidence identity distinct from the base default."""

    prepared: PreparedQ5Scenario
    mode: str
    plan_identity_sha256: str | None
    binding_algorithm_id: str
    profile: TwoStreamProfileContract
    profile_classes: Mapping[str, VesselClassProfile]
    identity_material: Mapping[str, object]
    physics_identity_sha256: str
    scope_status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_classes", MappingProxyType(dict(self.profile_classes)))
        object.__setattr__(self, "identity_material", MappingProxyType(dict(self.identity_material)))


class _Sampler:
    def resolve(self, stream_id: str, sample_index: int) -> ResolvedVesselSample:
        raise NotImplementedError


class _DistributionSampler(_Sampler):
    def __init__(self, prepared: PreparedQ5Scenario, plan: PhysicsDistributionPlan) -> None:
        expected = set(_stream_ids(prepared))
        declared = {item.stream_id for item in plan.streams}
        unknown = declared - expected
        if unknown:
            raise CustomPhysicsBindingError(f"distribution plan contains unknown streams: {sorted(unknown)}")
        self.prepared = prepared
        self.plan = plan
        self.base = {stream_id: _base_vessel_spec(prepared, stream_id) for stream_id in expected}

    def resolve(self, stream_id: str, sample_index: int) -> ResolvedVesselSample:
        if stream_id not in self.base:
            raise CustomPhysicsBindingError(f"unknown stream {stream_id!r}")
        samples = {item.stream_id: item for item in sample_distribution_plan(self.plan, sample_index)}
        sample = samples.get(stream_id)
        if sample is None:
            # A distribution-mode plan may deliberately randomize only one stream; the
            # other stream remains the declared fixed base profile, but still receives a
            # plan-bound sample identity so evidence cannot collapse to fixed/default.
            source = self.base[stream_id]
            material = {
                "schema": CUSTOM_PHYSICS_BINDING_SCHEMA,
                "mode": _MODE_DISTRIBUTIONS,
                "plan_identity_sha256": self.plan.plan_identity_sha256,
                "stream_id": stream_id,
                "sample_index": sample_index,
                "fixed_base_profile": source.canonical_dict(),
            }
            sid = _sha(material)
            class_id = f"dist-{self.plan.plan_identity_sha256[:12]}-{stream_id}-{sample_index:06d}-{sid[:12]}"
            profile = replace(source, class_id=class_id)
            return ResolvedVesselSample(stream_id, sample_index, source.class_id, class_id, profile, sid)
        class_id = (
            f"dist-{self.plan.plan_identity_sha256[:12]}-{stream_id}-"
            f"{sample_index:06d}-{sample.sample_identity_sha256[:12]}"
        )
        profile = _replace_spec_scalar(self.base[stream_id], sample, class_id=class_id)
        return ResolvedVesselSample(
            stream_id,
            sample_index,
            self.base[stream_id].class_id,
            class_id,
            profile,
            sample.sample_identity_sha256,
        )


class _FleetMixtureSampler(_Sampler):
    def __init__(self, prepared: PreparedQ5Scenario, plan: FleetMixturePlan) -> None:
        expected = set(_stream_ids(prepared))
        declared = {item.stream_id for item in plan.streams}
        if declared != expected:
            raise CustomPhysicsBindingError(
                "fleet-mixture binding requires exactly the canonical stream IDs; "
                f"expected={sorted(expected)}, got={sorted(declared)}"
            )
        self.prepared = prepared
        self.plan = plan

    def resolve(self, stream_id: str, sample_index: int) -> ResolvedVesselSample:
        samples: dict[str, FleetMixtureSample] = {
            item.stream_id: item for item in sample_fleet_mixture(self.plan, sample_index)
        }
        try:
            sample = samples[stream_id]
        except KeyError as exc:
            raise CustomPhysicsBindingError(f"unknown stream {stream_id!r}") from exc
        runtime_id = f"mix-{self.plan.plan_identity_sha256[:12]}-{sample.profile.class_id}"
        profile = replace(sample.profile, class_id=runtime_id)
        return ResolvedVesselSample(
            stream_id,
            sample_index,
            sample.profile.class_id,
            runtime_id,
            profile,
            sample.sample_identity_sha256,
        )


def _build_sampled_profile(
    prepared: PreparedQ5Scenario,
    sampler: _Sampler,
) -> tuple[TwoStreamProfileContract, dict[str, VesselClassProfile], str]:
    base = prepared.physical.profile
    count = base.b6_reference.saturated_demands_per_stream
    payload = base.canonical_dict()
    payload.pop("component_contract_ids", None)
    templates: list[dict[str, object]] = []
    catalog: dict[str, VesselClassProfile] = {}
    manifest: list[dict[str, object]] = []
    for stream_id in _stream_ids(prepared):
        heading = _stream_heading(prepared, stream_id)
        cycle: list[dict[str, object]] = []
        for sample_index in range(count):
            resolved = sampler.resolve(stream_id, sample_index)
            runtime_profile = _runtime_profile(resolved.profile, class_id=resolved.runtime_class_id)
            previous = catalog.get(resolved.runtime_class_id)
            if previous is not None and previous != runtime_profile:
                raise CustomPhysicsBindingError(
                    f"runtime class id {resolved.runtime_class_id!r} maps to inconsistent profiles"
                )
            catalog[resolved.runtime_class_id] = runtime_profile
            cycle.append({
                "profile_class_id": resolved.runtime_class_id,
                "through_water_heading_deg": heading,
                "through_water_speed_nm_per_min": knots_to_nm_per_minute(
                    resolved.profile.cruise_speed_knots
                ),
            })
            manifest.append(resolved.canonical_dict())
        templates.append({"stream_id": stream_id, "cycle": cycle})
    payload["b6_reference"]["supply_templates"] = templates
    try:
        profile = profile_contract_from_mapping(payload)
    except ValueError as exc:
        raise CustomPhysicsBindingError(f"sampled B6 profile is invalid: {exc}") from exc
    return profile, catalog, _sha(manifest)


def _bind_sampled(
    prepared: PreparedQ5Scenario,
    *,
    mode: str,
    plan_identity_sha256: str,
    plan_schema: str,
    algorithm_id: str,
    sampler: _Sampler,
) -> BoundCustomPhysicsScenario:
    profile, catalog, manifest_sha = _build_sampled_profile(prepared, sampler)
    material = {
        "schema": CUSTOM_PHYSICS_BINDING_SCHEMA,
        "mode": mode,
        "binding_algorithm_id": algorithm_id,
        "base_fixed_physics_identity_sha256": prepared.physical.physics_identity_sha256,
        "plan_schema": plan_schema,
        "plan_identity_sha256": plan_identity_sha256,
        "sample_sequence_policy": "per_stream_zero_based_occurrence_index_v1",
        "b6_sample_count_per_stream": profile.b6_reference.saturated_demands_per_stream,
        "b6_sample_manifest_sha256": manifest_sha,
        "b6_profile_contract_sha256": profile.contract_sha256,
        "reference_runtime_binding_id": prepared.reference_runtime.binding_id,
        "reference_runtime_binding_sha256": prepared.reference_runtime.identity_sha256,
    }
    return BoundCustomPhysicsScenario(
        prepared=prepared,
        mode=mode,
        plan_identity_sha256=plan_identity_sha256,
        binding_algorithm_id=algorithm_id,
        profile=profile,
        profile_classes=catalog,
        identity_material=material,
        physics_identity_sha256=_sha(material),
        scope_status=SUPPORTED_EXPLORATORY,
    )


def bind_fixed_scenario(prepared: PreparedQ5Scenario) -> BoundCustomPhysicsScenario:
    """Expose the existing fixed path without changing its qualified/custom identity."""
    return BoundCustomPhysicsScenario(
        prepared=prepared,
        mode=_MODE_FIXED,
        plan_identity_sha256=None,
        binding_algorithm_id=FIXED_BINDING_ALGORITHM_ID,
        profile=prepared.physical.profile,
        profile_classes=prepared.physical.profile_classes,
        identity_material={
            "schema": CUSTOM_PHYSICS_BINDING_SCHEMA,
            "mode": _MODE_FIXED,
            "binding_algorithm_id": FIXED_BINDING_ALGORITHM_ID,
            "existing_physics_identity_sha256": prepared.physical.physics_identity_sha256,
        },
        # Fixed binding is a passthrough: changing this hash would destroy the frozen Q5
        # qualification identity and would be scientifically wrong.
        physics_identity_sha256=prepared.physical.physics_identity_sha256,
        scope_status=prepared.physical.scope_status,
    )


def bind_distribution_scenario(
    prepared: PreparedQ5Scenario,
    plan: PhysicsDistributionPlan,
) -> BoundCustomPhysicsScenario:
    return _bind_sampled(
        prepared,
        mode=_MODE_DISTRIBUTIONS,
        plan_identity_sha256=plan.plan_identity_sha256,
        plan_schema=plan.schema,
        algorithm_id=DISTRIBUTION_BINDING_ALGORITHM_ID,
        sampler=_DistributionSampler(prepared, plan),
    )


def bind_fleet_mixture_scenario(
    prepared: PreparedQ5Scenario,
    plan: FleetMixturePlan,
) -> BoundCustomPhysicsScenario:
    return _bind_sampled(
        prepared,
        mode=_MODE_FLEET_MIXTURE,
        plan_identity_sha256=plan.plan_identity_sha256,
        plan_schema=plan.schema,
        algorithm_id=FLEET_MIXTURE_BINDING_ALGORITHM_ID,
        sampler=_FleetMixtureSampler(prepared, plan),
    )


def _sampler_for_bound(
    bound: BoundCustomPhysicsScenario,
    *,
    distribution_plan: PhysicsDistributionPlan | None,
    fleet_mixture_plan: FleetMixturePlan | None,
) -> _Sampler | None:
    if bound.mode == _MODE_FIXED:
        if distribution_plan is not None or fleet_mixture_plan is not None:
            raise CustomPhysicsBindingError("fixed binding does not accept a sampled plan")
        return None
    if bound.mode == _MODE_DISTRIBUTIONS:
        if distribution_plan is None or fleet_mixture_plan is not None:
            raise CustomPhysicsBindingError("distribution binding requires its distribution plan")
        if distribution_plan.plan_identity_sha256 != bound.plan_identity_sha256:
            raise CustomPhysicsBindingError("distribution plan identity does not match binding")
        return _DistributionSampler(bound.prepared, distribution_plan)
    if bound.mode == _MODE_FLEET_MIXTURE:
        if fleet_mixture_plan is None or distribution_plan is not None:
            raise CustomPhysicsBindingError("fleet-mixture binding requires its fleet-mixture plan")
        if fleet_mixture_plan.plan_identity_sha256 != bound.plan_identity_sha256:
            raise CustomPhysicsBindingError("fleet-mixture plan identity does not match binding")
        return _FleetMixtureSampler(bound.prepared, fleet_mixture_plan)
    raise CustomPhysicsBindingError(f"unsupported binding mode {bound.mode!r}")


def materialize_bound_traffic(
    bound: BoundCustomPhysicsScenario,
    *,
    config: object | None = None,
    distribution_plan: PhysicsDistributionPlan | None = None,
    fleet_mixture_plan: FleetMixturePlan | None = None,
) -> PoissonNetworkTrafficRealization:
    """Realize traffic with unchanged Poisson times/IDs and deterministic per-vessel physics."""
    cfg = bound.prepared.config if config is None else config
    sampler = _sampler_for_bound(
        bound,
        distribution_plan=distribution_plan,
        fleet_mixture_plan=fleet_mixture_plan,
    )
    scenario = cfg.scenario
    base = generate_poisson_network_traffic(
        cfg.network,
        cfg.streams,
        horizon=scenario.traffic_horizon_min,
        seed=scenario.seed,
        start_time=scenario.start_time,
    )
    if sampler is None:
        return base

    stream_by_vessel: dict[str, tuple[str, int]] = {}
    for stream in base.streams:
        for index, vessel_id in enumerate(stream.vessel_ids):
            if vessel_id in stream_by_vessel:
                raise CustomPhysicsBindingError(f"duplicate realized vessel id {vessel_id!r}")
            stream_by_vessel[vessel_id] = (stream.stream_id, index)

    arrivals: list[NominalVesselArrival] = []
    for arrival in base.arrivals:
        try:
            stream_id, sample_index = stream_by_vessel[arrival.vessel.vessel_id]
        except KeyError as exc:
            raise CustomPhysicsBindingError("traffic stream realization does not cover every arrival") from exc
        resolved = sampler.resolve(stream_id, sample_index)
        arrivals.append(
            _replace_arrival(
                arrival,
                spec=resolved.profile,
                class_id=resolved.runtime_class_id,
            )
        )
    arrivals.sort(key=lambda item: (item.at, item.vessel.vessel_id))
    return PoissonNetworkTrafficRealization(
        seed=base.seed,
        start_time=base.start_time,
        horizon=base.horizon,
        streams=base.streams,
        arrivals=tuple(arrivals),
    )


def run_bound_fifo_baseline(
    bound: BoundCustomPhysicsScenario,
    *,
    config: object | None = None,
    distribution_plan: PhysicsDistributionPlan | None = None,
    fleet_mixture_plan: FleetMixturePlan | None = None,
):
    """Execute one bound realization through D0's public pre-realized FIFO seam."""
    cfg = bound.prepared.config if config is None else config
    traffic = materialize_bound_traffic(
        bound,
        config=cfg,
        distribution_plan=distribution_plan,
        fleet_mixture_plan=fleet_mixture_plan,
    )
    return run_environmental_network_fifo_experiment_with_traffic(cfg, traffic)


def build_bound_reference_cache_identity(bound: BoundCustomPhysicsScenario):
    """Build the exact same-physics cache key using the bound sampled physics identity."""
    return build_reference_cache_identity(
        physics_identity_sha256=bound.physics_identity_sha256,
        profile=bound.profile,
        runtime=bound.prepared.reference_runtime,
        b6_spec_sha256=A69_SPEC_SHA256,
        implementation_manifest_sha256=A70_IMPLEMENTATION_MANIFEST_SHA256,
    )


def resolve_bound_dynamic_reference(
    bound: BoundCustomPhysicsScenario,
    *,
    cache_dir=None,
):
    """Resolve a fresh/cached B6 reference only for the exact bound sampled identity.

    A successful B6 reference remains reference evidence for a SUPPORTED_EXPLORATORY
    sampled scenario.  It is not robustness qualification and does not promote scope.
    """
    identity = build_bound_reference_cache_identity(bound)
    cache = None if cache_dir is None else DynamicReferenceCache(cache_dir)

    def compute():
        return run_balanced_saturated_reference(
            bound.profile,
            bound.profile_classes,
            bound.prepared.reference_runtime,
            b6_spec_sha256=A69_SPEC_SHA256,
            implementation_manifest_sha256=A70_IMPLEMENTATION_MANIFEST_SHA256,
        )

    result, status = resolve_dynamic_reference(cache, identity, compute)
    return result, status, identity
