"""R1 P4D microscopic AS/HS behavior contract.

P4D restores the original R1 separation between the independent physical CROSS target
and the macro/reference headway model.  This module therefore describes *microscopic*
AS/HS differences only.  It deliberately contains no pair-headway field and rejects any
attempt to smuggle reference-headway forcing into the independent target.

The v1 contract exposes two mechanisms already represented by the current architecture:

* ``command_delay_min`` -- controller/behavior response latency above the plant;
* ``operational_domain_scale`` -- multiplicative scale on the operational ship-domain
  geometry used by source clearance, encounter planning and safety-domain checks.

The second mechanism is intentionally marked source-coupled in the current architecture:
changing it can meter traffic before the conflict zone.  It is therefore qualification /
diagnostic-only unless a future architecture separates ingress clearance from the in-system
operational domain.  Confirmatory independent-target validation rejects any non-unit scale.

Neither mechanism changes hull dimensions, maneuver limits, plant dynamics, demand
arrival times, stream identity or route geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from typing import Mapping

from vessel_crossing.nextgen.ship_domain import ShipDomain
from vessel_crossing.nextgen.vessel_classes import VesselClassProfile
from vessel_crossing.research.r1_p4b_behavior_adapter import P4BBehaviorBinding
from vessel_crossing.testbed.behavior import BehaviorProfile


class P4DProvenanceKind(str, Enum):
    EXTERNAL_EMPIRICAL = "EXTERNAL_EMPIRICAL"
    EXTERNAL_LITERATURE = "EXTERNAL_LITERATURE"
    EXPERIMENTAL_HYPOTHESIS = "EXPERIMENTAL_HYPOTHESIS"
    CONTROL_ONLY = "CONTROL_ONLY"


@dataclass(frozen=True, slots=True)
class P4DParameterProvenance:
    kind: P4DProvenanceKind
    source_id: str
    note: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, P4DProvenanceKind):
            raise TypeError("kind must be P4DProvenanceKind")
        if not self.source_id or not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if not self.note or not self.note.strip():
            raise ValueError("note must be non-empty")


@dataclass(frozen=True, slots=True)
class P4DMicroscopicRoleSpec:
    role: str
    behavior_profile: BehaviorProfile
    command_delay_min: float
    operational_domain_scale: float
    command_delay_provenance: P4DParameterProvenance
    domain_scale_provenance: P4DParameterProvenance

    def __post_init__(self) -> None:
        if self.role not in {"AS", "HS"}:
            raise ValueError("role must be AS or HS")
        if self.role == "AS" and self.behavior_profile is not BehaviorProfile.AUTONOMOUS_POLICY:
            raise ValueError("AS role must use AUTONOMOUS_POLICY")
        if self.role == "HS" and self.behavior_profile is not BehaviorProfile.COMPLIANT_MERCHANT:
            raise ValueError("HS role must use COMPLIANT_MERCHANT")
        if not isfinite(self.command_delay_min) or self.command_delay_min < 0.0:
            raise ValueError("command_delay_min must be finite and non-negative")
        if not isfinite(self.operational_domain_scale) or self.operational_domain_scale <= 0.0:
            raise ValueError("operational_domain_scale must be finite and positive")
        if not isinstance(self.command_delay_provenance, P4DParameterProvenance):
            raise TypeError("command_delay_provenance must be P4DParameterProvenance")
        if not isinstance(self.domain_scale_provenance, P4DParameterProvenance):
            raise TypeError("domain_scale_provenance must be P4DParameterProvenance")

    @property
    def identity_payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "behavior_profile": self.behavior_profile.value,
            "command_delay_min": self.command_delay_min,
            "operational_domain_scale": self.operational_domain_scale,
            "command_delay_provenance": {
                "kind": self.command_delay_provenance.kind.value,
                "source_id": self.command_delay_provenance.source_id,
                "note": self.command_delay_provenance.note,
            },
            "domain_scale_provenance": {
                "kind": self.domain_scale_provenance.kind.value,
                "source_id": self.domain_scale_provenance.source_id,
                "note": self.domain_scale_provenance.note,
            },
        }


@dataclass(frozen=True, slots=True)
class P4DBehaviorContract:
    contract_id: str
    as_spec: P4DMicroscopicRoleSpec
    hs_spec: P4DMicroscopicRoleSpec
    reference_headway_forcing: bool = False
    scientific_parameter_status: str = "QUALIFICATION_OR_SCREENING_ONLY_NOT_CONFIRMATORY"

    def __post_init__(self) -> None:
        if not self.contract_id or not self.contract_id.strip():
            raise ValueError("contract_id must be non-empty")
        if self.as_spec.role != "AS" or self.hs_spec.role != "HS":
            raise ValueError("contract must bind AS spec then HS spec")
        if self.reference_headway_forcing:
            raise ValueError(
                "P4D independent target forbids macro/reference pair-headway forcing"
            )
        if not self.scientific_parameter_status:
            raise ValueError("scientific_parameter_status must be non-empty")

    @property
    def identity_sha256(self) -> str:
        payload = {
            "contract_id": self.contract_id,
            "as_spec": self.as_spec.identity_payload,
            "hs_spec": self.hs_spec.identity_payload,
            "reference_headway_forcing": self.reference_headway_forcing,
            "scientific_parameter_status": self.scientific_parameter_status,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(raw).hexdigest()

    def spec_for_role(self, role: str) -> P4DMicroscopicRoleSpec:
        if role == "AS":
            return self.as_spec
        if role == "HS":
            return self.hs_spec
        raise KeyError(role)


def scale_ship_domain(domain: ShipDomain, scale: float) -> ShipDomain:
    if not isinstance(domain, ShipDomain):
        raise TypeError("domain must be ShipDomain")
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    return ShipDomain(
        ahead_nm=domain.ahead_nm * scale,
        astern_nm=domain.astern_nm * scale,
        port_nm=domain.port_nm * scale,
        starboard_nm=domain.starboard_nm * scale,
    )


def profile_for_p4d_role(
    base: VesselClassProfile,
    spec: P4DMicroscopicRoleSpec,
) -> VesselClassProfile:
    if not isinstance(base, VesselClassProfile):
        raise TypeError("base must be VesselClassProfile")
    if base.safety_domain is None:
        raise ValueError("P4D domain scaling requires a base safety_domain")
    return replace(
        base,
        class_id=f"{base.class_id}/P4D-{spec.role}-D{spec.operational_domain_scale:.6g}",
        safety_domain=scale_ship_domain(base.safety_domain, spec.operational_domain_scale),
    )


def delay_binding_for_p4d_role(
    spec: P4DMicroscopicRoleSpec,
    *,
    contract_id: str,
) -> P4BBehaviorBinding:
    """Reuse the qualified command-delay backend without importing P4B semantics."""
    return P4BBehaviorBinding(
        profile=spec.behavior_profile,
        adapter_id=f"R1_P4D/{contract_id}/{spec.role}/COMMAND_DELAY",
        command_delay_min=spec.command_delay_min,
    )



def source_coupled_domain_scaling_active(contract: P4DBehaviorContract, *, tolerance: float = 1e-12) -> bool:
    """Return whether the current contract changes the domain seen by source clearance."""
    if tolerance < 0.0 or not isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    return (
        abs(contract.as_spec.operational_domain_scale - 1.0) > tolerance
        or abs(contract.hs_spec.operational_domain_scale - 1.0) > tolerance
    )


def validate_confirmatory_independent_target_contract(contract: P4DBehaviorContract) -> None:
    """Fail closed on mechanisms that would pre-meter the independent P4D target.

    P4D screening demonstrated that ``operational_domain_scale`` participates directly
    in source-clearance geometry.  Allowing a non-unit value in a confirmatory independent
    target could therefore recreate the P4C degeneracy by pre-metering the route before
    vessels reach the interaction zone.

    Command delay remains eligible, but only when the same-route following supervisor is
    given the same follower-specific delay as additional physical reaction allowance.
    That coupling may legitimately change dynamic source admission because a slow-reacting
    follower needs more braking room; it is a consequence of the microscopic behavior
    contract, not a prescribed reference headway.  A separate interaction-positive gate
    is still required before confirmatory freeze so this safety coupling cannot silently
    degenerate the experiment into pure pre-metering.

    Active confirmatory differences must also carry non-CONTROL_ONLY provenance.
    """
    if not isinstance(contract, P4DBehaviorContract):
        raise TypeError("contract must be P4DBehaviorContract")
    if source_coupled_domain_scaling_active(contract):
        raise ValueError(
            "confirmatory independent P4D target forbids non-unit operational_domain_scale "
            "while source clearance consumes that domain"
        )
    if abs(contract.as_spec.command_delay_min - contract.hs_spec.command_delay_min) > 1e-12:
        for spec in (contract.as_spec, contract.hs_spec):
            if spec.command_delay_provenance.kind is P4DProvenanceKind.CONTROL_ONLY:
                raise ValueError(
                    "active confirmatory command-delay contrast requires non-CONTROL_ONLY provenance"
                )

def validate_role_map(role_by_vessel: Mapping[str, str], expected_vessel_ids: set[str]) -> None:
    supplied = set(role_by_vessel)
    if supplied != expected_vessel_ids:
        raise ValueError(
            f"role map must match traffic vessel ids exactly; missing={sorted(expected_vessel_ids-supplied)}, "
            f"extra={sorted(supplied-expected_vessel_ids)}"
        )
    bad = sorted(v for v, role in role_by_vessel.items() if role not in {"AS", "HS"})
    if bad:
        raise ValueError(f"unsupported P4D role for vessels: {bad!r}")


def control_only_provenance(note: str) -> P4DParameterProvenance:
    return P4DParameterProvenance(
        P4DProvenanceKind.CONTROL_ONLY,
        "R1_P4D_CONTROL_ONLY",
        note,
    )


def experimental_hypothesis_provenance(source_id: str, note: str) -> P4DParameterProvenance:
    return P4DParameterProvenance(
        P4DProvenanceKind.EXPERIMENTAL_HYPOTHESIS,
        source_id,
        note,
    )
