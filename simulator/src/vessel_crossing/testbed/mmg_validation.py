"""Fail-closed provenance gate for publication-scale MMG validation protocols.

This module deliberately separates three questions that are easy to conflate:

1. Are all numerical inputs present, so an engineering run can execute?
2. Are the numerical inputs tied to a single internally coherent experiment?
3. Is every model/input transformation provenance-bound well enough to call the
   result publication-grade validation rather than an engineering sensitivity run?

The gate never infers or tunes a missing parameter from a desired trajectory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
from typing import Any


class ValidationReadiness(str, Enum):
    BLOCKED = "BLOCKED"
    ENGINEERING_EXECUTABLE = "ENGINEERING_EXECUTABLE"
    PUBLICATION_READY = "PUBLICATION_READY"


@dataclass(frozen=True, slots=True)
class ProvenanceBinding:
    """One numerical/model input tied to a source or an explicit assumption."""

    value: Any
    source_label: str | None
    assumption: bool = False
    note: str = ""

    @property
    def bound(self) -> bool:
        return self.value is not None and bool((self.source_label or "").strip())


@dataclass(frozen=True, slots=True)
class MmgValidationProtocol:
    protocol_id: str
    experiment_family: str
    hull_geometry: ProvenanceBinding
    environment: ProvenanceBinding
    initial_condition: ProvenanceBinding
    initial_equilibrium: ProvenanceBinding
    rudder_schedule: ProvenanceBinding
    neutral_rudder_offset_deg: ProvenanceBinding
    propeller_rate_rps: ProvenanceBinding
    propeller_geometry: ProvenanceBinding
    propeller_open_water_data: ProvenanceBinding
    propulsion_treatment: ProvenanceBinding
    resistance_treatment: ProvenanceBinding
    hydrodynamic_parameter_set: ProvenanceBinding
    validation_metrics: ProvenanceBinding
    source_conflicts: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def identity_sha256(self) -> str:
        payload = asdict(self)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class MmgValidationAudit:
    readiness: ValidationReadiness
    missing_fields: tuple[str, ...]
    unbound_fields: tuple[str, ...]
    assumption_fields: tuple[str, ...]
    source_conflicts: tuple[str, ...]

    @property
    def executable(self) -> bool:
        return self.readiness is not ValidationReadiness.BLOCKED

    @property
    def publication_ready(self) -> bool:
        return self.readiness is ValidationReadiness.PUBLICATION_READY


_REQUIRED_BINDINGS = (
    "hull_geometry",
    "environment",
    "initial_condition",
    "initial_equilibrium",
    "rudder_schedule",
    "neutral_rudder_offset_deg",
    "propeller_rate_rps",
    "propeller_geometry",
    "propeller_open_water_data",
    "propulsion_treatment",
    "resistance_treatment",
    "hydrodynamic_parameter_set",
    "validation_metrics",
)


def audit_validation_protocol(protocol: MmgValidationProtocol) -> MmgValidationAudit:
    """Return deterministic, fail-closed validation readiness.

    A run is blocked if any required value is missing or if the protocol records a
    source conflict that has not been resolved.  A numerically complete protocol can
    still be engineering-only when one or more bindings are assumptions or when a
    value is present without a source label.
    """

    missing: list[str] = []
    unbound: list[str] = []
    assumptions: list[str] = []
    for name in _REQUIRED_BINDINGS:
        binding = getattr(protocol, name)
        if binding.value is None:
            missing.append(name)
            continue
        if not (binding.source_label or "").strip():
            unbound.append(name)
        if binding.assumption:
            assumptions.append(name)

    conflicts = tuple(protocol.source_conflicts)
    if missing or conflicts:
        readiness = ValidationReadiness.BLOCKED
    elif unbound or assumptions:
        readiness = ValidationReadiness.ENGINEERING_EXECUTABLE
    else:
        readiness = ValidationReadiness.PUBLICATION_READY
    return MmgValidationAudit(
        readiness=readiness,
        missing_fields=tuple(missing),
        unbound_fields=tuple(unbound),
        assumption_fields=tuple(assumptions),
        source_conflicts=conflicts,
    )


def marin_2011_kvlcc2_candidate_protocol() -> MmgValidationProtocol:
    """MARIN-2011 free-running KVLCC2 candidate, intentionally not publication-ready.

    Public MARIN benchmark reporting binds the 7 m model, 1.18 m/s approach speed,
    620 rpm self-propulsion setting and finite helm-rate context.  The CROSS in-tree
    ShipMMG-derived hydrodynamic profile does *not* automatically inherit the exact
    MARIN resistance/propulsion identification, so those transformations remain
    explicit engineering assumptions here rather than being silently tuned.
    """

    marin = "Quadvlieg & Brouwer, MARINE 2011 KVLCC2 free-running benchmark"
    simman = "SIMMAN-2020 KVLCC2 Case 1.2 public protocol page"
    shipmmg = "ShipMMG 0.0.11 KVLCC2 L7 implementation fixture"
    return MmgValidationProtocol(
        protocol_id="KVLCC2_MARIN_2011_PLUS35_CANDIDATE",
        experiment_family="MARIN_2011_FREE_RUNNING",
        hull_geometry=ProvenanceBinding(
            {"L_pp_m": 7.0, "B_m": 1.27, "loaded_draft_m": 0.455}, marin
        ),
        environment=ProvenanceBinding(
            {"water_depth_m": 5.0, "calm_water": True}, marin
        ),
        initial_condition=ProvenanceBinding(
            {"approach_speed_m_s": 1.18, "rudder_deg": 0.0}, marin
        ),
        initial_equilibrium=ProvenanceBinding(
            "longitudinal self-propulsion equilibrium before manoeuvre", marin
        ),
        rudder_schedule=ProvenanceBinding(
            {"target_deg": 35.0, "full_scale_rate_deg_s": 2.3}, marin,
            note="Model-scale rate can be generated by Froude scaling when the scale is explicitly bound.",
        ),
        neutral_rudder_offset_deg=ProvenanceBinding(
            0.0,
            "engineering assumption pending exact MARIN neutral-rudder binding",
            assumption=True,
        ),
        propeller_rate_rps=ProvenanceBinding(620.0 / 60.0, marin),
        propeller_geometry=ProvenanceBinding(
            "MARIN/INSEAN KVLCC2 propeller geometry required",
            marin,
        ),
        propeller_open_water_data=ProvenanceBinding(
            "MARIN KVLCC2 POW data required",
            simman,
        ),
        propulsion_treatment=ProvenanceBinding(
            "ShipMMG-style propulsion terms provisionally used for engineering comparison",
            shipmmg,
            assumption=True,
        ),
        resistance_treatment=ProvenanceBinding(
            "ShipMMG constant R_0_dash treatment provisionally used for engineering comparison",
            shipmmg,
            assumption=True,
        ),
        hydrodynamic_parameter_set=ProvenanceBinding(
            "ShipMMG 0.0.11 KVLCC2 L7 coefficients",
            shipmmg,
            assumption=True,
            note="Not claimed to be the exact MARIN-2011 identified coefficient set.",
        ),
        validation_metrics=ProvenanceBinding(
            {
                "plus35_advance_m": 950.0,
                "plus35_tactical_diameter_m": 989.0,
            },
            marin,
            note="Keep MARIN-2011 metrics separate from later SIMMAN validation tables.",
        ),
        notes=(
            "620 rpm is MARIN-2011 model-scale self-propulsion RPM at 1.18 m/s.",
            "Do not substitute HSVA/SIMMAN-2008 614 rpm into this protocol.",
        ),
    )


def simman_2020_case12_protocol_shell() -> MmgValidationProtocol:
    """Publicly specified SIMMAN-2020 Case 1.2 shell with unresolved exact RPM.

    The public case page states that RPM must equal the longitudinal-equilibrium
    self-propulsion setting but does not expose the number in the page itself.
    Therefore the protocol is deliberately BLOCKED rather than borrowing 620 or 614 rpm.
    """

    simman = "SIMMAN-2020 KVLCC2 Case 1.2 public protocol page"
    return MmgValidationProtocol(
        protocol_id="KVLCC2_SIMMAN2020_CASE12_SHELL",
        experiment_family="SIMMAN_2020_CASE_1_2",
        hull_geometry=ProvenanceBinding(
            {"L_pp_m": 7.0, "draft_m": 0.455, "scale": 45.714}, simman
        ),
        environment=ProvenanceBinding(
            {"rho_kg_m3": 1000.0, "nu_m2_s": 1.27e-6, "calm_water": True}, simman
        ),
        initial_condition=ProvenanceBinding({"approach_speed_m_s": 1.18}, simman),
        initial_equilibrium=ProvenanceBinding(
            "longitudinal equilibrium at 15.5 kn full scale", simman
        ),
        rudder_schedule=ProvenanceBinding(
            {"target_deg": 35.0, "model_rate_deg_s": 15.68}, simman
        ),
        neutral_rudder_offset_deg=ProvenanceBinding(None, None),
        propeller_rate_rps=ProvenanceBinding(None, None),
        propeller_geometry=ProvenanceBinding("MARIN Case-1.2 propeller geometry", simman),
        propeller_open_water_data=ProvenanceBinding("MARIN Case-1.2 open-water data", simman),
        propulsion_treatment=ProvenanceBinding(None, None),
        resistance_treatment=ProvenanceBinding(None, None),
        hydrodynamic_parameter_set=ProvenanceBinding(None, None),
        validation_metrics=ProvenanceBinding("Case-1.2 turning/zigzag trajectory metrics", simman),
        source_conflicts=(
            "Public SIMMAN pages expose inconsistent Wd/T metadata (17.5 on Case 1.2; other overview material has reported 11.0). Bind absolute depth/draft before publication use.",
        ),
    )
