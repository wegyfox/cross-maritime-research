"""Stateful encounter observation, risk persistence and responsibility lifecycle."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import hypot, isfinite

from .dynamics import Vessel2DState, heading_unit_vector, normalize_heading
from .encounter_classification import EncounterClassification, classify_encounter
from .observation import (
    BearingTrendAssessment,
    BearingTrendConfig,
    PairObservation,
    assess_bearing_trend,
    observe_pair,
)
from .responsibility import (
    ResponsibilityBasis,
    ResponsibilityContext,
    ResponsibilityRole,
    ResponsibilityScaffold,
    assign_responsibility_from_classification,
)
from .risk import RiskAssessment, RiskGateConfig, assess_cpa_tcpa_risk
from .ship_domain import ShipDomain

_EPS = 1e-12


class AmbiguousCrossingPolicy(str, Enum):
    FAIL_CLOSED = "fail_closed"
    MUTUAL_KEEP_CLEAR = "mutual_keep_clear"


class LifecycleRiskBasis(str, Enum):
    NONE = "none"
    SNAPSHOT = "snapshot"
    STEADY_BEARING = "steady_bearing"
    SNAPSHOT_AND_STEADY_BEARING = "snapshot_and_steady_bearing"
    PERSISTED = "persisted"


@dataclass(frozen=True, slots=True)
class EncounterLifecycleConfig:
    risk_gate: RiskGateConfig
    bearing_trend: BearingTrendConfig
    clear_observations_required: int = 2
    overtaking_past_margin_nm: float = 0.0
    overtaking_clear_distance_nm: float = 0.5
    overtaking_clear_observations_required: int = 2
    max_observations: int = 16
    ambiguous_crossing_policy: AmbiguousCrossingPolicy = AmbiguousCrossingPolicy.FAIL_CLOSED

    def __post_init__(self) -> None:
        if self.clear_observations_required < 1:
            raise ValueError("clear_observations_required must be at least 1")
        if self.overtaking_past_margin_nm < 0.0:
            raise ValueError("overtaking_past_margin_nm must be non-negative")
        if self.overtaking_clear_distance_nm <= 0.0:
            raise ValueError("overtaking_clear_distance_nm must be positive")
        if self.overtaking_clear_observations_required < 1:
            raise ValueError("overtaking_clear_observations_required must be at least 1")
        if self.max_observations < 2:
            raise ValueError("max_observations must be at least 2")
        if not isinstance(self.ambiguous_crossing_policy, AmbiguousCrossingPolicy):
            raise TypeError("ambiguous_crossing_policy must be AmbiguousCrossingPolicy")




@dataclass(frozen=True, slots=True)
class LifecycleVesselGeometryEvidence:
    """Minimal hull-referenced geometry needed by lifecycle persistence semantics."""

    vessel_id: str
    position: tuple[float, float]
    hull_heading_deg: float

    def __post_init__(self) -> None:
        if not self.vessel_id:
            raise ValueError("vessel_id must be non-empty")
        if len(self.position) != 2 or not all(isfinite(float(v)) for v in self.position):
            raise ValueError("position must contain two finite components")
        if not isfinite(self.hull_heading_deg):
            raise ValueError("hull_heading_deg must be finite")
        object.__setattr__(self, "position", (float(self.position[0]), float(self.position[1])))
        object.__setattr__(self, "hull_heading_deg", normalize_heading(self.hull_heading_deg))


@dataclass(frozen=True, slots=True)
class EncounterLifecycleEvidence:
    """State-class-independent input contract for one lifecycle assimilation step.

    Classification and snapshot risk may be produced by different physical adapters,
    but they must describe the same ordered vessel pair and the same observation time.
    Rule-13 release geometry intentionally stores hull heading separately from motion
    velocity so environmental drift cannot replace the overtaken vessel's aspect.
    """

    observation: PairObservation
    classification: EncounterClassification
    snapshot_risk: RiskAssessment
    first_geometry: LifecycleVesselGeometryEvidence
    second_geometry: LifecycleVesselGeometryEvidence
    responsibility_context: ResponsibilityContext | None = None
    responsibility_delegated: bool = False

    def __post_init__(self) -> None:
        pair = (self.observation.first_id, self.observation.second_id)
        if (self.first_geometry.vessel_id, self.second_geometry.vessel_id) != pair:
            raise ValueError("lifecycle geometry ids must match ordered observation ids")
        if {self.classification.prediction.first_id, self.classification.prediction.second_id} != set(pair):
            raise ValueError("classification vessel ids do not match lifecycle observation")
        if not self.snapshot_risk.involves(*pair):
            raise ValueError("snapshot risk vessel ids do not match lifecycle observation")
        if not isinstance(self.responsibility_delegated, bool):
            raise TypeError("responsibility_delegated must be bool")
        if self.responsibility_context is not None and not isinstance(
            self.responsibility_context, ResponsibilityContext
        ):
            raise TypeError("responsibility_context must be ResponsibilityContext or None")


def build_state_lifecycle_evidence(
    first: Vessel2DState,
    second: Vessel2DState,
    *,
    time_min: float,
    config: EncounterLifecycleConfig,
    uncertain: bool = False,
    responsibility_context: ResponsibilityContext | None = None,
    first_domain: ShipDomain | None = None,
    second_domain: ShipDomain | None = None,
) -> EncounterLifecycleEvidence:
    """Build the historical ground-state evidence consumed by ``EncounterLifecycle``."""
    if not isinstance(first, Vessel2DState) or not isinstance(second, Vessel2DState):
        raise TypeError("first and second must be Vessel2DState")
    if not isinstance(config, EncounterLifecycleConfig):
        raise TypeError("config must be EncounterLifecycleConfig")
    observation = observe_pair(first, second, time_min=time_min)
    classification = classify_encounter(first, second)
    snapshot_risk = assess_cpa_tcpa_risk(
        first,
        second,
        config=config.risk_gate,
        uncertain=uncertain,
        first_domain=first_domain,
        second_domain=second_domain,
    )
    return EncounterLifecycleEvidence(
        observation=observation,
        classification=classification,
        snapshot_risk=snapshot_risk,
        first_geometry=LifecycleVesselGeometryEvidence(
            vessel_id=first.vessel_id,
            position=first.position,
            hull_heading_deg=first.heading_deg,
        ),
        second_geometry=LifecycleVesselGeometryEvidence(
            vessel_id=second.vessel_id,
            position=second.position,
            hull_heading_deg=second.heading_deg,
        ),
        responsibility_context=responsibility_context,
    )


@dataclass(frozen=True, slots=True)
class EncounterLifecycleDecision:
    observation: PairObservation
    classification: EncounterClassification
    current_scaffold: ResponsibilityScaffold
    effective_scaffold: ResponsibilityScaffold | None
    snapshot_risk: RiskAssessment
    bearing_trend: BearingTrendAssessment
    risk_active: bool
    risk_basis: LifecycleRiskBasis
    risk_clear_streak: int
    rule13_persisted: bool
    overtaking_past: bool
    overtaking_clear: bool
    overtaking_clear_streak: int
    responsibility_delegated: bool = False

    @property
    def responsibility_active(self) -> bool:
        return self.effective_scaffold is not None

    def role_for(self, vessel_id: str) -> ResponsibilityRole:
        ids = {self.current_scaffold.first_id, self.current_scaffold.second_id}
        if vessel_id not in ids:
            raise KeyError(vessel_id)
        if self.effective_scaffold is None:
            return ResponsibilityRole.NONE
        return self.effective_scaffold.role_for(vessel_id)


class EncounterLifecycle:
    """Maintain a stable pair identity across classification changes and maneuvers.

    The class deliberately stores the responsibility scaffold that became active.  A
    give-way maneuver can change instantaneous relative geometry; recomputing roles from
    that new geometry alone would make the obligation disappear exactly because the
    vessel started complying.  Rule-14/15 scaffolds are retained until risk has been
    clear for the configured number of observations.  Rule 13 is retained separately
    until the overtaking vessel is modeled as past *and* clear.
    """

    def __init__(self, first_id: str, second_id: str, *, config: EncounterLifecycleConfig) -> None:
        if not first_id or not second_id or first_id == second_id:
            raise ValueError("lifecycle requires two distinct non-empty vessel ids")
        self.first_id = first_id
        self.second_id = second_id
        self.config = config
        self._history: deque[PairObservation] = deque(maxlen=config.max_observations)
        self._last_time: float | None = None
        self._risk_active = False
        self._risk_clear_streak = 0
        self._risk_scaffold: ResponsibilityScaffold | None = None
        # Dormant continuity hint for the same tracked Rule-14/15 encounter.
        # It survives clear-release only until the registry retires this lifecycle.
        # Fresh lifecycle instances therefore still fail closed on ambiguous crossing.
        self._last_risk_scaffold: ResponsibilityScaffold | None = None
        self._rule13_scaffold: ResponsibilityScaffold | None = None
        self._rule13_overtaker_id: str | None = None
        self._rule13_overtaken_id: str | None = None
        self._rule13_clear_streak = 0

    @property
    def observations(self) -> tuple[PairObservation, ...]:
        return tuple(self._history)

    def _ordered_states(
        self, first: Vessel2DState, second: Vessel2DState
    ) -> tuple[Vessel2DState, Vessel2DState]:
        supplied = {first.vessel_id, second.vessel_id}
        expected = {self.first_id, self.second_id}
        if supplied != expected:
            raise ValueError("supplied vessel ids do not match lifecycle pair")
        if first.vessel_id == self.first_id:
            return first, second
        return second, first

    def _rule13_geometry(
        self,
        first: LifecycleVesselGeometryEvidence,
        second: LifecycleVesselGeometryEvidence,
    ) -> tuple[bool, bool]:
        if not self._rule13_overtaker_id or not self._rule13_overtaken_id:
            return False, False
        by_id = {first.vessel_id: first, second.vessel_id: second}
        overtaker = by_id[self._rule13_overtaker_id]
        overtaken = by_id[self._rule13_overtaken_id]
        rel_x = overtaker.position[0] - overtaken.position[0]
        rel_y = overtaker.position[1] - overtaken.position[1]
        ux, uy = heading_unit_vector(overtaken.hull_heading_deg)
        ahead_projection = rel_x * ux + rel_y * uy
        separation = hypot(rel_x, rel_y)
        past = ahead_projection >= self.config.overtaking_past_margin_nm - _EPS
        clear = separation >= self.config.overtaking_clear_distance_nm - _EPS
        return past, clear

    def _validate_evidence_pair(self, evidence: EncounterLifecycleEvidence) -> None:
        if not isinstance(evidence, EncounterLifecycleEvidence):
            raise TypeError("evidence must be EncounterLifecycleEvidence")
        if (evidence.observation.first_id, evidence.observation.second_id) != (
            self.first_id,
            self.second_id,
        ):
            raise ValueError("evidence ordered vessel ids do not match lifecycle pair")
        if evidence.snapshot_risk.config != self.config.risk_gate:
            raise ValueError("evidence risk config does not match lifecycle risk gate")

    def update(
        self,
        first: Vessel2DState,
        second: Vessel2DState,
        *,
        time_min: float,
        uncertain: bool = False,
        responsibility_context: ResponsibilityContext | None = None,
        first_domain: ShipDomain | None = None,
        second_domain: ShipDomain | None = None,
    ) -> EncounterLifecycleDecision:
        """Historical state-based API, now an evidence-building compatibility wrapper."""
        first, second = self._ordered_states(first, second)
        evidence = build_state_lifecycle_evidence(
            first,
            second,
            time_min=time_min,
            config=self.config,
            uncertain=uncertain,
            responsibility_context=responsibility_context,
            first_domain=first_domain,
            second_domain=second_domain,
        )
        return self.update_evidence(evidence)

    def update_evidence(
        self,
        evidence: EncounterLifecycleEvidence,
    ) -> EncounterLifecycleDecision:
        """Assimilate one state-class-independent lifecycle evidence snapshot.

        All persistence semantics live here. Physical adapters are responsible only for
        producing trustworthy classification, snapshot-risk, observation and hull
        geometry evidence for the same ordered pair.
        """
        self._validate_evidence_pair(evidence)
        time_min = evidence.observation.time_min
        if self._last_time is not None and time_min <= self._last_time + _EPS:
            raise ValueError("lifecycle update times must be strictly increasing")
        self._last_time = time_min

        observation = evidence.observation
        self._history.append(observation)
        trend = assess_bearing_trend(tuple(self._history), config=self.config.bearing_trend)

        classification = evidence.classification
        ambiguous_continuity_scaffold: ResponsibilityScaffold | None = None
        if evidence.responsibility_delegated:
            current_scaffold = ResponsibilityScaffold(
                first_id=self.first_id,
                second_id=self.second_id,
                first_role=ResponsibilityRole.NONE,
                second_role=ResponsibilityRole.NONE,
                basis=ResponsibilityBasis.NONE,
                encounter_type=classification.encounter_type,
                requires_collision_risk_gate=False,
            )
        else:
            try:
                current_scaffold = assign_responsibility_from_classification(
                    self.first_id,
                    self.second_id,
                    classification=classification,
                    context=evidence.responsibility_context,
                )
            except ValueError as exc:
                ambiguous_crossing = (
                    classification.encounter_type.value == "crossing"
                    and "does not yield a unique starboard-side give-way vessel" in str(exc)
                )
                ambiguous_continuity_scaffold = (
                    self._rule13_scaffold or self._risk_scaffold or self._last_risk_scaffold
                )
                if ambiguous_crossing and ambiguous_continuity_scaffold is not None:
                    current_scaffold = ResponsibilityScaffold(
                        first_id=self.first_id,
                        second_id=self.second_id,
                        first_role=ResponsibilityRole.NONE,
                        second_role=ResponsibilityRole.NONE,
                        basis=ResponsibilityBasis.NONE,
                        encounter_type=classification.encounter_type,
                        requires_collision_risk_gate=False,
                    )
                elif (
                    ambiguous_crossing
                    and self.config.ambiguous_crossing_policy is AmbiguousCrossingPolicy.MUTUAL_KEEP_CLEAR
                ):
                    if evidence.snapshot_risk.risk_exists:
                        current_scaffold = ResponsibilityScaffold(
                            first_id=self.first_id,
                            second_id=self.second_id,
                            first_role=ResponsibilityRole.MUTUAL_ACTION,
                            second_role=ResponsibilityRole.MUTUAL_ACTION,
                            basis=ResponsibilityBasis.MODELED_MUTUAL_KEEP_CLEAR,
                            encounter_type=classification.encounter_type,
                            requires_collision_risk_gate=True,
                        )
                    else:
                        current_scaffold = ResponsibilityScaffold(
                            first_id=self.first_id,
                            second_id=self.second_id,
                            first_role=ResponsibilityRole.NONE,
                            second_role=ResponsibilityRole.NONE,
                            basis=ResponsibilityBasis.NONE,
                            encounter_type=classification.encounter_type,
                            requires_collision_risk_gate=False,
                        )
                else:
                    raise
        if (
            not evidence.responsibility_delegated
            and self._rule13_scaffold is None
            and current_scaffold.basis
            in {ResponsibilityBasis.RULE_14_HEAD_ON, ResponsibilityBasis.RULE_15_CROSSING}
        ):
            self._last_risk_scaffold = current_scaffold

        snapshot_risk = evidence.snapshot_risk
        snapshot_signal = snapshot_risk.risk_exists
        bearing_signal = trend.indicates_collision_risk
        signal = snapshot_signal or bearing_signal

        if signal:
            self._risk_active = True
            self._risk_clear_streak = 0
            if snapshot_signal and bearing_signal:
                risk_basis = LifecycleRiskBasis.SNAPSHOT_AND_STEADY_BEARING
            elif snapshot_signal:
                risk_basis = LifecycleRiskBasis.SNAPSHOT
            else:
                risk_basis = LifecycleRiskBasis.STEADY_BEARING
            if self._risk_scaffold is None:
                if current_scaffold.basis in {
                    ResponsibilityBasis.RULE_14_HEAD_ON,
                    ResponsibilityBasis.RULE_15_CROSSING,
                    ResponsibilityBasis.MODELED_MUTUAL_KEEP_CLEAR,
                }:
                    self._risk_scaffold = current_scaffold
                    self._last_risk_scaffold = current_scaffold
                elif (
                    ambiguous_continuity_scaffold is not None
                    and ambiguous_continuity_scaffold.basis
                    in {
                        ResponsibilityBasis.RULE_14_HEAD_ON,
                        ResponsibilityBasis.RULE_15_CROSSING,
                        ResponsibilityBasis.MODELED_MUTUAL_KEEP_CLEAR,
                    }
                ):
                    self._risk_scaffold = ambiguous_continuity_scaffold
        elif self._risk_active:
            self._risk_clear_streak += 1
            if self._risk_clear_streak >= self.config.clear_observations_required:
                self._risk_active = False
                self._risk_clear_streak = 0
                self._risk_scaffold = None
                risk_basis = LifecycleRiskBasis.NONE
            else:
                risk_basis = LifecycleRiskBasis.PERSISTED
        else:
            self._risk_clear_streak = 0
            risk_basis = LifecycleRiskBasis.NONE

        # Rule 13 has its own lifecycle and takes precedence over a momentary later
        # crossing classification. The saved identities survive subsequent bearing
        # changes until the explicit past-and-clear release condition is satisfied.
        if (
            not evidence.responsibility_delegated
            and self._rule13_scaffold is None
            and current_scaffold.basis is ResponsibilityBasis.RULE_13_OVERTAKING
        ):
            self._rule13_scaffold = current_scaffold
            if current_scaffold.first_role is ResponsibilityRole.GIVE_WAY:
                self._rule13_overtaker_id = current_scaffold.first_id
                self._rule13_overtaken_id = current_scaffold.second_id
            else:
                self._rule13_overtaker_id = current_scaffold.second_id
                self._rule13_overtaken_id = current_scaffold.first_id
            self._rule13_clear_streak = 0
            self._risk_scaffold = None
            self._last_risk_scaffold = None

        overtaking_past, overtaking_clear = self._rule13_geometry(
            evidence.first_geometry,
            evidence.second_geometry,
        )
        if self._rule13_scaffold is not None:
            if overtaking_past and overtaking_clear:
                self._rule13_clear_streak += 1
            else:
                self._rule13_clear_streak = 0
            if self._rule13_clear_streak >= self.config.overtaking_clear_observations_required:
                self._rule13_scaffold = None
                self._rule13_overtaker_id = None
                self._rule13_overtaken_id = None
                self._rule13_clear_streak = 0
                overtaking_past = False
                overtaking_clear = False

        if evidence.responsibility_delegated:
            effective_scaffold = None
        elif self._rule13_scaffold is not None:
            effective_scaffold = self._rule13_scaffold
        elif self._risk_active and self._risk_scaffold is not None:
            effective_scaffold = self._risk_scaffold
        else:
            effective_scaffold = None

        return EncounterLifecycleDecision(
            observation=observation,
            classification=classification,
            current_scaffold=current_scaffold,
            effective_scaffold=effective_scaffold,
            snapshot_risk=snapshot_risk,
            bearing_trend=trend,
            risk_active=self._risk_active,
            risk_basis=risk_basis,
            risk_clear_streak=self._risk_clear_streak,
            rule13_persisted=self._rule13_scaffold is not None,
            overtaking_past=overtaking_past,
            overtaking_clear=overtaking_clear,
            overtaking_clear_streak=self._rule13_clear_streak,
            responsibility_delegated=evidence.responsibility_delegated,
        )

