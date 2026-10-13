"""Certified grid slot solver for the two-stream PSP v1 prediction domain.

This module deliberately implements only planned zero-drift, straight, constant-speed
motion and delegates continuous-time pair-domain proof to the frozen CROSS D0
``certify_pair_domain_clearance`` primitive.  It does not implement B4/B5/B6 bindings.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from math import cos, isfinite, pi, sin
from typing import Mapping

from vessel_crossing.nextgen.domain_clearance import (
    DomainClearanceCertification,
    certify_pair_domain_clearance,
)
from vessel_crossing.nextgen.ship_domain import ShipDomain

from .contract import SLOT_SOLVER_CONTRACT_ID, SafeSlotNumericalContract

SOLVER_ALGORITHM_ID = "cross-safe-conflict-slot-grid-v1"
CERTIFIED_GRID_SLOT = "CERTIFIED_GRID_SLOT"
NO_CERTIFIED_SLOT_WITHIN_SEARCH = "NO_CERTIFIED_SLOT_WITHIN_SEARCH"

_POINT_CERTIFIED = "CERTIFIED_CLEAR"
_POINT_NOT_CERTIFIED = "NOT_CERTIFIED"
_POINT_NO_OVERLAP = "CLEAR_BY_NONOVERLAP"
_EPS = 1e-12


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Mapping[str, object]) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _finite_positive(name: str, value: float) -> float:
    out = float(value)
    if not isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return out


def _finite_nonnegative(name: str, value: float) -> float:
    out = float(value)
    if not isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return out


def _heading_unit_vector(heading_deg: float) -> tuple[float, float]:
    theta = heading_deg * pi / 180.0
    return (sin(theta), cos(theta))


def _domain_dict(domain: ShipDomain) -> dict[str, float]:
    return {
        "ahead_nm": float(domain.ahead_nm),
        "astern_nm": float(domain.astern_nm),
        "port_nm": float(domain.port_nm),
        "starboard_nm": float(domain.starboard_nm),
    }


def _clearance_dict(
    certificate: DomainClearanceCertification | None,
) -> dict[str, object] | None:
    if certificate is None:
        return None
    return {
        "evaluated_duration_min": certificate.evaluated_duration_min,
        "sampled_min_margin_nm": certificate.sampled_min_margin_nm,
        "conservative_lower_bound_nm": certificate.conservative_lower_bound_nm,
        "sampled_min_separation_nm": certificate.sampled_min_separation_nm,
        "sampled_max_required_distance_nm": certificate.sampled_max_required_distance_nm,
        "state_evaluations": certificate.state_evaluations,
        "certified_clear": certificate.certified_clear,
    }


@dataclass(frozen=True, slots=True)
class LinearMotionState:
    """Minimal exact state consumed by the D0 domain-clearance certificate."""

    position: tuple[float, float]
    heading_deg: float


@dataclass(frozen=True, slots=True)
class PlannedCrossingMotion:
    """Immutable planned straight motion through one conflict center."""

    stream_id: str
    heading_deg: float
    speed_nm_per_min: float
    source_to_center_distance_nm: float
    center_to_exit_distance_nm: float
    ship_domain: ShipDomain
    profile_physics_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, str) or not self.stream_id.strip():
            raise ValueError("stream_id must be a non-empty string")
        if not isinstance(self.profile_physics_identity, str) or not self.profile_physics_identity.strip():
            raise ValueError("profile_physics_identity must be a non-empty string")
        heading = float(self.heading_deg)
        if not isfinite(heading):
            raise ValueError("heading_deg must be finite")
        object.__setattr__(self, "heading_deg", heading % 360.0)
        object.__setattr__(self, "speed_nm_per_min", _finite_positive("speed_nm_per_min", self.speed_nm_per_min))
        object.__setattr__(
            self,
            "source_to_center_distance_nm",
            _finite_positive("source_to_center_distance_nm", self.source_to_center_distance_nm),
        )
        object.__setattr__(
            self,
            "center_to_exit_distance_nm",
            _finite_positive("center_to_exit_distance_nm", self.center_to_exit_distance_nm),
        )
        if not isinstance(self.ship_domain, ShipDomain):
            raise TypeError("ship_domain must be a canonical ShipDomain")

    def active_interval(self, *, center_time_min: float) -> tuple[float, float]:
        center = float(center_time_min)
        if not isfinite(center):
            raise ValueError("center_time_min must be finite")
        return (
            center - self.source_to_center_distance_nm / self.speed_nm_per_min,
            center + self.center_to_exit_distance_nm / self.speed_nm_per_min,
        )

    def state_at(self, *, global_time_min: float, center_time_min: float) -> LinearMotionState:
        time = float(global_time_min)
        center = float(center_time_min)
        if not isfinite(time) or not isfinite(center):
            raise ValueError("motion times must be finite")
        ux, uy = _heading_unit_vector(self.heading_deg)
        displacement = self.speed_nm_per_min * (time - center)
        return LinearMotionState(
            position=(ux * displacement, uy * displacement),
            heading_deg=self.heading_deg,
        )

    def canonical_dict(self) -> dict[str, object]:
        return {
            "stream_id": self.stream_id,
            "heading_deg": self.heading_deg,
            "speed_nm_per_min": self.speed_nm_per_min,
            "source_to_center_distance_nm": self.source_to_center_distance_nm,
            "center_to_exit_distance_nm": self.center_to_exit_distance_nm,
            "ship_domain": _domain_dict(self.ship_domain),
            "profile_physics_identity": self.profile_physics_identity,
        }


@dataclass(frozen=True, slots=True)
class SafeSlotPointEvaluation:
    """Audit evidence for one declared grid point."""

    grid_index: int
    separation_min: float
    status: str
    overlap_start_min: float | None
    overlap_end_min: float | None
    clearance: DomainClearanceCertification | None

    @property
    def certified_clear(self) -> bool:
        return self.status in {_POINT_CERTIFIED, _POINT_NO_OVERLAP}

    @property
    def clear_by_nonoverlap(self) -> bool:
        return self.status == _POINT_NO_OVERLAP

    def canonical_dict(self) -> dict[str, object]:
        return {
            "grid_index": self.grid_index,
            "separation_min": self.separation_min,
            "status": self.status,
            "overlap_start_min": self.overlap_start_min,
            "overlap_end_min": self.overlap_end_min,
            "clear_by_nonoverlap": self.clear_by_nonoverlap,
            "clearance": _clearance_dict(self.clearance),
        }


@dataclass(frozen=True, slots=True)
class SafeConflictSlotCertificate:
    """Deterministic solver evidence for one ordered predecessor->candidate pair."""

    status: str
    solver_algorithm_id: str
    solver_contract_id: str
    numerical_contract_sha256: str
    numerical_contract: SafeSlotNumericalContract
    predecessor_identity: str
    candidate_identity: str
    separation_min: float | None
    grid_index: int | None
    evaluated_candidate_count: int
    certified_candidate_count: int
    not_certified_candidate_count: int
    selected_overlap_start_min: float | None
    selected_overlap_end_min: float | None
    selected_clear_by_nonoverlap: bool
    selected_clearance: DomainClearanceCertification | None
    cache_key_sha256: str

    @property
    def certified(self) -> bool:
        return self.status == CERTIFIED_GRID_SLOT

    def canonical_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "solver_algorithm_id": self.solver_algorithm_id,
            "solver_contract_id": self.solver_contract_id,
            "numerical_contract_sha256": self.numerical_contract_sha256,
            "numerical_contract": {
                "separation_grid_min": self.numerical_contract.separation_grid_min,
                "max_search_min": self.numerical_contract.max_search_min,
                "clearance_time_tolerance_min": self.numerical_contract.clearance_time_tolerance_min,
                "clearance_margin_tolerance_nm": self.numerical_contract.clearance_margin_tolerance_nm,
                "max_clearance_subdivisions": self.numerical_contract.max_clearance_subdivisions,
            },
            "predecessor_identity": self.predecessor_identity,
            "candidate_identity": self.candidate_identity,
            "separation_min": self.separation_min,
            "grid_index": self.grid_index,
            "evaluated_candidate_count": self.evaluated_candidate_count,
            "certified_candidate_count": self.certified_candidate_count,
            "not_certified_candidate_count": self.not_certified_candidate_count,
            "selected_overlap_start_min": self.selected_overlap_start_min,
            "selected_overlap_end_min": self.selected_overlap_end_min,
            "selected_clear_by_nonoverlap": self.selected_clear_by_nonoverlap,
            "selected_clearance": _clearance_dict(self.selected_clearance),
            "cache_key_sha256": self.cache_key_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.canonical_dict())

    @property
    def evidence_sha256(self) -> str:
        return sha256(self.canonical_bytes()).hexdigest()


class SafeSlotTable:
    """In-memory deterministic cache; it never changes solver evidence bytes."""

    def __init__(self) -> None:
        self._rows: dict[str, SafeConflictSlotCertificate] = {}

    def get(self, key_sha256: str) -> SafeConflictSlotCertificate | None:
        return self._rows.get(key_sha256)

    def put(self, certificate: SafeConflictSlotCertificate) -> None:
        existing = self._rows.get(certificate.cache_key_sha256)
        if existing is not None and existing.canonical_bytes() != certificate.canonical_bytes():
            raise RuntimeError("deterministic safe-slot cache collision/evidence mismatch")
        self._rows[certificate.cache_key_sha256] = certificate

    def __len__(self) -> int:
        return len(self._rows)


class SafeConflictSlotSolver:
    """Find the earliest certified point on the declared center-separation grid."""

    def __init__(
        self,
        numerical_contract: SafeSlotNumericalContract,
        *,
        table: SafeSlotTable | None = None,
    ) -> None:
        if not isinstance(numerical_contract, SafeSlotNumericalContract):
            raise TypeError("numerical_contract must be SafeSlotNumericalContract")
        self.numerical_contract = numerical_contract
        self.table = table
        self._numerical_dict = {
            "separation_grid_min": numerical_contract.separation_grid_min,
            "max_search_min": numerical_contract.max_search_min,
            "clearance_time_tolerance_min": numerical_contract.clearance_time_tolerance_min,
            "clearance_margin_tolerance_nm": numerical_contract.clearance_margin_tolerance_nm,
            "max_clearance_subdivisions": numerical_contract.max_clearance_subdivisions,
        }
        self.numerical_contract_sha256 = _sha(self._numerical_dict)

    @staticmethod
    def _validate_qualified_pair(
        predecessor: PlannedCrossingMotion,
        candidate: PlannedCrossingMotion,
    ) -> None:
        if predecessor.stream_id == candidate.stream_id:
            raise ValueError(
                "solver v1 is qualified only for distinct competing streams; "
                "same-stream safety remains canonical source/FIFO responsibility"
            )
        raw = abs(predecessor.heading_deg - candidate.heading_deg) % 360.0
        angle = min(raw, 360.0 - raw)
        if abs(angle - 90.0) > 1e-10:
            raise ValueError("solver v1 is qualified only for orthogonal stream headings")

    def cache_key_sha256(
        self,
        predecessor: PlannedCrossingMotion,
        candidate: PlannedCrossingMotion,
    ) -> str:
        return _sha(
            {
                "solver_algorithm_id": SOLVER_ALGORITHM_ID,
                "solver_contract_id": SLOT_SOLVER_CONTRACT_ID,
                "numerical_contract_sha256": self.numerical_contract_sha256,
                "predecessor": predecessor.canonical_dict(),
                "candidate": candidate.canonical_dict(),
            }
        )

    def evaluate_separation(
        self,
        predecessor: PlannedCrossingMotion,
        candidate: PlannedCrossingMotion,
        *,
        grid_index: int,
        separation_min: float,
    ) -> SafeSlotPointEvaluation:
        self._validate_qualified_pair(predecessor, candidate)
        if isinstance(grid_index, bool) or not isinstance(grid_index, int) or grid_index < 0:
            raise ValueError("grid_index must be a non-negative integer")
        delta = _finite_nonnegative("separation_min", separation_min)

        predecessor_start, predecessor_end = predecessor.active_interval(center_time_min=0.0)
        candidate_start, candidate_end = candidate.active_interval(center_time_min=delta)
        overlap_start = max(predecessor_start, candidate_start)
        overlap_end = min(predecessor_end, candidate_end)

        if overlap_end < overlap_start - _EPS:
            return SafeSlotPointEvaluation(
                grid_index=grid_index,
                separation_min=delta,
                status=_POINT_NO_OVERLAP,
                overlap_start_min=None,
                overlap_end_min=None,
                clearance=None,
            )

        # A one-instant touch is a real simultaneous boundary state and can be sent
        # through the canonical duration=0 certificate.  Strict non-overlap above is
        # the only by-construction shortcut.
        duration = max(0.0, overlap_end - overlap_start)

        def first_state_at(relative_time_min: float) -> LinearMotionState:
            return predecessor.state_at(
                global_time_min=overlap_start + relative_time_min,
                center_time_min=0.0,
            )

        def second_state_at(relative_time_min: float) -> LinearMotionState:
            return candidate.state_at(
                global_time_min=overlap_start + relative_time_min,
                center_time_min=delta,
            )

        clearance = certify_pair_domain_clearance(
            first_state_at=first_state_at,
            second_state_at=second_state_at,
            first_domain=predecessor.ship_domain,
            second_domain=candidate.ship_domain,
            duration_min=duration,
            first_speed_bound_nm_per_min=predecessor.speed_nm_per_min,
            second_speed_bound_nm_per_min=candidate.speed_nm_per_min,
            first_turn_rate_bound_deg_per_min=0.0,
            second_turn_rate_bound_deg_per_min=0.0,
            time_tolerance_min=self.numerical_contract.clearance_time_tolerance_min,
            margin_tolerance_nm=self.numerical_contract.clearance_margin_tolerance_nm,
            max_subdivisions=self.numerical_contract.max_clearance_subdivisions,
        )
        return SafeSlotPointEvaluation(
            grid_index=grid_index,
            separation_min=delta,
            status=_POINT_CERTIFIED if clearance.certified_clear else _POINT_NOT_CERTIFIED,
            overlap_start_min=overlap_start,
            overlap_end_min=overlap_end,
            clearance=clearance,
        )

    def solve(
        self,
        predecessor: PlannedCrossingMotion,
        candidate: PlannedCrossingMotion,
        *,
        use_cache: bool = True,
    ) -> SafeConflictSlotCertificate:
        if not isinstance(predecessor, PlannedCrossingMotion) or not isinstance(candidate, PlannedCrossingMotion):
            raise TypeError("predecessor and candidate must be PlannedCrossingMotion")
        self._validate_qualified_pair(predecessor, candidate)
        key = self.cache_key_sha256(predecessor, candidate)
        if use_cache and self.table is not None:
            cached = self.table.get(key)
            if cached is not None:
                return cached

        grid = Decimal(str(self.numerical_contract.separation_grid_min))
        max_search = Decimal(str(self.numerical_contract.max_search_min))
        max_index = int(max_search // grid)
        evaluated = 0
        not_certified = 0
        selected: SafeSlotPointEvaluation | None = None

        for grid_index in range(max_index + 1):
            separation = float(grid * grid_index)
            point = self.evaluate_separation(
                predecessor,
                candidate,
                grid_index=grid_index,
                separation_min=separation,
            )
            evaluated += 1
            if point.certified_clear:
                selected = point
                break
            not_certified += 1

        if selected is None:
            result = SafeConflictSlotCertificate(
                status=NO_CERTIFIED_SLOT_WITHIN_SEARCH,
                solver_algorithm_id=SOLVER_ALGORITHM_ID,
                solver_contract_id=SLOT_SOLVER_CONTRACT_ID,
                numerical_contract_sha256=self.numerical_contract_sha256,
                numerical_contract=self.numerical_contract,
                predecessor_identity=predecessor.profile_physics_identity,
                candidate_identity=candidate.profile_physics_identity,
                separation_min=None,
                grid_index=None,
                evaluated_candidate_count=evaluated,
                certified_candidate_count=0,
                not_certified_candidate_count=not_certified,
                selected_overlap_start_min=None,
                selected_overlap_end_min=None,
                selected_clear_by_nonoverlap=False,
                selected_clearance=None,
                cache_key_sha256=key,
            )
        else:
            result = SafeConflictSlotCertificate(
                status=CERTIFIED_GRID_SLOT,
                solver_algorithm_id=SOLVER_ALGORITHM_ID,
                solver_contract_id=SLOT_SOLVER_CONTRACT_ID,
                numerical_contract_sha256=self.numerical_contract_sha256,
                numerical_contract=self.numerical_contract,
                predecessor_identity=predecessor.profile_physics_identity,
                candidate_identity=candidate.profile_physics_identity,
                separation_min=selected.separation_min,
                grid_index=selected.grid_index,
                evaluated_candidate_count=evaluated,
                certified_candidate_count=1,
                not_certified_candidate_count=not_certified,
                selected_overlap_start_min=selected.overlap_start_min,
                selected_overlap_end_min=selected.overlap_end_min,
                selected_clear_by_nonoverlap=selected.clear_by_nonoverlap,
                selected_clearance=selected.clearance,
                cache_key_sha256=key,
            )

        if use_cache and self.table is not None:
            self.table.put(result)
        return result
