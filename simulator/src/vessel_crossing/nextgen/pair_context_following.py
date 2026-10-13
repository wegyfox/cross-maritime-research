"""Typed pair-context extensions for same-route longitudinal following.

This module promotes the exact response decomposition used by the accepted CROSS R2
V4/V5 campaigns into a reusable NextGen component.  It does not prescribe target
headways.  Instead it adds non-negative response allowances to the existing physical
same-route following budget based on follower type, leader type, and the ordered
(follower_type, leader_type) pair.

The component is deliberately generic with respect to type labels.  ``AS``/``HS`` are
one research use; later behavior studies may use other labels without changing the
physical following kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from .same_route_following import SameRouteFollowingConfig, SameRouteFollowingSupervisor

_EPS = 1e-10
_MECHANISM_ID = "CROSS_PAIR_CONTEXT_MICRO_RESPONSE_HEADWAY_V1"


class PairContextTypeUnavailable(ValueError):
    """A typed pair-context rule cannot be resolved for a concrete vessel pair."""


def _validated_label_map(value: Mapping[str, str] | None) -> MappingProxyType:
    raw = {} if value is None else dict(value)
    for vessel_id, label in raw.items():
        if not isinstance(vessel_id, str) or not vessel_id:
            raise ValueError("vessel type-map ids must be non-empty strings")
        if not isinstance(label, str) or not label:
            raise ValueError("vessel type labels must be non-empty strings")
    return MappingProxyType({str(k): str(v) for k, v in raw.items()})


def _validated_type_interval_map(
    value: Mapping[str, float] | None,
    *,
    name: str,
) -> MappingProxyType:
    raw = {} if value is None else dict(value)
    for label, interval in raw.items():
        if not isinstance(label, str) or not label:
            raise ValueError(f"{name} type labels must be non-empty strings")
        if not isfinite(interval) or interval < 0.0:
            raise ValueError(f"{name} intervals must be finite and non-negative")
    return MappingProxyType({str(k): float(v) for k, v in raw.items()})


def _validated_pair_interval_map(
    value: Mapping[tuple[str, str], float] | None,
) -> MappingProxyType:
    raw = {} if value is None else dict(value)
    normalized: dict[tuple[str, str], float] = {}
    for key, interval in raw.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not all(isinstance(label, str) and label for label in key)
        ):
            raise ValueError(
                "pair-context keys must be (follower_type, leader_type) string tuples"
            )
        if not isfinite(interval) or interval < 0.0:
            raise ValueError("pair-context allowances must be finite and non-negative")
        normalized[(str(key[0]), str(key[1]))] = float(interval)
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class PairContextResponseConfig:
    """Immutable typed response allowances layered on same-route following.

    ``follower_extra_min_by_type`` reproduces the accepted V3/V4/V5 follower-response
    axis. ``leader_extra_min_by_type`` reproduces the leader-observation axis.
    ``pair_context_extra_min_by_type_pair`` is the non-additive ordered-pair term from
    V4/V5.  All terms are response allowances in minutes, not target headways.
    """

    base_following: SameRouteFollowingConfig = SameRouteFollowingConfig()
    vessel_type_by_id: Mapping[str, str] = field(default_factory=dict)
    follower_extra_min_by_type: Mapping[str, float] = field(default_factory=dict)
    leader_extra_min_by_type: Mapping[str, float] = field(default_factory=dict)
    pair_context_extra_min_by_type_pair: Mapping[tuple[str, str], float] = field(
        default_factory=dict
    )
    mechanism_id: str = _MECHANISM_ID

    def __post_init__(self) -> None:
        if not isinstance(self.base_following, SameRouteFollowingConfig):
            raise TypeError("base_following must be SameRouteFollowingConfig")
        if self.mechanism_id != _MECHANISM_ID:
            raise ValueError(f"mechanism_id must remain {_MECHANISM_ID!r}")
        object.__setattr__(
            self,
            "vessel_type_by_id",
            _validated_label_map(self.vessel_type_by_id),
        )
        object.__setattr__(
            self,
            "follower_extra_min_by_type",
            _validated_type_interval_map(
                self.follower_extra_min_by_type,
                name="follower-extra",
            ),
        )
        object.__setattr__(
            self,
            "leader_extra_min_by_type",
            _validated_type_interval_map(
                self.leader_extra_min_by_type,
                name="leader-extra",
            ),
        )
        object.__setattr__(
            self,
            "pair_context_extra_min_by_type_pair",
            _validated_pair_interval_map(self.pair_context_extra_min_by_type_pair),
        )

    def identity_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-safe mechanism/config identity payload."""
        return {
            "mechanism_id": self.mechanism_id,
            "base_following": {
                "reaction_interval_min": self.base_following.reaction_interval_min,
                "near_route_cross_track_tolerance_nm": (
                    self.base_following.near_route_cross_track_tolerance_nm
                ),
                "heading_tolerance_deg": self.base_following.heading_tolerance_deg,
                "require_zero_environmental_drift": (
                    self.base_following.require_zero_environmental_drift
                ),
            },
            "vessel_type_by_id": dict(sorted(self.vessel_type_by_id.items())),
            "follower_extra_min_by_type": dict(
                sorted(self.follower_extra_min_by_type.items())
            ),
            "leader_extra_min_by_type": dict(
                sorted(self.leader_extra_min_by_type.items())
            ),
            "pair_context_extra_min_by_type_pair": [
                {
                    "follower_type": follower_type,
                    "leader_type": leader_type,
                    "extra_min": interval,
                }
                for (follower_type, leader_type), interval in sorted(
                    self.pair_context_extra_min_by_type_pair.items()
                )
            ],
        }

    def identity_sha256(self) -> str:
        raw = (
            json.dumps(
                self.identity_payload(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class PairContextResponseEvaluation:
    """Transparent decomposition of one ordered follower/leader response budget."""

    follower_id: str
    leader_id: str
    follower_type: str | None
    leader_type: str | None
    base_reaction_interval_min: float
    follower_extra_min: float
    leader_extra_min: float
    pair_context_extra_min: float

    @property
    def total_reaction_interval_min(self) -> float:
        return (
            self.base_reaction_interval_min
            + self.follower_extra_min
            + self.leader_extra_min
            + self.pair_context_extra_min
        )

    @property
    def pair_type(self) -> str | None:
        if self.follower_type is None or self.leader_type is None:
            return None
        return f"{self.follower_type}_{self.leader_type}"

    def to_json_object(self) -> dict[str, object]:
        return {
            "follower_id": self.follower_id,
            "leader_id": self.leader_id,
            "follower_type": self.follower_type,
            "leader_type": self.leader_type,
            "pair_type": self.pair_type,
            "base_reaction_interval_min": self.base_reaction_interval_min,
            "follower_extra_min": self.follower_extra_min,
            "leader_extra_min": self.leader_extra_min,
            "pair_context_extra_min": self.pair_context_extra_min,
            "total_reaction_interval_min": self.total_reaction_interval_min,
        }


class PairContextFollowingSupervisor(SameRouteFollowingSupervisor):
    """Same-route supervisor with typed follower/leader/pair response allowances."""

    def __init__(self, response: PairContextResponseConfig) -> None:
        if not isinstance(response, PairContextResponseConfig):
            raise TypeError("response must be PairContextResponseConfig")

        follower_by_vessel: dict[str, float] = {}
        leader_by_vessel: dict[str, float] = {}
        for vessel_id, label in response.vessel_type_by_id.items():
            follower_extra = response.follower_extra_min_by_type.get(label, 0.0)
            leader_extra = response.leader_extra_min_by_type.get(label, 0.0)
            if follower_extra > _EPS:
                follower_by_vessel[vessel_id] = follower_extra
            if leader_extra > _EPS:
                leader_by_vessel[vessel_id] = leader_extra

        super().__init__(
            response.base_following,
            additional_reaction_interval_min_by_vessel=follower_by_vessel,
            additional_leader_observation_interval_min_by_vessel=leader_by_vessel,
        )
        self.response = response

    @property
    def mechanism_id(self) -> str:
        return self.response.mechanism_id

    @property
    def config_identity_sha256(self) -> str:
        return self.response.identity_sha256()

    def evaluate_pair(
        self, follower_id: str, leader_id: str
    ) -> PairContextResponseEvaluation:
        """Return the exact response decomposition used by ``config_for_pair``.

        The method is observation-only.  It exposes the already accepted V4/V5
        mechanism contributions for diagnostics without introducing a target headway or
        changing same-route following semantics.
        """

        if not isinstance(follower_id, str) or not follower_id:
            raise ValueError("follower_id must be a non-empty string")
        if not isinstance(leader_id, str) or not leader_id:
            raise ValueError("leader_id must be a non-empty string")

        follower_type = self.response.vessel_type_by_id.get(follower_id)
        leader_type = self.response.vessel_type_by_id.get(leader_id)
        if self.response.pair_context_extra_min_by_type_pair and (
            follower_type is None or leader_type is None
        ):
            raise PairContextTypeUnavailable(
                "pair-context response requires a type label for both vessels"
            )

        follower_extra = (
            0.0
            if follower_type is None
            else self.response.follower_extra_min_by_type.get(follower_type, 0.0)
        )
        leader_extra = (
            0.0
            if leader_type is None
            else self.response.leader_extra_min_by_type.get(leader_type, 0.0)
        )
        pair_extra = (
            0.0
            if follower_type is None or leader_type is None
            else self.response.pair_context_extra_min_by_type_pair.get(
                (follower_type, leader_type), 0.0
            )
        )
        return PairContextResponseEvaluation(
            follower_id=follower_id,
            leader_id=leader_id,
            follower_type=follower_type,
            leader_type=leader_type,
            base_reaction_interval_min=self.response.base_following.reaction_interval_min,
            follower_extra_min=follower_extra,
            leader_extra_min=leader_extra,
            pair_context_extra_min=pair_extra,
        )

    def pair_context_extra_min_for_pair(self, follower_id: str, leader_id: str) -> float:
        return self.evaluate_pair(follower_id, leader_id).pair_context_extra_min

    def config_for_pair(self, follower_id: str, leader_id: str) -> SameRouteFollowingConfig:
        config = super().config_for_pair(follower_id, leader_id)
        extra = self.pair_context_extra_min_for_pair(follower_id, leader_id)
        if extra <= _EPS:
            return config
        return replace(
            config,
            reaction_interval_min=config.reaction_interval_min + extra,
        )
