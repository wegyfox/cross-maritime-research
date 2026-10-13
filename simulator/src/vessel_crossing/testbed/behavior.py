"""C3 pluggable behavior-model interface for heterogeneous traffic agents.

Behavior models are scenario/testbed infrastructure.  They produce normalized behavior
outputs but do not own A162 physics, perception, COLREG interpretation or evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import atan2, degrees, hypot, isfinite
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

from vessel_crossing.kinematics import nm_per_minute_to_knots
from vessel_crossing.nextgen.dynamics import ManeuverCommand
from vessel_crossing.nextgen.recorded_track import RecordedVesselTrack
from vessel_crossing.nextgen.recorded_track_kinematics import (
    RecordedTrackTimeSide,
    recorded_ground_motion_at,
)

from .scope import TestbedFeature
from .world_objects import Point2D

_EPS = 1e-12


class BehaviorProfile(str, Enum):
    COMPLIANT_MERCHANT = "COMPLIANT_MERCHANT"
    AUTONOMOUS_POLICY = "AUTONOMOUS_POLICY"
    DELAYED_ACTION = "DELAYED_ACTION"
    NO_ACTION = "NO_ACTION"
    ROUTE_FIXATED = "ROUTE_FIXATED"
    EXPECT_OTHER_TO_GIVE_WAY = "EXPECT_OTHER_TO_GIVE_WAY"
    FISHING_ACTIVITY = "FISHING_ACTIVITY"
    ERRATIC_OR_INTERMITTENT_COURSE = "ERRATIC_OR_INTERMITTENT_COURSE"
    REPLAYED_TRAJECTORY = "REPLAYED_TRAJECTORY"


class BehaviorDecisionMode(str, Enum):
    COMMAND = "COMMAND"
    REPLAY_STATE = "REPLAY_STATE"


@dataclass(frozen=True, slots=True)
class BehaviorModelSpec:
    profile: BehaviorProfile
    adapter_id: str
    seed: int | None = None
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.adapter_id:
            raise ValueError("adapter_id must be non-empty")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if tuple(sorted(self.parameters)) != self.parameters:
            raise ValueError("behavior parameters must be sorted for canonical identity")

    def identity_sha256(self) -> str:
        payload = {
            "profile": self.profile.value,
            "adapter_id": self.adapter_id,
            "seed": self.seed,
            "parameters": [list(item) for item in self.parameters],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class BehaviorState:
    vessel_id: str
    time_min: float
    x_east_nm: float
    y_north_nm: float
    heading_deg: float
    speed_knots: float
    route_heading_deg: float | None = None
    route_speed_knots: float | None = None

    def __post_init__(self) -> None:
        if not self.vessel_id:
            raise ValueError("vessel_id must be non-empty")
        for name in ("time_min", "x_east_nm", "y_north_nm", "heading_deg", "speed_knots"):
            if not isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.time_min < 0:
            raise ValueError("time_min must be non-negative")
        if not 0 <= self.heading_deg <= 360:
            raise ValueError("heading_deg must be within [0, 360]")
        if self.speed_knots < 0:
            raise ValueError("speed_knots must be non-negative")
        if self.route_heading_deg is not None and (
            not isfinite(self.route_heading_deg) or not 0 <= self.route_heading_deg <= 360
        ):
            raise ValueError("route_heading_deg must be within [0, 360]")
        if self.route_speed_knots is not None and (
            not isfinite(self.route_speed_knots) or self.route_speed_knots < 0
        ):
            raise ValueError("route_speed_knots must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RequestedManeuver:
    requested_at_min: float
    target_heading_deg: float
    target_speed_knots: float
    source: str = "external"

    def __post_init__(self) -> None:
        if not isfinite(self.requested_at_min) or self.requested_at_min < 0:
            raise ValueError("requested_at_min must be finite and non-negative")
        if not isfinite(self.target_heading_deg) or not 0 <= self.target_heading_deg <= 360:
            raise ValueError("target_heading_deg must be within [0, 360]")
        if not isfinite(self.target_speed_knots) or self.target_speed_knots < 0:
            raise ValueError("target_speed_knots must be finite and non-negative")
        if not self.source:
            raise ValueError("source must be non-empty")


@dataclass(frozen=True, slots=True)
class BehaviorContext:
    state: BehaviorState
    requested_maneuver: RequestedManeuver | None = None


@dataclass(frozen=True, slots=True)
class BehaviorDecision:
    profile: BehaviorProfile
    mode: BehaviorDecisionMode
    reason: str
    target_heading_deg: float | None = None
    target_speed_knots: float | None = None
    replay_position: tuple[float, float] | None = None
    replay_course_deg: float | None = None
    replay_speed_knots: float | None = None

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("reason must be non-empty")
        if self.mode is BehaviorDecisionMode.COMMAND:
            if self.target_heading_deg is None or self.target_speed_knots is None:
                raise ValueError("COMMAND decision requires target heading and speed")
            if self.replay_position is not None or self.replay_course_deg is not None or self.replay_speed_knots is not None:
                raise ValueError("COMMAND decision cannot contain replay state")
            if not isfinite(self.target_heading_deg) or not 0 <= self.target_heading_deg <= 360:
                raise ValueError("target_heading_deg must be within [0, 360]")
            if not isfinite(self.target_speed_knots) or self.target_speed_knots < 0:
                raise ValueError("target_speed_knots must be finite and non-negative")
        else:
            if self.replay_position is None or self.replay_speed_knots is None:
                raise ValueError("REPLAY_STATE decision requires position and speed")
            if self.target_heading_deg is not None or self.target_speed_knots is not None:
                raise ValueError("REPLAY_STATE decision cannot contain command targets")
            if len(self.replay_position) != 2 or not all(isfinite(v) for v in self.replay_position):
                raise ValueError("replay_position must contain two finite coordinates")
            if self.replay_course_deg is not None and (
                not isfinite(self.replay_course_deg) or not 0 <= self.replay_course_deg <= 360
            ):
                raise ValueError("replay_course_deg must be within [0, 360]")
            if not isfinite(self.replay_speed_knots) or self.replay_speed_knots < 0:
                raise ValueError("replay_speed_knots must be finite and non-negative")

    @classmethod
    def command(
        cls,
        *,
        profile: BehaviorProfile,
        heading_deg: float,
        speed_knots: float,
        reason: str,
    ) -> "BehaviorDecision":
        return cls(
            profile=profile,
            mode=BehaviorDecisionMode.COMMAND,
            reason=reason,
            target_heading_deg=heading_deg % 360.0,
            target_speed_knots=speed_knots,
        )

    @classmethod
    def replay(
        cls,
        *,
        position: tuple[float, float],
        course_deg: float | None,
        speed_knots: float,
        reason: str = "recorded trajectory",
    ) -> "BehaviorDecision":
        return cls(
            profile=BehaviorProfile.REPLAYED_TRAJECTORY,
            mode=BehaviorDecisionMode.REPLAY_STATE,
            reason=reason,
            replay_position=position,
            replay_course_deg=course_deg,
            replay_speed_knots=speed_knots,
        )


class BehaviorModel(Protocol):
    component_id: TestbedFeature
    spec: BehaviorModelSpec

    def decide(self, context: BehaviorContext) -> BehaviorDecision:
        ...


@dataclass(frozen=True, slots=True)
class NoActionBehavior:
    spec: BehaviorModelSpec = BehaviorModelSpec(BehaviorProfile.NO_ACTION, "cross-c3/no-action/v1")
    component_id = TestbedFeature.C3_BEHAVIOR_MODELS

    def decide(self, context: BehaviorContext) -> BehaviorDecision:
        state = context.state
        return BehaviorDecision.command(
            profile=self.spec.profile,
            heading_deg=state.heading_deg,
            speed_knots=state.speed_knots,
            reason="hold current course and speed",
        )


@dataclass(frozen=True, slots=True)
class RouteFixatedBehavior:
    expect_other_to_give_way: bool = False
    component_id = TestbedFeature.C3_BEHAVIOR_MODELS

    @property
    def spec(self) -> BehaviorModelSpec:
        profile = (
            BehaviorProfile.EXPECT_OTHER_TO_GIVE_WAY
            if self.expect_other_to_give_way
            else BehaviorProfile.ROUTE_FIXATED
        )
        return BehaviorModelSpec(profile, "cross-c3/route-fixated/v1")

    def decide(self, context: BehaviorContext) -> BehaviorDecision:
        state = context.state
        if state.route_heading_deg is None or state.route_speed_knots is None:
            raise ValueError("route-fixated behavior requires route heading and route speed")
        reason = "maintain route; expects other vessel to give way" if self.expect_other_to_give_way else "maintain route target"
        return BehaviorDecision.command(
            profile=self.spec.profile,
            heading_deg=state.route_heading_deg,
            speed_knots=state.route_speed_knots,
            reason=reason,
        )


Delegate = Callable[[BehaviorContext], BehaviorDecision]


@dataclass(frozen=True, slots=True)
class DelegatedBehavior:
    profile: BehaviorProfile
    delegate: Delegate
    adapter_id: str
    component_id = TestbedFeature.C3_BEHAVIOR_MODELS

    def __post_init__(self) -> None:
        if self.profile not in {BehaviorProfile.COMPLIANT_MERCHANT, BehaviorProfile.AUTONOMOUS_POLICY}:
            raise ValueError("delegated profile must be COMPLIANT_MERCHANT or AUTONOMOUS_POLICY")
        if not callable(self.delegate):
            raise TypeError("delegate must be callable")
        if not self.adapter_id:
            raise ValueError("adapter_id must be non-empty")

    @property
    def spec(self) -> BehaviorModelSpec:
        return BehaviorModelSpec(self.profile, self.adapter_id)

    def decide(self, context: BehaviorContext) -> BehaviorDecision:
        decision = self.delegate(context)
        if not isinstance(decision, BehaviorDecision):
            raise TypeError("behavior delegate must return BehaviorDecision")
        if decision.profile is not self.profile:
            raise ValueError("delegated decision profile does not match adapter profile")
        return decision


@dataclass(frozen=True, slots=True)
class DelayedActionBehavior:
    delay_min: float
    component_id = TestbedFeature.C3_BEHAVIOR_MODELS

    def __post_init__(self) -> None:
        if not isfinite(self.delay_min) or self.delay_min < 0:
            raise ValueError("delay_min must be finite and non-negative")

    @property
    def spec(self) -> BehaviorModelSpec:
        return BehaviorModelSpec(
            BehaviorProfile.DELAYED_ACTION,
            "cross-c3/delayed-action/v1",
            parameters=(("delay_min", format(self.delay_min, ".17g")),),
        )

    def decide(self, context: BehaviorContext) -> BehaviorDecision:
        state = context.state
        requested = context.requested_maneuver
        if requested is None or state.time_min + _EPS < requested.requested_at_min + self.delay_min:
            return BehaviorDecision.command(
                profile=self.spec.profile,
                heading_deg=state.heading_deg,
                speed_knots=state.speed_knots,
                reason="action delay not elapsed",
            )
        return BehaviorDecision.command(
            profile=self.spec.profile,
            heading_deg=requested.target_heading_deg,
            speed_knots=requested.target_speed_knots,
            reason=f"delayed action from {requested.source}",
        )


def _seeded_unit(seed: int, vessel_id: str, bucket: int, channel: str) -> float:
    payload = f"cross-c3|{seed}|{vessel_id}|{bucket}|{channel}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64)


def _heading_to(start: Point2D, target: Point2D) -> float:
    dx = target.x_east_nm - start.x_east_nm
    dy = target.y_north_nm - start.y_north_nm
    if hypot(dx, dy) <= _EPS:
        return 0.0
    return degrees(atan2(dx, dy)) % 360.0


@dataclass(frozen=True, slots=True)
class FishingActivityBehavior:
    goals: tuple[Point2D, ...]
    seed: int
    decision_interval_min: float = 5.0
    min_speed_knots: float = 1.0
    max_speed_knots: float = 6.0
    component_id = TestbedFeature.C3_BEHAVIOR_MODELS

    def __post_init__(self) -> None:
        if not self.goals:
            raise ValueError("fishing behavior requires at least one opportunistic goal")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not isfinite(self.decision_interval_min) or self.decision_interval_min <= 0:
            raise ValueError("decision_interval_min must be finite and positive")
        if not isfinite(self.min_speed_knots) or not isfinite(self.max_speed_knots):
            raise ValueError("fishing speeds must be finite")
        if self.min_speed_knots < 0 or self.max_speed_knots < self.min_speed_knots:
            raise ValueError("invalid fishing speed bounds")

    @property
    def spec(self) -> BehaviorModelSpec:
        goal_text = ";".join(f"{p.x_east_nm:.12g},{p.y_north_nm:.12g}" for p in self.goals)
        params = tuple(sorted({
            "decision_interval_min": format(self.decision_interval_min, ".17g"),
            "goals": goal_text,
            "max_speed_knots": format(self.max_speed_knots, ".17g"),
            "min_speed_knots": format(self.min_speed_knots, ".17g"),
        }.items()))
        return BehaviorModelSpec(BehaviorProfile.FISHING_ACTIVITY, "cross-c3/fishing-activity/v1", self.seed, params)

    def decide(self, context: BehaviorContext) -> BehaviorDecision:
        state = context.state
        bucket = int(state.time_min // self.decision_interval_min)
        goal_index = min(len(self.goals) - 1, int(_seeded_unit(self.seed, state.vessel_id, bucket, "goal") * len(self.goals)))
        goal = self.goals[goal_index]
        heading = _heading_to(Point2D(state.x_east_nm, state.y_north_nm), goal)
        u = _seeded_unit(self.seed, state.vessel_id, bucket, "speed")
        speed = self.min_speed_knots + u * (self.max_speed_knots - self.min_speed_knots)
        return BehaviorDecision.command(
            profile=self.spec.profile,
            heading_deg=heading,
            speed_knots=speed,
            reason=f"opportunistic fishing goal {goal_index}",
        )


@dataclass(frozen=True, slots=True)
class ErraticIntermittentCourseBehavior:
    seed: int
    decision_interval_min: float = 3.0
    max_heading_offset_deg: float = 45.0
    min_speed_factor: float = 0.5
    max_speed_factor: float = 1.1
    hold_probability: float = 0.25
    component_id = TestbedFeature.C3_BEHAVIOR_MODELS

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not isfinite(self.decision_interval_min) or self.decision_interval_min <= 0:
            raise ValueError("decision_interval_min must be finite and positive")
        if not isfinite(self.max_heading_offset_deg) or self.max_heading_offset_deg < 0 or self.max_heading_offset_deg > 180:
            raise ValueError("max_heading_offset_deg must be within [0, 180]")
        if not isfinite(self.min_speed_factor) or not isfinite(self.max_speed_factor):
            raise ValueError("speed factors must be finite")
        if self.min_speed_factor < 0 or self.max_speed_factor < self.min_speed_factor:
            raise ValueError("invalid speed-factor bounds")
        if not isfinite(self.hold_probability) or not 0 <= self.hold_probability <= 1:
            raise ValueError("hold_probability must be within [0, 1]")

    @property
    def spec(self) -> BehaviorModelSpec:
        params = tuple(sorted({
            "decision_interval_min": format(self.decision_interval_min, ".17g"),
            "hold_probability": format(self.hold_probability, ".17g"),
            "max_heading_offset_deg": format(self.max_heading_offset_deg, ".17g"),
            "max_speed_factor": format(self.max_speed_factor, ".17g"),
            "min_speed_factor": format(self.min_speed_factor, ".17g"),
        }.items()))
        return BehaviorModelSpec(
            BehaviorProfile.ERRATIC_OR_INTERMITTENT_COURSE,
            "cross-c3/erratic-intermittent/v1",
            self.seed,
            params,
        )

    def decide(self, context: BehaviorContext) -> BehaviorDecision:
        state = context.state
        bucket = int(state.time_min // self.decision_interval_min)
        if _seeded_unit(self.seed, state.vessel_id, bucket, "hold") < self.hold_probability:
            return BehaviorDecision.command(
                profile=self.spec.profile,
                heading_deg=state.heading_deg,
                speed_knots=state.speed_knots,
                reason="intermittent course hold",
            )
        base_heading = state.route_heading_deg if state.route_heading_deg is not None else state.heading_deg
        base_speed = state.route_speed_knots if state.route_speed_knots is not None else state.speed_knots
        offset_u = _seeded_unit(self.seed, state.vessel_id, bucket, "heading")
        offset = (2.0 * offset_u - 1.0) * self.max_heading_offset_deg
        factor_u = _seeded_unit(self.seed, state.vessel_id, bucket, "speed")
        factor = self.min_speed_factor + factor_u * (self.max_speed_factor - self.min_speed_factor)
        return BehaviorDecision.command(
            profile=self.spec.profile,
            heading_deg=(base_heading + offset) % 360.0,
            speed_knots=base_speed * factor,
            reason="seeded erratic/intermittent course",
        )


@dataclass(frozen=True, slots=True)
class BehaviorScriptStep:
    at_min: float
    target_heading_deg: float
    target_speed_knots: float

    def __post_init__(self) -> None:
        if not isfinite(self.at_min) or self.at_min < 0:
            raise ValueError("script at_min must be finite and non-negative")
        if not isfinite(self.target_heading_deg) or not 0 <= self.target_heading_deg <= 360:
            raise ValueError("script heading must be within [0, 360]")
        if not isfinite(self.target_speed_knots) or self.target_speed_knots < 0:
            raise ValueError("script speed must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ScriptedBehavior:
    steps: tuple[BehaviorScriptStep, ...]
    profile: BehaviorProfile = BehaviorProfile.NO_ACTION
    component_id = TestbedFeature.C3_BEHAVIOR_MODELS

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("scripted behavior requires at least one step")
        times = tuple(step.at_min for step in self.steps)
        if times != tuple(sorted(times)) or len(times) != len(set(times)):
            raise ValueError("script step times must be strictly increasing")
        if self.profile is BehaviorProfile.REPLAYED_TRAJECTORY:
            raise ValueError("REPLAYED_TRAJECTORY uses ReplayedTrajectoryBehavior")

    @property
    def spec(self) -> BehaviorModelSpec:
        script = ";".join(f"{s.at_min:.12g},{s.target_heading_deg:.12g},{s.target_speed_knots:.12g}" for s in self.steps)
        return BehaviorModelSpec(self.profile, "cross-c3/scripted/v1", parameters=(("script", script),))

    def decide(self, context: BehaviorContext) -> BehaviorDecision:
        state = context.state
        selected: BehaviorScriptStep | None = None
        for step in self.steps:
            if step.at_min <= state.time_min + _EPS:
                selected = step
            else:
                break
        if selected is None:
            return BehaviorDecision.command(
                profile=self.profile,
                heading_deg=state.heading_deg,
                speed_knots=state.speed_knots,
                reason="before first scripted action",
            )
        return BehaviorDecision.command(
            profile=self.profile,
            heading_deg=selected.target_heading_deg,
            speed_knots=selected.target_speed_knots,
            reason=f"scripted action at {selected.at_min:.12g} min",
        )


@dataclass(frozen=True, slots=True)
class ReplayedTrajectoryBehavior:
    track: RecordedVesselTrack
    component_id = TestbedFeature.C3_BEHAVIOR_MODELS

    @property
    def spec(self) -> BehaviorModelSpec:
        raw = [
            [p.time, p.position[0], p.position[1], p.reported_course_over_ground_deg, p.reported_speed_over_ground]
            for p in self.track.points
        ]
        digest = hashlib.sha256(json.dumps(raw, separators=(",", ":")).encode("utf-8")).hexdigest()
        return BehaviorModelSpec(
            BehaviorProfile.REPLAYED_TRAJECTORY,
            "cross-c3/recorded-track/v1",
            parameters=(("track_sha256", digest), ("vessel_id", self.track.vessel_id)),
        )

    def decide(self, context: BehaviorContext) -> BehaviorDecision:
        state = context.state
        if state.vessel_id != self.track.vessel_id:
            raise ValueError("replayed track vessel_id does not match behavior state")
        side = RecordedTrackTimeSide.LEFT if abs(state.time_min - self.track.end_time) <= _EPS else RecordedTrackTimeSide.RIGHT
        sample = recorded_ground_motion_at(self.track, state.time_min, side=side)
        return BehaviorDecision.replay(
            position=sample.position,
            course_deg=sample.course_over_ground_deg,
            speed_knots=nm_per_minute_to_knots(sample.speed_over_ground),
        )


@dataclass(frozen=True, slots=True)
class BehaviorRegistry:
    models: tuple[tuple[str, BehaviorModel], ...]
    component_id = TestbedFeature.C3_BEHAVIOR_MODELS

    def __post_init__(self) -> None:
        ids = [vessel_id for vessel_id, _ in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate behavior vessel_id")
        if tuple(sorted(ids)) != tuple(ids):
            raise ValueError("behavior registry must be sorted by vessel_id for canonical identity")
        for vessel_id, model in self.models:
            if not vessel_id:
                raise ValueError("behavior vessel_id must be non-empty")
            if not hasattr(model, "spec") or not hasattr(model, "decide"):
                raise TypeError("behavior model does not satisfy C3 contract")

    @classmethod
    def from_mapping(cls, models: Mapping[str, BehaviorModel]) -> "BehaviorRegistry":
        return cls(tuple(sorted(models.items(), key=lambda item: item[0])))

    def model_for(self, vessel_id: str) -> BehaviorModel:
        for key, model in self.models:
            if key == vessel_id:
                return model
        raise KeyError(vessel_id)

    def identity_sha256(self) -> str:
        payload = [[vessel_id, model.spec.identity_sha256()] for vessel_id, model in self.models]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()

    def spec_map(self) -> Mapping[str, BehaviorModelSpec]:
        return MappingProxyType({vessel_id: model.spec for vessel_id, model in self.models})


def to_nextgen_maneuver_command(decision: BehaviorDecision) -> ManeuverCommand:
    """Normalize one C3 COMMAND decision into the inherited bounded-physics command type."""
    if decision.mode is not BehaviorDecisionMode.COMMAND:
        raise ValueError("replayed trajectory state is not a ManeuverCommand")
    assert decision.target_heading_deg is not None
    assert decision.target_speed_knots is not None
    return ManeuverCommand.from_knots(
        target_heading_deg=decision.target_heading_deg,
        target_speed_knots=decision.target_speed_knots,
    )
