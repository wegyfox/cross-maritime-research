"""Read-only mission safety recording and pair-episode aggregation.

A64 introduced continuous pair outcome metrics for one explicit bounded-command interval.
A65 connects those primitives to a mission without making metrics part of control.  The
recorder observes immutable world state/current commands before a physical advance and
never schedules a command, changes ownership, or calls the encounter/risk stack.

A safety episode is deliberately modest: a maximal contiguous run of non-``CLEAR``
intervals for one pair.  ``CLEAR`` closes an episode; a time gap also closes it.  This is
an outcome-analysis construct, not a COLREG encounter-lifecycle replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from math import isfinite
from typing import Iterable

from .safety_metrics import (
    FleetSafetyMetricSummary,
    PairSafetyIntervalMetric,
    PairSafetyMetricConfig,
    PairSafetyOutcome,
    SafetyMetricUnavailable,
    evaluate_pair_safety_interval,
    summarize_fleet_safety_metrics,
)
from .vessel_classes import VesselProfileCatalog
from .world import World2D

_EPS = 1e-10
_UNRESOLVED = {
    PairSafetyOutcome.HULL_CLEARANCE_UNRESOLVED,
    PairSafetyOutcome.SAFETY_DOMAIN_CLEARANCE_UNRESOLVED,
    PairSafetyOutcome.NEAR_MISS_UNRESOLVED,
}


@dataclass(frozen=True, slots=True)
class TimedPairSafetyIntervalMetric:
    start_time_min: float
    end_time_min: float
    metric: PairSafetyIntervalMetric

    def __post_init__(self) -> None:
        if not isfinite(self.start_time_min) or not isfinite(self.end_time_min):
            raise ValueError("safety interval times must be finite")
        if self.end_time_min < self.start_time_min - _EPS:
            raise ValueError("safety interval end must not precede start")
        if abs((self.end_time_min - self.start_time_min) - self.metric.duration_min) > 1e-8:
            raise ValueError("timed interval duration must match pair metric duration")

    @property
    def pair_key(self) -> tuple[str, str]:
        return self.metric.pair_key

    @property
    def outcome(self) -> PairSafetyOutcome:
        return self.metric.outcome


@dataclass(frozen=True, slots=True)
class PairSafetyEpisode:
    first_vessel_id: str
    second_vessel_id: str
    start_time_min: float
    end_time_min: float
    interval_count: int
    outcomes: tuple[PairSafetyOutcome, ...]
    unresolved_interval_count: int
    sampled_min_center_separation_nm: float
    sampled_min_hull_margin_nm: float
    sampled_min_safety_domain_margin_nm: float

    @property
    def pair_key(self) -> tuple[str, str]:
        return (self.first_vessel_id, self.second_vessel_id)

    @property
    def duration_min(self) -> float:
        return self.end_time_min - self.start_time_min

    @property
    def modeled_hull_overlap_observed(self) -> bool:
        return PairSafetyOutcome.MODELED_HULL_OVERLAP in self.outcomes

    @property
    def safety_domain_intrusion_observed(self) -> bool:
        return PairSafetyOutcome.SAFETY_DOMAIN_INTRUSION in self.outcomes

    @property
    def near_miss_observed(self) -> bool:
        return PairSafetyOutcome.NEAR_MISS in self.outcomes


@dataclass(frozen=True, slots=True)
class MissionSafetySnapshot:
    intervals: tuple[TimedPairSafetyIntervalMetric, ...]
    episodes: tuple[PairSafetyEpisode, ...]
    summary: FleetSafetyMetricSummary


class ObservedSafetySeverity(str, Enum):
    """Worst *observed* physical evidence, deliberately excluding unresolved states."""

    NO_DATA = "no_data"
    NO_OBSERVED_BREACH = "no_observed_breach"
    NEAR_MISS = "near_miss"
    SAFETY_DOMAIN_INTRUSION = "safety_domain_intrusion"
    MODELED_HULL_OVERLAP = "modeled_hull_overlap"


@dataclass(frozen=True, slots=True)
class MissionSafetyExperimentSummary:
    interval_summary: FleetSafetyMetricSummary
    episode_count: int
    distinct_episode_pair_count: int
    affected_vessel_count: int
    total_pair_exposure_min: float
    nonclear_pair_exposure_min: float
    nonclear_pair_exposure_fraction: float | None
    mean_episode_duration_min: float | None
    max_episode_duration_min: float | None
    modeled_hull_overlap_episode_count: int
    safety_domain_intrusion_episode_count: int
    near_miss_episode_count: int
    unresolved_episode_count: int
    worst_observed_evidence: ObservedSafetySeverity


def evaluate_world_safety_interval(
    world: World2D,
    *,
    duration_min: float,
    profile_catalog: VesselProfileCatalog,
    config: PairSafetyMetricConfig,
) -> tuple[TimedPairSafetyIntervalMetric, ...]:
    """Evaluate all active unordered pairs under the world's *current* commands.

    The caller must ensure no queued command change lies inside ``duration_min``.  Mission
    integration enforces that by splitting physical advancement at ``next_command_time``.
    """
    if not isfinite(duration_min) or duration_min < 0.0:
        raise ValueError("duration_min must be finite and non-negative")
    if duration_min <= _EPS:
        return ()

    snapshot = world.snapshot()
    states = {state.vessel_id: state for state in snapshot.vessels}
    results: list[TimedPairSafetyIntervalMetric] = []
    for first_id, second_id in combinations(sorted(states), 2):
        first_profile = profile_catalog.profile_for(first_id)
        second_profile = profile_catalog.profile_for(second_id)
        if first_profile is None or second_profile is None:
            missing = first_id if first_profile is None else second_id
            raise SafetyMetricUnavailable(
                f"profile is required for mission safety metrics for {missing}"
            )
        metric = evaluate_pair_safety_interval(
            first_state=states[first_id],
            second_state=states[second_id],
            first_command=world.command_for(first_id),
            second_command=world.command_for(second_id),
            first_limits=world.limits_for(first_id),
            second_limits=world.limits_for(second_id),
            first_profile=first_profile,
            second_profile=second_profile,
            duration_min=duration_min,
            config=config,
        )
        results.append(
            TimedPairSafetyIntervalMetric(
                start_time_min=snapshot.time,
                end_time_min=snapshot.time + duration_min,
                metric=metric,
            )
        )
    return tuple(results)


def aggregate_pair_safety_episodes(
    intervals: Iterable[TimedPairSafetyIntervalMetric],
    *,
    adjacency_tolerance_min: float = 1e-8,
) -> tuple[PairSafetyEpisode, ...]:
    """Aggregate maximal contiguous non-clear runs independently for each pair."""
    if not isfinite(adjacency_tolerance_min) or adjacency_tolerance_min < 0.0:
        raise ValueError("adjacency_tolerance_min must be finite and non-negative")

    by_pair: dict[tuple[str, str], list[TimedPairSafetyIntervalMetric]] = {}
    for item in intervals:
        by_pair.setdefault(item.pair_key, []).append(item)

    episodes: list[PairSafetyEpisode] = []
    for pair_key in sorted(by_pair):
        items = sorted(by_pair[pair_key], key=lambda item: (item.start_time_min, item.end_time_min))
        active: list[TimedPairSafetyIntervalMetric] = []

        def flush() -> None:
            if not active:
                return
            unique_outcomes = tuple(sorted({item.outcome for item in active}, key=lambda value: value.value))
            episodes.append(
                PairSafetyEpisode(
                    first_vessel_id=pair_key[0],
                    second_vessel_id=pair_key[1],
                    start_time_min=active[0].start_time_min,
                    end_time_min=active[-1].end_time_min,
                    interval_count=len(active),
                    outcomes=unique_outcomes,
                    unresolved_interval_count=sum(item.outcome in _UNRESOLVED for item in active),
                    sampled_min_center_separation_nm=min(
                        item.metric.sampled_min_center_separation_nm for item in active
                    ),
                    sampled_min_hull_margin_nm=min(
                        item.metric.sampled_min_hull_margin_nm for item in active
                    ),
                    sampled_min_safety_domain_margin_nm=min(
                        item.metric.sampled_min_safety_domain_margin_nm for item in active
                    ),
                )
            )
            active.clear()

        for item in items:
            if item.outcome is PairSafetyOutcome.CLEAR:
                flush()
                continue
            if active and item.start_time_min > active[-1].end_time_min + adjacency_tolerance_min:
                flush()
            elif active and item.start_time_min < active[-1].end_time_min - adjacency_tolerance_min:
                raise ValueError(f"overlapping safety intervals for pair {pair_key}")
            active.append(item)
        flush()

    return tuple(sorted(episodes, key=lambda item: (item.start_time_min, item.pair_key, item.end_time_min)))


def summarize_mission_safety(
    intervals: Iterable[TimedPairSafetyIntervalMetric],
) -> MissionSafetyExperimentSummary:
    """Build one experiment-facing summary without conflating uncertainty with breaches.

    Episode evidence counts are intentionally non-exclusive: one episode may contain a
    near-miss interval and later a domain intrusion or modeled hull overlap.  The separate
    ``worst_observed_evidence`` field provides the strongest actually observed class, while
    unresolved episodes are counted independently.
    """
    items = tuple(intervals)
    episodes = aggregate_pair_safety_episodes(items)
    interval_summary = summarize_fleet_safety_metrics(item.metric for item in items)
    total_exposure = sum(item.metric.duration_min for item in items)
    nonclear_exposure = sum(
        item.metric.duration_min for item in items if item.outcome is not PairSafetyOutcome.CLEAR
    )
    fraction = None if total_exposure <= _EPS else nonclear_exposure / total_exposure
    durations = tuple(item.duration_min for item in episodes)

    hull_count = sum(item.modeled_hull_overlap_observed for item in episodes)
    domain_count = sum(item.safety_domain_intrusion_observed for item in episodes)
    near_count = sum(item.near_miss_observed for item in episodes)
    unresolved_count = sum(item.unresolved_interval_count > 0 for item in episodes)

    if not items:
        worst = ObservedSafetySeverity.NO_DATA
    elif hull_count:
        worst = ObservedSafetySeverity.MODELED_HULL_OVERLAP
    elif domain_count:
        worst = ObservedSafetySeverity.SAFETY_DOMAIN_INTRUSION
    elif near_count:
        worst = ObservedSafetySeverity.NEAR_MISS
    else:
        worst = ObservedSafetySeverity.NO_OBSERVED_BREACH

    affected_vessels = {vessel_id for episode in episodes for vessel_id in episode.pair_key}
    return MissionSafetyExperimentSummary(
        interval_summary=interval_summary,
        episode_count=len(episodes),
        distinct_episode_pair_count=len({item.pair_key for item in episodes}),
        affected_vessel_count=len(affected_vessels),
        total_pair_exposure_min=total_exposure,
        nonclear_pair_exposure_min=nonclear_exposure,
        nonclear_pair_exposure_fraction=fraction,
        mean_episode_duration_min=(None if not durations else sum(durations) / len(durations)),
        max_episode_duration_min=(None if not durations else max(durations)),
        modeled_hull_overlap_episode_count=hull_count,
        safety_domain_intrusion_episode_count=domain_count,
        near_miss_episode_count=near_count,
        unresolved_episode_count=unresolved_count,
        worst_observed_evidence=worst,
    )


class _OnlineEpisodeState:
    """Bounded state for one currently-open non-clear pair episode."""

    __slots__ = (
        "pair_key",
        "start_time_min",
        "end_time_min",
        "outcomes",
        "unresolved_interval_count",
        "sampled_min_center_separation_nm",
        "sampled_min_hull_margin_nm",
        "sampled_min_safety_domain_margin_nm",
    )

    def __init__(self, item: TimedPairSafetyIntervalMetric) -> None:
        self.pair_key = item.pair_key
        self.start_time_min = item.start_time_min
        self.end_time_min = item.end_time_min
        self.outcomes = {item.outcome}
        self.unresolved_interval_count = int(item.outcome in _UNRESOLVED)
        self.sampled_min_center_separation_nm = item.metric.sampled_min_center_separation_nm
        self.sampled_min_hull_margin_nm = item.metric.sampled_min_hull_margin_nm
        self.sampled_min_safety_domain_margin_nm = item.metric.sampled_min_safety_domain_margin_nm

    def observe(self, item: TimedPairSafetyIntervalMetric) -> None:
        self.end_time_min = item.end_time_min
        self.outcomes.add(item.outcome)
        self.unresolved_interval_count += int(item.outcome in _UNRESOLVED)
        self.sampled_min_center_separation_nm = min(
            self.sampled_min_center_separation_nm,
            item.metric.sampled_min_center_separation_nm,
        )
        self.sampled_min_hull_margin_nm = min(
            self.sampled_min_hull_margin_nm,
            item.metric.sampled_min_hull_margin_nm,
        )
        self.sampled_min_safety_domain_margin_nm = min(
            self.sampled_min_safety_domain_margin_nm,
            item.metric.sampled_min_safety_domain_margin_nm,
        )


class MissionSafetySummaryAccumulator:
    """Online exact equivalent of :func:`summarize_mission_safety` for bounded telemetry.

    The accumulator retains only aggregate counters, distinct pair/vessel sets, and at
    most one open episode per active pair.  It therefore avoids retaining every full
    ``DomainClearanceCertification`` while preserving the experiment-facing summary.
    ``finalize`` is terminal: additional observations after finalization are rejected.
    """

    def __init__(self, *, adjacency_tolerance_min: float = 1e-8) -> None:
        if not isfinite(adjacency_tolerance_min) or adjacency_tolerance_min < 0.0:
            raise ValueError("adjacency_tolerance_min must be finite and non-negative")
        self.adjacency_tolerance_min = adjacency_tolerance_min
        self._finalized: MissionSafetyExperimentSummary | None = None
        self._interval_count = 0
        self._distinct_pairs: set[tuple[str, str]] = set()
        self._modeled_hull_overlap_count = 0
        self._safety_domain_intrusion_count = 0
        self._near_miss_count = 0
        self._unresolved_count = 0
        self._clear_count = 0
        self._sampled_min_center_separation_nm: float | None = None
        self._sampled_min_hull_margin_nm: float | None = None
        self._sampled_min_safety_domain_margin_nm: float | None = None
        self._total_pair_exposure_min = 0.0
        self._nonclear_pair_exposure_min = 0.0
        self._active: dict[tuple[str, str], _OnlineEpisodeState] = {}
        self._episode_count = 0
        self._episode_pairs: set[tuple[str, str]] = set()
        self._affected_vessels: set[str] = set()
        self._episode_duration_sum_min = 0.0
        self._max_episode_duration_min: float | None = None
        self._modeled_hull_overlap_episode_count = 0
        self._safety_domain_intrusion_episode_count = 0
        self._near_miss_episode_count = 0
        self._unresolved_episode_count = 0

    @property
    def interval_count(self) -> int:
        return self._interval_count

    @staticmethod
    def _minimum(current: float | None, value: float) -> float:
        return value if current is None else min(current, value)

    def _flush_pair(self, pair_key: tuple[str, str]) -> None:
        active = self._active.pop(pair_key, None)
        if active is None:
            return
        duration = active.end_time_min - active.start_time_min
        self._episode_count += 1
        self._episode_pairs.add(pair_key)
        self._affected_vessels.update(pair_key)
        self._episode_duration_sum_min += duration
        self._max_episode_duration_min = (
            duration
            if self._max_episode_duration_min is None
            else max(self._max_episode_duration_min, duration)
        )
        self._modeled_hull_overlap_episode_count += int(
            PairSafetyOutcome.MODELED_HULL_OVERLAP in active.outcomes
        )
        self._safety_domain_intrusion_episode_count += int(
            PairSafetyOutcome.SAFETY_DOMAIN_INTRUSION in active.outcomes
        )
        self._near_miss_episode_count += int(PairSafetyOutcome.NEAR_MISS in active.outcomes)
        self._unresolved_episode_count += int(active.unresolved_interval_count > 0)

    def observe(self, item: TimedPairSafetyIntervalMetric) -> None:
        if self._finalized is not None:
            raise RuntimeError("cannot observe safety intervals after accumulator finalization")
        if not isinstance(item, TimedPairSafetyIntervalMetric):
            raise TypeError("item must be TimedPairSafetyIntervalMetric")

        metric = item.metric
        pair_key = item.pair_key
        self._interval_count += 1
        self._distinct_pairs.add(pair_key)
        self._modeled_hull_overlap_count += int(
            item.outcome is PairSafetyOutcome.MODELED_HULL_OVERLAP
        )
        self._safety_domain_intrusion_count += int(
            item.outcome is PairSafetyOutcome.SAFETY_DOMAIN_INTRUSION
        )
        self._near_miss_count += int(item.outcome is PairSafetyOutcome.NEAR_MISS)
        self._unresolved_count += int(item.outcome in _UNRESOLVED)
        self._clear_count += int(item.outcome is PairSafetyOutcome.CLEAR)
        self._sampled_min_center_separation_nm = self._minimum(
            self._sampled_min_center_separation_nm,
            metric.sampled_min_center_separation_nm,
        )
        self._sampled_min_hull_margin_nm = self._minimum(
            self._sampled_min_hull_margin_nm,
            metric.sampled_min_hull_margin_nm,
        )
        self._sampled_min_safety_domain_margin_nm = self._minimum(
            self._sampled_min_safety_domain_margin_nm,
            metric.sampled_min_safety_domain_margin_nm,
        )
        self._total_pair_exposure_min += metric.duration_min
        if item.outcome is not PairSafetyOutcome.CLEAR:
            self._nonclear_pair_exposure_min += metric.duration_min

        active = self._active.get(pair_key)
        if item.outcome is PairSafetyOutcome.CLEAR:
            self._flush_pair(pair_key)
            return
        if active is not None:
            if item.start_time_min > active.end_time_min + self.adjacency_tolerance_min:
                self._flush_pair(pair_key)
                active = None
            elif item.start_time_min < active.end_time_min - self.adjacency_tolerance_min:
                raise ValueError(f"overlapping safety intervals for pair {pair_key}")
        if active is None:
            self._active[pair_key] = _OnlineEpisodeState(item)
        else:
            active.observe(item)

    def finalize(self) -> MissionSafetyExperimentSummary:
        if self._finalized is not None:
            return self._finalized
        for pair_key in tuple(self._active):
            self._flush_pair(pair_key)

        interval_summary = FleetSafetyMetricSummary(
            interval_count=self._interval_count,
            distinct_pair_count=len(self._distinct_pairs),
            modeled_hull_overlap_count=self._modeled_hull_overlap_count,
            safety_domain_intrusion_count=self._safety_domain_intrusion_count,
            near_miss_count=self._near_miss_count,
            unresolved_count=self._unresolved_count,
            clear_count=self._clear_count,
            sampled_min_center_separation_nm=self._sampled_min_center_separation_nm,
            sampled_min_hull_margin_nm=self._sampled_min_hull_margin_nm,
            sampled_min_safety_domain_margin_nm=self._sampled_min_safety_domain_margin_nm,
        )
        fraction = (
            None
            if self._total_pair_exposure_min <= _EPS
            else self._nonclear_pair_exposure_min / self._total_pair_exposure_min
        )
        mean_duration = (
            None
            if self._episode_count == 0
            else self._episode_duration_sum_min / self._episode_count
        )
        if self._interval_count == 0:
            worst = ObservedSafetySeverity.NO_DATA
        elif self._modeled_hull_overlap_episode_count:
            worst = ObservedSafetySeverity.MODELED_HULL_OVERLAP
        elif self._safety_domain_intrusion_episode_count:
            worst = ObservedSafetySeverity.SAFETY_DOMAIN_INTRUSION
        elif self._near_miss_episode_count:
            worst = ObservedSafetySeverity.NEAR_MISS
        else:
            worst = ObservedSafetySeverity.NO_OBSERVED_BREACH

        self._finalized = MissionSafetyExperimentSummary(
            interval_summary=interval_summary,
            episode_count=self._episode_count,
            distinct_episode_pair_count=len(self._episode_pairs),
            affected_vessel_count=len(self._affected_vessels),
            total_pair_exposure_min=self._total_pair_exposure_min,
            nonclear_pair_exposure_min=self._nonclear_pair_exposure_min,
            nonclear_pair_exposure_fraction=fraction,
            mean_episode_duration_min=mean_duration,
            max_episode_duration_min=self._max_episode_duration_min,
            modeled_hull_overlap_episode_count=self._modeled_hull_overlap_episode_count,
            safety_domain_intrusion_episode_count=self._safety_domain_intrusion_episode_count,
            near_miss_episode_count=self._near_miss_episode_count,
            unresolved_episode_count=self._unresolved_episode_count,
            worst_observed_evidence=worst,
        )
        return self._finalized


class MissionSafetyRecorder:
    """Persistent read-only collector owned by ``UnifiedMissionSession`` when enabled."""

    def __init__(self, *, profile_catalog: VesselProfileCatalog, config: PairSafetyMetricConfig) -> None:
        self.profile_catalog = profile_catalog
        self.config = config
        self._history: list[TimedPairSafetyIntervalMetric] = []

    @property
    def history(self) -> tuple[TimedPairSafetyIntervalMetric, ...]:
        return tuple(self._history)

    @property
    def episodes(self) -> tuple[PairSafetyEpisode, ...]:
        return aggregate_pair_safety_episodes(self._history)

    @property
    def summary(self) -> FleetSafetyMetricSummary:
        return summarize_fleet_safety_metrics(item.metric for item in self._history)

    @property
    def experiment_summary(self) -> MissionSafetyExperimentSummary:
        return summarize_mission_safety(self._history)

    def observe_current_interval(
        self, world: World2D, *, duration_min: float
    ) -> tuple[TimedPairSafetyIntervalMetric, ...]:
        items = evaluate_world_safety_interval(
            world,
            duration_min=duration_min,
            profile_catalog=self.profile_catalog,
            config=self.config,
        )
        self._history.extend(items)
        return items

    def snapshot(self) -> MissionSafetySnapshot:
        return MissionSafetySnapshot(self.history, self.episodes, self.summary)
