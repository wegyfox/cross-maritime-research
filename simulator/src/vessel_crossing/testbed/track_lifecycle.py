"""C5 deterministic detection and track lifecycle / partial observability.

The layer consumes immutable C4 physical truth and emits immutable C4 ObservedWorld
snapshots.  It models whether a measurement exists, when it becomes available, when a
track is initiated, how long it can coast without a new measurement, loss/reacquisition
and deterministic measurement uncertainty.  Physical truth is never modified.

This is infrastructure, not a detector/tracker novelty claim.  Empirical profiles can
implement ``DetectionParameterProfile`` later without changing controller APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from math import atan2, cos, degrees, hypot, isfinite, log, pi, sin, sqrt
from typing import Protocol

from .observation import (
    ObservationSource,
    ObservationTrackStatus,
    ObservedObjectState,
    ObservedVesselState,
    ObservedWorldSnapshot,
    PerfectObservationProvider,
    PhysicalObjectState,
    PhysicalVesselState,
    PhysicalWorldSnapshot,
)
from .scope import TestbedFeature
from .world_objects import (
    CircleGeometry,
    Point2D,
    PointGeometry,
    PolygonGeometry,
    PolylineGeometry,
    WorldGeometry,
    WorldObjectKind,
)

_EPS = 1e-12


class DetectionTargetClass(str, Enum):
    MERCHANT = "MERCHANT"
    SMALL_CRAFT = "SMALL_CRAFT"
    FISHING_CRAFT = "FISHING_CRAFT"
    STATIC_HAZARD = "STATIC_HAZARD"
    MOVING_HAZARD = "MOVING_HAZARD"
    OTHER = "OTHER"


class AisAvailability(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    INTERMITTENT = "INTERMITTENT"
    STALE = "STALE"


class TrackCoastMode(str, Enum):
    HOLD_LAST = "HOLD_LAST"
    CONSTANT_VELOCITY = "CONSTANT_VELOCITY"


class TrackEventType(str, Enum):
    INITIATED = "INITIATED"
    UPDATED = "UPDATED"
    COASTING_STARTED = "COASTING_STARTED"
    LOST = "LOST"
    REACQUIRED = "REACQUIRED"
    GHOST_INJECTED = "GHOST_INJECTED"


@dataclass(frozen=True, slots=True)
class RelativeBlindSector:
    start_deg: float
    end_deg: float

    def __post_init__(self) -> None:
        for name in ("start_deg", "end_deg"):
            value = getattr(self, name)
            if not isfinite(value) or not 0.0 <= value <= 360.0:
                raise ValueError(f"{name} must be within [0, 360]")

    def contains(self, relative_bearing_deg: float) -> bool:
        value = relative_bearing_deg % 360.0
        start = self.start_deg % 360.0
        end = self.end_deg % 360.0
        if abs(start - end) <= _EPS:
            return False
        if start < end:
            return start <= value < end
        return value >= start or value < end


@dataclass(frozen=True, slots=True)
class DetectionCoverage:
    max_range_nm: float | None = None
    min_range_nm: float = 0.0
    blind_sectors: tuple[RelativeBlindSector, ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.min_range_nm) or self.min_range_nm < 0.0:
            raise ValueError("min_range_nm must be finite and non-negative")
        if self.max_range_nm is not None:
            if not isfinite(self.max_range_nm) or self.max_range_nm <= self.min_range_nm:
                raise ValueError("max_range_nm must be finite and greater than min_range_nm")

    def covers(self, *, observer: PhysicalVesselState, x_east_nm: float, y_north_nm: float) -> bool:
        dx = x_east_nm - observer.x_east_nm
        dy = y_north_nm - observer.y_north_nm
        distance = hypot(dx, dy)
        if distance + _EPS < self.min_range_nm:
            return False
        if self.max_range_nm is not None and distance > self.max_range_nm + _EPS:
            return False
        if distance <= _EPS:
            return True
        bearing = degrees(atan2(dx, dy)) % 360.0
        relative = (bearing - observer.heading_deg) % 360.0
        return not any(sector.contains(relative) for sector in self.blind_sectors)


@dataclass(frozen=True, slots=True)
class MeasurementUncertainty:
    range_std_nm: float = 0.0
    bearing_std_deg: float = 0.0
    heading_std_deg: float = 0.0
    course_std_deg: float = 0.0
    speed_std_knots: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "range_std_nm",
            "bearing_std_deg",
            "heading_std_deg",
            "course_std_deg",
            "speed_std_knots",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DetectionParameters:
    probability_of_detection: float = 1.0
    measurement_latency_min: float = 0.0
    track_initiation_delay_min: float = 0.0
    coast_duration_min: float = 0.0
    max_detection_gap_min: float = 0.0
    coast_mode: TrackCoastMode = TrackCoastMode.CONSTANT_VELOCITY
    uncertainty: MeasurementUncertainty = MeasurementUncertainty()

    def __post_init__(self) -> None:
        if not isfinite(self.probability_of_detection) or not 0.0 <= self.probability_of_detection <= 1.0:
            raise ValueError("probability_of_detection must be within [0, 1]")
        for name in (
            "measurement_latency_min",
            "track_initiation_delay_min",
            "coast_duration_min",
            "max_detection_gap_min",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TargetDetectionMetadata:
    target_class: DetectionTargetClass = DetectionTargetClass.OTHER
    ais: AisAvailability = AisAvailability.ABSENT
    ais_period_min: float = 5.0
    ais_on_fraction: float = 0.5
    ais_phase_min: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.ais_period_min) or self.ais_period_min <= 0.0:
            raise ValueError("ais_period_min must be finite and positive")
        if not isfinite(self.ais_on_fraction) or not 0.0 <= self.ais_on_fraction <= 1.0:
            raise ValueError("ais_on_fraction must be within [0, 1]")
        if not isfinite(self.ais_phase_min):
            raise ValueError("ais_phase_min must be finite")


@dataclass(frozen=True, slots=True)
class TargetDetectionRegistry:
    entries: tuple[tuple[str, TargetDetectionMetadata], ...] = ()

    def __post_init__(self) -> None:
        ids = [entity_id for entity_id, _ in self.entries]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("detection metadata entries must be unique and sorted by entity_id")

    @classmethod
    def from_mapping(cls, mapping: dict[str, TargetDetectionMetadata]) -> "TargetDetectionRegistry":
        return cls(tuple(sorted(mapping.items())))

    def metadata_for(self, entity_id: str, *, default_class: DetectionTargetClass) -> TargetDetectionMetadata:
        for key, metadata in self.entries:
            if key == entity_id:
                return metadata
        return TargetDetectionMetadata(target_class=default_class)


class DetectionParameterProfile(Protocol):
    profile_id: str

    def parameters_for(
        self,
        *,
        entity_id: str,
        target_class: DetectionTargetClass,
        time_min: float,
    ) -> DetectionParameters:
        ...


@dataclass(frozen=True, slots=True)
class FixedDetectionProfile:
    default: DetectionParameters = DetectionParameters()
    by_class: tuple[tuple[DetectionTargetClass, DetectionParameters], ...] = ()
    profile_id: str = "cross-c5/fixed-detection/v1"

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        classes = [target_class.value for target_class, _ in self.by_class]
        if classes != sorted(classes) or len(classes) != len(set(classes)):
            raise ValueError("by_class entries must be unique and sorted by target class")

    @classmethod
    def from_mapping(
        cls,
        *,
        default: DetectionParameters | None = None,
        by_class: dict[DetectionTargetClass, DetectionParameters] | None = None,
        profile_id: str = "cross-c5/fixed-detection/v1",
    ) -> "FixedDetectionProfile":
        return cls(
            default=default or DetectionParameters(),
            by_class=tuple(sorted((by_class or {}).items(), key=lambda item: item[0].value)),
            profile_id=profile_id,
        )

    def parameters_for(
        self,
        *,
        entity_id: str,
        target_class: DetectionTargetClass,
        time_min: float,
    ) -> DetectionParameters:
        del entity_id, time_min
        for key, value in self.by_class:
            if key is target_class:
                return value
        return self.default


class VisibilityHook(Protocol):
    hook_id: str

    def visible(
        self,
        *,
        time_min: float,
        entity_id: str,
        target_class: DetectionTargetClass,
    ) -> bool:
        ...


@dataclass(frozen=True, slots=True)
class HiddenInterval:
    start_min: float
    end_min: float
    entity_id: str | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.start_min) or self.start_min < 0.0:
            raise ValueError("start_min must be finite and non-negative")
        if not isfinite(self.end_min) or self.end_min <= self.start_min:
            raise ValueError("end_min must be finite and greater than start_min")

    def hides(self, time_min: float, entity_id: str) -> bool:
        return self.start_min <= time_min < self.end_min and (
            self.entity_id is None or self.entity_id == entity_id
        )


@dataclass(frozen=True, slots=True)
class IntervalVisibilityHook:
    hidden_intervals: tuple[HiddenInterval, ...]
    hook_id: str = "cross-c5/interval-visibility/v1"

    def visible(
        self,
        *,
        time_min: float,
        entity_id: str,
        target_class: DetectionTargetClass,
    ) -> bool:
        del target_class
        return not any(interval.hides(time_min, entity_id) for interval in self.hidden_intervals)


class GhostTrackHook(Protocol):
    hook_id: str

    def ghost_vessels(self, *, observation_time_min: float) -> tuple[ObservedVesselState, ...]:
        ...

    def ghost_objects(self, *, observation_time_min: float) -> tuple[ObservedObjectState, ...]:
        ...


@dataclass(frozen=True, slots=True)
class NoGhostTrackHook:
    hook_id: str = "cross-c5/no-ghost/v1"

    def ghost_vessels(self, *, observation_time_min: float) -> tuple[ObservedVesselState, ...]:
        del observation_time_min
        return ()

    def ghost_objects(self, *, observation_time_min: float) -> tuple[ObservedObjectState, ...]:
        del observation_time_min
        return ()


@dataclass(frozen=True, slots=True)
class TrackLifecycleEvent:
    time_min: float
    entity_id: str
    event_type: TrackEventType
    source_time_min: float | None


@dataclass(frozen=True, slots=True)
class NoOpTrackLifecycleProvider:
    """C5 NoOp mode: exact C4 PerfectObservation semantics."""

    component_id = TestbedFeature.C5_TRACK_LIFECYCLE
    provider_id: str = "cross-c5/noop-perfect/v1"

    def observe(self, truth: PhysicalWorldSnapshot) -> ObservedWorldSnapshot:
        return PerfectObservationProvider().observe(truth)


@dataclass(slots=True)
class _PendingMeasurement:
    entity_id: str
    is_vessel: bool
    target_class: DetectionTargetClass
    available_at_min: float
    candidate_start_min: float
    vessel: ObservedVesselState | None = None
    obj: ObservedObjectState | None = None


@dataclass(slots=True)
class _TrackRecord:
    active: bool = False
    ever_visible: bool = False
    coasting: bool = False
    target_class: DetectionTargetClass = DetectionTargetClass.OTHER
    candidate_start_min: float | None = None
    last_raw_detection_min: float | None = None
    last_received_at_min: float | None = None
    vessel: ObservedVesselState | None = None
    obj: ObservedObjectState | None = None


class TrackLifecycleObservationProvider:
    component_id = TestbedFeature.C5_TRACK_LIFECYCLE

    def __init__(
        self,
        *,
        observer_vessel_id: str,
        profile: DetectionParameterProfile | None = None,
        metadata: TargetDetectionRegistry | None = None,
        coverage: DetectionCoverage | None = None,
        seed: int = 0,
        visibility_hooks: tuple[VisibilityHook, ...] = (),
        ghost_hook: GhostTrackHook | None = None,
        provider_id: str = "cross-c5/track-lifecycle/v1",
    ) -> None:
        if not observer_vessel_id:
            raise ValueError("observer_vessel_id must be non-empty")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if not provider_id:
            raise ValueError("provider_id must be non-empty")
        self.observer_vessel_id = observer_vessel_id
        self.profile = profile or FixedDetectionProfile()
        self.metadata = metadata or TargetDetectionRegistry()
        self.coverage = coverage or DetectionCoverage()
        self.seed = seed
        self.visibility_hooks = tuple(visibility_hooks)
        self.ghost_hook = ghost_hook or NoGhostTrackHook()
        self.provider_id = provider_id
        self._last_time: float | None = None
        self._tracks: dict[str, _TrackRecord] = {}
        self._pending: list[_PendingMeasurement] = []
        self._ais_seen: set[str] = set()
        self._events: list[TrackLifecycleEvent] = []

    def configuration_sha256(self) -> str:
        payload = {
            "observer_vessel_id": self.observer_vessel_id,
            "profile_id": self.profile.profile_id,
            "metadata": [
                [entity_id, metadata.target_class.value, metadata.ais.value, metadata.ais_period_min,
                 metadata.ais_on_fraction, metadata.ais_phase_min]
                for entity_id, metadata in self.metadata.entries
            ],
            "coverage": {
                "min_range_nm": self.coverage.min_range_nm,
                "max_range_nm": self.coverage.max_range_nm,
                "blind_sectors": [[s.start_deg, s.end_deg] for s in self.coverage.blind_sectors],
            },
            "seed": self.seed,
            "visibility_hooks": [hook.hook_id for hook in self.visibility_hooks],
            "ghost_hook": self.ghost_hook.hook_id,
            "provider_id": self.provider_id,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def events(self) -> tuple[TrackLifecycleEvent, ...]:
        return tuple(self._events)

    def observe(self, truth: PhysicalWorldSnapshot) -> ObservedWorldSnapshot:
        time_min = truth.time_min
        if self._last_time is not None and time_min < self._last_time - _EPS:
            raise ValueError("track provider requires monotonic truth snapshots")
        self._last_time = time_min
        observer = truth.by_vessel_id(self.observer_vessel_id)

        # Raw captures are generated from current physical truth only.
        for vessel in truth.vessels:
            if vessel.vessel_id == self.observer_vessel_id:
                continue
            metadata = self.metadata.metadata_for(vessel.vessel_id, default_class=DetectionTargetClass.OTHER)
            self._capture_vessel(truth, observer, vessel, metadata)
        for obj in truth.objects:
            default_class = (
                DetectionTargetClass.MOVING_HAZARD
                if obj.kind in {WorldObjectKind.PROTECTED_MOVING_OBJECT, WorldObjectKind.EMERGENT_MOVING_OBJECT,
                                WorldObjectKind.TEMPORARY_OBSTRUCTION}
                else DetectionTargetClass.STATIC_HAZARD
            )
            metadata = self.metadata.metadata_for(obj.object_id, default_class=default_class)
            self._capture_object(truth, observer, obj, metadata)

        updated = self._release_pending(time_min)

        vessels: list[ObservedVesselState] = [
            ObservedVesselState(
                vessel_id=observer.vessel_id,
                x_east_nm=observer.x_east_nm,
                y_north_nm=observer.y_north_nm,
                heading_deg=observer.heading_deg,
                course_over_ground_deg=observer.course_over_ground_deg,
                speed_over_ground_knots=observer.speed_over_ground_knots,
                source_time_min=time_min,
                track_status=ObservationTrackStatus.MEASURED,
                observation_source=ObservationSource.PERFECT,
            )
        ]
        objects: list[ObservedObjectState] = []

        for entity_id in sorted(self._tracks):
            record = self._tracks[entity_id]
            if not record.active:
                continue
            params = self.profile.parameters_for(
                entity_id=entity_id,
                target_class=record.target_class,
                time_min=time_min,
            )
            if entity_id not in updated:
                assert record.last_received_at_min is not None
                if time_min - record.last_received_at_min > params.coast_duration_min + _EPS:
                    record.active = False
                    record.coasting = False
                    record.candidate_start_min = None
                    self._events.append(TrackLifecycleEvent(time_min, entity_id, TrackEventType.LOST, self._source_time(record)))
                    continue
                if not record.coasting:
                    record.coasting = True
                    self._events.append(
                        TrackLifecycleEvent(time_min, entity_id, TrackEventType.COASTING_STARTED, self._source_time(record))
                    )
            if record.vessel is not None:
                vessels.append(self._render_vessel_track(record, time_min, params, entity_id in updated))
            elif record.obj is not None:
                objects.append(self._render_object_track(record, time_min, entity_id in updated))

        ghost_vessels = self.ghost_hook.ghost_vessels(observation_time_min=time_min)
        ghost_objects = self.ghost_hook.ghost_objects(observation_time_min=time_min)
        for ghost in ghost_vessels:
            if ghost.vessel_id == self.observer_vessel_id or any(v.vessel_id == ghost.vessel_id for v in vessels):
                raise ValueError(f"ghost vessel_id collides with observed entity: {ghost.vessel_id}")
            vessels.append(replace(ghost, track_status=ObservationTrackStatus.GHOST, observation_source=ObservationSource.GHOST))
            self._events.append(TrackLifecycleEvent(time_min, ghost.vessel_id, TrackEventType.GHOST_INJECTED, ghost.source_time_min))
        for ghost in ghost_objects:
            if any(o.object_id == ghost.object_id for o in objects):
                raise ValueError(f"ghost object_id collides with observed entity: {ghost.object_id}")
            objects.append(replace(ghost, track_status=ObservationTrackStatus.GHOST, observation_source=ObservationSource.GHOST))
            self._events.append(TrackLifecycleEvent(time_min, ghost.object_id, TrackEventType.GHOST_INJECTED, ghost.source_time_min))

        return ObservedWorldSnapshot(
            observation_time_min=time_min,
            vessels=tuple(sorted(vessels, key=lambda item: item.vessel_id)),
            objects=tuple(sorted(objects, key=lambda item: item.object_id)),
            provider_id=self.provider_id,
            source_truth_sha256=truth.identity_sha256(),
        )

    def _capture_vessel(
        self,
        truth: PhysicalWorldSnapshot,
        observer: PhysicalVesselState,
        vessel: PhysicalVesselState,
        metadata: TargetDetectionMetadata,
    ) -> None:
        params = self.profile.parameters_for(
            entity_id=vessel.vessel_id,
            target_class=metadata.target_class,
            time_min=truth.time_min,
        )
        sensor = self._sensor_detected(
            time_min=truth.time_min,
            observer=observer,
            entity_id=vessel.vessel_id,
            target_class=metadata.target_class,
            x_east_nm=vessel.x_east_nm,
            y_north_nm=vessel.y_north_nm,
            probability=params.probability_of_detection,
        )
        ais = self._ais_available(vessel.vessel_id, metadata, truth.time_min)
        if not (sensor or ais):
            self._mark_raw_miss(vessel.vessel_id, params, truth.time_min)
            return
        candidate_start = self._mark_raw_detection(vessel.vessel_id, params, truth.time_min)
        source = ObservationSource.FUSED if sensor and ais else (ObservationSource.SENSOR if sensor else ObservationSource.AIS)
        measured = self._measure_vessel(observer, vessel, params.uncertainty, truth.time_min, source)
        self._pending.append(
            _PendingMeasurement(
                entity_id=vessel.vessel_id,
                is_vessel=True,
                target_class=metadata.target_class,
                available_at_min=truth.time_min + params.measurement_latency_min,
                candidate_start_min=candidate_start,
                vessel=measured,
            )
        )

    def _capture_object(
        self,
        truth: PhysicalWorldSnapshot,
        observer: PhysicalVesselState,
        obj: PhysicalObjectState,
        metadata: TargetDetectionMetadata,
    ) -> None:
        params = self.profile.parameters_for(
            entity_id=obj.object_id,
            target_class=metadata.target_class,
            time_min=truth.time_min,
        )
        ref = _geometry_reference_point(obj.geometry)
        sensor = self._sensor_detected(
            time_min=truth.time_min,
            observer=observer,
            entity_id=obj.object_id,
            target_class=metadata.target_class,
            x_east_nm=ref.x_east_nm,
            y_north_nm=ref.y_north_nm,
            probability=params.probability_of_detection,
        )
        ais = self._ais_available(obj.object_id, metadata, truth.time_min)
        if not (sensor or ais):
            self._mark_raw_miss(obj.object_id, params, truth.time_min)
            return
        candidate_start = self._mark_raw_detection(obj.object_id, params, truth.time_min)
        source = ObservationSource.FUSED if sensor and ais else (ObservationSource.SENSOR if sensor else ObservationSource.AIS)
        dx, dy = self._position_error(observer, ref.x_east_nm, ref.y_north_nm, params.uncertainty, obj.object_id, truth.time_min)
        measured = ObservedObjectState(
            object_id=obj.object_id,
            kind=obj.kind,
            geometry=_translate_geometry(obj.geometry, dx, dy),
            source_time_min=truth.time_min,
            observation_source=source,
        )
        self._pending.append(
            _PendingMeasurement(
                entity_id=obj.object_id,
                is_vessel=False,
                target_class=metadata.target_class,
                available_at_min=truth.time_min + params.measurement_latency_min,
                candidate_start_min=candidate_start,
                obj=measured,
            )
        )

    def _sensor_detected(
        self,
        *,
        time_min: float,
        observer: PhysicalVesselState,
        entity_id: str,
        target_class: DetectionTargetClass,
        x_east_nm: float,
        y_north_nm: float,
        probability: float,
    ) -> bool:
        if not self.coverage.covers(observer=observer, x_east_nm=x_east_nm, y_north_nm=y_north_nm):
            return False
        if not all(
            hook.visible(time_min=time_min, entity_id=entity_id, target_class=target_class)
            for hook in self.visibility_hooks
        ):
            return False
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return _uniform01(self.seed, entity_id, time_min, "detection") < probability

    def _ais_available(self, entity_id: str, metadata: TargetDetectionMetadata, time_min: float) -> bool:
        if metadata.ais is AisAvailability.ABSENT:
            return False
        if metadata.ais is AisAvailability.PRESENT:
            return True
        if metadata.ais is AisAvailability.STALE:
            if entity_id in self._ais_seen:
                return False
            self._ais_seen.add(entity_id)
            return True
        phase = (time_min - metadata.ais_phase_min) % metadata.ais_period_min
        return phase < metadata.ais_period_min * metadata.ais_on_fraction

    def _mark_raw_detection(self, entity_id: str, params: DetectionParameters, time_min: float) -> float:
        record = self._tracks.setdefault(entity_id, _TrackRecord())
        if (
            record.candidate_start_min is None
            or record.last_raw_detection_min is None
            or time_min - record.last_raw_detection_min > params.max_detection_gap_min + _EPS
        ):
            record.candidate_start_min = time_min
        record.last_raw_detection_min = time_min
        return record.candidate_start_min

    def _mark_raw_miss(self, entity_id: str, params: DetectionParameters, time_min: float) -> None:
        record = self._tracks.setdefault(entity_id, _TrackRecord())
        if not record.active and record.last_raw_detection_min is not None:
            if time_min - record.last_raw_detection_min > params.max_detection_gap_min + _EPS:
                record.candidate_start_min = None

    def _release_pending(self, time_min: float) -> set[str]:
        matured = [item for item in self._pending if item.available_at_min <= time_min + _EPS]
        self._pending = [item for item in self._pending if item.available_at_min > time_min + _EPS]
        matured.sort(key=lambda item: (item.available_at_min, item.entity_id))
        updated: set[str] = set()
        for item in matured:
            record = self._tracks.setdefault(item.entity_id, _TrackRecord())
            source_time = item.vessel.source_time_min if item.vessel is not None else item.obj.source_time_min  # type: ignore[union-attr]
            params = self.profile.parameters_for(
                entity_id=item.entity_id,
                target_class=item.target_class,
                time_min=time_min,
            )
            if not record.active and source_time - item.candidate_start_min + _EPS < params.track_initiation_delay_min:
                continue
            was_visible = record.ever_visible
            record.active = True
            record.ever_visible = True
            record.coasting = False
            record.target_class = item.target_class
            record.last_received_at_min = time_min
            record.vessel = item.vessel
            record.obj = item.obj
            updated.add(item.entity_id)
            event_type = TrackEventType.UPDATED if was_visible and record.active else TrackEventType.INITIATED
            # If it was seen before but currently inactive, this is a reacquisition.
            if was_visible and item.entity_id not in updated - {item.entity_id}:
                # active was false on entry; distinguish via event history/previous flag below.
                pass
            # Reconstruct the pre-update active state from whether the last record had been lost.
            previous_lost = was_visible and any(
                event.entity_id == item.entity_id and event.event_type is TrackEventType.LOST
                for event in reversed(self._events)
            )
            if previous_lost and (not self._events or self._events[-1].event_type is not TrackEventType.REACQUIRED):
                event_type = TrackEventType.REACQUIRED
            self._events.append(TrackLifecycleEvent(time_min, item.entity_id, event_type, source_time))
        return updated

    def _measure_vessel(
        self,
        observer: PhysicalVesselState,
        vessel: PhysicalVesselState,
        uncertainty: MeasurementUncertainty,
        time_min: float,
        source: ObservationSource,
    ) -> ObservedVesselState:
        dx, dy = self._position_error(
            observer, vessel.x_east_nm, vessel.y_north_nm, uncertainty, vessel.vessel_id, time_min
        )
        heading = _perturb_angle(vessel.heading_deg, uncertainty.heading_std_deg, self.seed, vessel.vessel_id, time_min, "heading")
        course = None if vessel.course_over_ground_deg is None else _perturb_angle(
            vessel.course_over_ground_deg, uncertainty.course_std_deg, self.seed, vessel.vessel_id, time_min, "course"
        )
        speed = max(
            0.0,
            vessel.speed_over_ground_knots
            + uncertainty.speed_std_knots * _normal01(self.seed, vessel.vessel_id, time_min, "speed"),
        )
        return ObservedVesselState(
            vessel_id=vessel.vessel_id,
            x_east_nm=vessel.x_east_nm + dx,
            y_north_nm=vessel.y_north_nm + dy,
            heading_deg=heading,
            course_over_ground_deg=course,
            speed_over_ground_knots=speed,
            source_time_min=time_min,
            observation_source=source,
        )

    def _position_error(
        self,
        observer: PhysicalVesselState,
        x_east_nm: float,
        y_north_nm: float,
        uncertainty: MeasurementUncertainty,
        entity_id: str,
        time_min: float,
    ) -> tuple[float, float]:
        if uncertainty.range_std_nm <= _EPS and uncertainty.bearing_std_deg <= _EPS:
            return 0.0, 0.0
        dx = x_east_nm - observer.x_east_nm
        dy = y_north_nm - observer.y_north_nm
        true_range = hypot(dx, dy)
        true_bearing = degrees(atan2(dx, dy)) % 360.0 if true_range > _EPS else 0.0
        measured_range = max(
            0.0,
            true_range + uncertainty.range_std_nm * _normal01(self.seed, entity_id, time_min, "range"),
        )
        measured_bearing = (
            true_bearing
            + uncertainty.bearing_std_deg * _normal01(self.seed, entity_id, time_min, "bearing")
        ) % 360.0
        theta = measured_bearing * pi / 180.0
        mx = observer.x_east_nm + measured_range * sin(theta)
        my = observer.y_north_nm + measured_range * cos(theta)
        return mx - x_east_nm, my - y_north_nm

    def _render_vessel_track(
        self,
        record: _TrackRecord,
        time_min: float,
        params: DetectionParameters,
        updated: bool,
    ) -> ObservedVesselState:
        assert record.vessel is not None
        if updated:
            return replace(record.vessel, track_status=ObservationTrackStatus.MEASURED)
        vessel = record.vessel
        x, y = vessel.x_east_nm, vessel.y_north_nm
        if params.coast_mode is TrackCoastMode.CONSTANT_VELOCITY and vessel.speed_over_ground_knots is not None:
            course = vessel.course_over_ground_deg if vessel.course_over_ground_deg is not None else vessel.heading_deg
            if course is not None:
                elapsed = max(0.0, time_min - vessel.source_time_min)
                distance = vessel.speed_over_ground_knots * elapsed / 60.0
                theta = course * pi / 180.0
                x += distance * sin(theta)
                y += distance * cos(theta)
        return replace(
            vessel,
            x_east_nm=x,
            y_north_nm=y,
            track_status=ObservationTrackStatus.COASTED,
            observation_source=ObservationSource.PREDICTED,
        )

    def _render_object_track(self, record: _TrackRecord, time_min: float, updated: bool) -> ObservedObjectState:
        del time_min
        assert record.obj is not None
        if updated:
            return replace(record.obj, track_status=ObservationTrackStatus.MEASURED)
        return replace(
            record.obj,
            track_status=ObservationTrackStatus.COASTED,
            observation_source=ObservationSource.PREDICTED,
        )

    @staticmethod
    def _source_time(record: _TrackRecord) -> float | None:
        if record.vessel is not None:
            return record.vessel.source_time_min
        if record.obj is not None:
            return record.obj.source_time_min
        return None


def _geometry_reference_point(geometry: WorldGeometry) -> Point2D:
    if isinstance(geometry, PointGeometry):
        return geometry.point
    if isinstance(geometry, CircleGeometry):
        return geometry.center
    if isinstance(geometry, PolylineGeometry):
        points = geometry.points
    elif isinstance(geometry, PolygonGeometry):
        points = geometry.vertices
    else:
        raise TypeError(f"unsupported geometry {type(geometry)!r}")
    return Point2D(
        sum(p.x_east_nm for p in points) / len(points),
        sum(p.y_north_nm for p in points) / len(points),
    )


def _translate_geometry(geometry: WorldGeometry, dx: float, dy: float) -> WorldGeometry:
    def move(point: Point2D) -> Point2D:
        return Point2D(point.x_east_nm + dx, point.y_north_nm + dy)

    if isinstance(geometry, PointGeometry):
        return PointGeometry(move(geometry.point))
    if isinstance(geometry, CircleGeometry):
        return CircleGeometry(move(geometry.center), geometry.radius_nm)
    if isinstance(geometry, PolylineGeometry):
        return PolylineGeometry(tuple(move(p) for p in geometry.points))
    if isinstance(geometry, PolygonGeometry):
        return PolygonGeometry(tuple(move(p) for p in geometry.vertices))
    raise TypeError(f"unsupported geometry {type(geometry)!r}")


def _uniform01(seed: int, entity_id: str, time_min: float, lane: str) -> float:
    raw = f"{seed}|{entity_id}|{float(time_min).hex()}|{lane}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") >> 11
    return value / float(1 << 53)


def _normal01(seed: int, entity_id: str, time_min: float, lane: str) -> float:
    u1 = max(_uniform01(seed, entity_id, time_min, lane + ":u1"), 2.0**-53)
    u2 = _uniform01(seed, entity_id, time_min, lane + ":u2")
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


def _perturb_angle(value: float, std_deg: float, seed: int, entity_id: str, time_min: float, lane: str) -> float:
    if std_deg <= _EPS:
        return value % 360.0
    return (value + std_deg * _normal01(seed, entity_id, time_min, lane)) % 360.0
