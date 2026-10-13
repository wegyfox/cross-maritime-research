"""Reusable maritime autonomy V&V research-testbed infrastructure."""

from .scope import (
    A162_BASE_ARCHIVE,
    A162_BASE_SHA256,
    A162_BASE_VERSION,
    ComponentAdapter,
    DEFAULT_FEATURE_FLAGS,
    NoOpAdapter,
    SCOPE_IDENTITY,
    SCOPE_NAME,
    TESTBED_SCOPE_VERSION,
    ScopeIdentity,
    TestbedFeature,
    disabled_feature_flags,
    result_provenance,
    validate_feature_flags,
)
from .scenario import (
    CanonicalCoordinateSystem,
    CanonicalDimensions,
    CanonicalEnvironment,
    CanonicalInitialState,
    CanonicalRoute,
    CanonicalRouteLeg,
    CanonicalScenario,
    CanonicalSpeedProfile,
    CanonicalVessel,
    CanonicalWaypoint,
    ScenarioLineage,
    ScenarioVesselRole,
)
from .world_objects import (
    AppearanceMode,
    CanonicalWorldObject,
    CircleGeometry,
    ConsequenceRule,
    ConsequenceType,
    ConstraintViolation,
    LinearMotion,
    NavigationQuery,
    ObjectLifecycle,
    Point2D,
    PointGeometry,
    PolygonGeometry,
    PolylineGeometry,
    VesselObjectClass,
    WorldObjectKind,
    WorldObjectRegistry,
    emergent_moving_object,
)
from .behavior import (
    BehaviorContext,
    BehaviorDecision,
    BehaviorDecisionMode,
    BehaviorModel,
    BehaviorModelSpec,
    BehaviorProfile,
    BehaviorRegistry,
    BehaviorScriptStep,
    BehaviorState,
    DelayedActionBehavior,
    DelegatedBehavior,
    ErraticIntermittentCourseBehavior,
    FishingActivityBehavior,
    NoActionBehavior,
    ReplayedTrajectoryBehavior,
    RequestedManeuver,
    RouteFixatedBehavior,
    ScriptedBehavior,
    to_nextgen_maneuver_command,
)
from .observation import (
    BehaviorModelObservedController,
    ObservationProvider,
    ObservationSource,
    ObservationTrackStatus,
    ObservedObjectState,
    ObservedVesselState,
    ObservedWorldSnapshot,
    PerfectObservationProvider,
    PhysicalObjectState,
    PhysicalVesselState,
    PhysicalWorldSnapshot,
    behavior_context_from_observed,
    physical_snapshot_from_environmental_world2d,
    physical_snapshot_from_world2d,
)
from .track_lifecycle import (
    AisAvailability,
    DetectionCoverage,
    DetectionParameterProfile,
    DetectionParameters,
    DetectionTargetClass,
    FixedDetectionProfile,
    GhostTrackHook,
    HiddenInterval,
    IntervalVisibilityHook,
    MeasurementUncertainty,
    NoGhostTrackHook,
    NoOpTrackLifecycleProvider,
    RelativeBlindSector,
    TargetDetectionMetadata,
    TargetDetectionRegistry,
    TrackCoastMode,
    TrackEventType,
    TrackLifecycleEvent,
    TrackLifecycleObservationProvider,
    VisibilityHook,
)
from .odd import (
    BooleanODDPredicate, COMMUNICATION_AVAILABLE, COMMUNICATION_LATENCY_MIN,
    ENV_BELIEF_AGE_MIN, ENV_CURRENT_SPEED_KNOTS, ENV_WIND_DRIFT_EQUIVALENT_KNOTS,
    NoOpODDEvaluator, ODDCategory, ODDComparison, ODDContext, ODDEvaluation, ODDEnvelope,
    ODDMetricSource, ODDMetricValue, ODDPredicateResult, ODDResultReason, ODDStatus,
    NumericODDPredicate, PERCEPTION_CONTINUITY_FRACTION, PERCEPTION_MAX_TRACK_AGE_MIN,
    PERCEPTION_OBSERVED_TARGET_COUNT, SEA_ROOM_MIN_CLEARANCE_NM, SYSTEM_HEALTH_FRACTION,
    TRAFFIC_FISHING_CRAFT_COUNT, TRAFFIC_NON_AIS_FRACTION, TRAFFIC_SMALL_CRAFT_COUNT,
    TRAFFIC_TARGET_COUNT, odd_context_from_components,
)

from .operational_state import (
    AuthorityOwner, NoOpOperationalStateMachine, OperationalSignals, OperationalState,
    OperationalStateMachine, OperationalStateMachineConfig, OperationalStateSnapshot,
    OperationalTransition, TransitionCause,
)

from .evaluation_oracles import (
    BooleanMetricOracle, ExternalOracleAdapter, ExternalOracleOutcome, EvaluationOracle,
    MultiOracleEvaluation, NoOpOracleSuite, OracleComparison, OracleEvaluationContext,
    OracleFamily, OracleMetricSource, OracleMetricValue, OracleReportedMetric, OracleResult,
    OracleSpec, OracleSuite, OracleVerdict, ThresholdMetricOracle, oracle_context_from_components,
    HULL_COLLISION_COUNT, GROUNDING_COUNT, MIN_CPA_NM, MIN_SHIP_DOMAIN_MARGIN_NM,
    COLREG_COMPLIANT, STL_MIN_ROBUSTNESS, ROUTE_COMPLETION_FRACTION, ROUTE_DELAY_MIN,
    ROUTE_EFFICIENCY_FRACTION, TRAFFIC_THROUGHPUT, TRAFFIC_BACKLOG, TRAFFIC_FAIRNESS_JAIN,
    MANEUVER_BURDEN, RECOVERABILITY_OUTCOME, RECOVERABILITY_SAFETY_MARGIN_NM, ODD_STATUS,
    OPERATIONAL_STATE,
)

from .factor_composition import (
    FactorAssignment, FactorAxisKind, FactorDesignKind, FrozenJSONMap, NoOpScenarioFactorComposer,
    ScenarioFactorAxis, ScenarioFactorDesign, ScenarioFactorLevel, ScenarioFactorPoint,
)

from .recoverability import (
    FeasibleSetCollapseInterval, NoOpRecoverabilityEvaluator, PhysicalRecoverabilityEvaluator,
    RecoverabilityCandidateEvidence, RecoverabilityEvaluation, RecoverabilityOutcome,
    RecoverabilityTimeline, recoverability_candidate_from_environmental_pair_evaluation,
)

from .adapters import (
    DNV_ADAPTER_ID,
    DNV_SCHEMA_LINE,
    DnvMaritimeSchemaAdapter,
    DnvMaritimeSchemaError,
    export_dnv_traffic_situation,
    export_dnv_ship_traffic_generator_situation,
    import_dnv_ship_traffic_generator_situation,
    import_dnv_traffic_situation,
)


from .environment_belief import (
    AnomalousEnvironmentalField, CircularFieldAnomaly, DelayedBiasedEnvironmentalBeliefProvider,
    EnvironmentalBeliefProvider, EnvironmentalBeliefSnapshot, EnvironmentalFieldSample,
    EnvironmentalTruthField, EnvironmentalUncertaintyBounds, NoOpEnvironmentalBeliefProvider,
    PerfectEnvironmentalBeliefProvider, SeededStochasticEnvironmentalField, UniformEnvironmentalField,
    VectorUncertaintyBound, truth_sample_from_environmental_world2d,
)

from .fault_injection import (
    ActuatorFaultEffect, CommunicationFaultEffect, EnvironmentEstimateFaultEffect,
    FaultAwareEnvironmentalBeliefProvider, FaultAuditRecord, FaultBusVisibilityHook,
    FaultInjectionBus, FaultSpec, FaultTargetComponent, FaultType, ObservationFaultEffect,
    PlannerFaultEffect, degraded_maneuver_limits,
)

from .corpus_lineage import (
    CorpusCoverageAudit, CorpusCoverageEvaluator, CorpusManifest, CorpusScenarioRecord,
    CorpusSourceArtifact, CorpusSourceKind, CorpusSupportClass, CoverageDimension,
    CoverageStatus, CoverageStratumResult, CoverageTag, NoOpCorpusCoverageEvaluator,
    ScenarioDerivationKind, TargetPopulationDefinition, TargetPopulationStratum,
    source_artifact_from_scenario_lineage,
)


from .macro_reference import (
    AnalyticalMacroReferenceModel, MacroModelFamily, MacroReferencePrediction, MacroReferenceSpec,
    MacroTrafficClassFraction, MicroAggregateObservation, MicroMacroComparison, NoOpMacroReferenceModel,
    PairHeadway, compare_micro_to_macro, oracle_metrics_from_micro_macro_comparison,
    MACRO_HEADWAY_DIFFERENCE_MIN, MACRO_QUEUE_WAIT_DIFFERENCE_MIN, MACRO_THROUGHPUT_RELATIVE_ERROR,
)


from .external_controller import (
    ExternalControllerBridge, ExternalControllerRequest, ExternalControllerResult, ExternalControllerSpec,
    ExternalControllerStatus, ExternalControllerTransport, ExternalTransportOutcome, NoOpExternalControllerBridge,
)

__all__ = [
    "A162_BASE_ARCHIVE", "A162_BASE_SHA256", "A162_BASE_VERSION", "ComponentAdapter",
    "DEFAULT_FEATURE_FLAGS", "NoOpAdapter", "SCOPE_IDENTITY", "SCOPE_NAME", "TESTBED_SCOPE_VERSION",
    "ScopeIdentity", "TestbedFeature", "disabled_feature_flags", "result_provenance", "validate_feature_flags",
    "CanonicalCoordinateSystem", "CanonicalDimensions", "CanonicalEnvironment", "CanonicalInitialState",
    "CanonicalRoute", "CanonicalRouteLeg", "CanonicalScenario", "CanonicalSpeedProfile", "CanonicalVessel",
    "CanonicalWaypoint", "ScenarioLineage", "ScenarioVesselRole",
    "AppearanceMode", "CanonicalWorldObject", "CircleGeometry", "ConsequenceRule", "ConsequenceType",
    "ConstraintViolation", "LinearMotion", "NavigationQuery", "ObjectLifecycle", "Point2D", "PointGeometry",
    "PolygonGeometry", "PolylineGeometry", "VesselObjectClass", "WorldObjectKind", "WorldObjectRegistry",
    "emergent_moving_object",
    "BehaviorContext", "BehaviorDecision", "BehaviorDecisionMode", "BehaviorModel", "BehaviorModelSpec",
    "BehaviorProfile", "BehaviorRegistry", "BehaviorScriptStep", "BehaviorState", "DelayedActionBehavior",
    "DelegatedBehavior", "ErraticIntermittentCourseBehavior", "FishingActivityBehavior", "NoActionBehavior",
    "ReplayedTrajectoryBehavior", "RequestedManeuver", "RouteFixatedBehavior", "ScriptedBehavior",
    "to_nextgen_maneuver_command", "BehaviorModelObservedController", "ObservationProvider",
    "ObservationSource", "ObservationTrackStatus", "ObservedObjectState", "ObservedVesselState", "ObservedWorldSnapshot", "PerfectObservationProvider",
    "PhysicalObjectState", "PhysicalVesselState", "PhysicalWorldSnapshot", "behavior_context_from_observed",
    "physical_snapshot_from_environmental_world2d", "physical_snapshot_from_world2d",
    "AisAvailability", "DetectionCoverage", "DetectionParameterProfile", "DetectionParameters",
    "DetectionTargetClass", "FixedDetectionProfile", "GhostTrackHook", "HiddenInterval",
    "IntervalVisibilityHook", "MeasurementUncertainty", "NoGhostTrackHook", "NoOpTrackLifecycleProvider",
    "RelativeBlindSector", "TargetDetectionMetadata", "TargetDetectionRegistry", "TrackCoastMode",
    "TrackEventType", "TrackLifecycleEvent", "TrackLifecycleObservationProvider", "VisibilityHook",
    "AnomalousEnvironmentalField", "CircularFieldAnomaly", "DelayedBiasedEnvironmentalBeliefProvider",
    "EnvironmentalBeliefProvider", "EnvironmentalBeliefSnapshot", "EnvironmentalFieldSample",
    "EnvironmentalTruthField", "EnvironmentalUncertaintyBounds", "NoOpEnvironmentalBeliefProvider",
    "PerfectEnvironmentalBeliefProvider", "SeededStochasticEnvironmentalField", "UniformEnvironmentalField",
    "VectorUncertaintyBound", "truth_sample_from_environmental_world2d",
    "ActuatorFaultEffect", "CommunicationFaultEffect", "EnvironmentEstimateFaultEffect",
    "FaultAwareEnvironmentalBeliefProvider", "FaultAuditRecord", "FaultBusVisibilityHook",
    "FaultInjectionBus", "FaultSpec", "FaultTargetComponent", "FaultType",
    "ObservationFaultEffect", "PlannerFaultEffect", "degraded_maneuver_limits",
    "BooleanODDPredicate", "COMMUNICATION_AVAILABLE", "COMMUNICATION_LATENCY_MIN",
    "ENV_BELIEF_AGE_MIN", "ENV_CURRENT_SPEED_KNOTS", "ENV_WIND_DRIFT_EQUIVALENT_KNOTS",
    "NoOpODDEvaluator", "ODDCategory", "ODDComparison", "ODDContext", "ODDEvaluation", "ODDEnvelope",
    "ODDMetricSource", "ODDMetricValue", "ODDPredicateResult", "ODDResultReason", "ODDStatus",
    "NumericODDPredicate", "PERCEPTION_CONTINUITY_FRACTION", "PERCEPTION_MAX_TRACK_AGE_MIN",
    "PERCEPTION_OBSERVED_TARGET_COUNT", "SEA_ROOM_MIN_CLEARANCE_NM", "SYSTEM_HEALTH_FRACTION",
    "TRAFFIC_FISHING_CRAFT_COUNT", "TRAFFIC_NON_AIS_FRACTION", "TRAFFIC_SMALL_CRAFT_COUNT",
    "TRAFFIC_TARGET_COUNT", "odd_context_from_components",
    "AuthorityOwner", "NoOpOperationalStateMachine", "OperationalSignals", "OperationalState",
    "OperationalStateMachine", "OperationalStateMachineConfig", "OperationalStateSnapshot",
    "OperationalTransition", "TransitionCause",
    "BooleanMetricOracle", "ExternalOracleAdapter", "ExternalOracleOutcome", "EvaluationOracle",
    "MultiOracleEvaluation", "NoOpOracleSuite", "OracleComparison", "OracleEvaluationContext",
    "OracleFamily", "OracleMetricSource", "OracleMetricValue", "OracleReportedMetric",
    "OracleResult", "OracleSpec", "OracleSuite", "OracleVerdict", "ThresholdMetricOracle",
    "oracle_context_from_components", "HULL_COLLISION_COUNT", "GROUNDING_COUNT", "MIN_CPA_NM",
    "MIN_SHIP_DOMAIN_MARGIN_NM", "COLREG_COMPLIANT", "STL_MIN_ROBUSTNESS",
    "ROUTE_COMPLETION_FRACTION", "ROUTE_DELAY_MIN", "ROUTE_EFFICIENCY_FRACTION",
    "TRAFFIC_THROUGHPUT", "TRAFFIC_BACKLOG", "TRAFFIC_FAIRNESS_JAIN", "MANEUVER_BURDEN",
    "RECOVERABILITY_OUTCOME", "RECOVERABILITY_SAFETY_MARGIN_NM", "ODD_STATUS", "OPERATIONAL_STATE",
    "FactorAssignment", "FactorAxisKind", "FactorDesignKind", "FrozenJSONMap",
    "NoOpScenarioFactorComposer", "ScenarioFactorAxis", "ScenarioFactorDesign",
    "ScenarioFactorLevel", "ScenarioFactorPoint",
    "FeasibleSetCollapseInterval", "NoOpRecoverabilityEvaluator", "PhysicalRecoverabilityEvaluator",
    "RecoverabilityCandidateEvidence", "RecoverabilityEvaluation", "RecoverabilityOutcome",
    "RecoverabilityTimeline", "recoverability_candidate_from_environmental_pair_evaluation",
    "DNV_ADAPTER_ID", "DNV_SCHEMA_LINE",
    "DnvMaritimeSchemaAdapter", "DnvMaritimeSchemaError", "export_dnv_traffic_situation",
    "import_dnv_ship_traffic_generator_situation", "import_dnv_traffic_situation",

    "CorpusCoverageAudit", "CorpusCoverageEvaluator", "CorpusManifest", "CorpusScenarioRecord",
    "CorpusSourceArtifact", "CorpusSourceKind", "CorpusSupportClass", "CoverageDimension",
    "CoverageStatus", "CoverageStratumResult", "CoverageTag", "NoOpCorpusCoverageEvaluator",
    "ScenarioDerivationKind", "TargetPopulationDefinition", "TargetPopulationStratum",
    "source_artifact_from_scenario_lineage",
    "AnalyticalMacroReferenceModel", "MacroModelFamily", "MacroReferencePrediction", "MacroReferenceSpec",
    "MacroTrafficClassFraction", "MicroAggregateObservation", "MicroMacroComparison", "NoOpMacroReferenceModel",
    "PairHeadway", "compare_micro_to_macro", "oracle_metrics_from_micro_macro_comparison",
    "MACRO_HEADWAY_DIFFERENCE_MIN", "MACRO_QUEUE_WAIT_DIFFERENCE_MIN", "MACRO_THROUGHPUT_RELATIVE_ERROR",
    "ExternalControllerBridge", "ExternalControllerRequest", "ExternalControllerResult", "ExternalControllerSpec",
    "ExternalControllerStatus", "ExternalControllerTransport", "ExternalTransportOutcome", "NoOpExternalControllerBridge",
    "A162KinematicPlantAdapter", "ExternalCoSimulationPlantAdapter", "ExternalPlantOutcome", "ExternalPlantTransport",
    "FirstOrderResponseParameters", "FirstOrderResponsePlantAdapter", "LowOrderActuatedPlantAdapter",
    "LowOrderActuatedPlantParameters", "NoOpPlantAdapter", "PlantAdapter", "PlantActuatorCommand",
    "PlantActuatorState", "PlantActuatorTransientState", "PlantDynamicState", "PlantModelKind", "PlantSpec",
    "PlantStepRequest", "PlantStepResult", "PlantStepStatus", "PropulsionMapPoint", "SteerageAuthorityPoint",
    "TabulatedSteerageResponseParameters", "TabulatedSteerageResponsePlantAdapter",
    "InterpolatedStoppingData", "InterpolatedTurningData", "ManeuveringProfileError",
    "MissingManeuveringDataError", "StoppingTablePoint", "TurningTablePoint", "VesselManeuveringProfile",
    "HullReferencePoints", "PlantSweptPairSafety", "PlantSweptSafetyConfig", "PlantTrajectoryBounds",
    "RigidBodyHullKinematics", "certify_pair_plant_swept_clearance", "hull_reference_points",
    "rigid_body_hull_kinematics",
    "CalibrationDataRole", "CalibrationPartition", "CalibrationProtocol", "CalibrationResult", "NoOpCalibrationEvaluator",
    "ParameterEstimate", "ReplayFitMetrics", "SimulatedTrackPoint", "SourceSeparationPolicy",
    "compare_simulated_track_to_recorded",
    "CoordinationChannel", "CoordinationChannelSpec", "CoordinationInbox", "CoordinationMessage", "CoordinationMessageKind",
    "CoordinationMode", "CoordinationParticipant", "CoordinationParticipantRole", "InformationBasis", "NoOpCoordinationChannel",
    "TransmissionRecord", "TransmissionStatus",
    "HumanAssignmentPolicy", "HumanInformationBasis", "HumanOperatorCandidateEvidence", "HumanOperatorRole",
    "HumanOperatorSnapshot", "HumanOperatorSpec", "HumanROCCentreSpec", "HumanROCEvaluator", "HumanTakeoverAssessment",
    "HumanTakeoverOutcome", "HumanTakeoverRequest", "HumanTakeoverStatus", "NoOpHumanROCEvaluator",
    "operational_signals_with_human_response",
    "BASELINE_COMPONENTS", "BASELINE_INVARIANTS", "BASELINE_MANIFEST", "BaselineComponent",
    "ResearchTestbedBaselineManifest", "current_baseline_manifest",
]

from .plant_fidelity import (
    A162KinematicPlantAdapter, ExternalCoSimulationPlantAdapter, ExternalPlantOutcome, ExternalPlantTransport,
    FirstOrderResponseParameters, FirstOrderResponsePlantAdapter, LowOrderActuatedPlantAdapter,
    LowOrderActuatedPlantParameters, NoOpPlantAdapter, PlantAdapter,
    PlantActuatorCommand, PlantActuatorState, PlantActuatorTransientState, PlantDynamicState, PlantModelKind,
    PlantSpec, PlantStepRequest, PlantStepResult, PlantStepStatus, PropulsionMapPoint,
    SteerageAuthorityPoint, TabulatedSteerageResponseParameters, TabulatedSteerageResponsePlantAdapter,
)

from .maneuvering_profile import (
    InterpolatedStoppingData, InterpolatedTurningData, ManeuveringProfileError,
    MissingManeuveringDataError, StoppingTablePoint, TurningTablePoint, VesselManeuveringProfile,
)

from .plant_safety import (
    HullReferencePoints, PlantSweptPairSafety, PlantSweptSafetyConfig, PlantTrajectoryBounds,
    RigidBodyHullKinematics, certify_pair_plant_swept_clearance, hull_reference_points,
    rigid_body_hull_kinematics,
)

from .real_data_calibration import (
    CalibrationDataRole, CalibrationPartition, CalibrationProtocol, CalibrationResult, NoOpCalibrationEvaluator,
    ParameterEstimate, ReplayFitMetrics, SimulatedTrackPoint, SourceSeparationPolicy,
    compare_simulated_track_to_recorded,
)

from .coordination import (
    CoordinationChannel, CoordinationChannelSpec, CoordinationInbox, CoordinationMessage, CoordinationMessageKind,
    CoordinationMode, CoordinationParticipant, CoordinationParticipantRole, InformationBasis, NoOpCoordinationChannel,
    TransmissionRecord, TransmissionStatus,
)

from .human_roc import (
    HumanAssignmentPolicy, HumanInformationBasis, HumanOperatorCandidateEvidence, HumanOperatorRole,
    HumanOperatorSnapshot, HumanOperatorSpec, HumanROCCentreSpec, HumanROCEvaluator, HumanTakeoverAssessment,
    HumanTakeoverOutcome, HumanTakeoverRequest, HumanTakeoverStatus, NoOpHumanROCEvaluator,
    operational_signals_with_human_response,
)

from .baseline import (
    BASELINE_COMPONENTS, BASELINE_INVARIANTS, BASELINE_MANIFEST, BaselineComponent,
    ResearchTestbedBaselineManifest, current_baseline_manifest,
)
