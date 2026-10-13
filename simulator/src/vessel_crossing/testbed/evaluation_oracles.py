"""C12 multi-oracle verification/evaluation layer.

Oracles are read-only consumers of frozen evidence.  They must never alter controller,
actuator or plant behavior.  Oracle identity binds family, implementation/version and
parameter set so evaluation-criterion changes remain visible in result provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
from typing import Any, Callable, Iterable, Mapping, Protocol

from .scope import TestbedFeature


class OracleFamily(str, Enum):
    HULL_COLLISION_GROUNDING = "HULL_COLLISION_GROUNDING"
    CPA_TCPA = "CPA_TCPA"
    SHIP_DOMAIN_INTRUSION = "SHIP_DOMAIN_INTRUSION"
    COLREG_COMPLIANCE = "COLREG_COMPLIANCE"
    FORMAL_STL_RULE = "FORMAL_STL_RULE"
    ROUTE_PERFORMANCE = "ROUTE_PERFORMANCE"
    TRAFFIC_SYSTEM = "TRAFFIC_SYSTEM"
    MANEUVER_BURDEN = "MANEUVER_BURDEN"
    RECOVERABILITY_ODD = "RECOVERABILITY_ODD"
    CUSTOM = "CUSTOM"


class OracleVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class OracleMetricSource(str, Enum):
    PHYSICAL_TRUTH = "PHYSICAL_TRUTH"
    OBSERVED_WORLD = "OBSERVED_WORLD"
    CONTROLLER = "CONTROLLER"
    TRAFFIC_SYSTEM = "TRAFFIC_SYSTEM"
    RECOVERABILITY = "RECOVERABILITY"
    ODD = "ODD"
    OPERATIONAL_STATE = "OPERATIONAL_STATE"
    EXTERNAL_EVALUATOR = "EXTERNAL_EVALUATOR"
    DECLARED = "DECLARED"


class OracleComparison(str, Enum):
    AT_MOST = "AT_MOST"
    AT_LEAST = "AT_LEAST"


@dataclass(frozen=True, slots=True)
class OracleMetricValue:
    metric_id: str
    value: float | int | bool | str
    unit: str
    source: OracleMetricSource
    evidence_id: str = ""

    def __post_init__(self) -> None:
        if not self.metric_id or not self.unit:
            raise ValueError("metric_id and unit must be non-empty")
        if isinstance(self.value, float) and not isfinite(self.value):
            raise ValueError("oracle metric float must be finite")
        if not isinstance(self.value, (float, int, bool, str)):
            raise TypeError("oracle metric value must be a scalar")


@dataclass(frozen=True, slots=True)
class OracleEvaluationContext:
    run_id: str
    metrics: tuple[OracleMetricValue, ...]
    evidence_ids: tuple[str, ...] = ()

    @classmethod
    def from_unsorted(
        cls,
        *,
        run_id: str,
        metrics: Iterable[OracleMetricValue],
        evidence_ids: Iterable[str] = (),
    ) -> "OracleEvaluationContext":
        if not run_id:
            raise ValueError("run_id must be non-empty")
        ordered = tuple(sorted(tuple(metrics), key=lambda item: item.metric_id))
        ids = [metric.metric_id for metric in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("oracle context metric_id values must be unique")
        evidence = tuple(sorted(set(evidence_ids)))
        return cls(run_id=run_id, metrics=ordered, evidence_ids=evidence)

    def metric(self, metric_id: str) -> OracleMetricValue | None:
        for metric in self.metrics:
            if metric.metric_id == metric_id:
                return metric
        return None

    def require(self, metric_id: str) -> OracleMetricValue:
        value = self.metric(metric_id)
        if value is None:
            raise KeyError(metric_id)
        return value

    def identity_sha256(self) -> str:
        return _sha256_payload(
            {
                "run_id": self.run_id,
                "metrics": [
                    {
                        "metric_id": metric.metric_id,
                        "value": metric.value,
                        "unit": metric.unit,
                        "source": metric.source.value,
                        "evidence_id": metric.evidence_id,
                    }
                    for metric in self.metrics
                ],
                "evidence_ids": list(self.evidence_ids),
            }
        )


@dataclass(frozen=True, slots=True)
class OracleSpec:
    oracle_id: str
    oracle_version: str
    family: OracleFamily
    implementation_id: str
    parameters_json: str

    @classmethod
    def from_parameters(
        cls,
        *,
        oracle_id: str,
        oracle_version: str,
        family: OracleFamily,
        implementation_id: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> "OracleSpec":
        if not oracle_id or not oracle_version or not implementation_id:
            raise ValueError("oracle_id, oracle_version and implementation_id must be non-empty")
        payload = _canonical_json_value(dict(parameters or {}))
        return cls(
            oracle_id=oracle_id,
            oracle_version=oracle_version,
            family=family,
            implementation_id=implementation_id,
            parameters_json=json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        return json.loads(self.parameters_json)

    def identity_sha256(self) -> str:
        return _sha256_payload(
            {
                "oracle_id": self.oracle_id,
                "oracle_version": self.oracle_version,
                "family": self.family.value,
                "implementation_id": self.implementation_id,
                "parameters": json.loads(self.parameters_json),
            }
        )


@dataclass(frozen=True, slots=True)
class OracleReportedMetric:
    metric_id: str
    value: float | int | bool | str
    unit: str

    def __post_init__(self) -> None:
        if not self.metric_id or not self.unit:
            raise ValueError("reported metric_id and unit must be non-empty")
        if isinstance(self.value, float) and not isfinite(self.value):
            raise ValueError("reported oracle metric float must be finite")


@dataclass(frozen=True, slots=True)
class OracleResult:
    oracle_id: str
    family: OracleFamily
    verdict: OracleVerdict
    oracle_spec_sha256: str
    context_sha256: str
    reported_metrics: tuple[OracleReportedMetric, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.oracle_id or not self.oracle_spec_sha256 or not self.context_sha256:
            raise ValueError("oracle result identity fields must be non-empty")
        metrics = tuple(self.reported_metrics)
        metric_ids = [metric.metric_id for metric in metrics]
        if metric_ids != sorted(metric_ids) or len(metric_ids) != len(set(metric_ids)):
            raise ValueError("reported_metrics must be unique and sorted by metric_id")
        evidence = tuple(self.evidence_ids)
        if evidence != tuple(sorted(set(evidence))):
            raise ValueError("evidence_ids must be unique and sorted")

    def identity_sha256(self) -> str:
        return _sha256_payload(
            {
                "oracle_id": self.oracle_id,
                "family": self.family.value,
                "verdict": self.verdict.value,
                "oracle_spec_sha256": self.oracle_spec_sha256,
                "context_sha256": self.context_sha256,
                "reported_metrics": [
                    {"metric_id": m.metric_id, "value": m.value, "unit": m.unit}
                    for m in self.reported_metrics
                ],
                "evidence_ids": list(self.evidence_ids),
                "reason": self.reason,
            }
        )


class EvaluationOracle(Protocol):
    spec: OracleSpec

    def evaluate(self, context: OracleEvaluationContext) -> OracleResult:
        ...


class ThresholdMetricOracle:
    """Simple read-only numeric criterion usable across many oracle families."""

    def __init__(
        self,
        *,
        oracle_id: str,
        oracle_version: str,
        family: OracleFamily,
        metric_id: str,
        comparison: OracleComparison,
        limit: float,
        unit: str,
    ) -> None:
        if not metric_id or not unit or not isfinite(limit):
            raise ValueError("metric_id/unit must be non-empty and limit finite")
        self.metric_id = metric_id
        self.comparison = comparison
        self.limit = float(limit)
        self.unit = unit
        self.spec = OracleSpec.from_parameters(
            oracle_id=oracle_id,
            oracle_version=oracle_version,
            family=family,
            implementation_id="cross-c12/threshold-metric/v1",
            parameters={
                "metric_id": metric_id,
                "comparison": comparison.value,
                "limit": self.limit,
                "unit": unit,
            },
        )

    def evaluate(self, context: OracleEvaluationContext) -> OracleResult:
        metric = context.metric(self.metric_id)
        context_sha = context.identity_sha256()
        if metric is None:
            return _result(
                self.spec, context_sha, OracleVerdict.INCONCLUSIVE,
                reason=f"missing metric {self.metric_id}", evidence_ids=context.evidence_ids,
            )
        if metric.unit != self.unit:
            return _result(
                self.spec, context_sha, OracleVerdict.INCONCLUSIVE,
                reason=f"unit mismatch: expected {self.unit}, got {metric.unit}",
                evidence_ids=_merge_evidence(context, metric),
            )
        if isinstance(metric.value, bool) or not isinstance(metric.value, (int, float)):
            return _result(
                self.spec, context_sha, OracleVerdict.INCONCLUSIVE,
                reason="metric is not numeric", evidence_ids=_merge_evidence(context, metric),
            )
        actual = float(metric.value)
        if self.comparison is OracleComparison.AT_MOST:
            signed_margin = self.limit - actual
        else:
            signed_margin = actual - self.limit
        verdict = OracleVerdict.PASS if signed_margin >= 0.0 else OracleVerdict.FAIL
        return _result(
            self.spec,
            context_sha,
            verdict,
            metrics=(
                OracleReportedMetric("actual", actual, self.unit),
                OracleReportedMetric("limit", self.limit, self.unit),
                OracleReportedMetric("signed_margin", signed_margin, self.unit),
            ),
            evidence_ids=_merge_evidence(context, metric),
        )


class BooleanMetricOracle:
    def __init__(
        self,
        *,
        oracle_id: str,
        oracle_version: str,
        family: OracleFamily,
        metric_id: str,
        required_value: bool,
    ) -> None:
        if not metric_id:
            raise ValueError("metric_id must be non-empty")
        self.metric_id = metric_id
        self.required_value = bool(required_value)
        self.spec = OracleSpec.from_parameters(
            oracle_id=oracle_id,
            oracle_version=oracle_version,
            family=family,
            implementation_id="cross-c12/boolean-metric/v1",
            parameters={"metric_id": metric_id, "required_value": self.required_value},
        )

    def evaluate(self, context: OracleEvaluationContext) -> OracleResult:
        metric = context.metric(self.metric_id)
        context_sha = context.identity_sha256()
        if metric is None or not isinstance(metric.value, bool):
            return _result(
                self.spec, context_sha, OracleVerdict.INCONCLUSIVE,
                reason=f"missing/non-boolean metric {self.metric_id}",
                evidence_ids=context.evidence_ids if metric is None else _merge_evidence(context, metric),
            )
        verdict = OracleVerdict.PASS if metric.value is self.required_value else OracleVerdict.FAIL
        return _result(
            self.spec,
            context_sha,
            verdict,
            metrics=(OracleReportedMetric("actual", metric.value, "boolean"),),
            evidence_ids=_merge_evidence(context, metric),
        )


@dataclass(frozen=True, slots=True)
class ExternalOracleOutcome:
    verdict: OracleVerdict
    reported_metrics: tuple[OracleReportedMetric, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    reason: str = ""


class ExternalOracleAdapter:
    """Adapter boundary for independent COLREG/STL/benchmark evaluators.

    The callback is runtime plumbing, while `OracleSpec` is the durable protocol
    identity. External code is not copied into the CROSS core by this adapter.
    """

    def __init__(self, *, spec: OracleSpec, evaluator: Callable[[OracleEvaluationContext], ExternalOracleOutcome]) -> None:
        self.spec = spec
        self._evaluator = evaluator

    def evaluate(self, context: OracleEvaluationContext) -> OracleResult:
        outcome = self._evaluator(context)
        if not isinstance(outcome, ExternalOracleOutcome):
            raise TypeError("external oracle evaluator must return ExternalOracleOutcome")
        metrics = tuple(sorted(outcome.reported_metrics, key=lambda item: item.metric_id))
        evidence = tuple(sorted(set(context.evidence_ids).union(outcome.evidence_ids)))
        return _result(
            self.spec,
            context.identity_sha256(),
            outcome.verdict,
            metrics=metrics,
            evidence_ids=evidence,
            reason=outcome.reason,
        )


@dataclass(frozen=True, slots=True)
class MultiOracleEvaluation:
    suite_id: str
    suite_version: str
    suite_sha256: str
    context_sha256: str
    results: tuple[OracleResult, ...]
    component_id: TestbedFeature = TestbedFeature.C12_EVALUATION_ORACLES

    @property
    def fail_count(self) -> int:
        return sum(result.verdict is OracleVerdict.FAIL for result in self.results)

    @property
    def inconclusive_count(self) -> int:
        return sum(result.verdict is OracleVerdict.INCONCLUSIVE for result in self.results)

    def identity_sha256(self) -> str:
        return _sha256_payload(
            {
                "suite_id": self.suite_id,
                "suite_version": self.suite_version,
                "suite_sha256": self.suite_sha256,
                "context_sha256": self.context_sha256,
                "result_identities": [result.identity_sha256() for result in self.results],
            }
        )


class OracleSuite:
    component_id = TestbedFeature.C12_EVALUATION_ORACLES

    def __init__(self, *, suite_id: str, suite_version: str, oracles: Iterable[EvaluationOracle]) -> None:
        if not suite_id or not suite_version:
            raise ValueError("suite_id and suite_version must be non-empty")
        ordered = tuple(sorted(tuple(oracles), key=lambda oracle: oracle.spec.oracle_id))
        ids = [oracle.spec.oracle_id for oracle in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("oracle_id values must be unique within a suite")
        self.suite_id = suite_id
        self.suite_version = suite_version
        self.oracles = ordered

    def configuration_sha256(self) -> str:
        return _sha256_payload(
            {
                "suite_id": self.suite_id,
                "suite_version": self.suite_version,
                "oracle_spec_sha256": [oracle.spec.identity_sha256() for oracle in self.oracles],
            }
        )

    def evaluate(self, context: OracleEvaluationContext) -> MultiOracleEvaluation:
        context_sha = context.identity_sha256()
        results = tuple(oracle.evaluate(context) for oracle in self.oracles)
        for result, oracle in zip(results, self.oracles):
            if result.oracle_spec_sha256 != oracle.spec.identity_sha256():
                raise ValueError("oracle result does not bind its registered OracleSpec")
            if result.context_sha256 != context_sha:
                raise ValueError("oracle result does not bind the supplied evaluation context")
        return MultiOracleEvaluation(
            suite_id=self.suite_id,
            suite_version=self.suite_version,
            suite_sha256=self.configuration_sha256(),
            context_sha256=context_sha,
            results=results,
        )


class NoOpOracleSuite:
    component_id = TestbedFeature.C12_EVALUATION_ORACLES

    def evaluate(self, context: OracleEvaluationContext) -> MultiOracleEvaluation:
        return MultiOracleEvaluation(
            suite_id="cross-c12/noop",
            suite_version="1",
            suite_sha256=_sha256_payload({"noop": True}),
            context_sha256=context.identity_sha256(),
            results=(),
        )


# Stable metric IDs for common oracle families. They are evidence names, not claims.
HULL_COLLISION_COUNT = "physical.hull_collision_count"
GROUNDING_COUNT = "physical.grounding_count"
MIN_CPA_NM = "physical.min_cpa_nm"
MIN_SHIP_DOMAIN_MARGIN_NM = "physical.min_ship_domain_margin_nm"
COLREG_COMPLIANT = "external.colreg_compliant"
STL_MIN_ROBUSTNESS = "external.stl_min_robustness"
ROUTE_COMPLETION_FRACTION = "route.completion_fraction"
ROUTE_DELAY_MIN = "route.delay_min"
ROUTE_EFFICIENCY_FRACTION = "route.efficiency_fraction"
TRAFFIC_THROUGHPUT = "traffic.throughput"
TRAFFIC_BACKLOG = "traffic.backlog"
TRAFFIC_FAIRNESS_JAIN = "traffic.fairness_jain"
MANEUVER_BURDEN = "maneuver.burden"
RECOVERABILITY_OUTCOME = "recoverability.outcome"
RECOVERABILITY_SAFETY_MARGIN_NM = "recoverability.best_safety_margin_nm"
ODD_STATUS = "odd.status"
OPERATIONAL_STATE = "operational.state"


def oracle_context_from_components(
    *,
    run_id: str,
    recoverability: Any | None = None,
    odd_evaluation: Any | None = None,
    operational_snapshot: Any | None = None,
    extra_metrics: Iterable[OracleMetricValue] = (),
    evidence_ids: Iterable[str] = (),
) -> OracleEvaluationContext:
    metrics = list(extra_metrics)
    ids = set(evidence_ids)
    if recoverability is not None:
        metrics.append(OracleMetricValue(
            RECOVERABILITY_OUTCOME, recoverability.outcome.value, "enum",
            OracleMetricSource.RECOVERABILITY, recoverability.identity_sha256(),
        ))
        ids.add(recoverability.identity_sha256())
        if recoverability.best_safety_margin_nm is not None:
            metrics.append(OracleMetricValue(
                RECOVERABILITY_SAFETY_MARGIN_NM, recoverability.best_safety_margin_nm, "nm",
                OracleMetricSource.RECOVERABILITY, recoverability.identity_sha256(),
            ))
    if odd_evaluation is not None:
        metrics.append(OracleMetricValue(
            ODD_STATUS, odd_evaluation.status.value, "enum", OracleMetricSource.ODD,
            f"odd:{odd_evaluation.context_sha256}:{odd_evaluation.envelope_sha256}",
        ))
        ids.add(f"odd:{odd_evaluation.context_sha256}:{odd_evaluation.envelope_sha256}")
    if operational_snapshot is not None:
        op_id = _sha256_payload({
            "time_min": operational_snapshot.time_min,
            "state": operational_snapshot.state.value,
            "authority_owner": operational_snapshot.authority_owner.value,
            "transition_history_sha256": operational_snapshot.transition_history_sha256,
            "config_sha256": operational_snapshot.config_sha256,
        })
        metrics.append(OracleMetricValue(
            OPERATIONAL_STATE, operational_snapshot.state.value, "enum",
            OracleMetricSource.OPERATIONAL_STATE, f"operational:{op_id}",
        ))
        ids.add(f"operational:{op_id}")
    return OracleEvaluationContext.from_unsorted(run_id=run_id, metrics=metrics, evidence_ids=ids)


def _result(
    spec: OracleSpec,
    context_sha256: str,
    verdict: OracleVerdict,
    *,
    metrics: Iterable[OracleReportedMetric] = (),
    evidence_ids: Iterable[str] = (),
    reason: str = "",
) -> OracleResult:
    return OracleResult(
        oracle_id=spec.oracle_id,
        family=spec.family,
        verdict=verdict,
        oracle_spec_sha256=spec.identity_sha256(),
        context_sha256=context_sha256,
        reported_metrics=tuple(sorted(tuple(metrics), key=lambda item: item.metric_id)),
        evidence_ids=tuple(sorted(set(evidence_ids))),
        reason=reason,
    )


def _merge_evidence(context: OracleEvaluationContext, metric: OracleMetricValue) -> tuple[str, ...]:
    ids = set(context.evidence_ids)
    if metric.evidence_id:
        ids.add(metric.evidence_id)
    return tuple(sorted(ids))


def _canonical_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("oracle parameters must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("oracle parameter mapping keys must be strings")
            result[key] = _canonical_json_value(item)
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    raise TypeError(f"unsupported oracle parameter type: {type(value).__name__}")


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
