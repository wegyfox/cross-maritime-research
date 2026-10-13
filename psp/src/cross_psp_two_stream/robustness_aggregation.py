"""Explainable deterministic aggregation of completed A90 robustness campaigns.

A91 reads already-completed atomic A90 unit records.  It never executes or mutates the
simulator.  Main effects are transparent one-at-a-time slices through the full-factorial
grid with all other parameters held at baseline.  Two-factor interactions are the
explicit second-order residual at baseline values of all other parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite, sqrt
from pathlib import Path
import re
from statistics import mean, stdev
from types import MappingProxyType
from typing import Any, Mapping

from .robustness_execution import ROBUSTNESS_UNIT_RESULT_SCHEMA
from .robustness_matrix import RobustnessCampaignPlan

ROBUSTNESS_AGGREGATION_SCHEMA = "cross-psp-robustness-aggregation-v1"
ROBUSTNESS_AGGREGATION_ALGORITHM_ID = "cross-psp-explainable-factorial-aggregation-v2"
IMPORTANCE_METRIC_ID = "max_abs_oat_delta_from_baseline-v1"
INTERACTION_METRIC_ID = "baseline_conditioned_second_order_residual-v1"
NONLINEARITY_METRIC_ID = "endpoint_linear_interpolation_residual-v1"

REGION_STABLE = "STABLE"
REGION_BOUNDARY = "BOUNDARY"
REGION_SENSITIVE = "SENSITIVE"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RobustnessAggregationError(ValueError):
    """Raised when a completed robustness campaign cannot be aggregated safely."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RobustnessAggregationError(f"value is not canonical JSON: {exc}") from exc


def _sha(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _sha256(name: str, value: object) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise RobustnessAggregationError(f"{name} must be a lowercase 64-character SHA-256")
    return text


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise RobustnessAggregationError(f"{name} must be finite numeric")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RobustnessAggregationError(f"{name} must be finite numeric") from exc
    if not isfinite(out):
        raise RobustnessAggregationError(f"{name} must be finite numeric")
    return out


def _extract_metric(result: Mapping[str, Any], path: str) -> float:
    current: object = result
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise RobustnessAggregationError(f"primary metric path {path!r} is missing at {part!r}")
        current = current[part]
    return _finite(f"metric {path}", current)


def _evaluate_rule(value: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == ">":
        return value > threshold
    if operator == "<=":
        return value <= threshold
    if operator == "<":
        return value < threshold
    raise RobustnessAggregationError(f"unsupported conclusion operator {operator!r}")


def _validate_record(
    path: Path,
    *,
    plan: RobustnessCampaignPlan,
    code_sha: str,
    expected_unit: object,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise RobustnessAggregationError(f"invalid unit record {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != ROBUSTNESS_UNIT_RESULT_SCHEMA:
        raise RobustnessAggregationError(f"unit record {path.name} schema mismatch")
    unit = expected_unit
    checks = {
        "campaign_identity_sha256": plan.campaign_identity_sha256,
        "execution_code_identity_sha256": code_sha,
        "work_unit_index": unit.work_unit_index,
        "work_unit_identity_sha256": unit.work_unit_identity_sha256,
        "combination_identity_sha256": unit.combination_identity_sha256,
        "replication_index": unit.replication_index,
        "derived_seed": unit.derived_seed,
        "parameter_values": dict(unit.parameter_values),
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise RobustnessAggregationError(f"unit record {path.name} identity mismatch: {key}")
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise RobustnessAggregationError(f"unit record {path.name} has invalid result mapping")
    if payload.get("result_sha256") != _sha(dict(result)):
        raise RobustnessAggregationError(f"unit record {path.name} result hash mismatch")
    material = {key: value for key, value in payload.items() if key != "record_identity_sha256"}
    if payload.get("record_identity_sha256") != _sha(material):
        raise RobustnessAggregationError(f"unit record {path.name} record hash mismatch")
    return payload


@dataclass(frozen=True, slots=True)
class CombinationMetricSummary:
    combination_index: int
    parameter_values: Mapping[str, str]
    replication_count: int
    mean: float
    sample_stddev: float | None
    standard_error: float | None
    minimum: float
    maximum: float
    delta_from_baseline: float
    relative_delta_from_baseline: float | None
    region: str
    conclusion: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_values", MappingProxyType(dict(self.parameter_values)))

    def canonical_dict(self) -> dict[str, object]:
        return {
            "combination_index": self.combination_index,
            "parameter_values": dict(self.parameter_values),
            "replication_count": self.replication_count,
            "mean": self.mean,
            "sample_stddev": self.sample_stddev,
            "standard_error": self.standard_error,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "delta_from_baseline": self.delta_from_baseline,
            "relative_delta_from_baseline": self.relative_delta_from_baseline,
            "region": self.region,
            "conclusion": self.conclusion,
        }


@dataclass(frozen=True, slots=True)
class ParameterEffectSummary:
    parameter_id: str
    unit: str
    direction: str
    max_abs_effect: float
    max_abs_relative_effect: float | None
    max_linearity_residual_abs: float | None
    level_means: tuple[tuple[str, float], ...]
    importance_rank: int

    def canonical_dict(self) -> dict[str, object]:
        return {
            "parameter_id": self.parameter_id,
            "unit": self.unit,
            "direction": self.direction,
            "max_abs_effect": self.max_abs_effect,
            "max_abs_relative_effect": self.max_abs_relative_effect,
            "max_linearity_residual_abs": self.max_linearity_residual_abs,
            "level_means": [{"value": value, "mean": metric} for value, metric in self.level_means],
            "importance_rank": self.importance_rank,
            "importance_metric_id": IMPORTANCE_METRIC_ID,
            "nonlinearity_metric_id": NONLINEARITY_METRIC_ID,
        }


@dataclass(frozen=True, slots=True)
class InteractionSummary:
    parameter_a: str
    parameter_b: str
    max_abs_interaction: float
    strongest_values: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "strongest_values", MappingProxyType(dict(self.strongest_values)))

    def canonical_dict(self) -> dict[str, object]:
        return {
            "parameter_a": self.parameter_a,
            "parameter_b": self.parameter_b,
            "max_abs_interaction": self.max_abs_interaction,
            "strongest_values": dict(self.strongest_values),
            "interaction_metric_id": INTERACTION_METRIC_ID,
        }


def _runtime_replication_summary(
    records: Mapping[int, Mapping[str, Any]],
    *,
    plan: RobustnessCampaignPlan,
) -> dict[str, object] | None:
    semantics_rows: list[Mapping[str, Any]] = []
    for index in sorted(records):
        result = records[index].get("result")
        if not isinstance(result, Mapping):
            raise RobustnessAggregationError("unit result mapping missing during replication audit")
        row = result.get("replication_semantics")
        if row is None:
            continue
        if not isinstance(row, Mapping):
            raise RobustnessAggregationError("replication_semantics must be a mapping")
        semantics_rows.append(row)
    if not semantics_rows:
        return None
    if len(semantics_rows) != len(records):
        raise RobustnessAggregationError(
            "replication_semantics must be present on every unit record or on none of them"
        )

    required = {
        "schema",
        "campaign_replication_index",
        "campaign_replication_axis",
        "q5_internal_replication_axis",
        "q5_internal_replication_count",
        "q5_internal_seed_schedule",
        "q5_internal_paired",
        "q5_internal_confidence_level",
        "uncertainty_note",
    }
    first = semantics_rows[0]
    if set(first) != required:
        raise RobustnessAggregationError("replication_semantics fields mismatch")
    invariant_keys = required - {"campaign_replication_index"}
    for row in semantics_rows:
        if set(row) != required:
            raise RobustnessAggregationError("replication_semantics fields mismatch")
        for key in invariant_keys:
            if row.get(key) != first.get(key):
                raise RobustnessAggregationError(
                    f"replication_semantics mismatch across work units: {key}"
                )

    internal_count = int(first["q5_internal_replication_count"])
    seed_schedule = [int(value) for value in first["q5_internal_seed_schedule"]]
    if internal_count < 1 or len(seed_schedule) != internal_count:
        raise RobustnessAggregationError(
            "q5 internal replication count must match q5_internal_seed_schedule"
        )
    indices = sorted(int(row["campaign_replication_index"]) for row in semantics_rows)
    expected_indices = sorted(unit.replication_index for unit in plan.work_units)
    if indices != expected_indices:
        raise RobustnessAggregationError("campaign replication indices do not match A89 work units")

    axis = str(first["campaign_replication_axis"])
    if plan.baseline.mode == "fixed":
        if plan.replications != 1 or axis != "NONE_FIXED_Q5_INTERNAL_ONLY":
            raise RobustnessAggregationError(
                "fixed-physics runtime replication semantics are inconsistent with the campaign plan"
            )
        campaign_uncertainty = "NONE_OUTER_FIXED_Q5_INTERNAL_ONLY"
    else:
        if axis != "SAMPLED_PHYSICS_SEED":
            raise RobustnessAggregationError(
                "sampled-physics runtime replication semantics are inconsistent with the campaign plan"
            )
        campaign_uncertainty = "SAMPLED_PHYSICS_SEED_ACROSS_WORK_UNITS"

    return {
        "campaign_replication_count": plan.replications,
        "campaign_replication_axis": axis,
        "campaign_uncertainty_scope": campaign_uncertainty,
        "q5_internal_replication_axis": first["q5_internal_replication_axis"],
        "q5_internal_replication_count": internal_count,
        "q5_internal_seed_schedule": seed_schedule,
        "q5_internal_paired": first["q5_internal_paired"],
        "q5_internal_confidence_level": first["q5_internal_confidence_level"],
        "standard_error_scope": (
            "CAMPAIGN_REPLICATE_MEANS_ONLY" if plan.replications > 1 else "NOT_AVAILABLE_AT_CAMPAIGN_LEVEL"
        ),
        "note": first["uncertainty_note"],
    }


@dataclass(frozen=True, slots=True)
class RobustnessAggregationResult:
    campaign_identity_sha256: str
    execution_code_identity_sha256: str
    primary_metric_path: str
    baseline_combination_index: int
    baseline_mean: float
    combination_summaries: tuple[CombinationMetricSummary, ...]
    parameter_effects: tuple[ParameterEffectSummary, ...]
    interactions: tuple[InteractionSummary, ...]
    region_counts: Mapping[str, int]
    conclusion_stable: bool
    conclusion_flip_combination_indices: tuple[int, ...]
    replication_summary: Mapping[str, object] | None
    aggregation_identity_sha256: str
    claim_status: str = "SUPPORTED_EXPLORATORY"

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_counts", MappingProxyType(dict(self.region_counts)))
        if self.replication_summary is not None:
            object.__setattr__(self, "replication_summary", MappingProxyType(dict(self.replication_summary)))

    def canonical_dict(self) -> dict[str, object]:
        return {
            "schema": ROBUSTNESS_AGGREGATION_SCHEMA,
            "algorithm_id": ROBUSTNESS_AGGREGATION_ALGORITHM_ID,
            "campaign_identity_sha256": self.campaign_identity_sha256,
            "execution_code_identity_sha256": self.execution_code_identity_sha256,
            "primary_metric_path": self.primary_metric_path,
            "baseline_combination_index": self.baseline_combination_index,
            "baseline_mean": self.baseline_mean,
            "combination_summaries": [item.canonical_dict() for item in self.combination_summaries],
            "parameter_effects": [item.canonical_dict() for item in self.parameter_effects],
            "interactions": [item.canonical_dict() for item in self.interactions],
            "region_counts": dict(self.region_counts),
            "conclusion_stable": self.conclusion_stable,
            "conclusion_flip_combination_indices": list(self.conclusion_flip_combination_indices),
            "replication_summary": None if self.replication_summary is None else dict(self.replication_summary),
            "claim_status": self.claim_status,
            "automatic_qualification_promotion": False,
            "claim_boundary": {
                "causal": False,
                "field_calibrated": False,
                "deployment_recommendation": False,
                "global_optimum_claim": False,
            },
            "aggregation_identity_sha256": self.aggregation_identity_sha256,
        }

    def human_report_dict(self) -> dict[str, object]:
        strongest = self.parameter_effects[:3]
        strongest_interactions = self.interactions[:3]
        return {
            "headline": (
                "Primary conclusion remained stable over the tested envelope."
                if self.conclusion_stable
                else "Primary conclusion changed within the tested envelope."
            ),
            "primary_metric": self.primary_metric_path,
            "baseline_mean": self.baseline_mean,
            "strongest_influences": [
                {
                    "parameter_id": item.parameter_id,
                    "direction": item.direction,
                    "max_abs_effect": item.max_abs_effect,
                    "importance_rank": item.importance_rank,
                }
                for item in strongest
            ],
            "strongest_interactions": [item.canonical_dict() for item in strongest_interactions],
            "region_counts": dict(self.region_counts),
            "conclusion_flip_combination_indices": list(self.conclusion_flip_combination_indices),
            "replication_summary": None if self.replication_summary is None else dict(self.replication_summary),
            "claim_status": self.claim_status,
            "qualification_note": "Robustness evidence does not automatically promote the scenario to QUALIFIED_PROFILE.",
        }


def aggregate_robustness_campaign(
    plan: RobustnessCampaignPlan,
    output_dir: str | Path,
    config: Mapping[str, Any],
) -> RobustnessAggregationResult:
    required = {
        "schema",
        "execution_code_identity_sha256",
        "primary_metric_path",
        "stable_absolute_delta",
        "sensitive_absolute_delta",
        "conclusion_rule",
    }
    if not isinstance(config, Mapping) or set(config) != required:
        actual = set(config) if isinstance(config, Mapping) else set()
        raise RobustnessAggregationError(
            f"aggregation fields mismatch; missing={sorted(required - actual)}, extra={sorted(actual - required)}"
        )
    if config["schema"] != ROBUSTNESS_AGGREGATION_SCHEMA:
        raise RobustnessAggregationError(f"schema must be {ROBUSTNESS_AGGREGATION_SCHEMA!r}")
    code_sha = _sha256("execution_code_identity_sha256", config["execution_code_identity_sha256"])
    metric_path = str(config["primary_metric_path"]).strip()
    if not metric_path or any(not part for part in metric_path.split(".")):
        raise RobustnessAggregationError("primary_metric_path must be a non-empty dot path")
    stable_threshold = _finite("stable_absolute_delta", config["stable_absolute_delta"])
    sensitive_threshold = _finite("sensitive_absolute_delta", config["sensitive_absolute_delta"])
    if stable_threshold < 0 or sensitive_threshold < 0 or stable_threshold > sensitive_threshold:
        raise RobustnessAggregationError(
            "thresholds must satisfy 0 <= stable_absolute_delta <= sensitive_absolute_delta"
        )
    rule = config["conclusion_rule"]
    if not isinstance(rule, Mapping) or set(rule) != {"operator", "threshold"}:
        raise RobustnessAggregationError("conclusion_rule must contain exactly operator and threshold")
    operator = str(rule["operator"])
    threshold = _finite("conclusion_rule.threshold", rule["threshold"])
    _evaluate_rule(0.0, operator, threshold)  # operator validation only

    root = Path(output_dir)
    units_dir = root / "units"
    expected_by_name = {
        f"{unit.work_unit_index:08d}_{unit.work_unit_identity_sha256}.json": unit
        for unit in plan.work_units
    }
    if not units_dir.exists():
        raise RobustnessAggregationError("completed A90 units directory is missing")
    paths = sorted(units_dir.glob("*.json"))
    if len(paths) != plan.work_unit_count:
        raise RobustnessAggregationError(
            f"campaign is incomplete: expected {plan.work_unit_count} unit records, found {len(paths)}"
        )
    records: dict[int, dict[str, Any]] = {}
    for path in paths:
        unit = expected_by_name.get(path.name)
        if unit is None:
            raise RobustnessAggregationError(f"unexpected unit record {path.name}")
        records[unit.work_unit_index] = _validate_record(
            path, plan=plan, code_sha=code_sha, expected_unit=unit
        )

    replication_summary = _runtime_replication_summary(records, plan=plan)

    metrics_by_combination: dict[int, list[float]] = {combo.combination_index: [] for combo in plan.combinations}
    for unit in plan.work_units:
        record = records[unit.work_unit_index]
        metrics_by_combination[unit.combination_index].append(
            _extract_metric(record["result"], metric_path)
        )
    if any(len(values) != plan.replications for values in metrics_by_combination.values()):
        raise RobustnessAggregationError("replication count mismatch in completed unit records")

    baseline_values = {item.parameter_id: item.baseline_value for item in plan.parameters}
    baseline_matches = [
        combo for combo in plan.combinations if dict(combo.values) == baseline_values
    ]
    if len(baseline_matches) != 1:
        raise RobustnessAggregationError("campaign must contain exactly one all-parameter baseline combination")
    baseline_combo = baseline_matches[0]
    baseline_mean = mean(metrics_by_combination[baseline_combo.combination_index])
    baseline_conclusion = _evaluate_rule(baseline_mean, operator, threshold)

    combination_summaries: list[CombinationMetricSummary] = []
    summary_by_values: dict[tuple[tuple[str, str], ...], CombinationMetricSummary] = {}
    region_counts = {REGION_STABLE: 0, REGION_BOUNDARY: 0, REGION_SENSITIVE: 0}
    flips: list[int] = []
    for combo in plan.combinations:
        values = metrics_by_combination[combo.combination_index]
        metric_mean = mean(values)
        sd = stdev(values) if len(values) > 1 else None
        se = sd / sqrt(len(values)) if sd is not None else None
        delta = metric_mean - baseline_mean
        abs_delta = abs(delta)
        if abs_delta <= stable_threshold:
            region = REGION_STABLE
        elif abs_delta >= sensitive_threshold:
            region = REGION_SENSITIVE
        else:
            region = REGION_BOUNDARY
        region_counts[region] += 1
        conclusion = _evaluate_rule(metric_mean, operator, threshold)
        if conclusion != baseline_conclusion:
            flips.append(combo.combination_index)
        summary = CombinationMetricSummary(
            combination_index=combo.combination_index,
            parameter_values=combo.values,
            replication_count=len(values),
            mean=metric_mean,
            sample_stddev=sd,
            standard_error=se,
            minimum=min(values),
            maximum=max(values),
            delta_from_baseline=delta,
            relative_delta_from_baseline=(delta / abs(baseline_mean) if baseline_mean != 0 else None),
            region=region,
            conclusion=conclusion,
        )
        combination_summaries.append(summary)
        summary_by_values[tuple(sorted(dict(combo.values).items()))] = summary

    raw_effects: list[dict[str, object]] = []
    for parameter in plan.parameters:
        level_means: list[tuple[str, float]] = []
        for level in parameter.values:
            values = dict(baseline_values)
            values[parameter.parameter_id] = level
            summary = summary_by_values.get(tuple(sorted(values.items())))
            if summary is None:
                raise RobustnessAggregationError(
                    f"full-factorial OAT slice missing for {parameter.parameter_id}={level}"
                )
            level_means.append((level, summary.mean))
        series = [metric for _, metric in level_means]
        increasing = all(b >= a for a, b in zip(series, series[1:]))
        decreasing = all(b <= a for a, b in zip(series, series[1:]))
        if all(abs(value - series[0]) <= 1e-15 for value in series):
            direction = "FLAT"
        elif increasing and not decreasing:
            direction = "POSITIVE_OVER_RANGE"
        elif decreasing and not increasing:
            direction = "NEGATIVE_OVER_RANGE"
        else:
            direction = "NON_MONOTONIC"
        max_abs = max(abs(value - baseline_mean) for value in series)
        max_rel = max_abs / abs(baseline_mean) if baseline_mean != 0 else None
        linearity = None
        if len(series) >= 3:
            xs = [float(value) for value, _ in level_means]
            x0, x1 = xs[0], xs[-1]
            y0, y1 = series[0], series[-1]
            if x1 != x0:
                residuals = []
                for x, observed in zip(xs[1:-1], series[1:-1]):
                    expected = y0 + (y1 - y0) * ((x - x0) / (x1 - x0))
                    residuals.append(abs(observed - expected))
                linearity = max(residuals, default=0.0)
        raw_effects.append({
            "parameter_id": parameter.parameter_id,
            "unit": parameter.unit,
            "direction": direction,
            "max_abs_effect": max_abs,
            "max_abs_relative_effect": max_rel,
            "max_linearity_residual_abs": linearity,
            "level_means": tuple(level_means),
        })
    ranked = sorted(raw_effects, key=lambda row: (-float(row["max_abs_effect"]), str(row["parameter_id"])))
    rank_by_parameter = {str(row["parameter_id"]): index + 1 for index, row in enumerate(ranked)}
    effects = tuple(
        ParameterEffectSummary(
            parameter_id=str(row["parameter_id"]),
            unit=str(row["unit"]),
            direction=str(row["direction"]),
            max_abs_effect=float(row["max_abs_effect"]),
            max_abs_relative_effect=(
                None if row["max_abs_relative_effect"] is None else float(row["max_abs_relative_effect"])
            ),
            max_linearity_residual_abs=(
                None if row["max_linearity_residual_abs"] is None else float(row["max_linearity_residual_abs"])
            ),
            level_means=tuple(row["level_means"]),
            importance_rank=rank_by_parameter[str(row["parameter_id"])],
        )
        for row in ranked
    )

    interactions: list[InteractionSummary] = []
    for i, parameter_a in enumerate(plan.parameters):
        for parameter_b in plan.parameters[i + 1:]:
            strongest = 0.0
            strongest_values = {
                parameter_a.parameter_id: parameter_a.baseline_value,
                parameter_b.parameter_id: parameter_b.baseline_value,
            }
            for level_a in parameter_a.values:
                for level_b in parameter_b.values:
                    values_ab = dict(baseline_values)
                    values_ab[parameter_a.parameter_id] = level_a
                    values_ab[parameter_b.parameter_id] = level_b
                    values_a = dict(baseline_values)
                    values_a[parameter_a.parameter_id] = level_a
                    values_b = dict(baseline_values)
                    values_b[parameter_b.parameter_id] = level_b
                    y_ab = summary_by_values[tuple(sorted(values_ab.items()))].mean
                    y_a = summary_by_values[tuple(sorted(values_a.items()))].mean
                    y_b = summary_by_values[tuple(sorted(values_b.items()))].mean
                    interaction = y_ab - y_a - y_b + baseline_mean
                    if abs(interaction) > abs(strongest):
                        strongest = interaction
                        strongest_values = {
                            parameter_a.parameter_id: level_a,
                            parameter_b.parameter_id: level_b,
                        }
            interactions.append(
                InteractionSummary(
                    parameter_a=parameter_a.parameter_id,
                    parameter_b=parameter_b.parameter_id,
                    max_abs_interaction=abs(strongest),
                    strongest_values=strongest_values,
                )
            )
    interactions.sort(key=lambda item: (-item.max_abs_interaction, item.parameter_a, item.parameter_b))

    identity_material = {
        "schema": ROBUSTNESS_AGGREGATION_SCHEMA,
        "algorithm_id": ROBUSTNESS_AGGREGATION_ALGORITHM_ID,
        "campaign_identity_sha256": plan.campaign_identity_sha256,
        "execution_code_identity_sha256": code_sha,
        "primary_metric_path": metric_path,
        "stable_absolute_delta": stable_threshold,
        "sensitive_absolute_delta": sensitive_threshold,
        "conclusion_rule": {"operator": operator, "threshold": threshold},
        "unit_record_identities": [records[index]["record_identity_sha256"] for index in sorted(records)],
        "replication_summary": replication_summary,
    }
    return RobustnessAggregationResult(
        campaign_identity_sha256=plan.campaign_identity_sha256,
        execution_code_identity_sha256=code_sha,
        primary_metric_path=metric_path,
        baseline_combination_index=baseline_combo.combination_index,
        baseline_mean=baseline_mean,
        combination_summaries=tuple(combination_summaries),
        parameter_effects=effects,
        interactions=tuple(interactions),
        region_counts=region_counts,
        conclusion_stable=not flips,
        conclusion_flip_combination_indices=tuple(flips),
        replication_summary=replication_summary,
        aggregation_identity_sha256=_sha(identity_material),
    )
