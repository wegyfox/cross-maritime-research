"""Post-run helpers for reading frozen result packages without changing them."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class PrimaryComparisonRow:
    """One pre-specified demand point from a paired policy comparison."""

    demand_rate_per_minute: float
    replication_count: int
    baseline_mean: float
    comparison_mean: float
    difference_mean: float
    ci_low: float
    ci_high: float
    relative_difference_percent: float | None


def load_summary_results(path: str | Path) -> dict[str, Any]:
    """Load a published ``summary_results.json`` as plain JSON data."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "demand_points" not in data or "manifest" not in data:
        raise ValueError("not a vessel-crossing summary_results document")
    return data


def _policy_summary(point: Mapping[str, Any], policy_name: str) -> Mapping[str, Any]:
    matches = [
        item for item in point["policy_summaries"]
        if item.get("policy_name") == policy_name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one policy summary for {policy_name!r}")
    return matches[0]


def _paired_difference(
    point: Mapping[str, Any],
    *,
    metric_name: str,
    baseline_policy: str,
    comparison_policy: str,
) -> Mapping[str, Any]:
    matches = [
        item for item in point["paired_differences"]
        if item.get("metric_name") == metric_name
        and item.get("baseline_policy") == baseline_policy
        and item.get("comparison_policy") == comparison_policy
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one paired difference for "
            f"{comparison_policy} - {baseline_policy} / {metric_name}"
        )
    return matches[0]["difference"]


def primary_comparison_rows(
    summary: Mapping[str, Any],
    *,
    metric_name: str,
    baseline_policy: str,
    comparison_policy: str,
) -> tuple[PrimaryComparisonRow, ...]:
    """Return the frozen paired primary comparison in demand-grid order."""

    rows: list[PrimaryComparisonRow] = []
    for point in summary["demand_points"]:
        baseline = _policy_summary(point, baseline_policy)
        comparison = _policy_summary(point, comparison_policy)
        paired = _paired_difference(
            point,
            metric_name=metric_name,
            baseline_policy=baseline_policy,
            comparison_policy=comparison_policy,
        )

        baseline_mean = baseline[metric_name]["mean"]
        comparison_mean = comparison[metric_name]["mean"]
        if baseline_mean is None or comparison_mean is None:
            raise ValueError(f"metric {metric_name!r} has a missing mean")
        difference_mean = paired["mean"]
        ci_low = paired["ci_low"]
        ci_high = paired["ci_high"]
        if difference_mean is None or ci_low is None or ci_high is None:
            raise ValueError(f"paired metric {metric_name!r} lacks a complete CI")

        relative = None
        if abs(float(baseline_mean)) > 1e-15:
            relative = 100.0 * float(difference_mean) / float(baseline_mean)

        rows.append(
            PrimaryComparisonRow(
                demand_rate_per_minute=float(point["total_demand_rate_per_minute"]),
                replication_count=int(point["replication_count"]),
                baseline_mean=float(baseline_mean),
                comparison_mean=float(comparison_mean),
                difference_mean=float(difference_mean),
                ci_low=float(ci_low),
                ci_high=float(ci_high),
                relative_difference_percent=relative,
            )
        )
    return tuple(rows)


def all_cis_exclude_zero(rows: tuple[PrimaryComparisonRow, ...]) -> bool:
    """True when every paired CI is strictly on one side of zero."""

    return bool(rows) and all(row.ci_high < 0.0 or row.ci_low > 0.0 for row in rows)
