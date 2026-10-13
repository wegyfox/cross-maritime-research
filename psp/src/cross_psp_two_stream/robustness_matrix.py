"""Deterministic Robustness Runner campaign matrix construction.

A89 deliberately stops at campaign planning.  It does not execute Q5/B6, mutate the
frozen scientific core, or infer qualification from a grid.  The builder accepts an
identity-bearing supported baseline plus explicit numeric parameter descriptors and
produces a canonical full-factorial grid, deterministic replication seeds, exact work
unit counts, and a machine-dependent workload estimate supplied by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from hashlib import sha256
from itertools import product
import json
from math import isfinite
import re
from types import MappingProxyType
from typing import Any, Mapping

ROBUSTNESS_CAMPAIGN_SCHEMA = "cross-psp-robustness-campaign-v1"
ROBUSTNESS_MATRIX_ALGORITHM_ID = "cross-psp-full-factorial-grid-v1"
ROBUSTNESS_WORK_UNIT_SCHEMA = "cross-psp-robustness-work-unit-v1"

SYMMETRIC_PERCENT = "symmetric_percent"
ASYMMETRIC_PERCENT = "asymmetric_percent"
ABSOLUTE_BOUNDS = "absolute_bounds"
FULL_FACTORIAL_GRID = "full_factorial_grid"

SUPPORTED_BASELINE_MODES = frozenset({"fixed", "distributions", "fleet_mixture"})
SUPPORTED_SCOPE_STATUSES = frozenset({"QUALIFIED_PROFILE", "SUPPORTED_EXPLORATORY"})
SUPPORTED_INTENTS = frozenset({"exploratory", "qualification_evidence"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RobustnessMatrixError(ValueError):
    """Raised when a robustness campaign cannot be planned safely."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _non_empty(name: str, value: object) -> str:
    text = str(value).strip()
    if not text:
        raise RobustnessMatrixError(f"{name} must be a non-empty string")
    return text


def _decimal(name: str, value: object) -> Decimal:
    if isinstance(value, bool):
        raise RobustnessMatrixError(f"{name} must be finite")
    try:
        out = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RobustnessMatrixError(f"{name} must be finite") from exc
    if not out.is_finite():
        raise RobustnessMatrixError(f"{name} must be finite")
    return out


def _canonical_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _sha256(name: str, value: object) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise RobustnessMatrixError(f"{name} must be a lowercase 64-character SHA-256")
    return text


def _positive_int(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RobustnessMatrixError(f"{name} must be an integer >= {minimum}")
    return value


def _fraction(name: str, value: object) -> str:
    number = _decimal(name, value)
    if not (Decimal(0) < number < Decimal(1)):
        raise RobustnessMatrixError(f"{name} must be strictly between 0 and 1")
    return _canonical_decimal(number)


def _linspace(start: Decimal, end: Decimal, count: int) -> list[Decimal]:
    if count <= 0:
        return []
    if count == 1:
        return [end]
    with localcontext() as ctx:
        ctx.prec = 50
        step = (end - start) / Decimal(count)
        return [start + step * Decimal(index) for index in range(1, count + 1)]


def _grid_values(
    *,
    baseline: Decimal,
    lower: Decimal,
    upper: Decimal,
    levels: int,
) -> tuple[str, ...]:
    if lower > baseline or upper < baseline or lower >= upper:
        raise RobustnessMatrixError(
            "resolved variation bounds must satisfy lower <= baseline <= upper and lower < upper"
        )
    has_lower = lower < baseline
    has_upper = upper > baseline
    if not has_lower and not has_upper:
        raise RobustnessMatrixError("variation must change the baseline")

    if has_lower and has_upper:
        if levels < 3 or levels % 2 == 0:
            raise RobustnessMatrixError(
                "two-sided variation requires an odd levels value >= 3 so baseline is explicit"
            )
        side = (levels - 1) // 2
        with localcontext() as ctx:
            ctx.prec = 50
            lower_step = (baseline - lower) / Decimal(side)
            below = [lower + lower_step * Decimal(i) for i in range(side)]
            above = _linspace(baseline, upper, side)
        values = below + [baseline] + above
    elif has_lower:
        if levels < 2:
            raise RobustnessMatrixError("one-sided variation requires levels >= 2")
        with localcontext() as ctx:
            ctx.prec = 50
            step = (baseline - lower) / Decimal(levels - 1)
            values = [lower + step * Decimal(i) for i in range(levels)]
    else:
        if levels < 2:
            raise RobustnessMatrixError("one-sided variation requires levels >= 2")
        with localcontext() as ctx:
            ctx.prec = 50
            step = (upper - baseline) / Decimal(levels - 1)
            values = [baseline + step * Decimal(i) for i in range(levels)]

    canonical = tuple(_canonical_decimal(value) for value in values)
    if len(set(canonical)) != len(canonical):
        raise RobustnessMatrixError("variation levels collapse to duplicate canonical values")
    return canonical


@dataclass(frozen=True, slots=True)
class RobustnessBaseline:
    mode: str
    physics_identity_sha256: str
    scope_status: str
    plan_identity_sha256: str | None = None

    def canonical_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "physics_identity_sha256": self.physics_identity_sha256,
            "scope_status": self.scope_status,
            "plan_identity_sha256": self.plan_identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class RobustnessParameterGrid:
    parameter_id: str
    unit: str
    baseline_value: str
    reference_value: str | None
    supported_lower: str | None
    supported_upper: str | None
    variation_mode: str
    requested_lower: str
    requested_upper: str
    values: tuple[str, ...]

    def canonical_dict(self) -> dict[str, object]:
        return {
            "parameter_id": self.parameter_id,
            "unit": self.unit,
            "baseline_value": self.baseline_value,
            "reference_value": self.reference_value,
            "supported_bounds": {
                "lower": self.supported_lower,
                "upper": self.supported_upper,
            },
            "variation": {
                "mode": self.variation_mode,
                "lower": self.requested_lower,
                "upper": self.requested_upper,
                "levels": len(self.values),
            },
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class RobustnessCombination:
    combination_index: int
    values: Mapping[str, str]
    combination_identity_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def canonical_dict(self) -> dict[str, object]:
        return {
            "combination_index": self.combination_index,
            "values": dict(self.values),
            "combination_identity_sha256": self.combination_identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class RobustnessWorkUnit:
    work_unit_index: int
    combination_index: int
    replication_index: int
    derived_seed: int
    parameter_values: Mapping[str, str]
    combination_identity_sha256: str
    work_unit_identity_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_values", MappingProxyType(dict(self.parameter_values)))

    def canonical_dict(self) -> dict[str, object]:
        return {
            "schema": ROBUSTNESS_WORK_UNIT_SCHEMA,
            "work_unit_index": self.work_unit_index,
            "combination_index": self.combination_index,
            "replication_index": self.replication_index,
            "derived_seed": self.derived_seed,
            "parameter_values": dict(self.parameter_values),
            "combination_identity_sha256": self.combination_identity_sha256,
            "work_unit_identity_sha256": self.work_unit_identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class RobustnessCampaignPlan:
    schema: str
    algorithm_id: str
    baseline: RobustnessBaseline
    scientific_question: str
    campaign_intent: str
    master_seed: int
    replications: int
    confidence_target: str | None
    estimated_seconds_per_work_unit: str
    parameters: tuple[RobustnessParameterGrid, ...]
    combinations: tuple[RobustnessCombination, ...]
    work_units: tuple[RobustnessWorkUnit, ...]
    campaign_identity_sha256: str

    @property
    def combination_count(self) -> int:
        return len(self.combinations)

    @property
    def work_unit_count(self) -> int:
        return len(self.work_units)

    @property
    def estimated_total_seconds(self) -> str:
        total = Decimal(self.estimated_seconds_per_work_unit) * Decimal(self.work_unit_count)
        return _canonical_decimal(total)

    def preflight_dict(self) -> dict[str, object]:
        return {
            "schema": ROBUSTNESS_CAMPAIGN_SCHEMA,
            "algorithm_id": self.algorithm_id,
            "campaign_identity_sha256": self.campaign_identity_sha256,
            "baseline": self.baseline.canonical_dict(),
            "scientific_question": self.scientific_question,
            "campaign_intent": self.campaign_intent,
            "automatic_qualification_promotion": False,
            "resulting_claim_status": "SUPPORTED_EXPLORATORY",
            "claim_boundary": {
                "causal": False,
                "field_calibrated": False,
                "deployment_recommendation": False,
                "global_optimum_claim": False,
            },
            "parameters": [item.canonical_dict() for item in self.parameters],
            "combination_count": self.combination_count,
            "replications": self.replications,
            "work_unit_count": self.work_unit_count,
            "master_seed": self.master_seed,
            "confidence_target": self.confidence_target,
            "estimated_seconds_per_work_unit": self.estimated_seconds_per_work_unit,
            "estimated_total_seconds": self.estimated_total_seconds,
            "checkpoint_resume": "planned_next_layer_not_executed_by_a89",
        }


def _parse_baseline(raw: object) -> RobustnessBaseline:
    if not isinstance(raw, Mapping):
        raise RobustnessMatrixError("baseline must be a mapping")
    required = {"mode", "physics_identity_sha256", "scope_status", "plan_identity_sha256"}
    if set(raw) != required:
        raise RobustnessMatrixError(
            f"baseline fields mismatch; missing={sorted(required - set(raw))}, extra={sorted(set(raw) - required)}"
        )
    mode = str(raw["mode"])
    if mode not in SUPPORTED_BASELINE_MODES:
        raise RobustnessMatrixError(f"unsupported baseline mode {mode!r}")
    scope = str(raw["scope_status"])
    if scope not in SUPPORTED_SCOPE_STATUSES:
        raise RobustnessMatrixError(f"unsupported scope_status {scope!r}")
    plan_identity = raw["plan_identity_sha256"]
    if mode == "fixed":
        if plan_identity is not None:
            raise RobustnessMatrixError("fixed baseline must have plan_identity_sha256=null")
        normalized_plan = None
    else:
        if plan_identity is None:
            raise RobustnessMatrixError(f"{mode} baseline requires plan_identity_sha256")
        normalized_plan = _sha256("baseline.plan_identity_sha256", plan_identity)
    return RobustnessBaseline(
        mode=mode,
        physics_identity_sha256=_sha256("baseline.physics_identity_sha256", raw["physics_identity_sha256"]),
        scope_status=scope,
        plan_identity_sha256=normalized_plan,
    )


def _resolve_grid(raw: object) -> RobustnessParameterGrid:
    if not isinstance(raw, Mapping):
        raise RobustnessMatrixError("each parameter must be a mapping")
    required = {
        "parameter_id",
        "unit",
        "baseline_value",
        "reference_value",
        "supported_bounds",
        "variation",
    }
    if set(raw) != required:
        raise RobustnessMatrixError(
            f"parameter fields mismatch; missing={sorted(required - set(raw))}, extra={sorted(set(raw) - required)}"
        )
    parameter_id = _non_empty("parameter_id", raw["parameter_id"])
    unit = _non_empty("unit", raw["unit"])
    baseline = _decimal(f"{parameter_id}.baseline_value", raw["baseline_value"])
    reference_raw = raw["reference_value"]
    reference = None if reference_raw is None else _decimal(f"{parameter_id}.reference_value", reference_raw)

    supported = raw["supported_bounds"]
    if not isinstance(supported, Mapping) or set(supported) != {"lower", "upper"}:
        raise RobustnessMatrixError("supported_bounds must contain exactly lower and upper")
    supported_lower = None if supported["lower"] is None else _decimal(f"{parameter_id}.supported_lower", supported["lower"])
    supported_upper = None if supported["upper"] is None else _decimal(f"{parameter_id}.supported_upper", supported["upper"])
    if supported_lower is not None and supported_upper is not None and supported_lower >= supported_upper:
        raise RobustnessMatrixError(f"{parameter_id} supported lower bound must be < upper bound")
    if supported_lower is not None and baseline < supported_lower:
        raise RobustnessMatrixError(f"{parameter_id} baseline is below its supported range")
    if supported_upper is not None and baseline > supported_upper:
        raise RobustnessMatrixError(f"{parameter_id} baseline is above its supported range")

    variation = raw["variation"]
    if not isinstance(variation, Mapping) or "mode" not in variation or "levels" not in variation:
        raise RobustnessMatrixError("variation must contain mode, levels, and mode-specific fields")
    mode = str(variation["mode"])
    levels = _positive_int(f"{parameter_id}.variation.levels", variation["levels"], minimum=2)

    if mode == SYMMETRIC_PERCENT:
        if set(variation) != {"mode", "percent", "levels"}:
            raise RobustnessMatrixError("symmetric_percent requires exactly mode, percent, levels")
        percent = _decimal(f"{parameter_id}.percent", variation["percent"])
        if percent <= 0:
            raise RobustnessMatrixError("symmetric_percent percent must be > 0")
        if baseline == 0:
            raise RobustnessMatrixError("percentage variation around a zero baseline is undefined for robustness planning")
        with localcontext() as ctx:
            ctx.prec = 50
            fraction = percent / Decimal(100)
            lower = baseline * (Decimal(1) - fraction)
            upper = baseline * (Decimal(1) + fraction)
    elif mode == ASYMMETRIC_PERCENT:
        if set(variation) != {"mode", "lower_percent", "upper_percent", "levels"}:
            raise RobustnessMatrixError(
                "asymmetric_percent requires exactly mode, lower_percent, upper_percent, levels"
            )
        lower_percent = _decimal(f"{parameter_id}.lower_percent", variation["lower_percent"])
        upper_percent = _decimal(f"{parameter_id}.upper_percent", variation["upper_percent"])
        if lower_percent < 0 or upper_percent < 0 or (lower_percent == 0 and upper_percent == 0):
            raise RobustnessMatrixError("asymmetric percentages must be non-negative and not both zero")
        if baseline == 0:
            raise RobustnessMatrixError("percentage variation around a zero baseline is undefined for robustness planning")
        with localcontext() as ctx:
            ctx.prec = 50
            lower = baseline * (Decimal(1) - lower_percent / Decimal(100))
            upper = baseline * (Decimal(1) + upper_percent / Decimal(100))
    elif mode == ABSOLUTE_BOUNDS:
        if set(variation) != {"mode", "lower", "upper", "levels"}:
            raise RobustnessMatrixError("absolute_bounds requires exactly mode, lower, upper, levels")
        lower = _decimal(f"{parameter_id}.absolute_lower", variation["lower"])
        upper = _decimal(f"{parameter_id}.absolute_upper", variation["upper"])
    else:
        raise RobustnessMatrixError(f"unsupported variation mode {mode!r}")

    if supported_lower is not None and lower < supported_lower:
        raise RobustnessMatrixError(
            f"{parameter_id} requested lower bound {_canonical_decimal(lower)} is below supported lower bound {_canonical_decimal(supported_lower)}"
        )
    if supported_upper is not None and upper > supported_upper:
        raise RobustnessMatrixError(
            f"{parameter_id} requested upper bound {_canonical_decimal(upper)} is above supported upper bound {_canonical_decimal(supported_upper)}"
        )

    values = _grid_values(baseline=baseline, lower=lower, upper=upper, levels=levels)
    return RobustnessParameterGrid(
        parameter_id=parameter_id,
        unit=unit,
        baseline_value=_canonical_decimal(baseline),
        reference_value=None if reference is None else _canonical_decimal(reference),
        supported_lower=None if supported_lower is None else _canonical_decimal(supported_lower),
        supported_upper=None if supported_upper is None else _canonical_decimal(supported_upper),
        variation_mode=mode,
        requested_lower=_canonical_decimal(lower),
        requested_upper=_canonical_decimal(upper),
        values=values,
    )


def _derived_seed(campaign_sha: str, combination_sha: str, replication_index: int, master_seed: int) -> int:
    material = {
        "schema": ROBUSTNESS_WORK_UNIT_SCHEMA,
        "campaign_identity_sha256": campaign_sha,
        "combination_identity_sha256": combination_sha,
        "replication_index": replication_index,
        "master_seed": master_seed,
    }
    return int.from_bytes(sha256(_canonical_bytes(material)).digest()[:8], "big", signed=False)


def robustness_campaign_plan_from_mapping(mapping: Mapping[str, Any]) -> RobustnessCampaignPlan:
    """Validate and deterministically materialize an A89 full-factorial campaign plan."""
    required = {
        "schema",
        "baseline",
        "scientific_question",
        "campaign_intent",
        "design",
        "master_seed",
        "replications",
        "confidence_target",
        "estimated_seconds_per_work_unit",
        "max_work_units",
        "parameters",
    }
    if not isinstance(mapping, Mapping) or set(mapping) != required:
        actual = set(mapping) if isinstance(mapping, Mapping) else set()
        raise RobustnessMatrixError(
            f"campaign fields mismatch; missing={sorted(required - actual)}, extra={sorted(actual - required)}"
        )
    if mapping["schema"] != ROBUSTNESS_CAMPAIGN_SCHEMA:
        raise RobustnessMatrixError(f"schema must be {ROBUSTNESS_CAMPAIGN_SCHEMA!r}")
    if mapping["design"] != FULL_FACTORIAL_GRID:
        raise RobustnessMatrixError(f"A89 supports only design={FULL_FACTORIAL_GRID!r}")

    baseline = _parse_baseline(mapping["baseline"])
    question = _non_empty("scientific_question", mapping["scientific_question"])
    intent = str(mapping["campaign_intent"])
    if intent not in SUPPORTED_INTENTS:
        raise RobustnessMatrixError(f"unsupported campaign_intent {intent!r}")
    master_seed = _positive_int("master_seed", mapping["master_seed"], minimum=0)
    replications = _positive_int("replications", mapping["replications"], minimum=1)
    max_work_units = _positive_int("max_work_units", mapping["max_work_units"], minimum=1)
    confidence = None if mapping["confidence_target"] is None else _fraction("confidence_target", mapping["confidence_target"])
    seconds = _decimal("estimated_seconds_per_work_unit", mapping["estimated_seconds_per_work_unit"])
    if seconds <= 0 or not isfinite(float(seconds)):
        raise RobustnessMatrixError("estimated_seconds_per_work_unit must be finite and > 0")

    raw_parameters = mapping["parameters"]
    if not isinstance(raw_parameters, (list, tuple)) or not raw_parameters:
        raise RobustnessMatrixError("parameters must be a non-empty sequence")
    parameters = tuple(sorted((_resolve_grid(item) for item in raw_parameters), key=lambda item: item.parameter_id))
    ids = [item.parameter_id for item in parameters]
    if len(ids) != len(set(ids)):
        raise RobustnessMatrixError("parameter_id values must be unique")

    combination_count = 1
    for item in parameters:
        combination_count *= len(item.values)
    work_unit_count = combination_count * replications
    if work_unit_count > max_work_units:
        raise RobustnessMatrixError(
            f"campaign would require {work_unit_count} work units, exceeding max_work_units={max_work_units}; reduce parameters/levels/replications or raise the explicit guard"
        )

    config_material = {
        "schema": ROBUSTNESS_CAMPAIGN_SCHEMA,
        "algorithm_id": ROBUSTNESS_MATRIX_ALGORITHM_ID,
        "baseline": baseline.canonical_dict(),
        "scientific_question": question,
        "campaign_intent": intent,
        "design": FULL_FACTORIAL_GRID,
        "master_seed": master_seed,
        "replications": replications,
        "confidence_target": confidence,
        "estimated_seconds_per_work_unit": _canonical_decimal(seconds),
        "parameters": [item.canonical_dict() for item in parameters],
    }
    campaign_sha = _sha(config_material)

    combinations: list[RobustnessCombination] = []
    work_units: list[RobustnessWorkUnit] = []
    for combination_index, level_tuple in enumerate(product(*(item.values for item in parameters))):
        values = {item.parameter_id: level for item, level in zip(parameters, level_tuple)}
        combination_material = {
            "campaign_identity_sha256": campaign_sha,
            "combination_index": combination_index,
            "parameter_values": values,
        }
        combination_sha = _sha(combination_material)
        combinations.append(RobustnessCombination(combination_index, values, combination_sha))
        for replication_index in range(replications):
            seed = _derived_seed(campaign_sha, combination_sha, replication_index, master_seed)
            unit_index = len(work_units)
            unit_material = {
                "schema": ROBUSTNESS_WORK_UNIT_SCHEMA,
                "campaign_identity_sha256": campaign_sha,
                "combination_identity_sha256": combination_sha,
                "combination_index": combination_index,
                "replication_index": replication_index,
                "derived_seed": seed,
                "parameter_values": values,
            }
            work_units.append(
                RobustnessWorkUnit(
                    work_unit_index=unit_index,
                    combination_index=combination_index,
                    replication_index=replication_index,
                    derived_seed=seed,
                    parameter_values=values,
                    combination_identity_sha256=combination_sha,
                    work_unit_identity_sha256=_sha(unit_material),
                )
            )

    return RobustnessCampaignPlan(
        schema=ROBUSTNESS_CAMPAIGN_SCHEMA,
        algorithm_id=ROBUSTNESS_MATRIX_ALGORITHM_ID,
        baseline=baseline,
        scientific_question=question,
        campaign_intent=intent,
        master_seed=master_seed,
        replications=replications,
        confidence_target=confidence,
        estimated_seconds_per_work_unit=_canonical_decimal(seconds),
        parameters=parameters,
        combinations=tuple(combinations),
        work_units=tuple(work_units),
        campaign_identity_sha256=campaign_sha,
    )
