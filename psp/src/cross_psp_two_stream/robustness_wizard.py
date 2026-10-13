"""Guided robustness-campaign UX over the A89/A90/A91 orchestration layers.

This module is deliberately orchestration-only.  It resolves an identity-bearing
Custom Physics v2 baseline, builds the deterministic A89 campaign plan, drives the A90
checkpoint/resume executor through an injected scientific work-unit callable, and
renders the A91 aggregation result.  It never rewrites D0/Q5 mechanisms or silently
promotes a scenario to ``QUALIFIED_PROFILE``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import yaml

from vessel_crossing.diagnostic.contract import load_diagnostic_scenario

from .custom_physics_binding import (
    bind_distribution_scenario,
    bind_fixed_scenario,
    bind_fleet_mixture_scenario,
)
from .fleet_mixtures import fleet_mixture_plan_from_mapping
from .physics_distributions import physics_distribution_plan_from_mapping
from .q5_integration import prepare_q5_scenario
from .robustness_aggregation import (
    ROBUSTNESS_AGGREGATION_SCHEMA,
    RobustnessAggregationError,
    aggregate_robustness_campaign,
)
from .robustness_execution import (
    STATUS_CANCELLED_RESUMABLE,
    STATUS_COMPLETE,
    STATUS_FAILED_RESUMABLE,
    STATUS_PAUSED_RESUMABLE,
    RobustnessExecutionError,
    run_robustness_campaign,
)
from .robustness_matrix import (
    ROBUSTNESS_CAMPAIGN_SCHEMA,
    RobustnessBaseline,
    RobustnessCampaignPlan,
    RobustnessWorkUnit,
    robustness_campaign_plan_from_mapping,
)

ROBUSTNESS_WIZARD_SCHEMA = "cross-psp-robustness-wizard-v1"
ROBUSTNESS_WIZARD_CONFIG_ALGORITHM_ID = "cross-psp-robustness-wizard-config-v1"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class RobustnessWizardError(ValueError):
    """Raised when a saved/public robustness wizard configuration is incompatible."""



class _WizardCancelled(Exception):
    pass


class _WizardBack(Exception):
    pass


def _prompt(io, label: str, default: str, parse):
    while True:
        raw = io.read(f"{label} [{default}] > ").strip()
        token = raw.lower()
        if token in {"q", "quit", "cancel", "/cancel"}:
            raise _WizardCancelled
        if token in {"b", "back", "/back"}:
            raise _WizardBack
        if raw == "":
            raw = default
        try:
            return parse(raw)
        except (TypeError, ValueError) as exc:
            io.write(f"Invalid value: {exc}")


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise ValueError("must be an integer >= 1")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not isfinite(value) or value <= 0:
        raise ValueError("must be a positive finite number")
    return value


def _nonnegative_float(text: str) -> float:
    value = float(text)
    if not isfinite(value) or value < 0:
        raise ValueError("must be a finite number >= 0")
    return value


def _fraction_or_none(text: str):
    token = text.strip().lower()
    if token in {"none", "null", "n/a"}:
        return None
    value = float(text)
    if not isfinite(value) or not 0 < value < 1:
        raise ValueError("must be strictly between 0 and 1, or 'none'")
    return value


def _optional_float(text: str):
    token = text.strip().lower()
    if token in {"none", "null", "n/a"}:
        return None
    value = float(text)
    if not isfinite(value):
        raise ValueError("must be finite or 'none'")
    return value


def _levels(text: str) -> int:
    value = int(text)
    if value < 2:
        raise ValueError("must be an integer >= 2")
    return value


def _default_robustness_config_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("cross_custom_configs") / f"cross_robustness_{stamp}.yaml"


def _interactive_config(io) -> dict[str, Any]:
    io.write("Robustness campaign setup. Type 'back' or 'cancel' at any prompt.")
    scenario_path = Path(str(_prompt(io, "Baseline diagnostic scenario YAML", "cross_psp_scenario.yaml", str)))
    mode_choice = str(_prompt(io, "Physics mode: 1=fixed, 2=distributions, 3=fleet mixture", "1", str))
    mode = {"1": "fixed", "2": "distributions", "3": "fleet_mixture"}.get(mode_choice)
    if mode is None:
        raise RobustnessWizardError("physics mode must be 1, 2, or 3")
    plan_path = None
    if mode != "fixed":
        plan_path = Path(str(_prompt(io, "Sampled physics plan YAML", "custom_physics_plan.yaml", str)))
    source = baseline_source_from_files(mode=mode, scenario_path=scenario_path, plan_path=plan_path)

    scientific_question = str(_prompt(
        io,
        "Scientific question",
        "Does the primary conclusion remain stable over the declared envelope?",
        str,
    )).strip()
    intent_choice = str(_prompt(io, "Campaign intent: 1=exploratory, 2=qualification evidence", "1", str))
    campaign_intent = {"1": "exploratory", "2": "qualification_evidence"}.get(intent_choice)
    if campaign_intent is None:
        raise RobustnessWizardError("campaign intent must be 1 or 2")
    master_seed = int(_prompt(io, "Master seed", "1234", int))
    if master_seed < 0:
        raise RobustnessWizardError("master seed must be >= 0")
    replication_default = "1" if mode == "fixed" else "2"
    replications = int(_prompt(io, "Campaign replications per combination", replication_default, _positive_int))
    if mode == "fixed" and replications != 1:
        raise RobustnessWizardError(
            "fixed physics requires campaign replications=1: Q5 already performs its own paired "
            "traffic replications; use sampled-physics mode for an outer sampled-physics replication axis"
        )
    confidence = _prompt(io, "Confidence target (or none)", "0.95", _fraction_or_none)
    seconds_per_unit = float(_prompt(io, "Estimated seconds per work unit", "60", _positive_float))
    max_work_units = int(_prompt(io, "Maximum allowed work units", "1000", _positive_int))
    parameter_count = int(_prompt(io, "Number of robustness parameters", "1", _positive_int))

    parameters: list[dict[str, Any]] = []
    help_text: dict[str, str] = {}
    for index in range(parameter_count):
        io.write(f"\nParameter {index + 1}/{parameter_count}")
        default_id = "stream.alpha.length_m" if index == 0 else f"parameter.{index + 1}"
        parameter_id = str(_prompt(io, "Parameter id", default_id, str)).strip()
        meaning = str(_prompt(io, "Meaning", "User-declared robustness parameter", str)).strip()
        unit = str(_prompt(io, "Unit", "m", str)).strip()
        baseline = float(_prompt(io, "Baseline value", "120", float))
        if not isfinite(baseline):
            raise RobustnessWizardError("baseline value must be finite")
        reference = _prompt(io, "Reference/qualified value (or none)", str(baseline), _optional_float)
        supported_lower = _prompt(io, "Model-supported lower bound (or none)", "none", _optional_float)
        supported_upper = _prompt(io, "Model-supported upper bound (or none)", "none", _optional_float)
        variation_choice = str(_prompt(
            io,
            "Variation: 1=symmetric %, 2=asymmetric %, 3=absolute bounds",
            "1",
            str,
        ))
        if variation_choice == "1":
            variation = {
                "mode": "symmetric_percent",
                "percent": float(_prompt(io, "Symmetric percentage", "10", _positive_float)),
                "levels": int(_prompt(io, "Grid levels (odd for two-sided)", "3", _levels)),
            }
        elif variation_choice == "2":
            variation = {
                "mode": "asymmetric_percent",
                "lower_percent": float(_prompt(io, "Lower percentage", "10", _nonnegative_float)),
                "upper_percent": float(_prompt(io, "Upper percentage", "10", _nonnegative_float)),
                "levels": int(_prompt(io, "Grid levels (odd for two-sided)", "3", _levels)),
            }
        elif variation_choice == "3":
            default_lower = baseline - 1 if baseline == 0 else baseline * 0.9
            default_upper = baseline + 1 if baseline == 0 else baseline * 1.1
            variation = {
                "mode": "absolute_bounds",
                "lower": float(_prompt(io, "Absolute lower bound", str(default_lower), float)),
                "upper": float(_prompt(io, "Absolute upper bound", str(default_upper), float)),
                "levels": int(_prompt(io, "Grid levels", "3", _levels)),
            }
        else:
            raise RobustnessWizardError("variation mode must be 1, 2, or 3")
        parameters.append({
            "parameter_id": parameter_id,
            "unit": unit,
            "baseline_value": baseline,
            "reference_value": reference,
            "supported_bounds": {"lower": supported_lower, "upper": supported_upper},
            "variation": variation,
        })
        help_text[parameter_id] = meaning

    metric_path = str(_prompt(io, "Primary result metric path", "metrics.primary", str)).strip()
    stable_delta = float(_prompt(io, "Stable absolute-delta threshold", "1", _nonnegative_float))
    sensitive_delta = float(_prompt(io, "Sensitive absolute-delta threshold", "10", _nonnegative_float))
    operator = str(_prompt(io, "Primary conclusion operator (>=, >, <=, <)", ">=", str)).strip()
    threshold = float(_prompt(io, "Primary conclusion threshold", "0", float))

    return {
        "schema": ROBUSTNESS_WIZARD_SCHEMA,
        "baseline_source": source,
        "campaign": {
            "scientific_question": scientific_question,
            "campaign_intent": campaign_intent,
            "design": "full_factorial_grid",
            "master_seed": master_seed,
            "replications": replications,
            "confidence_target": confidence,
            "estimated_seconds_per_work_unit": seconds_per_unit,
            "max_work_units": max_work_units,
            "parameters": parameters,
        },
        "parameter_help": help_text,
        "aggregation": {
            "schema": ROBUSTNESS_AGGREGATION_SCHEMA,
            "primary_metric_path": metric_path,
            "stable_absolute_delta": stable_delta,
            "sensitive_absolute_delta": sensitive_delta,
            "conclusion_rule": {"operator": operator, "threshold": threshold},
        },
    }


@dataclass(frozen=True, slots=True)
class RobustnessWizardPreflight:
    plan: RobustnessCampaignPlan
    aggregation_config: Mapping[str, object]
    parameter_help: Mapping[str, str]
    config_identity_sha256: str
    config_path: Path
    checkpoint_path: Path


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


def _sha256(name: str, value: object) -> str:
    text = str(value).strip().lower()
    if not _SHA_RE.fullmatch(text):
        raise RobustnessWizardError(f"{name} must be a lowercase 64-character SHA-256")
    return text


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise RobustnessWizardError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise RobustnessWizardError(f"{path} must contain a mapping")
    return dict(raw)


def _baseline_from_source(source: Mapping[str, Any]) -> RobustnessBaseline:
    required = {
        "mode",
        "scenario_path",
        "plan_path",
        "expected_physics_identity_sha256",
        "expected_plan_identity_sha256",
    }
    if set(source) != required:
        raise RobustnessWizardError(
            "baseline_source fields mismatch; "
            f"missing={sorted(required - set(source))}, extra={sorted(set(source) - required)}"
        )
    mode = str(source["mode"])
    if mode not in {"fixed", "distributions", "fleet_mixture"}:
        raise RobustnessWizardError(f"unsupported baseline mode {mode!r}")
    scenario_path = Path(str(source["scenario_path"]))
    if not scenario_path.is_file():
        raise RobustnessWizardError(f"baseline scenario not found: {scenario_path}")
    prepared = prepare_q5_scenario(load_diagnostic_scenario(scenario_path))

    plan_path_raw = source["plan_path"]
    if mode == "fixed":
        if plan_path_raw is not None:
            raise RobustnessWizardError("fixed baseline must have plan_path=null")
        bound = bind_fixed_scenario(prepared)
    elif mode == "distributions":
        if plan_path_raw is None:
            raise RobustnessWizardError("distributions baseline requires plan_path")
        plan_path = Path(str(plan_path_raw))
        plan = physics_distribution_plan_from_mapping(_load_mapping(plan_path))
        bound = bind_distribution_scenario(prepared, plan)
    else:
        if plan_path_raw is None:
            raise RobustnessWizardError("fleet_mixture baseline requires plan_path")
        plan_path = Path(str(plan_path_raw))
        plan = fleet_mixture_plan_from_mapping(_load_mapping(plan_path))
        bound = bind_fleet_mixture_scenario(prepared, plan)

    expected_physics = _sha256(
        "baseline_source.expected_physics_identity_sha256",
        source["expected_physics_identity_sha256"],
    )
    if bound.physics_identity_sha256 != expected_physics:
        raise RobustnessWizardError(
            "baseline physics identity mismatch: saved config does not match current baseline files"
        )
    expected_plan = source["expected_plan_identity_sha256"]
    if bound.plan_identity_sha256 is None:
        if expected_plan is not None:
            raise RobustnessWizardError("fixed baseline must have expected_plan_identity_sha256=null")
    else:
        if expected_plan is None:
            raise RobustnessWizardError("sampled baseline requires expected_plan_identity_sha256")
        normalized = _sha256("baseline_source.expected_plan_identity_sha256", expected_plan)
        if bound.plan_identity_sha256 != normalized:
            raise RobustnessWizardError(
                "baseline plan identity mismatch: saved config does not match current sampled plan"
            )
    return RobustnessBaseline(
        mode=mode,
        physics_identity_sha256=bound.physics_identity_sha256,
        scope_status=bound.scope_status,
        plan_identity_sha256=bound.plan_identity_sha256,
    )


def baseline_source_from_files(
    *,
    mode: str,
    scenario_path: str | Path,
    plan_path: str | Path | None = None,
) -> dict[str, object]:
    """Resolve current files once and return an identity-pinned baseline source mapping."""
    provisional = {
        "mode": str(mode),
        "scenario_path": str(Path(scenario_path)),
        "plan_path": None if plan_path is None else str(Path(plan_path)),
        "expected_physics_identity_sha256": "0" * 64,
        "expected_plan_identity_sha256": None,
    }
    # Resolve without using the placeholder identity, then pin the returned identities.
    scenario = Path(str(provisional["scenario_path"]))
    if not scenario.is_file():
        raise RobustnessWizardError(f"baseline scenario not found: {scenario}")
    prepared = prepare_q5_scenario(load_diagnostic_scenario(scenario))
    if mode == "fixed":
        if plan_path is not None:
            raise RobustnessWizardError("fixed baseline must not provide a sampled plan")
        bound = bind_fixed_scenario(prepared)
    elif mode == "distributions":
        if plan_path is None:
            raise RobustnessWizardError("distributions baseline requires plan_path")
        plan = physics_distribution_plan_from_mapping(_load_mapping(Path(plan_path)))
        bound = bind_distribution_scenario(prepared, plan)
    elif mode == "fleet_mixture":
        if plan_path is None:
            raise RobustnessWizardError("fleet_mixture baseline requires plan_path")
        plan = fleet_mixture_plan_from_mapping(_load_mapping(Path(plan_path)))
        bound = bind_fleet_mixture_scenario(prepared, plan)
    else:
        raise RobustnessWizardError(f"unsupported baseline mode {mode!r}")
    provisional["expected_physics_identity_sha256"] = bound.physics_identity_sha256
    provisional["expected_plan_identity_sha256"] = bound.plan_identity_sha256
    return provisional


def write_robustness_wizard_config(mapping: Mapping[str, Any], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(dict(mapping), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination


def load_robustness_wizard_config(path: str | Path) -> dict[str, Any]:
    return _load_mapping(Path(path))


def _validate_aggregation(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise RobustnessWizardError("aggregation must be a mapping")
    required = {
        "schema",
        "primary_metric_path",
        "stable_absolute_delta",
        "sensitive_absolute_delta",
        "conclusion_rule",
    }
    if set(raw) != required:
        raise RobustnessWizardError(
            f"aggregation fields mismatch; missing={sorted(required - set(raw))}, extra={sorted(set(raw) - required)}"
        )
    if raw["schema"] != ROBUSTNESS_AGGREGATION_SCHEMA:
        raise RobustnessWizardError(f"aggregation.schema must be {ROBUSTNESS_AGGREGATION_SCHEMA!r}")
    metric = str(raw["primary_metric_path"]).strip()
    if not metric or any(not item for item in metric.split(".")):
        raise RobustnessWizardError("primary_metric_path must be a non-empty dot path")
    try:
        stable = float(raw["stable_absolute_delta"])
        sensitive = float(raw["sensitive_absolute_delta"])
    except (TypeError, ValueError) as exc:
        raise RobustnessWizardError("aggregation thresholds must be finite numbers") from exc
    if not (isfinite(stable) and isfinite(sensitive)) or stable < 0 or sensitive < 0 or stable > sensitive:
        raise RobustnessWizardError(
            "aggregation thresholds must satisfy 0 <= stable_absolute_delta <= sensitive_absolute_delta"
        )
    rule = raw["conclusion_rule"]
    if not isinstance(rule, Mapping) or set(rule) != {"operator", "threshold"}:
        raise RobustnessWizardError("conclusion_rule must contain exactly operator and threshold")
    operator = str(rule["operator"])
    if operator not in {">=", ">", "<=", "<"}:
        raise RobustnessWizardError("conclusion_rule.operator must be one of >=, >, <=, <")
    try:
        threshold = float(rule["threshold"])
    except (TypeError, ValueError) as exc:
        raise RobustnessWizardError("conclusion_rule.threshold must be finite") from exc
    if not isfinite(threshold):
        raise RobustnessWizardError("conclusion_rule.threshold must be finite")
    return {
        "schema": ROBUSTNESS_AGGREGATION_SCHEMA,
        "primary_metric_path": metric,
        "stable_absolute_delta": stable,
        "sensitive_absolute_delta": sensitive,
        "conclusion_rule": {"operator": operator, "threshold": threshold},
    }


def preflight_robustness_wizard_config(
    path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> RobustnessWizardPreflight:
    config_path = Path(path)
    mapping = load_robustness_wizard_config(config_path)
    required = {"schema", "baseline_source", "campaign", "parameter_help", "aggregation"}
    if set(mapping) != required:
        raise RobustnessWizardError(
            f"wizard config fields mismatch; missing={sorted(required - set(mapping))}, extra={sorted(set(mapping) - required)}"
        )
    if mapping["schema"] != ROBUSTNESS_WIZARD_SCHEMA:
        raise RobustnessWizardError(f"schema must be {ROBUSTNESS_WIZARD_SCHEMA!r}")
    if not isinstance(mapping["baseline_source"], Mapping):
        raise RobustnessWizardError("baseline_source must be a mapping")
    baseline = _baseline_from_source(mapping["baseline_source"])
    campaign = mapping["campaign"]
    if not isinstance(campaign, Mapping):
        raise RobustnessWizardError("campaign must be a mapping")
    if "schema" in campaign or "baseline" in campaign:
        raise RobustnessWizardError("campaign must omit schema and baseline; wizard derives them from baseline_source")
    plan = robustness_campaign_plan_from_mapping({
        "schema": ROBUSTNESS_CAMPAIGN_SCHEMA,
        "baseline": baseline.canonical_dict(),
        **dict(campaign),
    })
    if baseline.mode == "fixed" and plan.replications != 1:
        raise RobustnessWizardError(
            "fixed-physics robustness requires campaign.replications=1: Q5 already executes paired "
            "internal traffic replications. Outer duplicate fixed replications would not add an "
            "independent uncertainty axis."
        )
    help_raw = mapping["parameter_help"]
    if not isinstance(help_raw, Mapping):
        raise RobustnessWizardError("parameter_help must be a mapping")
    expected_parameters = {item.parameter_id for item in plan.parameters}
    if set(help_raw) != expected_parameters:
        raise RobustnessWizardError(
            "parameter_help keys must exactly match campaign parameters; "
            f"missing={sorted(expected_parameters - set(help_raw))}, extra={sorted(set(help_raw) - expected_parameters)}"
        )
    help_text: dict[str, str] = {}
    for key, value in help_raw.items():
        text = str(value).strip()
        if not text:
            raise RobustnessWizardError(f"parameter_help[{key!r}] must be non-empty")
        help_text[str(key)] = text
    aggregation = _validate_aggregation(mapping["aggregation"])
    canonical = {
        "schema": ROBUSTNESS_WIZARD_SCHEMA,
        "baseline": baseline.canonical_dict(),
        "campaign_identity_sha256": plan.campaign_identity_sha256,
        "parameter_help": help_text,
        "aggregation": aggregation,
        "config_algorithm_id": ROBUSTNESS_WIZARD_CONFIG_ALGORITHM_ID,
    }
    root = Path(output_dir) if output_dir is not None else Path("cross_psp_robustness_outputs") / config_path.stem
    return RobustnessWizardPreflight(
        plan=plan,
        aggregation_config=aggregation,
        parameter_help=help_text,
        config_identity_sha256=_sha(canonical),
        config_path=config_path,
        checkpoint_path=root / "checkpoint.json",
    )


def _report_markdown(human: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    lines = [
        "# CROSS PSP Robustness Report",
        "",
        str(human["headline"]),
        "",
        f"- primary metric: `{human['primary_metric']}`",
        f"- baseline mean: `{human['baseline_mean']}`",
        f"- claim status: `{human['claim_status']}`",
        f"- region counts: `{json.dumps(human['region_counts'], sort_keys=True)}`",
        f"- conclusion flips: `{human['conclusion_flip_combination_indices']}`",
        f"- qualification note: {human['qualification_note']}",
        "",
        "## Strongest influences",
    ]
    influences = human.get("strongest_influences", [])
    if influences:
        for item in influences:
            lines.append(
                f"- `{item['parameter_id']}`: direction={item['direction']}, "
                f"max_abs_effect={item['max_abs_effect']}, rank={item['importance_rank']}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Strongest interactions"])
    interactions = human.get("strongest_interactions", [])
    if interactions:
        for item in interactions:
            lines.append(
                f"- `{item['parameter_a']}` × `{item['parameter_b']}`: "
                f"max_abs_interaction={item['max_abs_interaction']}"
            )
    else:
        lines.append("- none or unavailable")
    replication = human.get("replication_summary")
    lines.extend(["", "## Uncertainty / replication semantics"])
    if replication:
        lines.append(
            f"- campaign replication axis: `{replication['campaign_replication_axis']}`; "
            f"count={replication['campaign_replication_count']}"
        )
        lines.append(
            f"- Q5 internal replication axis: `{replication['q5_internal_replication_axis']}`; "
            f"count={replication['q5_internal_replication_count']}; "
            f"paired={replication['q5_internal_paired']}; "
            f"confidence_level={replication['q5_internal_confidence_level']}"
        )
        lines.append(f"- internal seed schedule: `{replication['q5_internal_seed_schedule']}`")
        lines.append(f"- standard-error scope: `{replication['standard_error_scope']}`")
        lines.append(f"- note: {replication['note']}")
    else:
        lines.append("- runtime replication semantics unavailable for this legacy/synthetic result")
    lines.extend([
        "",
        "## Claim boundary",
        "",
        "This is model-based robustness evidence within the declared tested envelope. "
        "It is not field calibration, deployment advice, a causal claim, or a global-optimum claim.",
        "",
        f"Aggregation identity: `{result['aggregation_identity_sha256']}`",
    ])
    return "\n".join(lines) + "\n"


def _yes(io, prompt: str, default: bool = False) -> bool:
    token = "Y/n" if default else "y/N"
    while True:
        raw = io.read(f"{prompt} [{token}] > ").strip().lower()
        if raw in {"q", "quit", "cancel"}:
            return False
        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        io.write("Please answer Y or N.")


def _show_preflight(io, preflight: RobustnessWizardPreflight, output_dir: Path) -> None:
    plan = preflight.plan
    io.write("\nRobustness preflight PASS")
    io.write(f"  config: {preflight.config_path}")
    io.write(f"  config identity: {preflight.config_identity_sha256}")
    io.write(f"  campaign identity: {plan.campaign_identity_sha256}")
    io.write(f"  baseline mode: {plan.baseline.mode}")
    io.write(f"  baseline scope: {plan.baseline.scope_status}")
    io.write(f"  baseline physics identity: {plan.baseline.physics_identity_sha256}")
    if plan.baseline.plan_identity_sha256:
        io.write(f"  baseline sampled-plan identity: {plan.baseline.plan_identity_sha256}")
    io.write(f"  scientific question: {plan.scientific_question}")
    for parameter in plan.parameters:
        meaning = preflight.parameter_help[parameter.parameter_id]
        io.write(
            f"  parameter {parameter.parameter_id}: {meaning}; unit={parameter.unit}; "
            f"baseline={parameter.baseline_value}; reference={parameter.reference_value}; "
            f"supported=[{parameter.supported_lower},{parameter.supported_upper}]; "
            f"requested=[{parameter.requested_lower},{parameter.requested_upper}]; "
            f"levels={len(parameter.values)}; mode={parameter.variation_mode}"
        )
    io.write(
        f"  combinations / campaign replications / work units: "
        f"{plan.combination_count} / {plan.replications} / {plan.work_unit_count}"
    )
    if plan.baseline.mode == "fixed":
        io.write(
            "  replication semantics: campaign replications=1; Q5 owns paired internal "
            "traffic-realization replications for each physics point"
        )
    else:
        io.write(
            "  replication semantics: campaign replications vary sampled-physics seed; Q5 also "
            "owns paired internal traffic-realization replications inside each work unit"
        )
    io.write(
        f"  workload estimate: {plan.estimated_total_seconds} sec total "
        f"({plan.estimated_seconds_per_work_unit} sec/work unit; machine dependent)"
    )
    io.write(f"  checkpoint/resume: {output_dir / 'checkpoint.json'}")
    io.write("  resulting claim status: SUPPORTED_EXPLORATORY")
    io.write(
        "  claim boundary: causal=false; field_calibrated=false; "
        "deployment_recommendation=false; global_optimum_claim=false"
    )


def run_robustness_wizard(
    *,
    io,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    execute_unit: Callable[[RobustnessWorkUnit], Mapping[str, Any]] | None = None,
    execution_code_identity_sha256: str | None = None,
    open_result: Callable[[Path], None] | None = None,
) -> int:
    """Run a saved A92 robustness config through preflight, A90 execution, and A91 report.

    The scientific work-unit callable is injected.  A92 owns the public UX and durable
    orchestration only; the callable remains the explicit supported-simulator adapter.
    """
    try:
        destination = Path(config_path) if config_path is not None else _default_robustness_config_path()
        if destination.is_file():
            io.write(f"Loading saved robustness configuration: {destination}")
        else:
            mapping = _interactive_config(io)
            write_robustness_wizard_config(mapping, destination)
            io.write(f"Saved robustness configuration: {destination}")
        root = Path(output_dir) if output_dir is not None else Path("cross_psp_robustness_outputs") / destination.stem
        preflight = preflight_robustness_wizard_config(destination, output_dir=root)
        _show_preflight(io, preflight, root)
        if execute_unit is None or execution_code_identity_sha256 is None:
            io.write(
                "Configuration is saved and preflighted. Scientific work-unit execution "
                "is not attached to this invocation; no campaign was started."
            )
            return 0
        code_sha = _sha256("execution_code_identity_sha256", execution_code_identity_sha256)
        if not _yes(io, "Start or resume this robustness campaign now?", default=False):
            io.write("No robustness work unit was started. The saved config remains reusable.")
            return 0

        root.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                summary = run_robustness_campaign(
                    preflight.plan,
                    root,
                    execute_unit,
                    execution_code_identity_sha256=code_sha,
                    max_new_units=1,
                )
            except KeyboardInterrupt:
                io.write(
                    f"Campaign interrupted safely. Resume with the same config; checkpoint={root / 'checkpoint.json'}"
                )
                return 130
            io.write(
                f"Robustness progress: completed={summary.completed_count}/{preflight.plan.work_unit_count}; "
                f"remaining={summary.remaining_count}; status={summary.status}; checkpoint={summary.checkpoint_path}"
            )
            if summary.status == STATUS_COMPLETE:
                break
            if summary.status == STATUS_PAUSED_RESUMABLE:
                continue
            if summary.status in {STATUS_CANCELLED_RESUMABLE, STATUS_FAILED_RESUMABLE}:
                io.write(
                    f"Campaign stopped in resumable state {summary.status}. "
                    f"Resume with the same config and code identity."
                )
                return 2 if summary.status == STATUS_FAILED_RESUMABLE else 0
            raise RobustnessWizardError(f"unexpected execution status {summary.status!r}")

        aggregation_config = {
            **dict(preflight.aggregation_config),
            "execution_code_identity_sha256": code_sha,
        }
        result = aggregate_robustness_campaign(preflight.plan, root, aggregation_config)
        machine_path = root / "ROBUSTNESS_RESULT.json"
        report_path = root / "ROBUSTNESS_REPORT.md"
        machine_path.write_text(
            json.dumps(result.canonical_dict(), sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report_path.write_text(
            _report_markdown(result.human_report_dict(), result.canonical_dict()),
            encoding="utf-8",
        )
        human = result.human_report_dict()
        io.write("ROBUSTNESS CAMPAIGN COMPLETE")
        io.write(f"  {human['headline']}")
        io.write(f"  strongest influences: {human['strongest_influences']}")
        io.write(f"  region counts: {human['region_counts']}")
        io.write(f"  claim status: {human['claim_status']}")
        io.write(f"  primary report: {report_path.resolve()}")
        io.write(f"  machine result: {machine_path.resolve()}")
        if open_result is not None:
            try:
                open_result(report_path.resolve())
            except OSError as exc:
                io.write(f"  result could not be opened automatically: {exc}")
        return 0
    except _WizardBack:
        io.write("Back requested. No new robustness work unit was started.")
        return 3
    except _WizardCancelled:
        io.write("Cancelled. No new robustness work unit was started.")
        return 0
    except (RobustnessWizardError, RobustnessExecutionError, RobustnessAggregationError) as exc:
        io.write(f"FAIL-CLOSED robustness campaign: {exc}")
        location = Path(output_dir).resolve() if output_dir is not None else Path("cross_psp_robustness_outputs").resolve()
        io.write(
            f"  recoverable action: inspect/fix the saved config or identity mismatch; "
            f"valid completed units, if any, remain under {location}"
        )
        return 2
