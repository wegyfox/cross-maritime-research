"""User-facing custom-analysis validation and result summaries.

RC2 accepts traffic/statistical changes plus a narrow, explicitly validated physical
scenario surface.  Scientific domain validation itself lives in ``dynamic_physics``; this
module only controls which parts of the diagnostic scenario users may attempt to edit.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

ANALYSIS_SURFACE_ID = "cross-psp-dynamic-physics-analysis-v2"
ANALYSIS_SUMMARY_FORMAT = "cross-psp-analysis-summary-v2"

# Non-physical fields that may differ from the bundled Q5 scenario.
_ALLOWED_EXACT_PATHS = {
    "scenario_id",
    "title",
    "description",
    "model.parameters.label",
    "model.parameters.scenario.seed",
    "model.parameters.scenario.traffic_horizon_min",
    "model.parameters.scenario.end_time",
    "experiment.replications",
    "experiment.confidence_level",
    "experiment.seed_namespace",
    "experiment.resume",
    "metadata.analysis_note",
}
_PHYSICAL_PREFIXES = (
    "model.parameters.network.",
    "model.parameters.vessel_classes.",
    "model.parameters.scenario.default_environmental_drift_nm_per_min[",
    "model.parameters.scenario.stream_environmental_drift_nm_per_min.",
    "metadata.b6_target_stream_shares.",
)
_ALLOWED_STREAM_FIELDS = {"rate_per_hour", "cruise_speed_knots", "vessel_class_id"}
_PHYSICAL_STREAM_FIELDS = {"cruise_speed_knots", "vessel_class_id"}


class AnalysisSurfaceError(ValueError):
    """Raised when a custom analysis request leaves the supported v1 surface."""


@dataclass(frozen=True, slots=True)
class AnalysisValidation:
    changed_paths: tuple[str, ...]
    physical_changed_paths: tuple[str, ...]
    rate_per_hour_by_stream: Mapping[str, float]
    seed: int
    traffic_horizon_min: float
    end_time: float
    replications: int
    confidence_level: float

    def to_json_object(self) -> dict[str, Any]:
        return {
            "analysis_surface_id": ANALYSIS_SURFACE_ID,
            "changed_paths": list(self.changed_paths),
            "physical_changed_paths": list(self.physical_changed_paths),
            "rate_per_hour_by_stream": dict(self.rate_per_hour_by_stream),
            "seed": self.seed,
            "traffic_horizon_min": self.traffic_horizon_min,
            "end_time": self.end_time,
            "replications": self.replications,
            "confidence_level": self.confidence_level,
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise AnalysisSurfaceError(f"scenario YAML must contain an object: {path}")
    return raw


def load_baseline_analysis_scenario() -> dict[str, Any]:
    resource = files("cross_psp_two_stream.data").joinpath("q5_scenario.yaml")
    with as_file(resource) as path:
        return _load_yaml(path)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    rows: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.update(_flatten(value[key], child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            rows.update(_flatten(item, child))
    else:
        rows[prefix] = value
    return rows


def _allowed(path: str) -> bool:
    if path in _ALLOWED_EXACT_PATHS:
        return True
    if path.startswith(_PHYSICAL_PREFIXES):
        return True
    if path.startswith("model.parameters.streams["):
        suffix = path.rsplit(".", 1)[-1]
        return suffix in _ALLOWED_STREAM_FIELDS
    return False


def _is_physical_path(path: str) -> bool:
    if path.startswith(_PHYSICAL_PREFIXES):
        return True
    if path.startswith("model.parameters.streams["):
        return path.rsplit(".", 1)[-1] in _PHYSICAL_STREAM_FIELDS
    return False


def validate_analysis_scenario(path: Path) -> AnalysisValidation:
    baseline = load_baseline_analysis_scenario()
    candidate = _load_yaml(path)
    left = _flatten(baseline)
    right = _flatten(candidate)
    changed: list[str] = []
    forbidden: list[str] = []
    physical_changed: list[str] = []
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            changed.append(key)
            if not _allowed(key):
                forbidden.append(key)
            elif _is_physical_path(key):
                physical_changed.append(key)
    if forbidden:
        detail = "\n  - ".join(forbidden)
        raise AnalysisSurfaceError(
            "custom scenario changes a field outside the RC2 editable surface.\n"
            "RC2 accepts traffic/statistical fields and a narrow physical scenario surface; "
            "scientific-domain violations inside that surface are diagnosed separately before execution.\n"
            f"Unsupported changed paths:\n  - {detail}"
        )

    params = candidate.get("model", {}).get("parameters", {})
    streams = params.get("streams")
    if not isinstance(streams, list) or len(streams) != 2:
        raise AnalysisSurfaceError("custom analysis requires exactly two streams")
    rate_map: dict[str, float] = {}
    for row in streams:
        if not isinstance(row, dict):
            raise AnalysisSurfaceError("stream definitions must be objects")
        stream_id = str(row.get("stream_id", ""))
        rate = float(row.get("rate_per_hour"))
        if not stream_id or rate <= 0.0:
            raise AnalysisSurfaceError("each stream requires a positive rate_per_hour")
        rate_map[stream_id] = rate

    scenario = params.get("scenario", {})
    seed = int(scenario.get("seed"))
    traffic = float(scenario.get("traffic_horizon_min"))
    end = float(scenario.get("end_time"))
    start = float(scenario.get("start_time", 0.0))
    if traffic <= start:
        raise AnalysisSurfaceError("traffic_horizon_min must be greater than start_time")
    if end <= traffic:
        raise AnalysisSurfaceError("end_time must be greater than traffic_horizon_min so the network can drain")

    experiment = candidate.get("experiment", {})
    reps = int(experiment.get("replications"))
    confidence = float(experiment.get("confidence_level"))
    if reps < 2:
        raise AnalysisSurfaceError("custom analysis requires at least 2 replications for paired uncertainty evidence")
    if not (0.0 < confidence < 1.0):
        raise AnalysisSurfaceError("confidence_level must be between 0 and 1")

    return AnalysisValidation(
        changed_paths=tuple(changed),
        physical_changed_paths=tuple(physical_changed),
        rate_per_hour_by_stream=rate_map,
        seed=seed,
        traffic_horizon_min=traffic,
        end_time=end,
        replications=reps,
        confidence_level=confidence,
    )


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    validation = payload["input_validation"]
    lines = [
        "# CROSS PSP custom analysis summary",
        "",
        f"Status: **{payload['status']}**",
        f"Analysis surface: `{payload['analysis_surface_id']}`",
        "",
        "## Inputs",
        "",
        f"- rates: `{validation['rate_per_hour_by_stream']}` vessels/hour;",
        f"- seed: `{validation['seed']}`;",
        f"- traffic horizon: `{validation['traffic_horizon_min']}` min;",
        f"- end time: `{validation['end_time']}` min;",
        f"- replications: `{validation['replications']}`;",
        f"- confidence level: `{validation['confidence_level']}`.",
        "",
        "## Dynamic physical reference",
        "",
        f"- scope: `{payload.get('physics', {}).get('scope_status', 'UNAVAILABLE')}`;",
        f"- physics identity: `{payload.get('physics', {}).get('physics_identity_sha256', 'UNAVAILABLE')}`;",
        f"- reference mode: `{payload.get('physics', {}).get('reference_mode', 'DYNAMIC_SAME_PHYSICS')}`.",
        "",
        "## Modeled diagnosis",
        "",
    ]
    bottleneck = payload.get("dominant_modeled_bottleneck", {})
    lines += [
        f"- classification: **{bottleneck.get('classification', 'UNAVAILABLE')}**;",
        f"- basis: {bottleneck.get('basis', 'not available')};",
        f"- caveat: {bottleneck.get('caveat', 'not available')}.",
        "",
        "## Canonical next research counterfactual",
        "",
    ]
    counter = payload.get("canonical_next_counterfactual", {})
    for key in (
        "for_B0_like_operation",
        "calibration",
        "deployment_recommendation",
        "safety_feasibility_tradeoff",
    ):
        if key in counter:
            lines.append(f"- **{key}**: {counter[key]}")
    lines += ["", "## Scenario metrics", ""]
    for row in payload.get("scenarios", []):
        lines += [
            f"### {row.get('scenario_id')}",
            f"- B0/current: `{row.get('current_modeled_value')}`;",
            f"- B5/managed: `{row.get('managed_modeled_value')}`;",
            f"- recoverable modeled gap: `{row.get('managed_recoverable_gap')}`;",
            f"- B6 physical reference: `{row.get('physical_reference_value')}`;",
            f"- residual reference gap: `{row.get('residual_reference_gap')}`.",
            "",
        ]
    lines += [
        "## Interpretation boundary",
        "",
        "This is a modeled research diagnosis, not a deployment recommendation or field-capacity estimate.",
        "The B6 comparison uses a same-physics dynamic reference. Changed parameter points are model-supported exploratory results unless separately qualified.",
        "",
    ]
    return "\n".join(lines)


def write_analysis_summary(
    output: Path,
    validation: AnalysisValidation,
    *,
    physics_info: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    decision_path = output / "DECISION_VIEW.json"
    if not decision_path.is_file():
        raise AnalysisSurfaceError(f"diagnostic run did not produce {decision_path}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    payload = {
        "format": ANALYSIS_SUMMARY_FORMAT,
        "status": "PASS",
        "analysis_surface_id": ANALYSIS_SURFACE_ID,
        "input_validation": validation.to_json_object(),
        "physics": dict(physics_info or {}),
        "source_decision_view_sha256": decision.get("source", {}).get("decomposition_sha256"),
        "dominant_modeled_bottleneck": decision.get("dominant_modeled_bottleneck", {}),
        "canonical_next_counterfactual": decision.get("canonical_next_counterfactual", {}),
        "claim_scope": decision.get("claim_scope", {}),
        "scenarios": decision.get("scenarios", []),
        "warnings": [
            "Modeled research diagnosis only; not field calibrated or deployment-ready.",
            "B6 is recomputed from the same normalized physical configuration; the packaged A70 85/h qualification-fixture scalar is not reused for RC2 custom analysis.",
        ],
    }
    json_path = output / "ANALYSIS_SUMMARY.json"
    md_path = output / "ANALYSIS_SUMMARY.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_summary_markdown(payload), encoding="utf-8")
    return json_path, md_path


def classify_analysis_failure(error: str) -> tuple[str, str, str]:
    text = str(error or "").strip()
    lower = text.lower()
    if "unique starboard-side give-way vessel" in lower or "ambiguous crossing" in lower:
        return (
            "SEMANTIC_AMBIGUITY_FAIL_CLOSED",
            "The frozen encounter lifecycle could not assign one unique starboard-side give-way vessel for this realization.",
            "Inspect the encounter/seed that triggered the ambiguity. Treat it as a semantic domain boundary; do not interpret it as a traffic-capacity result.",
        )
    if "reservoir streams must exactly match the two profile streams" in lower:
        return (
            "INSUFFICIENT_REALIZED_DEMAND",
            "This stochastic realization did not provide pending demand for both qualified profile streams when the managed controller initialized.",
            "Use a longer traffic-generation horizon, a different seed, or sufficient stream demand. Treat this as realization coverage, not as a physical-capacity result.",
        )
    if "no safe progress-capable row" in lower or "noenvironmentalpaircandidate" in lower:
        return (
            "MODEL_ACTION_SPACE_BOUNDARY",
            "The frozen candidate/action-space produced no safe progress-capable command for the realized state.",
            "Inspect candidate coverage and the terminal state. Do not interpret this modeled terminal as proof of physical impossibility or a capacity threshold.",
        )
    if "dynamic_reference_fail_closed" in lower or "fail_non_promotable_reference" in lower:
        return (
            "DYNAMIC_REFERENCE_FAIL_CLOSED",
            "The same-physics B6 reference could not satisfy its required safety, balance, drain, convergence, or deterministic-replay gate.",
            "Inspect the dynamic-reference evidence and change only the scientific scenario or reference horizon in a separately versioned study; do not substitute the packaged 85/h qualification-fixture value.",
        )
    if "rc2 v1" in lower or "requires exactly" in lower or "cruise speed exceeds" in lower or "safety domain required" in lower:
        return (
            "UNSUPPORTED_PHYSICAL_CONFIGURATION",
            "The requested physical configuration is outside the RC2 v1 dynamic-reference domain.",
            "Use two straight orthogonal zero-drift streams with explicit valid vessel physics, or treat the requested extension as a later scientific-profile version.",
        )
    if "resource_unavailable" in lower or "resource unavailable" in lower or "cartesian" in lower and "limit" in lower:
        return (
            "RESOURCE_UNAVAILABLE",
            "The run exhausted a declared computational/resource budget.",
            "Increase or redesign the explicit resource budget separately from the physical model; do not classify this as modeled infeasibility.",
        )
    return (
        "UNEXPECTED_ERROR",
        "The run terminated with an error that is not part of the frozen custom-analysis terminal taxonomy.",
        "Preserve the exact error and reproduction inputs for engineering review before drawing a scientific conclusion.",
    )


def write_analysis_failure(
    output: Path,
    validation: AnalysisValidation,
    *,
    error: str,
) -> tuple[Path, Path, str]:
    output.mkdir(parents=True, exist_ok=True)
    classification, explanation, next_action = classify_analysis_failure(error)
    payload = {
        "format": ANALYSIS_SUMMARY_FORMAT,
        "status": "FAIL_CLOSED_DIAGNOSED" if classification != "UNEXPECTED_ERROR" else "ERROR",
        "analysis_surface_id": ANALYSIS_SURFACE_ID,
        "input_validation": validation.to_json_object(),
        "terminal_classification": classification,
        "terminal_error": str(error),
        "explanation": explanation,
        "research_next_action": next_action,
        "claim_scope": {
            "field_calibrated": False,
            "deployment_recommendation": False,
            "physical_capacity_conclusion": False,
        },
    }
    json_path = output / "ANALYSIS_SUMMARY.json"
    md_path = output / "ANALYSIS_SUMMARY.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join([
            "# CROSS PSP custom analysis summary",
            "",
            f"Status: **{payload['status']}**",
            f"Terminal classification: **{classification}**",
            "",
            "## What happened",
            "",
            explanation,
            "",
            f"Exact runtime error: `{error}`",
            "",
            "## Next research action",
            "",
            next_action,
            "",
            "## Claim boundary",
            "",
            "This terminal is a modeled/semantic/software-domain diagnostic. It is not a field-capacity value, deployment recommendation, or proof of physical impossibility.",
            "",
        ]),
        encoding="utf-8",
    )
    return json_path, md_path, classification
