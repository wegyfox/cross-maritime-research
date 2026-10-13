"""Canonical machine-readable and HTML reporting for B0..B6 reference envelopes.

The reporting layer is deliberately above simulation, control and research-specific
execution code.  It validates the common ideal-gap decomposition through the portable
``nextgen.reference_envelope`` contract and then renders every user surface from the
same immutable semantic object.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..nextgen.reference_envelope import (
    ReferenceEnvelopeLevel,
    ReferenceEnvelopePoint,
    ReferenceEnvelopeReport,
)

_INPUT_FORMAT = "cross-ideal-gap-decomposition-v1"
_REPORT_MANIFEST_FORMAT = "vessel-crossing-reference-envelope-report-manifest-v1"
_PRIMARY_METRIC = "service_yield_per_hour"

_LEVEL_PREFIX = {
    ReferenceEnvelopeLevel.B1_DEDICATED_AUTONOMOUS_CORRIDOR: "B1_",
    ReferenceEnvelopeLevel.B2_CONTROLLED_ENTRY: "B2_",
    ReferenceEnvelopeLevel.B3_SPEED_TRAJECTORY_HARMONIZATION: "B3_",
    ReferenceEnvelopeLevel.B4_SLOT_VTS_COORDINATION: "B4_",
    ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE: "B5_",
    ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE: "B6_",
}

_LEVEL_LABELS = {
    ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK: "B0 — actual modeled network",
    ReferenceEnvelopeLevel.B1_DEDICATED_AUTONOMOUS_CORRIDOR: "B1 — dedicated corridor/exclusion",
    ReferenceEnvelopeLevel.B2_CONTROLLED_ENTRY: "B2 — controlled entry/metering",
    ReferenceEnvelopeLevel.B3_SPEED_TRAJECTORY_HARMONIZATION: "B3 — speed/plant harmonization",
    ReferenceEnvelopeLevel.B4_SLOT_VTS_COORDINATION: "B4 — global slot coordination",
    ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE: "B5 — type/order organization",
    ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE: "B6 — physical/reference envelope",
}

_PRIMARY_SOURCE_KEYS = (
    "service_yield_per_hour",
    "mean_service_yield_per_hour",
    "fixed_window_service_yield_per_hour",
)


class ReferenceEnvelopeReportingError(ValueError):
    """Raised when a decomposition cannot satisfy the canonical report contract."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON portable") from exc


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReferenceEnvelopeReportingError(f"{name} must be an object")
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceEnvelopeReportingError(f"{name} must be a finite number")
    out = float(value)
    if not isfinite(out):
        raise ReferenceEnvelopeReportingError(f"{name} must be finite")
    return out


def _primary_value(record: Mapping[str, Any], *, name: str) -> float:
    for key in _PRIMARY_SOURCE_KEYS:
        if key in record:
            return _finite(record[key], name=f"{name}.{key}")
    raise ReferenceEnvelopeReportingError(
        f"{name} must contain one of {', '.join(_PRIMARY_SOURCE_KEYS)}"
    )


def _scalar_metrics(record: Mapping[str, Any], *, name: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for key, value in record.items():
        if value is None:
            metrics[key] = None
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)):
            metrics[key] = _finite(value, name=f"{name}.{key}")
    metrics[_PRIMARY_METRIC] = _primary_value(record, name=name)
    return metrics


def _one_level_record(
    ladder: Mapping[str, Any],
    level: ReferenceEnvelopeLevel,
    *,
    scenario_id: str,
) -> tuple[str, Mapping[str, Any]]:
    prefix = _LEVEL_PREFIX[level]
    matches = [(key, value) for key, value in ladder.items() if key.startswith(prefix)]
    if len(matches) != 1:
        raise ReferenceEnvelopeReportingError(
            f"reference_ladder[{scenario_id!r}] must contain exactly one {prefix} entry"
        )
    key, value = matches[0]
    return key, _mapping(value, name=f"reference_ladder[{scenario_id!r}].{key}")


def _point_status(level: ReferenceEnvelopeLevel, record: Mapping[str, Any]) -> str:
    if level in (
        ReferenceEnvelopeLevel.B1_DEDICATED_AUTONOMOUS_CORRIDOR,
        ReferenceEnvelopeLevel.B2_CONTROLLED_ENTRY,
        ReferenceEnvelopeLevel.B3_SPEED_TRAJECTORY_HARMONIZATION,
    ) and "identity_of" in record:
        return "IDENTITY_DERIVED"
    if level is ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE:
        return "DETERMINISTIC_REFERENCE" if record.get("deterministic") is True else "REFERENCE"
    if level is ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK:
        return "ACTUAL_MODELED"
    return "MODEL_DERIVED"


def _point_note(record: Mapping[str, Any]) -> str | None:
    reason = record.get("reason")
    identity = record.get("identity_of")
    if isinstance(reason, str) and reason.strip():
        if isinstance(identity, str) and identity.strip():
            return f"identity_of={identity.strip()}; reason={reason.strip()}"
        return reason.strip()
    if isinstance(identity, str) and identity.strip():
        return f"identity_of={identity.strip()}"
    return None


def _selected_safety(record: Mapping[str, Any]) -> dict[str, Any]:
    prefixes = (
        "near_miss",
        "safety_domain_intrusion",
        "modeled_hull_overlap",
        "sampled_min_",
        "median_run_min_",
        "status_counts",
        "terminal_status",
        "strict_balance_gate_pass",
    )
    return {
        key: _json_copy(value, name=f"safety.{key}")
        for key, value in record.items()
        if key.startswith(prefixes)
    }


def _close(left: Any, right: float, *, tolerance: float = 1e-10) -> bool:
    return isinstance(left, (int, float)) and not isinstance(left, bool) and abs(float(left) - right) <= tolerance


@dataclass(frozen=True, slots=True)
class ReferenceEnvelopeScenario:
    """One scenario-conditioned B0..B6 report plus its source decomposition evidence."""

    scenario_id: str
    report: ReferenceEnvelopeReport
    efficiency: Mapping[str, Any]
    loss_decomposition: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("scenario_id must be non-empty")
        object.__setattr__(self, "scenario_id", self.scenario_id.strip())
        object.__setattr__(
            self,
            "efficiency",
            MappingProxyType(_json_copy(dict(self.efficiency), name="efficiency")),
        )
        object.__setattr__(
            self,
            "loss_decomposition",
            MappingProxyType(
                _json_copy(dict(self.loss_decomposition), name="loss_decomposition")
            ),
        )


@dataclass(frozen=True, slots=True)
class IdealGapDecomposition:
    """Validated study-level reference-envelope input used by JSON, HTML, CLI and API."""

    source: Mapping[str, Any]
    scenarios: tuple[ReferenceEnvelopeScenario, ...]

    def __post_init__(self) -> None:
        source = _json_copy(dict(self.source), name="source")
        if source.get("format") != _INPUT_FORMAT:
            raise ReferenceEnvelopeReportingError(
                f"unsupported format: {source.get('format')!r}; expected {_INPUT_FORMAT!r}"
            )
        scenarios = tuple(self.scenarios)
        if not scenarios:
            raise ReferenceEnvelopeReportingError("at least one scenario is required")
        ids = [item.scenario_id for item in scenarios]
        if len(ids) != len(set(ids)):
            raise ReferenceEnvelopeReportingError("scenario ids must be unique")
        object.__setattr__(self, "source", MappingProxyType(source))
        object.__setattr__(self, "scenarios", scenarios)

    @classmethod
    def from_json_object(cls, payload: Mapping[str, Any]) -> "IdealGapDecomposition":
        source = _json_copy(dict(payload), name="payload")
        if source.get("format") != _INPUT_FORMAT:
            raise ReferenceEnvelopeReportingError(
                f"unsupported format: {source.get('format')!r}; expected {_INPUT_FORMAT!r}"
            )

        actual = _mapping(source.get("actual"), name="actual")
        ladders = _mapping(source.get("reference_ladder"), name="reference_ladder")
        efficiencies = _mapping(source.get("efficiency"), name="efficiency")
        losses = _mapping(
            source.get("incremental_loss_decomposition"),
            name="incremental_loss_decomposition",
        )
        bottleneck = source.get("dominant_bottleneck")
        next_counterfactual = source.get("recommended_next_counterfactual")
        claim_scope = _mapping(source.get("claim_scope", {}), name="claim_scope")

        scenario_sets = [set(actual), set(ladders), set(efficiencies), set(losses)]
        if not scenario_sets[0] or any(item != scenario_sets[0] for item in scenario_sets[1:]):
            raise ReferenceEnvelopeReportingError(
                "actual, reference_ladder, efficiency and incremental_loss_decomposition "
                "must contain the same non-empty scenario set"
            )

        if isinstance(bottleneck, Mapping):
            dominant = bottleneck.get("classification")
        else:
            dominant = bottleneck
        if not isinstance(dominant, str) or not dominant.strip():
            raise ReferenceEnvelopeReportingError("dominant_bottleneck classification is required")

        if isinstance(next_counterfactual, Mapping):
            next_name = next_counterfactual.get("for_B0_like_operation")
        else:
            next_name = next_counterfactual
        if not isinstance(next_name, str) or not next_name.strip():
            raise ReferenceEnvelopeReportingError(
                "recommended_next_counterfactual.for_B0_like_operation is required"
            )

        field_calibrated = claim_scope.get("field_calibrated", False)
        deployment_recommendation = claim_scope.get("deployment_recommendation", False)
        if not isinstance(field_calibrated, bool) or not isinstance(deployment_recommendation, bool):
            raise ReferenceEnvelopeReportingError(
                "claim_scope field_calibrated/deployment_recommendation must be booleans"
            )

        scenarios: list[ReferenceEnvelopeScenario] = []
        for scenario_id in sorted(scenario_sets[0]):
            actual_record = _mapping(actual[scenario_id], name=f"actual[{scenario_id!r}]")
            ladder = _mapping(ladders[scenario_id], name=f"reference_ladder[{scenario_id!r}]")
            points: list[ReferenceEnvelopePoint] = [
                ReferenceEnvelopePoint(
                    ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK,
                    _scalar_metrics(actual_record, name=f"actual[{scenario_id!r}]"),
                    model_status="ACTUAL_MODELED",
                )
            ]
            level_records: dict[ReferenceEnvelopeLevel, Mapping[str, Any]] = {
                ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK: actual_record
            }
            for level in tuple(ReferenceEnvelopeLevel)[1:]:
                _, record = _one_level_record(ladder, level, scenario_id=scenario_id)
                level_records[level] = record
                points.append(
                    ReferenceEnvelopePoint(
                        level,
                        _scalar_metrics(
                            record,
                            name=f"reference_ladder[{scenario_id!r}].{level.value}",
                        ),
                        model_status=_point_status(level, record),
                        notes=_point_note(record),
                    )
                )

            loss = _mapping(losses[scenario_id], name=f"losses[{scenario_id!r}]")
            efficiency = _mapping(
                efficiencies[scenario_id], name=f"efficiency[{scenario_id!r}]"
            )
            safety_summary = {
                "by_level": {
                    level.value: _selected_safety(record)
                    for level, record in level_records.items()
                },
                "tradeoff": (
                    next_counterfactual.get("safety_feasibility_tradeoff")
                    if isinstance(next_counterfactual, Mapping)
                    else None
                ),
            }
            uncertainty_summary = {
                "incremental_loss_decomposition": _json_copy(
                    dict(loss), name=f"losses[{scenario_id!r}]"
                ),
                "convergence_limit": _json_copy(
                    source.get("convergence_limit", {}), name="convergence_limit"
                ),
            }
            report = ReferenceEnvelopeReport(
                tuple(points),
                primary_metric=_PRIMARY_METRIC,
                dominant_bottleneck=dominant,
                recommended_next_counterfactual=next_name,
                field_calibrated=field_calibrated,
                deployment_recommendation=deployment_recommendation,
                uncertainty_summary=uncertainty_summary,
                safety_feasibility_summary=safety_summary,
            )

            expected = (
                ("C_actual", report.actual),
                ("C_managed_current_demand", report.managed_corridor),
                ("C_balanced_saturated_reference", report.physical_upper),
            )
            for key, value in expected:
                if key in efficiency and not _close(efficiency[key], value):
                    raise ReferenceEnvelopeReportingError(
                        f"efficiency[{scenario_id!r}].{key} disagrees with reference ladder"
                    )

            scenarios.append(
                ReferenceEnvelopeScenario(
                    scenario_id=scenario_id,
                    report=report,
                    efficiency=efficiency,
                    loss_decomposition=loss,
                )
            )
        return cls(source=source, scenarios=tuple(scenarios))

    def to_json_object(self) -> dict[str, Any]:
        return _json_copy(dict(self.source), name="source")

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def result_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(item.scenario_id for item in self.scenarios)


@dataclass(frozen=True, slots=True)
class ReferenceEnvelopeArtifacts:
    output_directory: str
    diagnostics_path: str
    report_html_path: str
    manifest_path: str
    diagnostics_sha256: str
    report_html_sha256: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class ReferenceEnvelopeArtifactVerification:
    passed: bool
    errors: tuple[str, ...]


def load_ideal_gap_decomposition(path: str | Path) -> IdealGapDecomposition:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceEnvelopeReportingError(f"cannot read decomposition {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ReferenceEnvelopeReportingError("decomposition root must be an object")
    return IdealGapDecomposition.from_json_object(payload)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{100.0 * float(value):.2f}%"
    return "—"


def _pretty_key(key: str) -> str:
    return key.replace("_", " ").strip().capitalize()


def _render_mapping(mapping: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            shown = escape(json.dumps(value, ensure_ascii=False, sort_keys=True))
        elif isinstance(value, (list, tuple)):
            shown = escape(json.dumps(value, ensure_ascii=False))
        else:
            shown = escape(str(value))
        rows.append(
            f"<tr><th>{escape(_pretty_key(str(key)))}</th><td>{shown}</td></tr>"
        )
    return "<table class=\"kv\"><tbody>" + "".join(rows) + "</tbody></table>"


def render_reference_envelope_html(
    decomposition: IdealGapDecomposition,
    *,
    title: str | None = None,
) -> str:
    """Render one standalone deterministic HTML report from the canonical decomposition."""

    source = decomposition.to_json_object()
    title = title or str(source.get("scope") or "CROSS Reference / Ideal Envelope Report")
    date = source.get("date")
    scenario_headers = "".join(
        f"<th>{escape(level.value.split('_', 1)[0])}</th>" for level in ReferenceEnvelopeLevel
    )
    ladder_rows: list[str] = []
    for scenario in decomposition.scenarios:
        values = [point.metrics[_PRIMARY_METRIC] for point in scenario.report.points]
        cells = "".join(f"<td>{_fmt(float(value))}</td>" for value in values if value is not None)
        ladder_rows.append(f"<tr><th>{escape(scenario.scenario_id)}</th>{cells}</tr>")

    efficiency_rows: list[str] = []
    for scenario in decomposition.scenarios:
        report = scenario.report
        efficiency_rows.append(
            "<tr>"
            f"<th>{escape(scenario.scenario_id)}</th>"
            f"<td>{_fmt_pct(report.efficiency_vs_managed_corridor)}</td>"
            f"<td>{_fmt_pct(report.efficiency_vs_physical_upper)}</td>"
            f"<td>{_fmt(report.managed_corridor - report.actual)}</td>"
            f"<td>{_fmt(report.physical_upper - report.actual)}</td>"
            "</tr>"
        )

    detail_sections: list[str] = []
    for scenario in decomposition.scenarios:
        rows: list[str] = []
        increments = scenario.report.incremental_primary_metric_changes
        previous: ReferenceEnvelopeLevel | None = None
        for point in scenario.report.points:
            if previous is None:
                delta = None
            else:
                delta = increments[f"{previous.value}->{point.level.value}"]
            rows.append(
                "<tr>"
                f"<th>{escape(_LEVEL_LABELS[point.level])}</th>"
                f"<td>{_fmt(float(point.metrics[_PRIMARY_METRIC]))}</td>"
                f"<td>{_fmt(delta) if delta is not None else '—'}</td>"
                f"<td>{escape(point.model_status)}</td>"
                f"<td>{escape(point.notes or '')}</td>"
                "</tr>"
            )
            previous = point.level
        detail_sections.append(
            f"<section><h3>{escape(scenario.scenario_id)}</h3>"
            "<table><thead><tr><th>Level</th><th>Service yield / hour</th>"
            "<th>Increment</th><th>Status</th><th>Notes</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            "<details><summary>Uncertainty / conditional decomposition</summary>"
            f"{_render_mapping(scenario.loss_decomposition)}</details>"
            "<details><summary>Safety / feasibility evidence</summary>"
            f"{_render_mapping(scenario.report.safety_feasibility_summary)}</details>"
            "</section>"
        )

    bottleneck = _mapping(source.get("dominant_bottleneck"), name="dominant_bottleneck")
    intervention = _mapping(
        source.get("recommended_next_counterfactual"),
        name="recommended_next_counterfactual",
    )
    claims = _mapping(source.get("claim_scope", {}), name="claim_scope")
    claim_items = "".join(
        f"<li><code>{escape(str(key))}</code>: <strong>{escape(str(value))}</strong></li>"
        for key, value in claims.items()
    )
    extra_sections: list[str] = []
    for key in ("ordering_theory", "convergence_limit"):
        value = source.get(key)
        if isinstance(value, Mapping) and value:
            extra_sections.append(
                f"<section><h2>{escape(_pretty_key(key))}</h2>{_render_mapping(value)}</section>"
            )
    if "B6v1_disposition" in source:
        extra_sections.append(
            "<section><h2>Superseded / diagnostic reference</h2>"
            f"<p>{escape(str(source['B6v1_disposition']))}</p></section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{ font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; line-height: 1.45; }}
body {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px; }}
h1 {{ margin-bottom: 0.2rem; }}
h2 {{ margin-top: 2.2rem; border-bottom: 1px solid #bbb; padding-bottom: 0.35rem; }}
h3 {{ margin-top: 1.7rem; }}
.meta {{ opacity: 0.75; margin-top: 0; }}
table {{ width: 100%; border-collapse: collapse; margin: 0.8rem 0 1.3rem; font-variant-numeric: tabular-nums; }}
th, td {{ border: 1px solid #bbb; padding: 0.45rem 0.55rem; text-align: right; vertical-align: top; }}
th:first-child, td:first-child, .kv th, .kv td {{ text-align: left; }}
.kv th {{ width: 32%; }}
code {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
.notice {{ border-left: 4px solid #777; padding: 0.65rem 0.9rem; background: rgba(127,127,127,0.08); }}
details {{ margin: 0.7rem 0; }}
summary {{ cursor: pointer; font-weight: 600; }}
</style>
</head>
<body>
<header>
<h1>{escape(title)}</h1>
<p class="meta">Date: {escape(str(date or 'unspecified'))} · decomposition SHA-256: <code>{decomposition.result_sha256}</code></p>
</header>
<section>
<h2>Actual → managed → reference ladder</h2>
<p>Primary metric: <code>{_PRIMARY_METRIC}</code>. Identity and negative increments are preserved.</p>
<table><thead><tr><th>Scenario</th>{scenario_headers}</tr></thead><tbody>{''.join(ladder_rows)}</tbody></table>
</section>
<section>
<h2>Efficiency and gap</h2>
<table><thead><tr><th>Scenario</th><th>Actual / managed B5</th><th>Actual / B6</th><th>Recoverable gap to B5</th><th>Residual gap to B6</th></tr></thead><tbody>{''.join(efficiency_rows)}</tbody></table>
<p class="notice">The B5→B6 difference is a reference/saturation residual unless the decomposition explicitly establishes a paired causal effect.</p>
</section>
<section>
<h2>Dominant modeled bottleneck</h2>
{_render_mapping(bottleneck)}
</section>
<section>
<h2>Largest modeled next intervention</h2>
{_render_mapping(intervention)}
</section>
<section>
<h2>Scenario evidence</h2>
{''.join(detail_sections)}
</section>
{''.join(extra_sections)}
<section>
<h2>Claim scope</h2>
<ul>{claim_items}</ul>
</section>
</body>
</html>
"""


def write_reference_envelope_artifacts(
    decomposition: IdealGapDecomposition,
    output_directory: str | Path,
    *,
    title: str | None = None,
    source_file_sha256: str | None = None,
) -> ReferenceEnvelopeArtifacts:
    output_directory = Path(output_directory)
    diagnostics_directory = output_directory / "diagnostics"
    diagnostics_directory.mkdir(parents=True, exist_ok=True)

    diagnostics_path = diagnostics_directory / "IDEAL_GAP_DECOMPOSITION.json"
    report_html_path = output_directory / "REPORT.html"
    manifest_path = output_directory / "REPORT_MANIFEST.json"

    diagnostics_bytes = decomposition.to_json_bytes()
    report_bytes = render_reference_envelope_html(decomposition, title=title).encode("utf-8")
    diagnostics_path.write_bytes(diagnostics_bytes)
    report_html_path.write_bytes(report_bytes)

    diagnostics_digest = sha256(diagnostics_bytes).hexdigest()
    report_digest = sha256(report_bytes).hexdigest()
    if source_file_sha256 is not None:
        if (
            not isinstance(source_file_sha256, str)
            or len(source_file_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in source_file_sha256)
        ):
            raise ValueError("source_file_sha256 must be a lowercase SHA-256 hex digest")

    manifest = {
        "format": _REPORT_MANIFEST_FORMAT,
        "source_decomposition_canonical_sha256": decomposition.result_sha256,
        "source_file_sha256": source_file_sha256,
        "files": {
            "diagnostics/IDEAL_GAP_DECOMPOSITION.json": {
                "sha256": diagnostics_digest,
                "size_bytes": len(diagnostics_bytes),
            },
            "REPORT.html": {
                "sha256": report_digest,
                "size_bytes": len(report_bytes),
            },
        },
        "scenario_ids": list(decomposition.scenario_ids),
    }
    manifest_bytes = _canonical_json(manifest)
    manifest_path.write_bytes(manifest_bytes)
    manifest_digest = sha256(manifest_bytes).hexdigest()

    return ReferenceEnvelopeArtifacts(
        output_directory=str(output_directory),
        diagnostics_path=str(diagnostics_path),
        report_html_path=str(report_html_path),
        manifest_path=str(manifest_path),
        diagnostics_sha256=diagnostics_digest,
        report_html_sha256=report_digest,
        manifest_sha256=manifest_digest,
    )


def verify_reference_envelope_artifacts(
    output_directory: str | Path,
) -> ReferenceEnvelopeArtifactVerification:
    output_directory = Path(output_directory)
    manifest_path = output_directory / "REPORT_MANIFEST.json"
    if not manifest_path.exists():
        return ReferenceEnvelopeArtifactVerification(False, ("REPORT_MANIFEST.json is missing",))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ReferenceEnvelopeArtifactVerification(False, (f"invalid REPORT_MANIFEST.json: {exc}",))
    errors: list[str] = []
    if manifest.get("format") != _REPORT_MANIFEST_FORMAT:
        errors.append("unsupported report manifest format")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        return ReferenceEnvelopeArtifactVerification(False, tuple(errors + ["manifest files object is missing"]))
    for relative_name, evidence in files.items():
        path = output_directory / relative_name
        if not path.exists():
            errors.append(f"missing report artifact: {relative_name}")
            continue
        if not isinstance(evidence, Mapping):
            errors.append(f"invalid manifest evidence: {relative_name}")
            continue
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != evidence.get("sha256"):
            errors.append(f"SHA-256 mismatch: {relative_name}")
        if len(raw) != evidence.get("size_bytes"):
            errors.append(f"size mismatch: {relative_name}")
    return ReferenceEnvelopeArtifactVerification(not errors, tuple(errors))
