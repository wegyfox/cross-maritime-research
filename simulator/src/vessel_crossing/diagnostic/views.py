"""Compact operator/decision projection from the canonical D0 decomposition.

This module is intentionally presentation-only.  It accepts the already validated
``IdealGapDecomposition`` and does not classify bottlenecks, select interventions, or
recompute scientific effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from typing import Any, Mapping

from ..reporting.reference_envelope import IdealGapDecomposition

_FORMAT = "cross-diagnostic-decision-view-v1"
_SCHEMA_VERSION = 1


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


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def build_diagnostic_decision_view(
    decomposition: IdealGapDecomposition,
) -> dict[str, Any]:
    """Project canonical report semantics into a compact decision-facing object.

    The only derived numerical values are direct arithmetic gaps between already
    validated B0/B5/B6 primary-metric points.  No new intervention ranking or causal
    inference is performed here.
    """

    if not isinstance(decomposition, IdealGapDecomposition):
        raise TypeError("decomposition must be IdealGapDecomposition")

    source = decomposition.to_json_object()
    bottleneck = _mapping(source.get("dominant_bottleneck"), name="dominant_bottleneck")
    counterfactual = _mapping(
        source.get("recommended_next_counterfactual"),
        name="recommended_next_counterfactual",
    )
    claim_scope = _mapping(source.get("claim_scope", {}), name="claim_scope")

    scenarios: list[dict[str, Any]] = []
    for scenario in decomposition.scenarios:
        report = scenario.report
        scenarios.append(
            {
                "scenario_id": scenario.scenario_id,
                "primary_metric": report.primary_metric,
                "current_modeled_value": report.actual,
                "managed_modeled_value": report.managed_corridor,
                "physical_reference_value": report.physical_upper,
                "managed_recoverable_gap": report.managed_corridor - report.actual,
                "residual_reference_gap": report.physical_upper - report.managed_corridor,
                "actual_to_reference_gap": report.physical_upper - report.actual,
                "efficiency_vs_managed": report.efficiency_vs_managed_corridor,
                "efficiency_vs_reference": report.efficiency_vs_physical_upper,
                "incremental_modeled_changes": dict(
                    report.incremental_primary_metric_changes
                ),
                "uncertainty_and_conditional_evidence": _json_copy(
                    dict(scenario.loss_decomposition)
                ),
                "safety_and_feasibility_evidence": _json_copy(
                    dict(report.safety_feasibility_summary)
                ),
            }
        )

    return {
        "format": _FORMAT,
        "schema_version": _SCHEMA_VERSION,
        "source": {
            "decomposition_format": source.get("format"),
            "decomposition_sha256": decomposition.result_sha256,
            "canonical_diagnostics_path": "diagnostics/IDEAL_GAP_DECOMPOSITION.json",
            "researcher_report_path": "REPORT.html",
        },
        "report_metadata": {
            "date": source.get("date"),
            "scope": source.get("scope"),
        },
        "dominant_modeled_bottleneck": _json_copy(dict(bottleneck)),
        "canonical_next_counterfactual": _json_copy(dict(counterfactual)),
        "claim_scope": _json_copy(dict(claim_scope)),
        "scenarios": scenarios,
    }


def render_diagnostic_decision_view_html(
    decomposition: IdealGapDecomposition,
    *,
    title: str | None = None,
) -> str:
    """Render a compact HTML view from the same canonical decision projection."""

    payload = build_diagnostic_decision_view(decomposition)
    title = title or "CROSS Diagnostic Decision View"
    bottleneck = payload["dominant_modeled_bottleneck"]
    counterfactual = payload["canonical_next_counterfactual"]
    claims = payload["claim_scope"]
    metadata = payload["report_metadata"]

    rows: list[str] = []
    for scenario in payload["scenarios"]:
        rows.append(
            "<tr>"
            f"<th>{escape(str(scenario['scenario_id']))}</th>"
            f"<td>{float(scenario['current_modeled_value']):.4f}</td>"
            f"<td>{float(scenario['managed_modeled_value']):.4f}</td>"
            f"<td>{float(scenario['managed_recoverable_gap']):.4f}</td>"
            f"<td>{float(scenario['physical_reference_value']):.4f}</td>"
            f"<td>{float(scenario['residual_reference_gap']):.4f}</td>"
            "</tr>"
        )

    claim_rows = "".join(
        "<tr>"
        f"<th>{escape(str(key))}</th>"
        f"<td>{escape(str(value))}</td>"
        "</tr>"
        for key, value in sorted(claims.items())
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{ font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; line-height: 1.45; }}
body {{ max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0 1.6rem; }}
th, td {{ border: 1px solid #bbb; padding: 0.5rem 0.6rem; text-align: right; }}
th:first-child, td:first-child, .kv th, .kv td {{ text-align: left; }}
.kv th {{ width: 34%; }}
.notice {{
  border-left: 4px solid #777; padding: 0.7rem 0.9rem;
  background: rgba(127,127,127,.08);
}}
code {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
</style>
</head>
<body>
<h1>{escape(title)}</h1>
<p>Date: {escape(str(metadata.get('date') or 'unspecified'))}<br>
Scope: {escape(str(metadata.get('scope') or 'unspecified'))}</p>
<p>Source decomposition SHA-256: <code>{escape(decomposition.result_sha256)}</code></p>
<p class="notice">This surface is a projection of the canonical modeled evidence.
It does not add a new bottleneck classification, intervention ranking, field-calibration
claim, or deployment recommendation.</p>
<h2>Modeled throughput and gaps</h2>
<table>
<thead><tr><th>Scenario</th><th>B0 actual</th><th>B5 managed</th>
<th>Gap to B5</th><th>B6 reference</th><th>B5→B6 residual</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
<h2>Dominant modeled bottleneck</h2>
<table class="kv"><tbody>
<tr><th>Classification</th><td>{escape(str(bottleneck.get('classification', '')))}</td></tr>
<tr><th>Basis</th><td>{escape(str(bottleneck.get('basis', '')))}</td></tr>
<tr><th>Caveat</th><td>{escape(str(bottleneck.get('caveat', '')))}</td></tr>
</tbody></table>
<h2>Canonical next counterfactual</h2>
<table class="kv"><tbody>
<tr><th>Modeled intervention</th>
<td>{escape(str(counterfactual.get('for_B0_like_operation', '')))}</td></tr>
<tr><th>Calibration</th><td>{escape(str(counterfactual.get('calibration', '')))}</td></tr>
<tr><th>Deployment recommendation</th>
<td>{escape(str(counterfactual.get('deployment_recommendation', '')))}</td></tr>
<tr><th>Safety / feasibility</th>
<td>{escape(str(counterfactual.get('safety_feasibility_tradeoff', '')))}</td></tr>
</tbody></table>
<h2>Claim scope</h2>
<table class="kv"><tbody>{claim_rows}</tbody></table>
<p>Full uncertainty, conditional decomposition, and safety evidence remain in
<code>REPORT.html</code> and <code>diagnostics/IDEAL_GAP_DECOMPOSITION.json</code>.</p>
</body>
</html>
"""


@dataclass(frozen=True, slots=True)
class DiagnosticDecisionViewArtifacts:
    json_path: str
    html_path: str
    json_sha256: str
    html_sha256: str


def write_diagnostic_decision_view_artifacts(
    decomposition: IdealGapDecomposition,
    output_directory: str | Path,
    *,
    title: str | None = None,
) -> DiagnosticDecisionViewArtifacts:
    """Write deterministic JSON/HTML decision surfaces beside the canonical report."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "DECISION_VIEW.json"
    html_path = output_directory / "DECISION_VIEW.html"

    json_bytes = _canonical_json(build_diagnostic_decision_view(decomposition))
    html_bytes = render_diagnostic_decision_view_html(
        decomposition,
        title=title,
    ).encode("utf-8")
    json_path.write_bytes(json_bytes)
    html_path.write_bytes(html_bytes)

    return DiagnosticDecisionViewArtifacts(
        json_path=str(json_path),
        html_path=str(html_path),
        json_sha256=sha256(json_bytes).hexdigest(),
        html_sha256=sha256(html_bytes).hexdigest(),
    )
