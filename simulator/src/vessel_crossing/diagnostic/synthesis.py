"""Generic synthesis from D0 evidence/effects into the existing report contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..nextgen.reference_envelope import ReferenceEnvelopeLevel
from .classifier import DiagnosticBottleneckClassification
from .effects import DiagnosticEffectEstimate
from .evidence import DiagnosticStudyEvidence


@dataclass(frozen=True, slots=True)
class DiagnosticSynthesisProfile:
    date: str
    scope: str
    ladder_keys: Mapping[str, str]
    intervention_display_names: Mapping[str, str]
    bottleneck_basis: str
    bottleneck_caveat: str
    safety_feasibility_tradeoff: str
    no_additional_recovery: str
    calibration_label: str
    b5_activation_note: str
    residual_interpretation: str
    ordering_theory: Mapping[str, Any] = field(default_factory=dict)
    b6v1_disposition: str | None = None
    claim_scope_extensions: Mapping[str, bool] = field(default_factory=dict)
    convergence_note: str | None = None
    required_b5_event: str | None = None
    required_b5_event_count: int | None = None
    require_common_reference_yield: bool = False
    require_common_reference_headway: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "ladder_keys", MappingProxyType(dict(self.ladder_keys)))
        object.__setattr__(
            self,
            "intervention_display_names",
            MappingProxyType(dict(self.intervention_display_names)),
        )
        object.__setattr__(self, "ordering_theory", MappingProxyType(dict(self.ordering_theory)))
        object.__setattr__(
            self,
            "claim_scope_extensions",
            MappingProxyType(dict(self.claim_scope_extensions)),
        )


def r2_frozen_acceptance_synthesis_profile() -> DiagnosticSynthesisProfile:
    """Static wording/claim-limit profile used only for frozen R2 compatibility acceptance."""

    return DiagnosticSynthesisProfile(
        date="2026-09-30",
        scope="R2 full-network reference-envelope synthesis",
        ladder_keys={
            ReferenceEnvelopeLevel.B1_DEDICATED_AUTONOMOUS_CORRIDOR.value: "B1_dedicated_corridor",
            ReferenceEnvelopeLevel.B2_CONTROLLED_ENTRY.value: "B2_controlled_entry",
            ReferenceEnvelopeLevel.B3_SPEED_TRAJECTORY_HARMONIZATION.value: "B3_speed_harmonization",
            ReferenceEnvelopeLevel.B4_SLOT_VTS_COORDINATION.value: "B4_coordination",
            ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE.value: "B5_platoon_structure",
            ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE.value: "B6_balanced_saturated_reference",
        },
        intervention_display_names={
            "global_slot_coordination": "B4 rolling physical center-slot coordination",
        },
        bottleneck_basis=(
            "B3->B4 is the only positive qualified implementable stochastic step; "
            "paired CI excludes zero in all three Delta arms"
        ),
        bottleneck_caveat=(
            "classification is conditional on the frozen ordered ladder and currently modeled factors"
        ),
        safety_feasibility_tradeoff=(
            "modeled infeasibility falls strongly, but minimum safety-domain margin "
            "compresses and near-miss burden increases"
        ),
        no_additional_recovery=(
            "NO_ADDITIONAL_POSITIVE_B1_B5_RECOVERY_IDENTIFIED_AT_TESTED_6_VESSELS_PER_HOUR"
        ),
        calibration_label="MODEL_DERIVED_NOT_FIELD_CALIBRATED",
        b5_activation_note=(
            "near-zero activation at tested stochastic demand; one swap total across all arms, in POS050"
        ),
        residual_interpretation=(
            "bundled residual to deterministic balanced saturation; not a paired causal effect "
            "and not attributable to one mechanism"
        ),
        ordering_theory={
            "penetration_alone_sufficient": False,
            "B5_stochastic_result": (
                "NULL_PRIMARY_GAIN_AT_TESTED_LOAD_DUE_TO_NEAR_ZERO_STAGING_ACTIVATION"
            ),
            "B6v2_result": (
                "NEG050 exercises bounded ordering actions but all Delta arms share the same "
                "balanced saturated service yield"
            ),
            "interpretation": (
                "under balanced saturation the cross-stream physical/safety separation is "
                "binding before pair-order differences can change throughput"
            ),
        },
        b6v1_disposition=(
            "RETAIN_AS_UNCONSTRAINED_STREAM_STARVATION_DIAGNOSTIC_NOT_BALANCED_FULL_NETWORK_ENVELOPE"
        ),
        claim_scope_extensions={
            "exact_Guo_section3_replication": False,
            "empirical_AS_HS_calibration": False,
            "ordered_ladder_is_order_independent_causal_decomposition": False,
            "B6v2_is_asymptotic_capacity_estimate": False,
        },
        convergence_note=(
            "B6v2 is a deterministic finite-window reference point; no horizon-convergence campaign was run."
        ),
        required_b5_event="B5_ADJACENT_SWAP",
        required_b5_event_count=1,
        require_common_reference_yield=True,
        require_common_reference_headway=True,
    )


def _effect_to(
    effects: Sequence[DiagnosticEffectEstimate], level: ReferenceEnvelopeLevel
) -> DiagnosticEffectEstimate:
    for effect in effects:
        if effect.to_level is level:
            return effect
    raise ValueError(f"missing effect ending at {level.value}")


def _primary(study: DiagnosticStudyEvidence, level: ReferenceEnvelopeLevel) -> float:
    return study.level(level).primary_kpi


_GENERIC_LADDER_KEYS: Mapping[ReferenceEnvelopeLevel, str] = MappingProxyType(
    {
        ReferenceEnvelopeLevel.B1_DEDICATED_AUTONOMOUS_CORRIDOR: "B1_dedicated_corridor",
        ReferenceEnvelopeLevel.B2_CONTROLLED_ENTRY: "B2_controlled_entry",
        ReferenceEnvelopeLevel.B3_SPEED_TRAJECTORY_HARMONIZATION: "B3_speed_harmonization",
        ReferenceEnvelopeLevel.B4_SLOT_VTS_COORDINATION: "B4_coordination",
        ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE: "B5_type_order_structure",
        ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE: "B6_physical_reference",
    }
)
_PRIMARY_METRIC_KEYS = (
    "service_yield_per_hour",
    "mean_service_yield_per_hour",
    "fixed_window_service_yield_per_hour",
)


def _generic_level_record(
    study: DiagnosticStudyEvidence, level: ReferenceEnvelopeLevel
) -> dict[str, Any]:
    evidence = study.level(level)
    record = dict(evidence.aggregate_metrics)
    if not any(key in record for key in _PRIMARY_METRIC_KEYS):
        record["service_yield_per_hour"] = evidence.primary_kpi
    if evidence.identity_of is not None:
        record["identity_of"] = evidence.identity_of.value
    if evidence.reason is not None:
        record["reason"] = evidence.reason
    if evidence.intervention_id is not None:
        record["intervention_id"] = evidence.intervention_id
    if evidence.target_factor is not None:
        record["target_factor"] = evidence.target_factor
    record["causal"] = evidence.causal
    return record


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if abs(denominator) <= 1e-12 else numerator / denominator


def _generic_loss_key(effect: DiagnosticEffectEstimate) -> str:
    left = effect.from_level.value.split("_", 1)[0]
    right = effect.to_level.value.split("_", 1)[0]
    suffix = effect.intervention_id or effect.target_factor or "transition"
    return f"{left}_to_{right}_{suffix}"


def _validate_generic_effects(
    study: DiagnosticStudyEvidence,
    effects: Sequence[DiagnosticEffectEstimate],
) -> tuple[DiagnosticEffectEstimate, ...]:
    rows = tuple(effects)
    expected = tuple(zip(study.levels, study.levels[1:]))
    if len(rows) != len(expected):
        raise ValueError("effects must cover all six adjacent B0-B6 transitions")
    for effect, (previous, current) in zip(rows, expected):
        if effect.from_level is not previous.level or effect.to_level is not current.level:
            raise ValueError("effects must follow canonical adjacent B0-B6 order")
        if effect.intervention_id != current.intervention_id:
            raise ValueError("effect intervention_id disagrees with destination evidence")
        if effect.target_factor != current.target_factor:
            raise ValueError("effect target_factor disagrees with destination evidence")
        if effect.causal is not current.causal:
            raise ValueError("effect causal flag disagrees with destination evidence")
    return rows


def synthesize_generic_ideal_gap_decomposition(
    study: DiagnosticStudyEvidence,
    effects: Sequence[DiagnosticEffectEstimate],
    classification: DiagnosticBottleneckClassification,
    *,
    date: str,
    scope: str,
    intervention_display_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Project one generic B0-B6 diagnostic study into the shared report contract.

    This path deliberately carries only evidence-derived values and generic claim-limit
    wording.  It does not reuse the frozen R2 compatibility profile or assume that any
    particular ladder step is identity, causal, positive, or headway-backed.
    """

    if not isinstance(study, DiagnosticStudyEvidence):
        raise TypeError("study must be DiagnosticStudyEvidence")
    if not isinstance(classification, DiagnosticBottleneckClassification):
        raise TypeError("classification must be DiagnosticBottleneckClassification")
    if not isinstance(date, str) or not date.strip():
        raise ValueError("date must be a non-empty string")
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("scope must be a non-empty string")
    display_names = dict(intervention_display_names or {})
    if any(not isinstance(key, str) or not key.strip() for key in display_names):
        raise ValueError("intervention display-name keys must be non-empty strings")
    if any(not isinstance(value, str) or not value.strip() for value in display_names.values()):
        raise ValueError("intervention display names must be non-empty strings")

    effect_rows = _validate_generic_effects(study, effects)
    scenario = study.scenario_id
    b0 = study.level(ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK)
    b5 = study.level(ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE)
    b6 = study.level(ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE)
    c0, c5, c6 = b0.primary_kpi, b5.primary_kpi, b6.primary_kpi

    actual_record = dict(b0.aggregate_metrics)
    if not any(key in actual_record for key in _PRIMARY_METRIC_KEYS):
        actual_record["service_yield_per_hour"] = c0

    ladder = {
        _GENERIC_LADDER_KEYS[level]: _generic_level_record(study, level)
        for level in tuple(ReferenceEnvelopeLevel)[1:]
    }

    loss_records: dict[str, Any] = {}
    for effect in effect_rows:
        record: dict[str, Any] = {
            "from_level": effect.from_level.value,
            "to_level": effect.to_level.value,
            "intervention_id": effect.intervention_id,
            "target_factor": effect.target_factor,
            "causal": effect.causal,
            "paired": effect.paired,
            "mean_effect_service_yield_per_hour": effect.mean_effect,
            "qualified_recoverable": effect.qualified_recoverable,
        }
        if effect.ci_low is not None or effect.ci_high is not None:
            record["confidence_interval_service_yield_per_hour"] = [
                effect.ci_low,
                effect.ci_high,
            ]
        if (
            effect.positive_count is not None
            or effect.zero_count is not None
            or effect.negative_count is not None
        ):
            record["positive_zero_negative_blocks"] = [
                effect.positive_count,
                effect.zero_count,
                effect.negative_count,
            ]
        if effect.replication_count is not None:
            record["replication_count"] = effect.replication_count
        loss_records[_generic_loss_key(effect)] = record

    selected_effect = None
    if classification.intervention_id is not None:
        selected_effect = next(
            (
                effect
                for effect in effect_rows
                if effect.intervention_id == classification.intervention_id
                and effect.target_factor == classification.target_factor
            ),
            None,
        )
        if selected_effect is None:
            raise ValueError("classified intervention is not present in adjacent effect evidence")
        if not selected_effect.qualified_recoverable:
            raise ValueError(
                "classified intervention is not supported by qualified effect evidence"
            )

    if classification.intervention_id is None:
        next_name = "NO_QUALIFIED_RECOVERABLE_INTERVENTION"
        basis = (
            "No modeled causal adjacent ladder step had a strictly positive confidence-"
            "qualified recoverable effect."
        )
    else:
        next_name = display_names.get(
            classification.intervention_id,
            classification.intervention_id,
        )
        basis = (
            "Selected from modeled causal adjacent ladder effects whose confidence evidence "
            "qualified as recoverable."
        )

    claim_scope = dict(study.claim_scope)
    field_calibrated = bool(claim_scope.get("field_calibrated", False))
    deployment_recommendation = bool(
        claim_scope.get("deployment_recommendation", False)
    )
    claim_scope.setdefault("field_calibrated", field_calibrated)
    claim_scope.setdefault("deployment_recommendation", deployment_recommendation)

    efficiency = {
        "C_actual": c0,
        "C_managed_current_demand": c5,
        "C_balanced_saturated_reference": c6,
        "eta_managed": _safe_ratio(c0, c5),
        "eta_reference": _safe_ratio(c0, c6),
        "managed_recoverable_gap_abs": c5 - c0,
        "managed_recoverable_gain_pct_of_actual": (
            None if abs(c0) <= 1e-12 else 100.0 * (c5 - c0) / c0
        ),
        "actual_to_reference_gap_abs": c6 - c0,
        "actual_to_reference_gap_pct_of_reference": (
            None if abs(c6) <= 1e-12 else 100.0 * (c6 - c0) / c6
        ),
        "managed_level_fraction_of_reference": _safe_ratio(c5, c6),
    }

    expected_effect: dict[str, float] = {}
    if selected_effect is not None:
        expected_effect[scenario] = selected_effect.mean_effect

    result: dict[str, Any] = {
        "format": "cross-ideal-gap-decomposition-v1",
        "date": date.strip(),
        "scope": scope.strip(),
        "actual": {scenario: actual_record},
        "reference_ladder": {scenario: ladder},
        "efficiency": {scenario: efficiency},
        "incremental_loss_decomposition": {scenario: loss_records},
        "dominant_bottleneck": {
            "classification": classification.classification,
            "basis": basis,
            "caveat": (
                "Classification is conditional on the configured B0-B6 ladder, modeled "
                "factors, executed counterfactuals, and declared claim scope."
            ),
        },
        "recommended_next_counterfactual": {
            "for_B0_like_operation": next_name,
            "expected_effect_service_yield_per_hour_by_scenario": expected_effect,
            "safety_feasibility_tradeoff": (
                "Use the modeled safety/feasibility evidence attached to the selected ladder "
                "step; no unmodeled deployment tradeoff is inferred."
            ),
            "calibration": (
                "FIELD_CALIBRATED" if field_calibrated else "MODEL_DERIVED_NOT_FIELD_CALIBRATED"
            ),
            "deployment_recommendation": deployment_recommendation,
        },
        "claim_scope": claim_scope,
    }

    headway = b6.aggregate_metrics.get("mean_center_headway_min")
    if isinstance(headway, (int, float)) and not isinstance(headway, bool) and headway > 0.0:
        result["convergence_limit"] = {
            "reference_yield_per_hour": c6,
            "observed_center_headway_min": float(headway),
            "headway_implied_long_run_rate_per_hour": 60.0 / float(headway),
            "note": (
                "Headway-derived rate is reported only because the registered B6 evidence "
                "supplied it."
            ),
        }
    return result


def synthesize_ideal_gap_decomposition(
    studies: Sequence[DiagnosticStudyEvidence],
    effects_by_scenario: Mapping[str, Sequence[DiagnosticEffectEstimate]],
    classifications_by_scenario: Mapping[str, DiagnosticBottleneckClassification],
    *,
    profile: DiagnosticSynthesisProfile,
    b6v1_stream_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    rows = tuple(studies)
    if not rows:
        raise ValueError("at least one study is required")
    scenario_ids = tuple(item.scenario_id for item in rows)
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("study scenario ids must be unique")
    if set(effects_by_scenario) != set(scenario_ids):
        raise ValueError("effects_by_scenario must match studies")
    if set(classifications_by_scenario) != set(scenario_ids):
        raise ValueError("classifications_by_scenario must match studies")

    classifications = {classifications_by_scenario[key].classification for key in scenario_ids}
    interventions = {classifications_by_scenario[key].intervention_id for key in scenario_ids}
    if len(classifications) != 1 or len(interventions) != 1:
        raise ValueError("study-level report requires one common qualified bottleneck/intervention")
    classification = next(iter(classifications))
    selected_intervention = next(iter(interventions))

    if profile.required_b5_event is not None and profile.required_b5_event_count is not None:
        observed = 0
        for study in rows:
            counts = study.level(
                ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE
            ).aggregate_metrics.get("controller_event_counts", {})
            if isinstance(counts, Mapping):
                observed += int(counts.get(profile.required_b5_event, 0))
        if observed != profile.required_b5_event_count:
            raise ValueError(
                f"required B5 event count mismatch: {observed} != {profile.required_b5_event_count}"
            )

    b6_yields = [
        _primary(study, ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE) for study in rows
    ]
    if profile.require_common_reference_yield and max(b6_yields) - min(b6_yields) > 1e-12:
        raise ValueError("reference yields are not common across scenarios")
    b6_headways = [
        float(
            study.level(ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE).aggregate_metrics[
                "mean_center_headway_min"
            ]
        )
        for study in rows
    ]
    if profile.require_common_reference_headway and max(b6_headways) - min(b6_headways) > 1e-12:
        raise ValueError("reference headways are not common across scenarios")
    if profile.b6v1_disposition is not None and b6v1_stream_counts is not None:
        if any(len(b6v1_stream_counts.get(scenario, {})) != 1 for scenario in scenario_ids):
            raise ValueError("B6v1 supporting evidence does not match the declared disposition")

    actual: dict[str, Any] = {}
    ladder: dict[str, Any] = {}
    efficiency: dict[str, Any] = {}
    losses: dict[str, Any] = {}
    expected_selected_effect: dict[str, float] = {}

    for study in rows:
        scenario = study.scenario_id
        effects = tuple(effects_by_scenario[scenario])
        b0 = study.level(ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK)
        b1 = study.level(ReferenceEnvelopeLevel.B1_DEDICATED_AUTONOMOUS_CORRIDOR)
        b2 = study.level(ReferenceEnvelopeLevel.B2_CONTROLLED_ENTRY)
        b3 = study.level(ReferenceEnvelopeLevel.B3_SPEED_TRAJECTORY_HARMONIZATION)
        b4 = study.level(ReferenceEnvelopeLevel.B4_SLOT_VTS_COORDINATION)
        b5 = study.level(ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE)
        b6 = study.level(ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE)
        c0, c4, c5, c6 = b0.primary_kpi, b4.primary_kpi, b5.primary_kpi, b6.primary_kpi
        b4_effect = _effect_to(effects, ReferenceEnvelopeLevel.B4_SLOT_VTS_COORDINATION)
        b5_effect = _effect_to(effects, ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE)
        if b4_effect.ci_low is None or b4_effect.ci_high is None:
            raise ValueError("B4 paired confidence interval is required by this synthesis profile")
        if b4_effect.positive_count is None or b4_effect.zero_count is None or b4_effect.negative_count is None:
            raise ValueError("B4 paired sign evidence is required")

        actual[scenario] = dict(b0.aggregate_metrics)
        level_records: dict[str, Any] = {}
        for item in (b1, b2, b3):
            key = profile.ladder_keys[item.level.value]
            level_records[key] = {
                "identity_of": item.identity_of.value.split("_")[0] if item.identity_of else None,
                "reason": item.reason,
                "service_yield_per_hour": item.primary_kpi,
            }
        level_records[profile.ladder_keys[b4.level.value]] = dict(b4.aggregate_metrics)
        level_records[profile.ladder_keys[b5.level.value]] = dict(b5.aggregate_metrics)
        level_records[profile.ladder_keys[b6.level.value]] = dict(b6.aggregate_metrics)
        ladder[scenario] = level_records

        efficiency[scenario] = {
            "C_actual": c0,
            "C_managed_current_demand": c5,
            "C_balanced_saturated_reference": c6,
            "eta_managed": c0 / c5,
            "eta_reference": c0 / c6,
            "managed_recoverable_gap_abs": c5 - c0,
            "managed_recoverable_gain_pct_of_actual": 100.0 * (c5 - c0) / c0,
            "actual_to_reference_gap_abs": c6 - c0,
            "actual_to_reference_gap_pct_of_reference": 100.0 * (c6 - c0) / c6,
            "managed_level_fraction_of_reference": c5 / c6,
        }
        losses[scenario] = {
            "B0_to_B1_external_corridor_exclusion": 0.0,
            "B1_to_B2_admission_organization": 0.0,
            "B2_to_B3_speed_plant_harmonization": 0.0,
            "B3_to_B4_global_slot_coordination": {
                # The report contract historically used the difference of aggregate means
                # while the paired CI is centered on the paired-replication mean.
                "mean_paired_effect_service_yield_per_hour": c4 - c0,
                "paired_ci95": [b4_effect.ci_low, b4_effect.ci_high],
                "positive_zero_negative_blocks": [
                    b4_effect.positive_count,
                    b4_effect.zero_count,
                    b4_effect.negative_count,
                ],
            },
            "B4_to_B5_type_order_organization": {
                "mean_effect_service_yield_per_hour": c5 - c4,
                "activation_note": profile.b5_activation_note,
            },
            "B5_to_B6_residual_to_balanced_saturation": {
                "difference_service_yield_per_hour": c6 - c5,
                "interpretation": profile.residual_interpretation,
            },
        }
        if selected_intervention == b4.intervention_id:
            expected_selected_effect[scenario] = b4_effect.mean_effect
        elif selected_intervention == b5.intervention_id:
            expected_selected_effect[scenario] = b5_effect.mean_effect
        elif selected_intervention is not None:
            match = next(
                (effect for effect in effects if effect.intervention_id == selected_intervention),
                None,
            )
            if match is None:
                raise ValueError("selected intervention is not present in effect evidence")
            expected_selected_effect[scenario] = match.mean_effect

    first = rows[0]
    claim_scope = dict(first.claim_scope)
    for study in rows[1:]:
        if dict(study.claim_scope) != dict(first.claim_scope):
            raise ValueError("claim scope must be common across scenarios")
    claim_scope.update(profile.claim_scope_extensions)

    if selected_intervention is None:
        next_name = "NO_QUALIFIED_RECOVERABLE_INTERVENTION"
    else:
        next_name = profile.intervention_display_names.get(
            selected_intervention, selected_intervention
        )

    convergence_headway = round(b6_headways[0], 2)
    result: dict[str, Any] = {
        "format": "cross-ideal-gap-decomposition-v1",
        "date": profile.date,
        "scope": profile.scope,
        "actual": actual,
        "reference_ladder": ladder,
        "efficiency": efficiency,
        "incremental_loss_decomposition": losses,
        "dominant_bottleneck": {
            "classification": classification,
            "basis": profile.bottleneck_basis,
            "caveat": profile.bottleneck_caveat,
        },
        "recommended_next_counterfactual": {
            "for_B0_like_operation": next_name,
            "expected_paired_effect_service_yield_per_hour_by_arm": expected_selected_effect,
            "safety_feasibility_tradeoff": profile.safety_feasibility_tradeoff,
            "after_B4": profile.no_additional_recovery,
            "calibration": profile.calibration_label,
            "deployment_recommendation": False,
        },
        "ordering_theory": dict(profile.ordering_theory),
        "B6v1_disposition": profile.b6v1_disposition,
        "claim_scope": claim_scope,
        "convergence_limit": {
            "B6v2_fixed_window_yield_per_hour": b6_yields[0],
            "observed_center_headway_min": convergence_headway,
            "headway_implied_long_run_rate_per_hour": 60.0 / convergence_headway,
            "note": profile.convergence_note,
        },
    }
    return result
