"""Reusable templates for result-credibility challenge audits."""

from __future__ import annotations

from .records import (
    AuditChallenge,
    AuditChallengeStatus,
    CredibilityKillAudit,
)

_CHALLENGES = (
    ("software_bug", "SOFTWARE", "Could this be a bug?"),
    ("event_ordering", "SOFTWARE", "Could event ordering create it?"),
    ("state_persistence", "SOFTWARE", "Could state persistence create it?"),
    ("queue_accounting", "SOFTWARE", "Could queue accounting create it?"),
    ("analysis_script", "SOFTWARE", "Could an analysis script create it?"),
    (
        "time_resolution",
        "NUMERICS",
        "Could timestep/event resolution/tolerance create it?",
    ),
    ("rounding_precision", "NUMERICS", "Could rounding/precision create it?"),
    ("boundary_artefact", "NUMERICS", "Could a boundary artefact create it?"),
    ("seed_specific", "STOCHASTICS", "Could it be seed-specific?"),
    ("monte_carlo_noise", "STOCHASTICS", "Could it be Monte Carlo noise?"),
    ("crn_pairing", "STOCHASTICS", "Are CRN/pairing assumptions correct?"),
    ("single_parameter", "PARAMETERS", "Does it depend on one arbitrary parameter?"),
    ("parameter_interpretation", "PARAMETERS", "Are parameter meanings supported?"),
    ("motion_model", "MODEL_FORM", "Does it depend on the motion model?"),
    ("behavior_model", "MODEL_FORM", "Does it depend on the behavior model?"),
    ("encounter_geometry", "MODEL_FORM", "Does it depend on one geometry?"),
    ("source_queue_form", "MODEL_FORM", "Does it depend on source/queue form?"),
    (
        "independent_reference",
        "MODEL_FORM",
        "Does it survive an independent/reference implementation?",
    ),
    ("confounding", "EXPERIMENT_DESIGN", "Is any factor unintentionally confounded?"),
    (
        "matched_exogenous",
        "EXPERIMENT_DESIGN",
        "Are exogenous assumptions correctly matched?",
    ),
    (
        "reference_leakage",
        "EXPERIMENT_DESIGN",
        "Does target inherit the reference mechanism?",
    ),
    ("posthoc_tuning", "EXPERIMENT_DESIGN", "Was anything tuned after viewing results?"),
    (
        "confirmatory_boundary",
        "EXPERIMENT_DESIGN",
        "Are confirmatory/exploratory analyses separated?",
    ),
    ("estimand", "STATISTICS", "Is the estimand the intended one?"),
    ("multiplicity", "STATISTICS", "Are multiplicity controls correct?"),
    ("ci_interpretation", "STATISTICS", "Are confidence intervals interpreted correctly?"),
    (
        "unresolved_as_zero",
        "STATISTICS",
        "Are unresolved effects misreported as zero?",
    ),
    (
        "exploratory_overstatement",
        "STATISTICS",
        "Are exploratory surfaces overstated?",
    ),
    ("claim_exceeds_evidence", "STATISTICS", "Does the claim exceed the evidence?"),
)


def required_kill_audit_challenge_ids() -> tuple[str, ...]:
    """Return the canonical challenge IDs required by the publication mandate."""
    return tuple(challenge_id for challenge_id, _category, _question in _CHALLENGES)


def build_kill_audit_template(study_id: str) -> CredibilityKillAudit:
    """Return a study-agnostic kill-audit template in NOT_RUN state."""
    challenges = tuple(
        AuditChallenge(
            challenge_id=challenge_id,
            category=category,
            question=question,
            status=AuditChallengeStatus.NOT_RUN,
        )
        for challenge_id, category, question in _CHALLENGES
    )
    return CredibilityKillAudit(
        schema_version=1,
        study_id=study_id,
        challenges=challenges,
    )


def build_r1_kill_audit_template(study_id: str) -> CredibilityKillAudit:
    """Backward-compatible R1 alias for the reusable generic template."""
    return build_kill_audit_template(study_id)
