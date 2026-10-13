"""Final R2 scientific-integration and simulator-campaign readiness gate.

R2-S9 does not add a new physical law.  It integrates the qualified S1-S8
scientific layers into an explicit fail-closed campaign contract.  The default
campaign is CROSS-owned and uses the CROSS operational measurement protocol;
it is not an exact Guo-2024 replication and it does not force reference
headways into simulator dynamics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from typing import Final

from .r2_s1_measurement_protocol import R2_S1_MEASUREMENT_BINDING
from .r2_s1_pair_semantics import GUO_2024_RUNNING_EXAMPLE_S1, UNRESOLVED
from .r2_s1_trust import R2_S1_TRUST_BINDING, TrustTransformHypothesis, TrustTransformStatus


class CampaignKind(str, Enum):
    CROSS_STRUCTURAL_MECHANISM = "CROSS_STRUCTURAL_MECHANISM"
    CROSS_MEASUREMENT_NOISE_SENSITIVITY = "CROSS_MEASUREMENT_NOISE_SENSITIVITY"
    CROSS_TRUST_HYPOTHESIS_SENSITIVITY = "CROSS_TRUST_HYPOTHESIS_SENSITIVITY"
    GUO_2024_EXACT_REPLICATION = "GUO_2024_EXACT_REPLICATION"


@dataclass(frozen=True, slots=True)
class R2QualifiedChain:
    s1: str = "QUALIFIED"
    s2: str = "QUALIFIED"
    s3: str = "QUALIFIED"
    s4: str = "QUALIFIED"
    s5: str = "QUALIFIED"
    s6: str = "QUALIFIED"
    s7: str = "QUALIFIED"
    s8: str = "QUALIFIED"

    @property
    def all_qualified(self) -> bool:
        return all(value == "QUALIFIED" for value in asdict(self).values())


@dataclass(frozen=True, slots=True)
class CampaignReadinessSnapshot:
    qualified_chain: R2QualifiedChain
    source_pair_orientation: str
    source_measurement_event: str
    cross_measurement_protocol_id: str
    runtime_measurement_event: str
    guo_exact_binding_status: str
    guo_exact_replication_authorized: bool
    trust_transform_status: str
    measurement_noise_calibration_status: str
    gaussian_interval_claim_authorized: bool
    default_campaign_forces_reference_headways: bool

    @property
    def cross_structural_prerequisites_ready(self) -> bool:
        return (
            self.qualified_chain.all_qualified
            and self.cross_measurement_protocol_id != ""
            and self.runtime_measurement_event != ""
            and not self.default_campaign_forces_reference_headways
        )


@dataclass(frozen=True, slots=True)
class CampaignAuthorizationDecision:
    campaign_kind: CampaignKind
    authorized: bool
    semantic_label: str
    reasons: tuple[str, ...]
    forbidden_claims: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StructuralCampaignProtocol:
    campaign_id: str
    campaign_kind: str
    scientific_owner: str
    model_baseline: str
    qualification_requirement: str
    measurement_protocol_id: str
    runtime_measurement_event: str
    reference_matrix_id: str
    reference_matrix_sha256: str
    pair_orientation: str
    role_penetrations: tuple[float, ...]
    ordering_families: tuple[str, ...]
    primary_outcomes: tuple[str, ...]
    secondary_outcomes: tuple[str, ...]
    exclusions: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    added_measurement_noise: str
    trust_transform: str
    reference_headway_forcing: bool
    evidence_role: str

    @property
    def identity_sha256(self) -> str:
        payload = asdict(self)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(raw).hexdigest()

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["identity_sha256"] = self.identity_sha256
        return out


QUALIFIED_R2_S1_S8_CHAIN: Final = R2QualifiedChain()


def qualified_s8_readiness_snapshot() -> CampaignReadinessSnapshot:
    """Return the frozen readiness state after PC qualification of R2-S8."""
    binding = R2_S1_MEASUREMENT_BINDING
    trust = R2_S1_TRUST_BINDING
    return CampaignReadinessSnapshot(
        qualified_chain=QUALIFIED_R2_S1_S8_CHAIN,
        source_pair_orientation="FOLLOWER_TO_LEADER",
        source_measurement_event=binding.source_measurement_event,
        cross_measurement_protocol_id=binding.cross_protocol_id,
        runtime_measurement_event=binding.runtime_event,
        guo_exact_binding_status="NOT_READY",
        guo_exact_replication_authorized=False,
        trust_transform_status=trust.transform_status.value,
        measurement_noise_calibration_status="UNBOUND_PARAMETER",
        gaussian_interval_claim_authorized=False,
        default_campaign_forces_reference_headways=False,
    )


def evaluate_campaign_authorization(
    campaign_kind: CampaignKind,
    *,
    s9_qualified: bool,
    measurement_noise_variance_min2: float | None = None,
    trust_hypothesis: TrustTransformHypothesis | None = None,
) -> CampaignAuthorizationDecision:
    """Evaluate a campaign request under the frozen S9 fail-closed rules."""
    state = qualified_s8_readiness_snapshot()
    forbidden = (
        "EXACT_GUO_2024_REPLICATION",
        "REAL_WORLD_CHANNEL_CAPACITY_VALIDATION",
        "EMPIRICALLY_CALIBRATED_MEASUREMENT_NOISE",
        "GAUSSIAN_INTERVAL_WITHOUT_ADDITIONAL_JUSTIFICATION",
    )
    if not s9_qualified:
        return CampaignAuthorizationDecision(
            campaign_kind, False, "BLOCKED_PENDING_R2_S9_QUALIFICATION",
            ("R2-S9 PC qualification has not been accepted.",), forbidden,
        )
    if not state.cross_structural_prerequisites_ready:
        return CampaignAuthorizationDecision(
            campaign_kind, False, "BLOCKED_PREREQUISITE_DRIFT",
            ("Qualified S1-S8 integration prerequisites are not satisfied.",), forbidden,
        )
    if campaign_kind is CampaignKind.GUO_2024_EXACT_REPLICATION:
        return CampaignAuthorizationDecision(
            campaign_kind, False, "BLOCKED_GUO_EXACT_BINDING_NOT_READY",
            ("Guo exact measurement/trust/equation binding remains NOT_READY.",), forbidden,
        )
    if campaign_kind is CampaignKind.CROSS_STRUCTURAL_MECHANISM:
        return CampaignAuthorizationDecision(
            campaign_kind, True, "AUTHORIZED_CROSS_OWNED_NOT_GUO_EXACT",
            (
                "S1-S8 are qualified.",
                "CROSS operational measurement protocol is explicit.",
                "Reference headways are diagnostic comparators and are not runtime forcing terms.",
            ), forbidden,
        )
    if campaign_kind is CampaignKind.CROSS_MEASUREMENT_NOISE_SENSITIVITY:
        if measurement_noise_variance_min2 is None:
            return CampaignAuthorizationDecision(
                campaign_kind, False, "BLOCKED_EXPLICIT_NOISE_PARAMETER_REQUIRED",
                ("Noise sensitivity requires an explicit CROSS hypothesis variance.",), forbidden,
            )
        value = float(measurement_noise_variance_min2)
        if not isfinite(value) or value < 0.0:
            raise ValueError("measurement_noise_variance_min2 must be finite and >= 0")
        return CampaignAuthorizationDecision(
            campaign_kind, True, "AUTHORIZED_CROSS_HYPOTHESIS_NOT_CALIBRATED",
            ("Noise variance is explicit and remains a CROSS sensitivity parameter.",), forbidden,
        )
    if campaign_kind is CampaignKind.CROSS_TRUST_HYPOTHESIS_SENSITIVITY:
        if trust_hypothesis is None:
            return CampaignAuthorizationDecision(
                campaign_kind, False, "BLOCKED_EXPLICIT_TRUST_HYPOTHESIS_REQUIRED",
                ("Trust sensitivity requires an explicit validated CROSS hypothesis object.",), forbidden,
            )
        if R2_S1_TRUST_BINDING.transform_status is not TrustTransformStatus.UNBOUND:
            raise RuntimeError("frozen S9 assumes exact source trust transform remains unbound")
        return CampaignAuthorizationDecision(
            campaign_kind, True, "AUTHORIZED_CROSS_HYPOTHESIS_NOT_GUO_EXACT",
            ("Trust transform is explicit CROSS_HYPOTHESIS and must remain labeled NOT_GUO_EXACT.",), forbidden,
        )
    raise AssertionError(f"unhandled campaign kind: {campaign_kind}")


def default_structural_campaign_protocol() -> StructuralCampaignProtocol:
    """Freeze the first post-S9 physical campaign's scientific contract.

    The later executable runner may materialize the declared factors into frozen
    simulator point identities.  This contract intentionally does not prescribe
    an empirical measurement-noise variance or a Guo trust transform.
    """
    state = qualified_s8_readiness_snapshot()
    return StructuralCampaignProtocol(
        campaign_id="CROSS_R2_POST_S9_STRUCTURAL_MECHANISM_CAMPAIGN_V1",
        campaign_kind=CampaignKind.CROSS_STRUCTURAL_MECHANISM.value,
        scientific_owner="CROSS",
        model_baseline="R2_S9_V1_QUALIFIED_REQUIRED",
        qualification_requirement="R2_S9_PC_GATE_ACCEPTED_BEFORE_EXECUTION",
        measurement_protocol_id=state.cross_measurement_protocol_id,
        runtime_measurement_event=state.runtime_measurement_event,
        reference_matrix_id=GUO_2024_RUNNING_EXAMPLE_S1.reference_id,
        reference_matrix_sha256=GUO_2024_RUNNING_EXAMPLE_S1.identity_sha256,
        pair_orientation="FOLLOWER_TO_LEADER",
        role_penetrations=(0.0, 0.25, 0.5, 0.75, 1.0),
        ordering_families=(
            "S2_EXACT_MIN_TOTAL_FINITE_SEQUENCE",
            "S2_EXACT_MAX_TOTAL_FINITE_SEQUENCE",
            "S3_UNIFORM_FIXED_COMPOSITION_RANDOM_ORDER",
        ),
        primary_outcomes=(
            "pair_typed_conflict_center_pass_headway_min",
            "pair_typed_reference_residual_min",
            "finite_block_mean_center_pass_headway_min",
            "finite_block_capacity_proxy_1440_over_mean",
        ),
        secondary_outcomes=(
            "qualification_pass",
            "terminal_status",
            "center_pass_pair_coverage",
            "ordering_inversion_count",
        ),
        exclusions=(
            "NO_GUO_EXACT_REPLICATION_CLAIM",
            "NO_REAL_WORLD_CHANNEL_CAPACITY_CLAIM",
            "NO_REFERENCE_HEADWAY_RUNTIME_FORCING",
            "NO_EMPIRICAL_NOISE_CALIBRATION_CLAIM",
            "NO_GUO_TRUST_TRANSFORM_CLAIM",
            "NO_GAUSSIAN_INTERVAL_CLAIM",
        ),
        stop_conditions=(
            "ABORT_ON_ROLE_ORIENTATION_DRIFT",
            "ABORT_ON_MEASUREMENT_EVENT_DRIFT",
            "ABORT_ON_REFERENCE_HEADWAY_FORCING",
            "PRESERVE_MODELED_INFEASIBLE_AND_NEGATIVE_RESULTS",
            "DO_NOT_SILENTLY_DROP_MISSING_PAIR_COVERAGE",
        ),
        added_measurement_noise="NONE_IN_DEFAULT_STRUCTURAL_CAMPAIGN",
        trust_transform="NONE_IN_DEFAULT_STRUCTURAL_CAMPAIGN",
        reference_headway_forcing=False,
        evidence_role="CROSS_OWNED_STRUCTURAL_MECHANISM_COMPARISON_NOT_GUO_EXACT",
    )
