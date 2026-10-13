"""Current-time fleet ownership gate for one environmental JOINT control cycle.

A113 mirrors the earlier PAIR ownership boundary without reusing PAIR-specific
semantics.  It is intentionally clockless and side-effect free: one already-committed
fleet dispatch is compared with the exact JOINT vessel/pair set owned by a112.

During avoidance, an exact JOINT owner may continue.  A temporary absence of any owner
for the controlled vessels is also tolerated so the local component lifecycles may
finish their own release hysteresis.  Any touching PAIR owner, changed JOINT component,
changed semantic edge set, or split/merged touching ownership requires recapture.

A local a110 ``HANDOFF_ELIGIBLE`` decision is stricter: route handoff is allowed only
when the current fleet dispatch has no active owner touching any controlled vessel.
This prevents stale local pair lifecycles from handing route control back while the
canonical fleet registry still sees active encounter work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .encounter_registry import PairKey, canonical_pair_key
from .environmental_joint_execution_policy import EnvironmentalJointExecutionPolicyAction
from .fleet_supervision import FleetCommandOwner, FleetSupervisionDispatch, FleetSupervisorBackend


class EnvironmentalJointOwnershipAction(str, Enum):
    CONTINUE_LOCAL_CYCLE = "continue_local_cycle"
    ALLOW_LOCAL_HANDOFF = "allow_local_handoff"
    RECAPTURE_REQUIRED = "recapture_required"


class EnvironmentalJointOwnershipReason(str, Enum):
    EXACT_JOINT_OWNER = "exact_joint_owner"
    NO_ACTIVE_OWNER = "no_active_owner"
    REPLAN_WITHOUT_ACTIVE_OWNER = "replan_without_active_owner"
    UNRELATED_ACTIVE_COMPONENTS_ONLY = "unrelated_active_components_only"
    CONTROLLED_COMPONENT_CHANGED = "controlled_component_changed"
    HANDOFF_NO_TOUCHING_OWNER = "handoff_no_touching_owner"
    HANDOFF_TOUCHING_OWNER = "handoff_touching_owner"


@dataclass(frozen=True, slots=True)
class EnvironmentalJointOwnershipDecision:
    time_min: float
    component_id: str
    vessel_ids: tuple[str, ...]
    pair_keys: tuple[PairKey, ...]
    requested_policy_action: EnvironmentalJointExecutionPolicyAction | None
    action: EnvironmentalJointOwnershipAction
    reason: EnvironmentalJointOwnershipReason
    touching_owners: tuple[FleetCommandOwner, ...]

    @property
    def recapture_required(self) -> bool:
        return self.action is EnvironmentalJointOwnershipAction.RECAPTURE_REQUIRED


def _normalize_pair_keys(pair_keys: tuple[PairKey, ...]) -> tuple[PairKey, ...]:
    normalized = tuple(canonical_pair_key(*key) for key in pair_keys)
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate pair key in JOINT ownership identity")
    return normalized


def assess_environmental_joint_cycle_ownership(
    component_id: str,
    vessel_ids: tuple[str, ...],
    pair_keys: tuple[PairKey, ...],
    dispatch: FleetSupervisionDispatch,
    *,
    requested_policy_action: EnvironmentalJointExecutionPolicyAction | None = None,
) -> EnvironmentalJointOwnershipDecision:
    """Validate a112 command ownership against one canonical fleet dispatch.

    ``requested_policy_action`` is the a110 action about to be consumed by a112.  It is
    optional for periodic fleet reviews.  HANDOFF is deliberately admitted only when no
    active fleet owner touches the controlled JOINT vessel set.
    """

    if not component_id:
        raise ValueError("component_id must be non-empty")
    if not isinstance(dispatch, FleetSupervisionDispatch):
        raise TypeError("dispatch must be FleetSupervisionDispatch")
    if requested_policy_action is not None and not isinstance(
        requested_policy_action, EnvironmentalJointExecutionPolicyAction
    ):
        raise TypeError("requested_policy_action must be EnvironmentalJointExecutionPolicyAction or None")
    if len(vessel_ids) < 3:
        raise ValueError("JOINT ownership requires at least three vessel ids")
    if len(set(vessel_ids)) != len(vessel_ids) or any(not item for item in vessel_ids):
        raise ValueError("vessel_ids must be unique and non-empty")
    if component_id != "|".join(vessel_ids):
        raise ValueError("component_id must match ordered vessel_ids")
    pair_keys = _normalize_pair_keys(pair_keys)
    vessel_set = set(vessel_ids)
    if any(set(key) - vessel_set for key in pair_keys):
        raise ValueError("pair_keys must reference only controlled JOINT vessels")

    touching = tuple(owner for owner in dispatch.owners if vessel_set.intersection(owner.vessel_ids))

    if requested_policy_action is EnvironmentalJointExecutionPolicyAction.HANDOFF_ELIGIBLE:
        if not touching:
            reason = (
                EnvironmentalJointOwnershipReason.UNRELATED_ACTIVE_COMPONENTS_ONLY
                if dispatch.owners
                else EnvironmentalJointOwnershipReason.HANDOFF_NO_TOUCHING_OWNER
            )
            return EnvironmentalJointOwnershipDecision(
                dispatch.time_min,
                component_id,
                vessel_ids,
                pair_keys,
                requested_policy_action,
                EnvironmentalJointOwnershipAction.ALLOW_LOCAL_HANDOFF,
                reason,
                touching,
            )
        return EnvironmentalJointOwnershipDecision(
            dispatch.time_min,
            component_id,
            vessel_ids,
            pair_keys,
            requested_policy_action,
            EnvironmentalJointOwnershipAction.RECAPTURE_REQUIRED,
            EnvironmentalJointOwnershipReason.HANDOFF_TOUCHING_OWNER,
            touching,
        )

    exact = tuple(
        owner
        for owner in touching
        if owner.backend is FleetSupervisorBackend.JOINT
        and owner.component_id == component_id
        and owner.vessel_ids == vessel_ids
        and owner.pair_keys == pair_keys
    )
    if len(exact) == 1 and len(touching) == 1:
        return EnvironmentalJointOwnershipDecision(
            dispatch.time_min,
            component_id,
            vessel_ids,
            pair_keys,
            requested_policy_action,
            EnvironmentalJointOwnershipAction.CONTINUE_LOCAL_CYCLE,
            EnvironmentalJointOwnershipReason.EXACT_JOINT_OWNER,
            touching,
        )

    if not touching:
        # An ownerless boundary may let an already-selected command continue long enough
        # for local release hysteresis to finish, but it cannot authorize a *new* plan.
        # REPLAN_REQUIRED consumes fresh current ownership evidence; if the canonical
        # dispatch has already released the component, the stale JOINT cycle must yield
        # to fleet recapture/release instead of trying to rebuild from its old identity.
        if requested_policy_action is EnvironmentalJointExecutionPolicyAction.REPLAN_REQUIRED:
            return EnvironmentalJointOwnershipDecision(
                dispatch.time_min,
                component_id,
                vessel_ids,
                pair_keys,
                requested_policy_action,
                EnvironmentalJointOwnershipAction.RECAPTURE_REQUIRED,
                EnvironmentalJointOwnershipReason.REPLAN_WITHOUT_ACTIVE_OWNER,
                touching,
            )
        reason = (
            EnvironmentalJointOwnershipReason.UNRELATED_ACTIVE_COMPONENTS_ONLY
            if dispatch.owners
            else EnvironmentalJointOwnershipReason.NO_ACTIVE_OWNER
        )
        return EnvironmentalJointOwnershipDecision(
            dispatch.time_min,
            component_id,
            vessel_ids,
            pair_keys,
            requested_policy_action,
            EnvironmentalJointOwnershipAction.CONTINUE_LOCAL_CYCLE,
            reason,
            touching,
        )

    return EnvironmentalJointOwnershipDecision(
        dispatch.time_min,
        component_id,
        vessel_ids,
        pair_keys,
        requested_policy_action,
        EnvironmentalJointOwnershipAction.RECAPTURE_REQUIRED,
        EnvironmentalJointOwnershipReason.CONTROLLED_COMPONENT_CHANGED,
        touching,
    )
