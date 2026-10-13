"""Deterministic R1 P4B seed schedule.

The schedule is separated from plan materialization so seed namespaces remain a small,
auditable contract with no simulator side effects.
"""
from __future__ import annotations

from dataclasses import dataclass

from vessel_crossing.research.r1_p4b_publication_reference import publication_seed

@dataclass(frozen=True, slots=True)
class P4BSeedRecord:
    replication: int
    traffic_arrivals_seed: int
    type_assignment_seed: int
    behavior_stochasticity_seed: int

def _seed_record(replication: int, campaign_id: str) -> P4BSeedRecord:
    return P4BSeedRecord(
        replication=replication,
        traffic_arrivals_seed=publication_seed(
            campaign_id=campaign_id, replication=replication, namespace="TRAFFIC_ARRIVALS"
        ),
        type_assignment_seed=publication_seed(
            campaign_id=campaign_id, replication=replication, namespace="TYPE_ASSIGNMENT"
        ),
        behavior_stochasticity_seed=publication_seed(
            campaign_id=campaign_id, replication=replication, namespace="BEHAVIOR_STOCHASTICITY"
        ),
    )
