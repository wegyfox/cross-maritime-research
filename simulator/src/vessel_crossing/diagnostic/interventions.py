"""Reusable intervention registry for the CROSS Diagnostic Product.

The registry decides only whether a canonical ladder step is identity/applicable/reference
for a configured model.  It never supplies an outcome, effect size or bottleneck answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from ..nextgen.reference_envelope import ReferenceEnvelopeLevel
from .contract import DiagnosticCapabilities


class InterventionDisposition(str, Enum):
    IDENTITY = "IDENTITY"
    APPLICABLE = "APPLICABLE"
    REFERENCE = "REFERENCE"
    NOT_REQUESTED = "NOT_REQUESTED"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class InterventionDefinition:
    level: ReferenceEnvelopeLevel
    intervention_id: str
    target_factor: str
    capability_id: str
    reference: bool = False
    identity_when_absent: bool = True
    identity_when_canonical: bool = True


@dataclass(frozen=True, slots=True)
class InterventionPlanEntry:
    level: ReferenceEnvelopeLevel
    intervention_id: str
    disposition: InterventionDisposition
    reason: str
    target_factor: str

    def to_json_object(self) -> dict[str, str]:
        return {
            "level": self.level.value,
            "intervention_id": self.intervention_id,
            "disposition": self.disposition.value,
            "reason": self.reason,
            "target_factor": self.target_factor,
        }


_DEFAULT_DEFINITIONS = (
    InterventionDefinition(
        ReferenceEnvelopeLevel.B1_DEDICATED_AUTONOMOUS_CORRIDOR,
        "dedicated_corridor_exclusion",
        "external_corridor_intrusion",
        "dedicated_corridor_exclusion",
    ),
    InterventionDefinition(
        ReferenceEnvelopeLevel.B2_CONTROLLED_ENTRY,
        "controlled_entry_metering",
        "source_admission",
        "controlled_entry_metering",
    ),
    InterventionDefinition(
        ReferenceEnvelopeLevel.B3_SPEED_TRAJECTORY_HARMONIZATION,
        "speed_plant_harmonization",
        "plant_speed_heterogeneity",
        "speed_plant_harmonization",
    ),
    InterventionDefinition(
        ReferenceEnvelopeLevel.B4_SLOT_VTS_COORDINATION,
        "global_slot_coordination",
        "global_slot_conflict",
        "global_slot_coordination",
    ),
    InterventionDefinition(
        ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE,
        "type_order_staging",
        "queue_order_structure",
        "type_order_staging",
    ),
    InterventionDefinition(
        ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE,
        "physical_balanced_reference",
        "cross_stream_physical_separation",
        "physical_balanced_reference",
        reference=True,
        identity_when_absent=False,
        identity_when_canonical=False,
    ),
)


class InterventionRegistry:
    def __init__(self, definitions: tuple[InterventionDefinition, ...] = _DEFAULT_DEFINITIONS) -> None:
        levels = [item.level for item in definitions]
        expected = list(ReferenceEnvelopeLevel)[1:]
        if levels != expected:
            raise ValueError("intervention definitions must cover B1..B6 in canonical order")
        ids = [item.intervention_id for item in definitions]
        if len(ids) != len(set(ids)):
            raise ValueError("intervention_id values must be unique")
        self._definitions = tuple(definitions)
        self._by_id: Mapping[str, InterventionDefinition] = MappingProxyType(
            {item.intervention_id: item for item in definitions}
        )

    @property
    def definitions(self) -> tuple[InterventionDefinition, ...]:
        return self._definitions

    @property
    def intervention_ids(self) -> tuple[str, ...]:
        return tuple(item.intervention_id for item in self._definitions)

    def definition(self, intervention_id: str) -> InterventionDefinition:
        try:
            return self._by_id[intervention_id]
        except KeyError as exc:
            raise KeyError(f"unknown diagnostic intervention {intervention_id!r}") from exc

    def plan(
        self,
        capabilities: DiagnosticCapabilities,
        *,
        requested_interventions: tuple[str, ...] = (),
    ) -> tuple[InterventionPlanEntry, ...]:
        requested = set(requested_interventions) if requested_interventions else set(self.intervention_ids)
        unknown = requested - set(self.intervention_ids)
        if unknown:
            raise ValueError(f"unknown requested_interventions: {sorted(unknown)}")

        entries: list[InterventionPlanEntry] = []
        for definition in self._definitions:
            if definition.intervention_id not in requested:
                entries.append(
                    InterventionPlanEntry(
                        definition.level,
                        definition.intervention_id,
                        InterventionDisposition.NOT_REQUESTED,
                        "NOT_REQUESTED_BY_SCENARIO",
                        definition.target_factor,
                    )
                )
                continue

            if definition.target_factor in capabilities.canonical_factors and definition.identity_when_canonical:
                entries.append(
                    InterventionPlanEntry(
                        definition.level,
                        definition.intervention_id,
                        InterventionDisposition.IDENTITY,
                        "TARGET_FACTOR_ALREADY_CANONICAL",
                        definition.target_factor,
                    )
                )
                continue

            if definition.target_factor not in capabilities.represented_factors and definition.identity_when_absent:
                entries.append(
                    InterventionPlanEntry(
                        definition.level,
                        definition.intervention_id,
                        InterventionDisposition.IDENTITY,
                        "TARGET_FACTOR_NOT_REPRESENTED",
                        definition.target_factor,
                    )
                )
                continue

            available = (
                definition.capability_id in capabilities.reference_models
                if definition.reference
                else definition.capability_id in capabilities.executable_interventions
            )
            if available:
                entries.append(
                    InterventionPlanEntry(
                        definition.level,
                        definition.intervention_id,
                        InterventionDisposition.REFERENCE if definition.reference else InterventionDisposition.APPLICABLE,
                        "REFERENCE_MODEL_AVAILABLE" if definition.reference else "EXECUTION_ADAPTER_AVAILABLE",
                        definition.target_factor,
                    )
                )
            else:
                entries.append(
                    InterventionPlanEntry(
                        definition.level,
                        definition.intervention_id,
                        InterventionDisposition.UNSUPPORTED,
                        "NO_REFERENCE_ADAPTER" if definition.reference else "NO_EXECUTION_ADAPTER",
                        definition.target_factor,
                    )
                )
        return tuple(entries)


DEFAULT_INTERVENTION_REGISTRY = InterventionRegistry()
