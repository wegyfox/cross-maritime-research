"""Portable reporting surfaces built above canonical CROSS result contracts."""

from .reference_envelope import (
    IdealGapDecomposition,
    ReferenceEnvelopeArtifacts,
    ReferenceEnvelopeReportingError,
    load_ideal_gap_decomposition,
    render_reference_envelope_html,
    verify_reference_envelope_artifacts,
    write_reference_envelope_artifacts,
)

__all__ = [
    "IdealGapDecomposition",
    "ReferenceEnvelopeArtifacts",
    "ReferenceEnvelopeReportingError",
    "load_ideal_gap_decomposition",
    "render_reference_envelope_html",
    "verify_reference_envelope_artifacts",
    "write_reference_envelope_artifacts",
]
