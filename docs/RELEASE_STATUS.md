# Release status — 2026-10-14

**Current state:** Two **public Zenodo v1.0.0 archival records** and a **public GitHub source repository** (visibility confirmed via GitHub API on 2026-10-14). The Ocean Engineering manuscript was **submitted on 14 October 2026** (portal status: Submitted; not peer reviewed, accepted or published).

## Published archival objects

- **R2 scientific reproduction dataset:** [Zenodo version 1.0.0, DOI 10.5281/zenodo.23345696](https://doi.org/10.5281/zenodo.23345696); record https://zenodo.org/records/23345696; source outer ZIP published as `CROSS_R2_Reproduction_Release_v1.0.0.zip`, exact pre-rename SHA-256 `d9a966cd58537619928283da47dea0534026fd875d9f5ce194e071621d8be735`.
- **PSP Unified LEAN research software:** [Zenodo version 1.0.0, DOI 10.5281/zenodo.23363285](https://doi.org/10.5281/zenodo.23363285); record https://zenodo.org/records/23363285; published outer ZIP `CROSS_PSP_Unified_LEAN_v1.0.0.zip`, exact pre-rename SHA-256 `0788a52d2cf3677d0393852dff2f222872087056e19d551295fc374c70213b62`.

Zenodo metadata and status were reviewed from owner-provided screenshots on 2026-10-14; direct authenticated read-back via this assistant's public web access was unavailable. The outer ZIP filenames changed only, not archived bytes. **Version DOIs** should be used for reproducible citations; Zenodo also assigns separate concept DOIs for all versions.

## GitHub source / CI evidence

- A131 imported **390 pinned source members** with checksums in `docs/SOURCE_IMPORT_SHA256.csv`, from frozen PSP source ZIP and D0 wheel. The Git working tree contains source, tests, derived install metadata and provenance, not the heavy scientific result archives.
- One Windows `START_CROSS.cmd` path has a documented CRLF/LF Git-normalization exception; its original bytes are checked via `SOURCE_IMPORT_NORMALIZATION.md`.
- [GitHub Actions run 38394402694](https://github.com/wegyfox/cross-maritime-research/actions/runs/38394402694) passed Python 3.13 source verification, install/CLI import, byte checks and **30/30** focused tests. This is not a full GUI or scientific replay gate.

## Independent release-archive evidence

- Frozen R2 reference-model results and reproduction archive retain their original integrity and external reproduction evidence.
- Earlier derived LEAN qualified PSP component passed Windows Python 3.13 clean-install CLI checks (22/22), and the unmodified A127 launcher passed targeted Windows dual-archive SHA verification with the trusted-checkpoint warning visible.
- Published A130 outer archives passed exact CRC, SHA-256 and manifest checks; renamed upload files preserve bytes.

These results do **not** imply AIS calibration, operational waterway capacity, real-world navigational safety, full GUI acceptance, a complete penetration test or an external legal opinion. Embedded third-party assets retain their component licenses. Only locally created trusted checkpoints may be deserialized.

## Next separately controlled steps

1. The **version-specific Zenodo DOIs** are linked in the English manuscript's Data and Code Availability section and are reflected in repository citation metadata; preserve that linkage during editorial review.
2. This repository is **public** by explicit owner decision; no GitHub Release has been created. Do not turn on automatic GitHub → Zenodo archiving for the already-deposited version, to avoid duplicate DOI records.
3. Owner completed the submission to Ocean Engineering on 14 October 2026, at 16:55 as shown in the portal. Full length article: *From Pair-Context Effects to Network Bottlenecks in Mixed Autonomous and Human-Driven Ship Traffic: A Reference-Envelope Simulation Study*. Wait for the actual manuscript tracking number and editorial correspondence; do not equate Submitted with peer review or acceptance.

Do not alter frozen scientific archive members or original results to match repository documentation.
