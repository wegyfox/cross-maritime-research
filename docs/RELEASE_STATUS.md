# Release status — 2026-10-13

**Current state: PRIVATE / PRE-PUBLICATION.** The pinned source trees and scoped CI are present in this GitHub repository; no public GitHub Release, Zenodo DOI or journal submission exists yet.

## GitHub source / CI evidence

- A131 imported **390 pinned source members**, identified by original-member SHA-256 in `docs/SOURCE_IMPORT_SHA256.csv`, from the PSP source ZIP and D0 frozen wheel. The tree contains source, tests, derived installation metadata and provenance, not full heavy result archives.
- Git storage normalizes one Windows `START_CROSS.cmd` file from CRLF to LF; exact original 333 bytes are reconstructed/verified by a strictly scoped rule. See `SOURCE_IMPORT_NORMALIZATION.md`.
- GitHub Actions run [38394402694](https://github.com/wegyfox/cross-maritime-research/actions/runs/38394402694) completed successfully on Python 3.13: source byte check, installation of both source projects, CLI help commands, compilation, **30/30 focused tests**.
- This single bounded Linux CI job is **not** exhaustive scientific qualification or Windows GUI acceptance.

## Independently verified release-archive evidence

- Frozen R2 scientific archive and research core retain their historical reproduction/integrity evidence.
- The earlier derived LEAN qualified component passed Windows Python 3.13 clean-install/CLI smoke (22/22 checks).
- The unchanged A127 PSP Unified launcher subsequently passed Windows startup, both component SHA checks and visible trusted-checkpoint warning.
- The separate **A130 R2** and **A130 PSP** candidate ZIP wrappers passed SHA/CRC/manifests; their scientific core and wheel bytes were not modified.

These results do **not** constitute AIS calibration, real-world navigation certification, full GUI acceptance, exhaustive security clearance or legal opinion.

## Outstanding before public release

1. Final asset-level license/notice and distribution-data privacy review on the exact intended public bytes.
2. Owner approval of the exact repository commit and each of the two A130 candidate ZIPs; any changed archive needs a fresh exact hash.
3. Public visibility and release only by separate owner approval; do **not** enable GitHub → Zenodo automatic release for this first publication.
4. Two manual Zenodo deposits and actual DOI URLs, followed by citation updates to code and article manuscript.
5. Separate editorial review and journal submission approval.

No frozen scientific archive or original results should be modified to match the packaging.
