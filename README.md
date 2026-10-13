# CROSS — Maritime Crossing Research

**Synthetic two-stream maritime traffic research and researcher-facing simulation tools.**

Author: **Aleksandr Gulyk**, Independent Researcher · [ORCID 0009-0005-8564-4175](https://orcid.org/0009-0005-8564-4175)

> **Private release-preparation repository.** Code and documentation are being imported from verified frozen research artifacts. This repository is **not yet a complete runnable source checkout**. No public release or DOI has been registered.

## Research scope

CROSS models controlled synthetic crossings of mixed autonomous and human-operated vessels. It studies throughput, interactions and modeled safety margins in explicitly documented simulation settings. The results are **not AIS-calibrated** and do **not** establish operational waterway capacity, collision-avoidance certification or navigational safety. Do not use them to direct actual vessel traffic.

## Components

- **R2 research reproduction:** frozen reference-envelope experiments, results and reproduction instructions. Its original scientific archive remains immutable.
- **PSP Unified:** a researcher launcher with a qualified-default reference mode and a **separate exploratory metered mode**. Changed physics, fleet mixtures and other configurations do not automatically inherit qualification.
- **Research documentation:** evidence/qualification boundaries, English canonical documentation and a Japanese overview.

## Current repository status

This is the initial documentation and attribution foundation. The exact source-code import, independent CI validation and publication review are pending. **Do not assume the Python modules are present in this revision.** The already verified A130 archive candidates are managed outside GitHub until exact-object owner approval. See [Release status](docs/RELEASE_STATUS.md).

## Source, data and citation

The authoritative R2 and PSP release candidates are distinct artifacts and will receive **separate Zenodo records/DOIs if published**. No DOI is asserted yet. Large experiment-result ZIPs are intended for Zenodo, not Git history. Citation metadata will be updated when actual DOI links exist.

## License and safety

Original author-owned code: [Apache-2.0](LICENSE). Original author-owned synthetic data and documentation may use CC BY 4.0 when explicitly identified; third-party material retains its own terms. See [License scope](LICENSE_SCOPE.md).

**Checkpoints:** Some program paths deserialize Python pickle/cloudpickle. Resume **only locally generated, trusted checkpoints** that have remained under your control. A companion `.sha256` does not authenticate their origin. See [Security](SECURITY.md).

Japanese overview: [README.ja.md](README.ja.md).
