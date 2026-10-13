# CROSS — Maritime Crossing Research

**Synthetic two-stream maritime traffic research and researcher-facing simulation tools.**

Author: **Aleksandr Gulyk**, Independent Researcher · [ORCID 0009-0005-8564-4175](https://orcid.org/0009-0005-8564-4175)

> **PRIVATE / PRE-PUBLICATION.** The pinned PSP and D0 source trees are now imported, and a bounded source-install/test CI gate has passed. This repository is not yet a public release; no DOI is registered.

## Research scope

CROSS models controlled synthetic crossings of mixed autonomous and human-operated vessels. It studies throughput, interactions and modeled safety margins in explicitly documented simulation settings. The results are **not AIS-calibrated** and do **not** establish operational waterway capacity, collision-avoidance certification or navigational safety. Do not use them to direct actual vessel traffic.

## Components

- `psp/` — qualified-default two-stream research analysis source, wizard and tests. Modified physical parameters and heterogeneous fleets are **exploratory**, not automatically qualified.
- `simulator/` — D0 model source extracted from the **pinned** `vessel-crossing-simulator==1.1.0a36` wheel. Its source-install metadata is derived, not the original development tree.
- `experiments/` and `docs/` — small reproducibility pointers and scientific qualification/provenance documentation.
- Separately packaged **PSP Unified** — a researcher launcher with a qualified-default mode and a distinct post-1.0 exploratory metered mode. Its Windows launcher and bundled data are in the independently reviewed distribution candidate, not a GitHub binary release.

## Source installation (research development)

With Python **3.13** and a fresh virtual environment:

```sh
python -m pip install -e ./simulator -e ./psp pytest
cross-psp --help
vessel-crossing --help
python tools/verify_source_import.py
python -m pytest -q psp/tests/test_release_cli.py psp/tests/test_contract.py psp/tests/test_custom_physics_binding.py psp/tests/test_dynamic_physics.py
```

The CI also compiles the Python source trees. The [focused CI run](https://github.com/wegyfox/cross-maritime-research/actions/runs/38394402694) passed **30/30 tests** and verified the 390 pinned source members. It does **not** run 726 scientific experiments, validate every GUI flow, or substitute for the frozen scientific release evidence.

## Source, data and citation

The authoritative frozen R2 reproduction archive and PSP Unified distribution are separate candidates, with separate Zenodo records/DOIs **if approved and published**. No DOI is asserted yet. Large research archives should remain on Zenodo, not in Git history. See [release status](docs/RELEASE_STATUS.md), [import normalization](docs/SOURCE_IMPORT_NORMALIZATION.md) and [research scope](docs/RESEARCH_SCOPE.md).

## Licenses and safety

Original author-owned code: [Apache-2.0](LICENSE). Original synthetic data/documentation may use CC BY 4.0 **only where specifically identified**; third-party components retain their own terms. See [license scope](LICENSE_SCOPE.md).

**Checkpoint warning:** Some paths use Python pickle/cloudpickle. Resume **only locally created, trusted checkpoints** kept under your control; a companion `.sha256` is not proof of origin. See [security notes](SECURITY.md).

Japanese overview: [README.ja.md](README.ja.md).
