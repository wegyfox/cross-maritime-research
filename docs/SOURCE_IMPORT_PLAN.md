# Exact-source import plan (not yet executed)

The verified PSP LEAN archival candidate contains an explicit `source/cross_psp_two_stream-1.0.0-source.zip` (74 entries) and pinned D0 `vessel_crossing_simulator-1.1.0a36-py3-none-any.whl` (321 entries). These are the intended immutable inputs for a **separate, exact-byte source import**, not permission to regenerate or revise the frozen simulator.

Planned path mapping, subject to per-file audit:

- `psp/`: PSP source tree, `pyproject.toml`, examples and matching tests.
- `simulator/`: D0 source tree extracted from the pinned wheel, with an explicit provenance record distinguishing wheel contents from a development source tree.
- `experiments/`: small reproducible experiment specifications and scripts; large archived scientific result ZIPs remain in the external, version-specific Zenodo releases.

For each imported path record original archive member name, component SHA-256, exact extracted SHA-256, rights classification and any executable/checkpoint ingress. Run focused import tests, Python lint/format checks where supported, then repository CI. Do not add `.venv`, `__pycache__`, local outputs, raw Windows logs, credentials or large archives to Git. This file is a plan, not a claim that the source tree is already populated.
