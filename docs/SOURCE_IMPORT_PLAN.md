# A131 exact-source import — executed, with guarded follow-up

## Imported components (2026-10-13)

The verified PSP source ZIP contains 74 original source members, and the pinned D0 simulator wheel supplies 316 Python source members. The A131 import placed them under `psp/` and `simulator/`, respectively, with per-file original path, byte-size and SHA-256 in `SOURCE_IMPORT_SHA256.csv`.

The D0 `simulator/pyproject.toml` was reconstructed separately for source installation from wheel metadata: it is **not** an original pinned D0 wheel member. No frozen model code or scientific data was regenerated.

The Windows batch file `psp/release_shell/START_CROSS.cmd` has one documented Git CRLF/LF normalization exception. See `SOURCE_IMPORT_NORMALIZATION.md`; source content is otherwise pinned and unmodified.

## Tested

GitHub Actions [run 38394402694](https://github.com/wegyfox/cross-maritime-research/actions/runs/38394402694) installed both derived source projects on Python 3.13, checked CLI help, compiled source Python modules and passed 30 selected tests. This is a **focused source-integrity check** and is not the previously qualified full release acceptance campaign.

## Still pending

Carry out per-file rights review where needed; validate future intentional source changes separately from the frozen baseline; expand CI selectively if valuable, avoiding expensive automatic simulation runs. The heavy R2 reproduction and PSP Unified v1.0.0 ZIPs are now separately published on Zenodo under their version-specific DOIs; do not duplicate these result archives in Git history. Do not commit `.venv`, `__pycache__`, local output, raw Windows home-path logs, credentials or heavy replay artifacts.
