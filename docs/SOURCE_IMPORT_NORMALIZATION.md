# A131 pinned source-byte import and line-ending exception

The source import carries **390 unchanged original source members** from the pinned PSP source archive and D0 wheel, recorded by exact path/size/SHA-256 in `SOURCE_IMPORT_SHA256.csv`. The Git working tree is checked by `tools/verify_source_import.py` in CI.

**Only one source member has a Git storage normalization exception:** `psp/release_shell/START_CROSS.cmd`. Its original Windows text was 333 bytes with 12 CRLF line endings (SHA-256 `e6b4b339d66a45ec3c0bf65d55b5581710f8c16fc9a01145f174230c0ca2d25c`). Git's LF-normalized blob is 321 bytes (SHA-256 `d2b118660c7c56f65a60623b5e90fe4950dd519f481b8f39c0d9e2f7a65a1bd3`). `.gitattributes` requests CRLF checkout for `*.cmd`, and the pin-verifier reconstructs CRLF for **this exact path only**, then checks its original SHA-256. No other member is given a hash-verification exception.

This source identity gate does not validate operational navigation safety, third-party licenses, GUI execution, scientific reproducibility of new configurations or the large frozen experiments. It is a lightweight guard against accidental source drift in the private repository, not a substitute for the previously recorded archive evidence.
