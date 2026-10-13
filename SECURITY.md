# Security and untrusted inputs

## Checkpoint trust boundary

Some CROSS execution paths use Python `pickle.loads` or `cloudpickle.loads` to restore saved state. These deserializers can execute arbitrary code in a crafted file **before** subsequent data or checksum verification.

**Resume only checkpoints created by your own trusted local CROSS runs and retained under your control.** Do not import third-party checkpoints, even when accompanied by a matching `.sha256`: a digest alone does not authenticate the file's origin.

CLI, API and potential GUI/config reachability have not been comprehensively audited. Do not treat ZIP integrity or modeled safety PASS indicators as a security audit or a certification of navigation safety.

Please avoid posting suspicious sample checkpoints or potentially sensitive full local paths to public issue threads.
