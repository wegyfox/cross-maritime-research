# CROSS PSP two-stream — 1.0.0 release candidate 1

Self-contained research release candidate for the qualified CROSS two-stream orthogonal-crossing profile and its supported exploratory analysis layer.

## Release identities

- User-facing distribution: `1.0.0`.
- Qualified Q5 integration identity: `0.1.0a8`.
- Frozen D0 dependency: `vessel-crossing-simulator==1.1.0a36`.
- The qualified default physics profile and its scientific evidence remain frozen; the release-candidate version change does not promote changed physics.

## What this release supports

The public surface provides:

1. end-to-end article reproduction on the frozen qualified profile;
2. custom traffic on qualified physics;
3. bounded custom fixed physical parameters with dynamic same-physics B6 reference generation;
4. deterministic parameter distributions;
5. coherent fleet-mixture/class declarations;
6. deterministic robustness matrices with checkpoint/resume;
7. sensitivity, interaction, uncertainty and claim-boundary reporting;
8. wizard-led creation/preflight/execution of supported analyses;
9. integrity-checked dynamic-reference caching keyed by complete physics/runtime identity.

Use `START_CROSS.cmd` on Windows for the normal interactive entry point.

## Scientific status

`QUALIFIED_PROFILE` applies only to the exact frozen Q5 physical profile. Changed fixed physics, deterministic distributions and fleet-mixture scenarios remain `SUPPORTED_EXPLORATORY` unless separate evidence justifies a stronger status.

A fresh B6 reference is necessary for changed physics but is not itself a robustness qualification. No output establishes field calibration, deployment safety, deployment recommendation, causal effects, universal capacity, or a global optimum.

## Heterogeneous fleet limitation in 1.0.0

Fleet-mixture declarations are deterministic, coherent and identity-safe, but not every declared mixture is guaranteed to obtain a promotable dynamic B6 reference under the frozen qualified controller.

The final staged realism check established the following boundary:

- a standardized coherent 145 m autonomous-style fleet produced a promotable balanced saturated B6 reference;
- a moderate two-class 90/145 m mixture failed closed during B6 warmup with `NoEnvironmentalPairCandidate`;
- diagnostics showed that the failing crossing pair had no safe row in the frozen pair-action grid (`30/60 deg` course changes and `0.8/1.0` speed fractions);
- counterfactual diagnostics showed that a wider combined course/speed action space can produce safe rows, so the observed limitation is controller-action-space expressiveness rather than proof that heterogeneous fleets are physically impossible.

For this release the frozen qualified controller is intentionally not broadened. Heterogeneous mixtures therefore remain explicitly exploratory and may return a fail-closed non-promotable reference. This preserves the qualified core instead of silently changing controller semantics immediately before 1.0.0.

## Advanced CLI

```text
cross-psp verify
cross-psp info
cross-psp wizard --mode traffic
cross-psp wizard --mode physics
cross-psp analyze scenario.yaml --output output
```

The dynamic B6 cache is accepted only when the complete reference identity matches and cached evidence passes integrity and semantic validation.
