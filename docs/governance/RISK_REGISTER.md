# Risk register

The frozen method boundary and mandatory tests are defined in `FROZEN_RESEARCH_CHARTER.md`.

| ID | Risk | Direct test/mitigation | Gate |
|---|---|---|---|
| R-001 | Direct or derived high-precision/data-split leakage influences development | physical separation; provenance/lineage manifests; guard tests; independent split and feature audit | every phase; mandatory P2/P4/P6 |
| R-002 | DrvFS case/permission semantics cause ambiguity or exposure | lowercase unique names; no secrets/reference on `/mnt/e`; WSL checks | P0/P1 |
| R-003 | `/mnt/e` becomes a training I/O bottleneck | keep source only there; representative benchmark on approved external WSL-native roots; 8 MiB probe is not a training benchmark | P2 before data runs |
| R-004 | Scientific dependencies are not reproducible | resolver ADR and generated hash lock before the first third-party scientific import | first dependency-bearing Phase |
| R-005 | WLS/SPP is not independent, conventional, or stable | official method specification; standardized outputs; synthetic/golden and cross-implementation checks | P2 gate |
| R-006 | Model simply copies SPP | compare residual/output to SPP; counterfactual perturbation and GNSS-degradation tests | mandatory P4/P6 |
| R-007 | A later predictive-covariance model inflates uncertainty to evade residual penalties | if a covariance head is separately approved, require proper scoring, calibration, sharpness, coverage, and bounded-output diagnostics | conditional P6 extension only |
| R-008 | Model rejects every GNSS update | update-acceptance distribution; forced-valid-GNSS cases; SPP-information ablation | mandatory P4/P6 |
| R-009 | A later joint `Q`, `R`, and bias model becomes unidentifiable through compensation | prohibit joint free learning in the primary model; require staged identification and single-factor tests if separately approved later | conditional extension only |
| R-010 | Route/receiver/time identity is memorized | route-level and cross-receiver holdouts; metadata ablation; nearest-route checks | mandatory P2/P4/P6 |
| R-011 | Constant common SPP bias is claimed recoverable without absolute information | record the observability limitation; add a constant-bias counterexample once the learned method exists | P1 wording; P4/P6 test |
| R-012 | PINN duplicates the ESKF mechanization identity | independent residual derivation with units/assumptions; dependency graph and ablation | P1 specification; P4 gate |
| R-013 | Symmetry group mishandles gravity or reflection/pseudovector behavior | `SO(2)` first; transformation tests; `O(2)` only with correct angular-velocity reflection | P1/P5 gate |
| R-014 | A later GNSS temporal branch adds capacity rather than stable causal value | do not require extra backbones in the primary method; use a matched-capacity comparison only if a temporal branch is separately approved | conditional extension only |
| R-015 | Final route is invalidated after viewing | immutable freeze and one-time access log; any change requires a new untouched route | P6/P7 |
| R-016 | Public release exposes data, cache, symlink/gitlink, secret, chat, or binary | ignore rules; exact staged blob/mode scanner; dedicated history/secret/license scan; human review | every commit/release |
| R-017 | License, institutional IP, patent intent, or dataset terms prohibit publication | keep public release blocked until owner/institution/legal decisions and official dataset review | before public GitHub/P2 data use |
| R-018 | Guard naming/content heuristics miss semantic leakage | treat scanner as defense-in-depth; require provenance, controlled roots, and independent review | every phase |
| R-019 | Deployable-only selection is gamed by teacher copying, covariance inflation, or GNSS rejection | keep the first model mean-only; use direct copying/zero-update tests and a small set of deployable selection checks | P4/P6 |
| R-020 | A supervised or reference-ranked oracle contaminates development | prohibit high-precision oracle training and reference-ranked checkpoints; preserve the sealed final-evaluation rule | every phase |
| R-021 | Mathematical equivariance is confused with physical sensor-remounting generalization | require the coordinate property test; claim mounting generalization only when suitable real or validated simulated data exist | P5/P6 |
| R-022 | Optional architecture work delays the primary loose-coupled question | tight coupling, full covariance, extra estimators, and large ablation grids require a separate controller scope decision | every phase |
| R-023 | A later Phase's unknown data or compute choices are frozen prematurely | each Phase owns its data-dependent constants and records them before the relevant experiment, without reference-error influence | P1–P6 |
