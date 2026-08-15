# Risk register

The frozen method boundary and mandatory tests are defined in `FROZEN_RESEARCH_CHARTER.md`.

| ID | Risk | Direct test/mitigation | Gate |
|---|---|---|---|
| R-001 | Direct or derived high-precision/data-split leakage influences development | physical separation; provenance/lineage manifests; guard tests; independent split and feature audit | every phase; mandatory P2/P4/P6 |
| R-002 | DrvFS case/permission semantics cause ambiguity or exposure | lowercase unique names; no secrets/reference on `/mnt/e`; WSL checks | P0/P1 |
| R-003 | `/mnt/e` becomes a training I/O bottleneck | keep source only there; representative benchmark on approved external WSL-native roots; 8 MiB probe is not a training benchmark | P2 before data runs |
| R-004 | Scientific dependencies are not reproducible | resolver ADR and generated hash lock before third-party scientific imports | P1 gate |
| R-005 | WLS/SPP is not independent, conventional, or stable | first-party method spec; standardized outputs; synthetic/golden and cross-implementation checks | P1/P2 gate |
| R-006 | Model simply copies SPP | compare residual/output to SPP; counterfactual perturbation and GNSS-degradation tests | mandatory P4/P6 |
| R-007 | Model inflates covariance to evade residual penalties | proper scoring/calibration, sharpness and coverage tests; cap/ablation checks | mandatory P3/P4/P6 |
| R-008 | Model rejects every GNSS update | update-acceptance distribution; forced-valid-GNSS cases; SPP-information ablation | mandatory P4/P6 |
| R-009 | `Q`, `R`, and bias compensate and become unidentifiable | staged identification order; hold-two-vary-one tests; parameter recovery on synthetic fixtures | mandatory P4/P6 |
| R-010 | Route/receiver/time identity is memorized | route-level and cross-receiver holdouts; metadata ablation; nearest-route checks | mandatory P2/P4/P6 |
| R-011 | Constant common SPP bias is claimed recoverable without absolute information | observability analysis and constant-bias counterexample; state limitation explicitly | P1 gate and P4/P6 tests |
| R-012 | PINN duplicates the ESKF mechanization identity | independent residual derivation with units/assumptions; dependency graph and ablation | P1 specification; P4 gate |
| R-013 | Symmetry group mishandles gravity or reflection/pseudovector behavior | `SO(2)` first; transformation tests; `O(2)` only with correct angular-velocity reflection | P1/P5 gate |
| R-014 | GNSS Transformer adds capacity rather than stable causal value | matched-capacity rule/TCN/GRU/causal-Transformer comparison across seeds/domains | P6 gate |
| R-015 | Final route is invalidated after viewing | immutable freeze and one-time access log; any change requires a new untouched route | P6/P7 |
| R-016 | Public release exposes data, cache, symlink/gitlink, secret, chat, or binary | ignore rules; exact staged blob/mode scanner; dedicated history/secret/license scan; human review | every commit/release |
| R-017 | License, institutional IP, patent intent, or dataset terms prohibit publication | keep public release blocked until owner/institution/legal decisions and official dataset review | before public GitHub/P2 data use |
| R-018 | Guard naming/content heuristics miss semantic leakage | treat scanner as defense-in-depth; require provenance, controlled roots, and independent review | every phase |
