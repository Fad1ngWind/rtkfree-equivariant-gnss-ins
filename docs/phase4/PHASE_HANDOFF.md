# PHASE_HANDOFF — Phase 4 ordinary physics-informed baseline

## Stage objective

Build the first ordinary, non-equivariant, loose-coupled causal student using the frozen Phase 3 forward ESKF states only as weak pseudo-labels. Compare one matched student without and with an independent inertial residual. The deployed output must be the recursive student position, velocity, and attitude, not the ESKF.

## Completed implementation

- Added one 32-hidden-unit ordinary GRU. It recursively consumes the previous student posterior and arrived IMU summary, freely emits a pre-GNSS position increment, bounded velocity mean, and attitude increment, then optionally applies a current-GNSS innovation correction.
- Kept current GNSS, teacher state/covariance, absolute time, and route/file/device identity out of the pre-GNSS branch. Invalid or controlled-masked GNSS produces no post correction.
- Added an independent reference `Phi(x^+_{theta,k-1}, IMU[k-1:k])` using the Phase 3 deterministic mechanization, fixed gravity/Earth rotation, and fixed-zero bias. The residual compares this reference with the student's free pre-GNSS state; it is not a network correction identity.
- Used NED position/velocity differences and `log(R_reference.T @ R_prediction)` for attitude error. Synthetic fixtures match the frozen Phase 3 mechanization and verify the small-angle branch and gradient boundary.
- Added fixed chronological train/guard/validation/guard/diagnostic splits, train-only robust normalization, timing-gap exclusion for physics loss, and the Phase 3 left-closed/right-open 20/30-second masks.
- Implemented exactly two configurations: physics weight `0.0` and `0.1`. Seed, initialization, data, architecture, masks, optimizer, step budget, validation rule, and every other setting are identical.
- Added repository-external smoke/development entries, direct no-reference diagnostics, and an artifact verifier. Model weights, normalizers, logs, data, and reports remain outside Git.

## Frozen development profile

- Profile: `ordinary_causal_student_v1`; seed `3407`; CPU-only PyTorch `2.13.0+cpu`.
- Records: 764 causal 1 Hz steps, 731 valid PVT records, 400–402 arrived IMU subsegments per step, and 27 timing-gap steps.
- Splits: train `[0,458)`, guard `[458,488)`, validation `[488,626)`, guard `[626,656)`, diagnostic `[656,764)`.
- Optimization: AdamW, learning rate `1e-3`, weight decay `1e-5`, gradient clip `1`, unroll 64, smoke 4 steps, bounded development 40 steps.
- Training masks cycle through `[0,20,30,0]` seconds. Benchmark masks begin 120 seconds after Phase 3 initialization.
- Checkpoint selection uses only the fixed validation weak-label mean-state objective and finite-value rules. Teacher covariance is never used.

## Matched smoke and development evidence

Two independent stable smoke runs are byte-identical at the summary level (`SHA-256 677d4055693fbbe53359f3a042caa482f98073720f98e3bca68c17cce3317f0c`) and share initial model tensor hash `d9ae040799f2cb16dcbaf6b163742203f54dd0e2ccde14e17b89b6656550c227`.

| Variant | Physics weight | Smoke weak loss before | Smoke weak loss after | Finite |
|---|---:|---:|---:|---|
| no-physics | 0.0 | 3.254259348 | 2.643389463 | yes |
| physics | 0.1 | 3.254259348 | 2.635092497 | yes |

Two independent stable development runs are identical for nine byte/tensor artifacts. The summary hash is `871a6757db0eeb551d6481be63448e15e1297c45f2e28f23a0cecb892ef193f9`; the detailed diagnostic hash is `31388c0724dae7821935d80ff3e24938c4355d5c321122719af6c3aad2bd447b`.

| Variant | Initial validation composite | Best step | Best validation composite | Best model tensor SHA-256 |
|---|---:|---:|---:|---|
| no-physics | 3.774408738 | 40 | 3.317065557 | `a33ccb0fadc1369fc8bdf8732c95e933f841f13ba0f62dae894dd43c101b3079` |
| physics | 3.774408738 | 40 | 3.265618006 | `b60d8d4d9c75dfb97384dbf147d5878d097232ba3ab38a0defda9f06e0518638` |

This weak-label validation difference is not a position-accuracy result. More importantly, the independent diagnostic physics Huber is `0.374354878` for no-physics and `0.418493815` for physics. The physics configuration therefore shows no positive physics-residual benefit in this bounded experiment. Phase 4 records that negative result and does not select or promote a scientific winner.

## Direct degeneracy and outage diagnostics

- Both variants report: not an exact teacher copy, not an exact SPP copy, not all-zero GNSS corrections, and not identical to the reject-all-GNSS rollout.
- Full versus truncated future prefixes are bit-identical. A float64 coordinate-translation structural audit passes at `1e-9`; it supports absence of direct absolute-coordinate memorization in the tested implementation, not route generalization.
- All checked states are finite with proper rotations. Maximum full-rollout speed is approximately `15.48` m/s for no-physics and `15.15` m/s for physics, below the pre-fixed 60 m/s numerical diagnostic.
- The 20/30-second masks hide exactly 20/30 valid PVT epochs, preserve the entire pre-mask prefix bit-for-bit, and restore GNSS only at the recovery timestamp.
- Same-start endpoint diagnostics separate the student from the repeated pure-IMU bridge:

| Variant | Mask | Student displacement | Pure-IMU displacement | Student endpoint speed |
|---|---:|---:|---:|---:|
| no-physics | 20 s | 69.31 m | 1082.01 m | 4.35 m/s |
| no-physics | 30 s | 116.47 m | 2381.84 m | 4.49 m/s |
| physics | 20 s | 70.32 m | 972.40 m | 4.57 m/s |
| physics | 30 s | 115.47 m | 2150.32 m | 4.76 m/s |

The large student-versus-pure-IMU separation is mainly caused by the fixed-zero-bias repeated pure-IMU bridge reaching about 94–156 m/s from the student starting attitude. It is not student numerical explosion. It also shows that the physics-trained student did not become a multi-step pure-INS copy. Without reference truth these figures establish causal, finite behavior only; they do not establish which trajectory is accurate.

## Failed attempts retained

- The initial additive unbounded velocity-increment form lowered weak-label loss but reached about `179.93` and `173.05` m/s, failing the pre-fixed 60 m/s diagnostic. It is superseded, not hidden. The minimal correction changed the pre-GNSS velocity head to a bounded direct mean output and reran both matched configurations from the same fixed seed.
- The first stable translation audit accumulated float32 millimetric roundoff under a large coordinate shift and false-failed an exact structural question. Its report is retained externally as `diagnostics_failed_float32_translation.json`. The audit was moved to float64 with a fixed `1e-9` tolerance; no model, data, loss, or checkpoint changed.
- Superseded smoke/development directories are not final evidence and are not used for model selection.

## Teacher and information limitations

- The frozen Phase 3 ESKF is a weak pseudo-label teacher, never truth. Its elevated NIS mismatch is retained: 352/731 updates exceed the 95% three-degree-of-freedom threshold and 288/731 exceed the 99% threshold.
- Its approximately 324.85 m recovery correction after ten consecutive naturally invalid SPP epochs is also retained. No Q/R change, additional gate, covariance weighting, or label-beautification replay was performed.
- No high-precision reference, RTK/PPK result, post-processed trajectory, receiver-native truth, or Deep sealed holdout was downloaded, opened, parsed, normalized against, selected against, or used for debugging.
- Only the approved Medium deployable PVT/IMU/extrinsic and frozen teacher state file were read. Development on one session cannot support route, device, or session generalization.
- No accuracy, superiority, uncertainty-calibration, teacher-quality, or paper-performance claim is supported at this gate.

## Reproducibility and verification

- External stable smoke runs: `~/rtkfree-runs/phase4/smoke_stable_run1_20260827` and `smoke_stable_user_run_20260827`.
- External stable development runs: `~/rtkfree-runs/phase4/development_stable_run1_20260827` and `development_stable_user_run_20260827`.
- `scripts/phase4/verify_medium_development.py` validates summaries, train-only normalizer, logs, checkpoint tensor hashes, information-boundary flags, direct degeneracy checks, detailed masks, and deterministic comparison without reading deployable data or a reference trajectory.
- Complete locked-environment health check: 96 tests pass with warnings as errors; public-release guard passes; `git diff --check` is clean.
- No commit, push, PR, merge, or research-notebook edit was performed.

## Controller review checklist

1. Inspect the config and student API for exactly one ordinary causal GRU, recursive state output, and pre/post-GNSS separation.
2. Rerun the synthetic state, student, masking, loss-fairness, dependency, and artifact-tamper tests.
3. Confirm that both training variants differ only in physics weight and share the recorded initialization hash.
4. Run the artifact verifier on both stable development directories and confirm all nine hashes match.
5. Independently review the failed speed diagnostic, the bounded correction, the pure-IMU outage behavior, and the negative independent-physics result.
6. Confirm the teacher is described only as a weak pseudo-label and its NIS/recovery limitations are retained without covariance weighting or Q/R changes.
7. Rerun the full health check, public-release guard, and repository diff review; confirm weights/logs/data remain external and no sealed reference was used.

## Executor state

**READY_FOR_CONTROLLER_REVIEW**

## Controller outcome

**ACCEPTED on 2026-08-27.**

The controller independently reran the locked 96-test health check, public-release guard, bounded Medium development training, direct diagnostics, and artifact comparison. The controller run reproduced all nine final byte/tensor artifacts exactly. Phase 4 is accepted as an implementation, causality, stability, and fair-comparison gate only.

The matched run does **not** demonstrate a beneficial physics effect: the independent diagnostic physics Huber is worse with the physics term (`0.418493815`) than without it (`0.374354878`). This result, the superseded approximately 180 m/s design, and the weak-teacher limitations remain part of the accepted evidence. They must not be converted into an accuracy, PINN-benefit, or superiority claim. Any later change to the physics formulation, weighting, bias treatment, budget, or architecture is a new predeclared experiment rather than a retroactive Phase 4 adjustment.
