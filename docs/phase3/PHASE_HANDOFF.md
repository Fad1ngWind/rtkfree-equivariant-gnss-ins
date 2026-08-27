# PHASE_HANDOFF — Phase 3 deterministic navigation baselines

## Stage objective

Produce reproducible SPP-only, pure-INS, and fixed real-time forward loose-coupled ESKF baselines from the Phase 2 frozen standardized PVT/IMU and selected calibration boundary. The fixed ESKF is intended for controller review as the Phase 4 weak-label teacher and conventional baseline; no learned method is part of this phase.

## Completed

- Kept SPP-only as a timestamp-preserving pass-through of all 786 valid and invalid frozen PVT records. It is not treated as truth.
- Established a per-sequence fixed local NED frame and a causal 20-second initialization. Horizontal velocity and yaw come from a PVT line fit available only at the window endpoint; fitted vertical PVT velocity is diagnostic while nominal vertical velocity is fixed to zero with larger uncertainty.
- Added fixed-local-NED INS propagation with midpoint attitude integration, specific-force/gravity composition, Earth-rotation compensation, Coriolis compensation, accelerometer bias, and gyroscope bias.
- Performed the required pre-use transform/sign test. The selected record maps IMU coordinates into antenna coordinates; its inverse gives the IMU-frame antenna lever `[0, -0.86, 0.31]` m, which is rotated before antenna/IMU position composition.
- Added a 15-state true-minus-nominal ESKF with state order position, velocity, right attitude error, accelerometer bias, and gyroscope bias. Covariance uses a second-order transition, PSD process-noise quadrature, a Joseph position update, and an attitude reset Jacobian.
- Added a finite-difference test under physical nonzero Earth rotation. It confirmed that right-attitude `F[att,att]` uses the bias-corrected inertial body rate; the earlier caller incorrectly subtracted `R.T * omega_ie_n` a second time and was corrected without changing nominal mechanization.
- Added causal zero-order-held IMU replay, valid PVT position updates, innovation/NIS diagnostics, and left-closed/right-open 20/30-second GNSS masks.
- Added one SPP-only entry, one INS-only entry, one fixed-ESKF entry, both controlled-outage entries, and a structural/determinism verifier.
- Did not implement the optional adaptive ESKF because it is not required for the fixed-baseline gate.

## Fixed profile

- Profile: `fixed_forward_eskf_v1`.
- Input hashes: PVT `c0ee6282191a3fc08676e6be32554a9bf3ef0c25b35c22f097e7b93a1b5d0723`; IMU `c2538e409091452c549c7bd380a6865e125b1adc4fdbd5e3c579dd3ad0c8e516`; official `xsens_imu_param.yaml` `55da278a30572015d2ab4024ae1dfc11abb1fb54f252498ce318152c6d39ecd5`; selected extrinsic `873d9da5c7957ba0fd61e903275ea5f281dde3baade9517c0555ad1b5ff3f693`.
- Position measurement covariance: rotated standardized PVT covariance without a gate or adaptive rescaling. Accepted floating-point asymmetry is canonicalized once at the PVT input boundary.
- Process noise uses the controller-preselected official `avg-axis` scalars without choosing an axis or changing a value after replay: `acc_n=1.1197412605492375e-02`, `gyr_n=1.0270904839480961e-02`, `acc_w=1.1751767903346351e-04`, and `gyr_w=9.1355383994881894e-05`.
- Continuous-time interpretation: `acc_n` is accelerometer white-noise density in `m/s^2/sqrt(Hz)` and `gyr_n` is gyroscope white-noise density in `rad/s/sqrt(Hz)`; `acc_w` and `gyr_w` drive bias random walks in `(m/s^2)/sqrt(s)` and `(rad/s)/sqrt(s)`, respectively. The source YAML explicitly provides the parent units while this stochastic dimensional interpretation defines the fixed Phase 3 SDE.
- The continuous diagonal is `Qc=diag(acc_n^2 I3, gyr_n^2 I3, acc_w^2 I3, gyr_w^2 I3)`. With `C=L Qc L.T`, the implemented interval mapping is `Qd=0.5*(C + Phi C Phi.T)*dt`, alongside the second-order `Phi`. The four official densities are not rescaled by the sample interval before this integration.
- Initial standard deviations: velocity `[2, 2, 5]` m/s; attitude `[10, 10, 30]` degrees; accelerometer bias `[0.1, 0.1, 0.1]` m/s2; gyroscope bias `[0.02, 0.02, 0.02]` rad/s; position from rotated PVT covariance.
- Controlled outages start 120 seconds after initialization and last exactly 20 or 30 seconds.

## Deployable replay evidence

Two independent post-rework runs, `medium_fixed_v1_rework_run1` and `medium_fixed_v1_rework_run2` under `~/rtkfree-runs/phase3/`, are byte-identical. The output hashes in both runs are:

| Output | SHA-256 |
|---|---|
| `spp_only.jsonl` | `89bbe4d62b96f2692c102f224f1bc2a5d41e98444ab18c1c251561e99de352cd` |
| `ins_only.jsonl` | `462228342541f0fc9f4cc77b30c31cec2db1c68b204508caf57293a559674126` |
| `fixed_eskf.jsonl` | `949d8ab68c1db1b29e3285d55da38c777f95465976f85d143751748c59d6b37d` |
| `fixed_eskf_outage_20s.jsonl` | `2ef636bdb4fc862b90ddd2cc31505e2c0d2ce3cd69feb38e0742091310027b8e` |
| `fixed_eskf_outage_30s.jsonl` | `a335a09e808715db168982f4cd6e18e957997cb83aa74be4e00a5ce7f8360b60` |

Initialization is valid at UTC-nanosecond timestamp `1621218796006000000`. Each navigation scenario contains 764 causal post-initialization output records. The fixed ESKF applies 731 valid PVT updates; INS-only applies zero. The 20-second scenario masks 20 valid PVT epochs and applies 711 updates, while the 30-second scenario masks 30 and applies 701. Their first masked timestamp is `1621218916006000000`; updates recover at `1621218936006000000` and `1621218946006000000`, respectively. Twenty-seven IMU timing-gap flags occur after initialization.

All checked states and covariances are finite. Covariances are exactly symmetric in serialized output and the minimum observed eigenvalue across all scenarios is `9.409015813792121e-07`. Fixed-ESKF speed remains finite with an observed maximum norm of about `45.51` m/s. The largest position-update correction norm is about `324.85` m. It occurs when a valid SPP epoch follows ten consecutive naturally invalid `insufficient_c1c_observations` epochs, with an innovation of approximately `[-179.81, 302.97, 53.22]` m. Time remains monotonic, no controlled mask applies, and covariance remains PSD. These are numerical/model-mismatch diagnostics, not accuracy measurements.

The fixed ESKF records 731 NIS values: minimum `0.2592614608445286`, median `7.258662982605188`, and maximum `305.1263011258299`. For three degrees of freedom, 352/731 (`0.48153214774281805`) exceed the fixed 95% chi-square threshold `7.814727903251179`, and 288/731 (`0.39398084815321477`) exceed the 99% threshold `11.344866730144373`. Innovation component RMS is approximately `[14.07, 31.45, 69.55]` m in NED, with the largest component in the vertical direction. This elevated NIS and the large natural-gap recovery correction are retained as evidence that the fixed model, SPP covariance, and non-Gaussian urban errors are not statistically well matched at many epochs. They are not hidden with gating, R scaling, adaptation, or post hoc noise tuning, and cannot establish position accuracy because no truth is used.

The earlier `medium_fixed_v1_run1` and `medium_fixed_v1_run2` outputs used the controller-rejected process-noise basis and are superseded. They are not Phase 3 freeze evidence.

## Information and scope boundary

- Only the Phase 2 frozen standardized PVT, standardized IMU, and selected extrinsic are read by the replay entry.
- No high-precision reference, RTK/PPK solution, post-processed trajectory, receiver-native PVT truth, or sealed holdout content was downloaded, opened, parsed, selected against, debugged against, or validated against in Phase 3.
- No PINN, learned model, equivariant network, tight coupling, training, hyperparameter search, covariance head, or GNSS Transformer was implemented.
- External data and run outputs remain outside the repository and are not Git candidates.
- No accuracy, superiority, teacher-quality, or paper-performance claim is supported at this gate.

## Verification

- Phase 3 focused suite: 29 tests pass with warnings treated as errors.
- Complete repository suite: 71 tests pass with warnings treated as errors.
- Two independent real deployable replays: byte-identical.
- Structural verifier: passes finite-state, strict-time, rotation, covariance, mask, count, hash, and deterministic-comparison checks; a one-byte tampering fixture is rejected.
- Locked-environment command: `source ~/rtkfree-venvs/phase3/bin/activate && bash scripts/healthcheck.sh`; WSL Python 3.12 and NumPy 2.5.2 check: pass.
- Public-release guard: pass.
- `git diff --check`: pass.
- No commit, push, PR, merge, or local research-notebook edit was performed.

## Controller review checklist

1. Recheck the fixed config, dependency lock, four official average-axis noise values, and exact four Phase 2 input hashes including `xsens_imu_param.yaml`.
2. Rerun the transform/sign, mechanization, right-attitude finite-difference, ESKF, causal scheduling, and full synthetic replay tests.
3. Run two new Medium replays into new external directories and compare every recorded output hash.
4. Run `verify_medium_baselines.py` and independently inspect time causality, mask endpoints, finite values, rotations, covariance symmetry/PSD, innovations, and NIS.
5. Review the elevated NIS exceedance rates and large natural-gap recovery correction as documented fixed-baseline limitations. Do not use a high-precision reference or post-result noise change for that decision.
6. Rerun the complete suite, health check, release guard, and repository diff review; confirm that data and outputs remain external.
7. Confirm the absence of adaptive ESKF work and all Phase 4+ implementation.

## Executor state

**READY_FOR_CONTROLLER_REVIEW**

## Controller outcome

**ACCEPTED — 2026-08-27**

The controller independently reran the complete Medium baseline into a new external directory and reproduced all five recorded output hashes. The official IMU-parameter hash, locked-environment health check, all 71 tests, structural validator, causality, covariance checks, public-release guard, and repository diff review passed without high-precision reference use.

The elevated NIS and the large recovery update after ten naturally invalid SPP epochs are frozen as explicit limitations of this conventional weak-label teacher rather than hidden through post-result tuning. Phase 4 must treat its states as weak pseudo-labels, must not use the current teacher covariance for loss weighting, and must retain robust label handling without describing the teacher as truth.
