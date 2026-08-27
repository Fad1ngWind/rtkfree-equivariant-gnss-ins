# Phase 4 controller review report

Date: 2026-08-27 (Asia/Shanghai)

Executor status: **reviewed**

Formal phase status: **ACCEPTED by the controller**

## Scope assessment

Phase 4 implements only one ordinary non-equivariant causal GRU student and two matched training configurations, without and with the independent inertial loss. It consumes only approved deployable Medium PVT/IMU fields and Phase 3 ESKF state weak pseudo-labels. It does not implement equivariant layers, rotation augmentation, a GNSS Transformer, tight coupling, learned bias/Q/R, a covariance head, or Phase 5/6 infrastructure.

Deployment inference is the recursively rolled student posterior. The teacher is absent from the model input and inference path. Current GNSS becomes visible only after the free pre-GNSS state and its independent IMU reference have been formed.

## Gate evidence

- Synthetic tests establish pre/post-GNSS isolation, future-prefix causality, recursive masked behavior, `SO(3)` attitude error, mechanization agreement, independent-reference construction, and differentiability through the free student prediction.
- The fixed interface and config prohibit teacher covariance, teacher state input, absolute time, route/file/device identity, planned outage duration, and current GNSS in the pre branch.
- Chronological guard-separated splits and train-only robust normalization are frozen before the bounded run.
- Both variants share seed, initial tensor hash, architecture, data, masks, optimizer, minibatch order, 40-step budget, and weak-label validation selection. Physics weight is the only difference.
- Two independent smoke runs and two independent development runs reproduce exactly. The final verifier matches nine byte/tensor artifacts and rejects a modified-artifact fixture.
- Direct diagnostics reject the simplest degenerate explanations: exact teacher copying, exact SPP copying, all-zero correction, all-GNSS rejection, future leakage, direct coordinate memory, and nonfinite/excess-speed output.
- Both 20/30-second masks use the Phase 3 half-open timing basis, hide the exact event count, leave the earlier prefix bit-identical, and expose GNSS at recovery only.
- The complete 96-test locked-environment suite and public-release guard pass.

These facts support implementation correctness, causality, bounded numerical behavior, and reproducibility. They do not support navigation accuracy.

## Matched comparison and required judgment

The no-physics validation weak-label composite improves from `3.774408738` to `3.317065557`; physics improves from the same initialization value to `3.265618006`. This is a weak-label objective only.

On the independent diagnostic split, normalized physics Huber is `0.374354878` for no-physics and `0.418493815` for physics. The physics-trained model is worse on this diagnostic. The repeated same-start pure-IMU bridge also separates strongly from both students over 20/30 seconds because it accelerates to about 94–156 m/s under fixed-zero-bias propagation from the student attitude. The students remain finite and bounded, but the evidence does not show that the physics penalty improved physical consistency.

The controller must therefore judge Phase 4 as an implementation/fair-comparison gate with an honestly negative physics-effect result, not as evidence that PINN regularization works. Phase 5 must not cite this run as a demonstrated physics benefit. Any later change of physics scaling, bias model, training budget, or architecture would be a separately predeclared experiment, not retroactive Phase 4 tuning.

## Failed-attempt review

The first pre-GNSS velocity design used an unbounded additive increment and reached approximately 180 m/s, failing the fixed 60 m/s no-reference diagnostic despite decreasing training loss. It was replaced with a bounded direct velocity mean, then smoke, development, and independent repetition were rerun from the fixed seed. This is the only stability-driven architecture correction.

A separate float32 translation audit initially failed because repeated large-coordinate arithmetic accumulated millimetric rounding. A float64 structural audit with fixed `1e-9` tolerance replaced it; the failed report remains external. Neither correction used a high-precision trajectory or changed the teacher.

## Teacher and claim boundary

The Phase 3 forward ESKF remains a weak pseudo-label source with frozen elevated NIS and a large recovery update after ten naturally invalid SPP epochs. Teacher covariance is not used for normalization, weighting, selection, diagnostics, or model input. Q/R, gating, and teacher outputs were not retuned.

No high-precision reference or sealed holdout was accessed. Consequently, this report makes no accuracy, superiority, route-generalization, teacher-quality, or uncertainty claim. External data, weights, logs, and diagnostics are not Git candidates.

## Controller disposition

**Accepted on 2026-08-27.** The controller independently reran the locked health check (96 tests), public-release guard, bounded Medium development training, direct diagnostics, and stable-run comparison. The new controller run reproduced all nine final byte/tensor artifacts exactly, including both checkpoint tensor hashes and the negative diagnostic result.

Acceptance is deliberately narrow: Phase 4 establishes a reproducible ordinary causal baseline and a valid matched mechanism for adding an independent inertial residual. It does not establish positioning accuracy or a positive physics effect. Phase 5 may build on the interface and comparison protocol, but it must retain the no-physics baseline and may not cite this Phase 4 run as evidence that PINN regularization helps. A changed physics loss or training design must be declared and evaluated as a new experiment.
