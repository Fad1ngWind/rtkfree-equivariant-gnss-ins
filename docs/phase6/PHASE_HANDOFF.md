# Phase 6 final-freeze handoff

Status: **READY_FOR_CONTROLLER_REVIEW**. The executor package is ready for independent review; Phase 6 is not formally accepted.

## Frozen method

The deployment primary is the accepted ordinary causal loose-coupled mean-state student with `physics_weight=0.1`. It produces recursive position, velocity, and attitude means and has no predictive covariance head. Its designated seed is `163736869`; the external checkpoint tensor SHA-256 is `bf4fc42ca6eb5b8a7acb90c7e3d59eda6b65e27868e8e74cec257ef00dc8fb44`.

The designated checkpoint's `best_optimizer_step` is `200`, exactly the frozen training-budget boundary. Applying the same 200-step budget to every arm and seed preserves the prospective comparison's fairness, and the budget cannot be extended after observing results. The boundary checkpoint does not demonstrate that optimization had fully converged; Phase 7 must interpret every frozen result as a **bounded-training freeze**.

This choice follows the pre-result RTK-free rule, not truth-referenced error. Rotation augmentation had the lowest three-seed median weak-label validation (`1.498722831`) but won only one seed. Ordinary won two seeds, so the required stability condition failed and simplicity selected ordinary. Strict SO(2) had median `1.758293748` and no seed wins.

For the selected ordinary structure, physics `0.1` met the separate rule: median diagnostic physics Huber changed from `0.117669680` to `0.105250198`, two of three paired seeds improved, and median common validation changed from `1.563463370` to `1.546205799`. One physics-Huber seed worsened. Phase 4's negative physics result remains valid, and Phase 6 does not establish truth-referenced physics benefit.

## Confirmatory arms

Phase 7 must retain ordinary/no-physics, ordinary/physics, strict-SO2/no-physics, and strict-SO2/physics using the checkpoints fixed by the external summary hash. Rotation-augmented/no-physics remains the Phase 5/6 empirical control. Because rotation augmentation was not selected, no rotation-physics arm was run.

The 2x2 interaction directions are mixed: common validation difference-of-differences median `-0.103235364` with two negative directions, while physics-Huber difference-of-differences median `+0.019837602` with all three directions positive. This does not support a Phase 6 combined strict-SO2+physics benefit claim. Phase 7 reports all confirmatory outcomes whether positive or negative and cannot use them to replace the deployment primary.

All strict checkpoints retained the Phase 5 classification: complete float64 recursive structural tests passed, while the preregistered float32 deployed-rollout tolerance failed. Coordinate equivariance is not sensor-remounting generalization.

## Robustness and limitations

- Every required arm/seed passed causality, finite/proper-rotation, degeneracy, mask/recovery, information-boundary, 60 m/s, and deployable-metadata low-quality gates.
- The designated checkpoint reached maximum checked speed `15.557231903 m/s`. Its 20/30-second masks hid exactly 20/30 valid epochs, preserved the prefix bit-for-bit, remained finite, and restored GNSS at recovery.
- The 53/108 low-quality union remained finite and used GNSS. Its weak-label Huber `1.547270674` and physics Huber `0.106839470` are proxies only.
- Medium is the only development session. Cross-route, cross-receiver, device, and remounting claims are deleted. Predictive covariance is future work.
- There is no truth-referenced accuracy, superiority, uncertainty-calibration, teacher-quality, or generalization conclusion at this gate.

## Reproducibility and artifacts

- Frozen config: `config/phase6/final_freeze_v1.json`, SHA-256 `7d6ec6c2995c63d31a57544d2b9baff59d586807dd5fbddb66772d66fda79de0`.
- External run: `~/rtkfree-runs/phase6/medium_final_freeze_user_20260828`; summary SHA-256 `54dec97a895772df171c31d2dcee9de178853c31158c9f620e27bb8750112718`.
- Fifteen required training runs were executed once. No failed or substituted seed exists.
- The final ordinary/physics seed-`163736869` run was reproduced once. Checkpoint file/tensor and training-log hashes match exactly.
- The read-only verifier accepted 81 artifacts. The complete locked suite passes 162 tests and the public-release guard.

## Phase 7 access rule

Deep and every high-precision reference remain sealed. `PHASE7_SEALED_EVALUATION_PROTOCOL.md` freezes the one-time interface, two endpoints, deterministic outage targets, support rules, statistical units, `0 m` directional comparison threshold, failure handling, and pre-access checklist. That threshold classifies only within-session descriptive error direction; it is not a practical-significance margin and cannot support route generalization or broad superiority. Before first access, the controller must verify that protocol, `FREEZE_MANIFEST.json`, the external summary hash, all checkpoint hashes, the absence of a failure artifact, and the exact reference-bearing command. Once any Deep or reference byte is opened, no method, parser semantics, code, config, seed, checkpoint, metric, statistical unit, arm role, or claim-selection rule may change. Reference ranking may be reported but cannot select a new primary model.

No commit, push, PR, merge, or `科研手记.md` modification was performed.

## Controller outcome

**ACCEPTED on 2026-08-29.**

The controller independently reproduced the 162-test health check, public-release guard, 81-artifact read-only verification, decision calculations, repository and external hashes, and all strict float64/float32 classifications. Acceptance freezes ordinary plus `physics_weight=0.1` as the deployment primary without claiming truth-referenced accuracy, convergence, practical significance, route generalization, or a general physics/equivariance benefit.
