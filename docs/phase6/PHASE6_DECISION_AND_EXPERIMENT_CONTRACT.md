# Phase 6 decision and experiment contract

Status: **implemented and accepted by the controller on 2026-08-29**.

This short contract records the Phase 6 decisions before any new result. It does not implement a model, open new data, or authorize training. `docs/governance/EXPERIMENT_REGISTRY.md` owns the executable selection details.

## Method decisions

- The deployment primary remains one causal loose-coupled position/velocity/attitude mean-state student. No Transformer, tight coupling, learned Q/R/bias, O(2)/SO(3), new filter, or complex GNSS time-series branch enters Phase 6.
- MC-001 is resolved: predictive covariance is deferred. Phase 6 adds no covariance head, covariance loss, calibration metric, or uncertainty claim.
- Deployment selection and scientific hypotheses use a dual-track freeze. The deployment primary is chosen prospectively from RTK-free evidence. Physics and strict SO(2) remain Phase 7 confirmatory arms even when they are not the deployment primary.
- Phase 4/5 negative and mixed evidence remains unchanged: physics did not improve independent residual, strict SO(2) did not improve the one-seed weak-label validation, and the float32 complete-rollout tolerance failed by about 0.30–0.51 mm.

## Minimal experiment sequence

1. Run the three no-physics structures once for each of the three fixed seeds in `P6-STRUCTURE-001`, using 200 optimizer steps and checkpoint cadence 10. All other accepted Phase 5 training settings remain unchanged. The 200-step budget is fixed before Phase 6 results and cannot be extended afterward.
2. Apply qualification/safety gates, then select structure only by the frozen common-validation rule. Physics Huber remains an independent secondary diagnostic, not a structure-selection gate. If support is unstable, select ordinary by simplicity and stop structural tuning.
3. Run the controller-approved minimum 2x2 confirmatory comparison: `structure={ordinary, strict SO(2)}` x `physics_weight={0.0,0.1}`. Reuse ordinary/no-physics and strict-SO(2)/no-physics from step 1 without retraining; add only ordinary/physics and strict-SO(2)/physics for the same three seeds, six new runs total. This comparison reports interaction direction only and does not participate in structure selection.
4. Apply `P6-PHYSICS-001` to the selected deployment structure. If ordinary or strict SO(2) was selected, reuse its 2x2 physics runs. Only if rotation-augmented was selected, run its matched `physics_weight=0.1` arm for the same three seeds; do not extend rotation augmentation into the factorial.
5. Freeze the deployment primary. Its designated deployable checkpoint is seed `163736869`; reproduce that one configuration and seed exactly once. Do not repeat the multi-seed matrix.

The statistical unit is the fixed seed. Validation and physics-consistency values are RTK-free proxy signals, not truth-referenced accuracy. Failed seeds remain failures and are never replaced.

## Data, degradation, and claim boundary

- Medium remains the sole development session. Deep and every high-precision reference remain unopened throughout Phase 6.
- Existing 20/30-second controlled outages and one prospectively defined GNSS-available robustness stratum are the entire degradation set. The `deployable-metadata low-quality stratum` is the union of the worst diagnostic-split quartile by deployable HDOP and the worst quartile by deployable SPP postfit residual RMS: 53 of 108 valid-PVT epochs under the frozen Medium file hash. The union is not itself called a worst quartile, does not enter model selection, and is not reference-error truth or a complete urban-severity definition.
- Phase 6 adds no receiver or route data. The already viewed Medium archive central-directory names do not establish time alignment or receiver-specific calibration; no candidate receiver member may be extracted or read. Cross-receiver and cross-route generalization claims are deleted.
- Coordinate SO(2) equivariance is not physical sensor-remounting generalization. No suitable remounting evidence exists, so the remounting claim is deleted.
- Receiver-native PVT, teacher covariance, future data, route/device identity, and reference-derived choices are prohibited.

## Phase 7 seal

Phase 6 will freeze method code, configs, data roles, checkpoints, seeds, metrics, statistical unit, failures, and allowed claims before Deep or reference access. Phase 7 may report whether reference ranking agrees with RTK-free selection, but it cannot replace the deployment primary or tune any arm after reference access.

## Pre-run implementation

- Frozen config: `config/phase6/final_freeze_v1.json`, SHA-256 `7d6ec6c2995c63d31a57544d2b9baff59d586807dd5fbddb66772d66fda79de0`.
- Single main entry: `scripts/phase6/run_medium_freeze.py`.
- Read-only validation entry: `scripts/phase6/verify_medium_freeze.py`.
- The complete locked health check passes 162 tests with warnings as errors, and the public-release guard passes.

There is no remaining controller research decision in this contract. The frozen workflow was executed once, the read-only verifier accepted 81 artifacts, and only the designated final configuration was reproduced once. Results and limitations are recorded in the Phase 6 review and handoff documents.

No commit, push, PR, merge, or `科研手记.md` modification is authorized. Final Phase 6 documents may use only **READY_FOR_CONTROLLER_REVIEW** until independent controller acceptance.
