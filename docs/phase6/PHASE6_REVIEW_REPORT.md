# Phase 6 executor review report

Date: 2026-08-29 (Asia/Shanghai)

Status: **ACCEPTED by the controller on 2026-08-29**

## Scope and contract fidelity

Phase 6 reused the Phase 4/5 mean-state models, data interface, masks, optimizer, losses, and diagnostics. It added one strict config, one main entry, one read-only verifier, pure selection functions, the low-quality metadata diagnostic, and focused tests. It did not add dependencies, a framework, covariance, a new filter, a Transformer, tight coupling, learned Q/R/bias, new data, or a hyperparameter search.

The accepted prospective contract was followed exactly: three fixed seeds, 200 optimizer steps, cadence 10, three no-physics structures, and the minimum ordinary/strict x no-physics/physics comparison. The two no-physics factorial arms were reused. Rotation-augmented did not enter the factorial and received no physics arm because it was not selected.

## Selection review

All 15 required arm/seed runs passed qualification. The structure rule selected ordinary by the preregistered stability/simplicity fallback: rotation augmentation had the lowest median but only one seed win, while ordinary won two seeds. Physics `0.1` then met all three ordinary matched conditions and entered the deployment primary.

This is a prospective RTK-free proxy selection, not evidence that ordinary+physics is more accurate. The Phase 4 negative physics result is not overwritten. The Phase 6 physics decision is also limited: Huber improved in two seeds and worsened in one. The 2x2 interaction is mixed across the validation and independent-physics proxies, so no combined strict-SO2+physics benefit is claimed.

The deployment-primary checkpoint's `best_optimizer_step` is `200`, the exact frozen budget boundary. The common 200-step budget keeps the arm/seed comparison fair and is not eligible for post-result extension, but a boundary optimum cannot establish that optimization fully converged. The frozen checkpoint and any Phase 7 finding therefore have a bounded-training interpretation.

## Numerical and claim review

The strict architecture passed every float64 structural classification and failed every float32 complete deployed-rollout classification, preserving the accepted limitation. All models passed finite, causal, degeneracy, mask, information, and robustness gates. The low-quality stratum and outage outputs are not reference-error metrics.

No route, receiver, device, physical remounting, calibrated uncertainty, truth accuracy, SPP/ESKF superiority, or teacher-quality claim is supported. Mathematical coordinate rotation remains separate from physical sensor remounting.

## Reproduction and verifier review

The designated ordinary/physics checkpoint at seed `163736869` was reproduced exactly once. Checkpoint file SHA-256, tensor SHA-256, training-log SHA-256, validation values, and boundary-selected step `200` match. The full multi-seed matrix was not repeated.

The first read-only verifier attempt exposed a verifier-only defect: JSON was written with sorted keys, while the validator incorrectly required insertion order for arm names. The validator was corrected to require exact arm and seed sets with no extras. No training artifact, selection result, config, seed, budget, metric, or checkpoint changed, and training was not rerun. The corrected verifier accepted 81 artifacts.

## Controller checklist

1. Recompute the Phase 6 config and method-file hashes in `FREEZE_MANIFEST.json`.
2. Run the 162-test locked health check, public-release guard, and `git diff --check`.
3. Run `scripts/phase6/verify_medium_freeze.py` on the external result and confirm 81 artifacts.
4. Recompute structure selection, physics selection, and both interaction directions from the external summary.
5. Inspect every strict float64/float32 classification and retain the float32 failure.
6. Compare the designated run and reproduction checkpoint/log hashes.
7. Confirm Deep, high-precision reference, candidate receiver contents, new routes, weights, logs, and data are absent from Git candidates.
8. Review `PHASE7_SEALED_EVALUATION_PROTOCOL.md`, independently lock the exact one-time reference-bearing command, and confirm no method or claim-selection change is possible afterward.

No commit, push, PR, merge, or `科研手记.md` modification was performed.

## Controller outcome

The controller independently reran the complete 162-test health check, public-release guard, and read-only verifier for all 81 external artifacts. Repository, configuration, external summary, normalizer, designated checkpoint, training log, and reproduction hashes matched the freeze. All six strict runs retained the float64 pass and float32 failure classifications.

Phase 6 is accepted as a bounded RTK-free model freeze. The deployment primary is ordinary plus `physics_weight=0.1`, selected by the prospective proxy rules rather than reference accuracy. The boundary-selected step 200 does not establish full optimization convergence, and the mixed 2x2 interaction does not establish a general physics or strict-SO2 benefit. These limitations remain mandatory in Phase 7 and the paper.
