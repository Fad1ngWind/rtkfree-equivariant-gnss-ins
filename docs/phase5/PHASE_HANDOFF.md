# Phase 5 controller handoff

Status: **READY_FOR_CONTROLLER_REVIEW**. This is not controller acceptance, and no commit, push, PR, or merge has been performed.

## Delivered scope

Phase 5 compares exactly three matched no-physics variants: the Phase 4 ordinary network, the identical ordinary architecture with pass-level yaw rotation augmentation, and a capacity-controlled typed scalar/vector strict `SO(2)` network. All use `physics_weight=0.0`, Phase 4 data/interface/splits/masks, 40 optimizer steps, AdamW settings, checkpoint cadence, weak-label selection rule, and seed `3407`. Parameter counts are `6994`, `6994`, and `6780`.

The active gravity-preserving action rotates navigation-frame position, velocity, GNSS innovation, and directional quantities by the same `G_psi`; left-multiplies `C_b^n`; and leaves body-frame IMU, body lever arm, time, masks, satellite/DOP/quality scalars, and right body attitude increments invariant. Only coordinate equivariance was studied, not sensor remounting.

## Findings requiring controller interpretation

- Reference-free safety checks passed for all three checkpoints: causality/prefix, masks and recovery, degeneracy controls, finite/proper rotations, translation consistency, and speed bound.
- The strict model passed the complete 764-step float64 structural commutation audit at the preregistered tolerance. Maximum absolute angle-case error was `9.78772618509538e-13`.
- The strict model did **not** pass the complete 764-step float32 deployed-recursive audit at the preregistered tolerance. Maximum absolute errors over the three fixed angles were approximately `0.2966–0.5102 mm`; the largest tolerance ratio was about `39.99`. No tolerance was changed.
- Ordinary and rotation-augmented controls have large float64 and float32 counterexamples, so neither is labeled strictly equivariant. Rotation augmentation remains an empirical control.
- The old dual-dtype JSON uses schema 1 and contains both `diagnostics_passed=true` and `float32_complete_mapping_gate_passed=false`. The first field was a legacy aggregate of reference-free safety plus float64 structural classification, not an overall pass. Source schema 2 now emits the three outcomes separately; diagnostics were not rerun under the controller's repetition cap.

## Matched development results

These values measure agreement with a weak pseudo-label teacher on the common unrotated validation view; they are not positioning accuracy.

| Variant | Initial validation | Best validation | Best step |
|---|---:|---:|---:|
| ordinary | `3.774408738` | `3.317065557` | 40 |
| rotation-augmented | `3.774408738` | `2.949578842` | 40 |
| strict SO(2) | `19.803268433` | `3.498847644` | 30 |

The strict model did not beat ordinary on this bounded weak-label metric. No tuning followed this negative result. A single independent repeat reproduced all 11 audited training artifacts exactly.

Post-hoc no-physics checkpoint residuals were ordinary `0.374354878`, rotation-augmented `0.297772611`, and strict `0.207513982`. They do not show a physics-training benefit or accuracy. The accepted Phase 4 negative comparison remains: no-physics `0.374354878` versus physics `0.418493815`.

## Failed attempts retained

1. `development_user_20260828` stopped after writing the normalizer because Phase 5 initially referenced a Phase 4 config attribute that was not exposed. The selection cadence was then explicitly frozen to the existing Phase 4 value of 10; the failed directory remains outside the repository.
2. The first `diagnostics.json` failed the strict float32 full-rollout tolerance. It remains unmodified as provenance.
3. The first artifact-verifier run assumed six rather than seven normalized IMU features; the verifier assumption was corrected and both complete evidence directories then validated.

## Validation and evidence

Evidence paths and hashes are listed in `docs/phase5/EVIDENCE_INDEX.md`. The one allowed complete repository test passed 146 tests and the public-release guard. The one allowed complete development comparison reported `byte_and_tensor_identical=true` for 11 artifacts. Smoke reproduction was also capped at one pair and matched across 8 artifacts.

## Limitations and explicitly unperformed work

No high-precision/RTK/PPK or sealed holdout was used. There is no positioning-accuracy, ESKF/SPP-superiority, route/device/domain-generalization, real sensor-remounting, or equivariant-benefit claim. There was no O(2), reflection, SO(3), new filter, tight coupling, Transformer, covariance prediction, Q/R/bias learning, new data, hyperparameter grid, cross-domain experiment, or Phase 6 work.

The executor did not modify or add `科研手记.md` to Git. It remains user-authored. The controller's treatment of the float64 structural result and retained float32 deployed-rollout failure is recorded below.

## Controller outcome

**ACCEPTED on 2026-08-28.**

The controller independently reran the complete 146-test health check, public-release guard, and repository-external artifact verifier. Both complete development runs reproduced all 11 audited training artifacts exactly, and the smoke pair reproduced all eight audited artifacts.

Acceptance is structural and bounded. The typed architecture and complete float64 recursive mapping satisfy the Phase 5 mathematical `SO(2)` gate, while the preregistered float32 complete-rollout tolerance remains failed and must be reported whenever numerical equivariance is discussed. Phase 5 does not establish an accuracy or equivariant-benefit result: rotation augmentation had the lowest weak-label validation value, and the strict model was worse than ordinary on that bounded metric. Phase 6 may not promote the strict model, the augmentation control, or the physics residual as a scientific winner without a separately frozen fair experiment.
