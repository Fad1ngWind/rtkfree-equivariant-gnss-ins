# Phase 5 gravity-aware SO(2) implementation contract

Status: **implemented and accepted by the controller on 2026-08-28**.

## Single scientific question

Test whether a gravity-preserving horizontal `SO(2)` structure changes the bounded, reference-free development behavior of the accepted Phase 4 causal mean-state student. Compare exactly three cases: the accepted ordinary architecture, the same ordinary architecture with training-time yaw rotation augmentation, and one strictly `SO(2)`-equivariant full learned mapping. This phase does not test `O(2)`, reflections, `SO(3)`, sensor remounting, new filters, covariance prediction, learned bias/Q/R, tight coupling, new data, or Phase 6 components.

## Frozen group action

Use the active left action about the fixed local-NED origin

\[
G_\psi=\begin{bmatrix}\cos\psi&-\sin\psi&0\\
\sin\psi&\cos\psi&0\\0&0&1\end{bmatrix}.
\]

Every navigation-frame vector in the learned interface, including position, velocity, GNSS innovation, and corresponding direction-valued quantities, transforms as `q^n -> G_psi q^n`. The body-to-navigation attitude transforms as `C_b^n -> G_psi C_b^n`. Body-frame accelerometer and gyroscope measurements, body-frame lever arm, elapsed time, availability/mask, satellite count, DOP/quality values, and other declared physical scalars remain unchanged. Navigation-frame directional constants transform consistently; gravity is numerically unchanged because `G_psi e_3 = e_3`. This is coordinate equivariance, not evidence of robustness to physically reinstalling a sensor.

The strict claim covers initialization, recurrent hidden state, pre-GNSS prediction, GNSS correction, posterior state, and multi-step causal rollout. It is not enough for one internal layer to commute.

## Frozen fair comparison

- Reuse the Phase 4 deployable data interface, hashes, chronological splits and guards, train-only normalizer source, causal masks, optimizer, learning rate, weight decay, clipping, unroll length, smoke/development step budgets, checkpoint cadence, weak-label selection rule, and seed `3407`.
- Use `physics_weight = 0` for all three Phase 5 variants. Preserve the accepted Phase 4 negative result verbatim: diagnostic physics Huber `0.374354878` for no-physics and `0.418493815` for physics. Do not retune or rerun a three-architecture physics cross-product.
- The augmented control uses the ordinary Phase 4 architecture and parameter count. One deterministic seeded yaw is held fixed for each complete training pass, changes only the coordinate presentation of eligible training tensors, adds no optimizer steps, and is an empirical control rather than a strict-equivariance claim.
- Trainable parameter counts are frozen at `6994` for ordinary, `6994` for rotation-augmented, and `6780` for strict `SO(2)` (`-3.1%`, inside the preregistered `±15%` interval). No architecture or hyperparameter grid search is allowed.

## Pre-training property and safety gates

For the complete learned step and rollout, test fixed non-special yaw angles, fixed random inputs, identity, and composition. Tolerances were fixed before training: for float64, `atol=1e-9`, `rtol=1e-8`; for float32, `atol=5e-6`, `rtol=5e-5`. Ordinary and augmented controls must have counterexamples showing they are not mislabeled as strictly equivariant; augmentation alone is never a proof.

Diagnostic reporting has three separate outcomes and no aggregate scientific `PASS`: (1) reference-free safety checks, (2) float64 structural `SO(2)` checks, and (3) float32 deployed full-recursive-rollout checks. A failure in one dtype must remain visible and cannot be converted into success by another dtype. The retained evidence has reference-free safety checks passing, the strict model's float64 structural check passing, and its float32 deployed recursive check failing the preregistered tolerance with approximately `0.30–0.51 mm` maximum absolute commutation error. The validator checks reported outcomes for internal consistency; it does not require a dtype failure as a success condition.

Retain Phase 4 causality, pre/post-GNSS isolation, GNSS mask endpoints and recovery, invalid-GNSS rejection, degeneracy checks, finite/proper-rotation checks, deterministic reproduction, and all prohibited-input guards. Data, checkpoints, logs, and run evidence remain outside the repository.

## Reporting boundary

Report structural commutation results and reference-free development metrics honestly, including negative outcomes. Do not claim accuracy, superiority over ESKF/SPP, route/device generalization, physical sensor-remounting generalization, or an equivariant benefit without later sealed evaluation. Final Phase 5 documents may say only `READY_FOR_CONTROLLER_REVIEW` until independent controller acceptance. `科研手记.md` is user-owned, remains unmodified and untracked, and is not a Phase 5 Git artifact.
