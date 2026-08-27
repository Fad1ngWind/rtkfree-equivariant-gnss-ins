# Phase 4 minimal implementation contract

Status: implementation boundary for the ordinary mean-state baseline. This is not a scientific result or phase acceptance.

## Information and data

- Use only the frozen Medium standardized PVT/IMU interfaces and the Phase 3 `fixed_eskf.jsonl` posterior states as weak pseudo-labels. Do not read Deep, a precise trajectory, receiver-native truth, or any derived reference quantity.
- Never expose teacher state or covariance to the deployed student. Do not use teacher covariance for loss weighting. Retain the frozen elevated-NIS and natural ten-second SPP-gap recovery limitations in reports.
- Use the one available Medium session only for bounded development. Split it chronologically with fixed guard intervals before training; do not claim route/session generalization from this phase.

## Causal student

- Run recursively on the Phase 3 PVT-event time grid. At step `k`, inputs are the previous student posterior, the arrived IMU samples since step `k-1`, and deployable GNSS availability/quality fields. Absolute time and route/file/device identity are prohibited.
- Use one small ordinary GRU trunk. From the previous posterior and causal IMU interval it freely predicts `x^-_theta,k = (p, v, R)` before current GNSS is visible. Position is represented recursively as an increment from the previous student position; the learned branch never receives absolute SPP position directly.
- After `x^-_theta,k` exists, a small correction head may use the current valid SPP innovation relative to the predicted antenna position plus deployable quality/availability. It produces the final posterior `x^+_theta,k`. Invalid or masked GNSS produces no GNSS correction. Deployment output is this student posterior.

## Independent physics and attitude errors

- Independently compute `bar{x}^-_k = Phi(x^+_theta,k-1, IMU[k-1:k])` with the deterministic Phase 3 mechanization, fixed known gravity/Earth rotation, the frozen lever arm, and zero student bias states.
- Compute the physics residual as `x^-_theta,k boxminus bar{x}^-_k`; do not define the learned pre-state as mechanization plus a correction. Current GNSS and teacher outputs are forbidden from both the learned pre-GNSS branch and its inertial reference.
- Use position and velocity differences in fixed local NED. Use the `SO(3)` logarithm of `R_reference.T @ R_prediction` for attitude error, with a tested small-angle branch.

## Fair comparison and selection

- Train exactly two configurations: the same initialization, split, masks, minibatch order, optimizer, step budget, architecture, and seed, with physics weight respectively zero and fixed nonzero. No architecture or grid search is authorized.
- Select checkpoints by the same finite held-out weak-label mean-state score and fixed numerical-stability rules. Record physics residual separately; do not use reference accuracy or teacher covariance.
- Apply causal left-closed/right-open 20-second and 30-second masks on the Phase 3 time basis. Hide all current GNSS-derived student fields while masked; expose neither planned duration nor recovery time.

## Minimum gates and artifacts

- Synthetic tests must prove recursive causality, future perturbation invariance, pre/post-GNSS isolation, attitude error correctness, independent physics residual correctness, and exact 20/30-second mask endpoints.
- Smoke and one bounded development run must report finite values, loss behavior, restart reproducibility, and direct diagnostics for teacher copying, SPP copying, zero correction, GNSS rejection, future leakage, absolute-time/route identity, and mask recovery behavior.
- Keep environments, weights, caches, data, and run logs under WSL-native repository-external roots. Repository outputs are limited to code, minimal configs, synthetic tests, concise reviewed evidence, and Phase 4 handoff/review documents.
