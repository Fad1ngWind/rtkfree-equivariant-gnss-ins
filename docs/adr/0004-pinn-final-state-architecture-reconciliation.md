# ADR-0004: Mentor-directed PINN final-state architecture

- Status: accepted by controller
- Date: 2026-08-24
- Scope: scientific architecture only; no implementation or Phase 1 acceptance is implied

## Context

The original charter placed the final navigation output in an ESKF. The mentor later clarified that a real-time forward GNSS/INS ESKF should generate non-RTK weak pseudo-labels, while the deployed PINN should use causal GNSS and IMU information and output the final navigation result.

The mentor also said that loose and tight coupling could both be explored. This is permission to investigate, not a requirement to implement two complete methods at once.

## Decision

The project adopts the mentor-directed architecture with a deliberately narrow first implementation:

1. A frozen real-time forward ESKF is the weak-label teacher and conventional baseline. It is not a deployment input.
2. The deployed PINN is a causal recursive GNSS/INS state estimator and produces the final position, velocity, and attitude.
3. Standardized WLS/SPP PVT plus low-cost IMU defines the single primary loose-coupled method.
4. The method is not direct absolute-position regression: it carries a causal navigation state, uses an explicit inertial propagation reference, and prohibits route or absolute-location shortcuts in the learned branch.
5. A per-satellite tight-coupled method is deferred. It may start only after the loose-coupled mean-state method is reliable and the controller approves a separate scope decision.
6. Predictive covariance remains a project objective but is staged after mean-state learning is stable. Its exact role is an explicit mentor decision rather than a Phase 1 assumption.

The method is described as ESKF-pseudo-label weak supervision without high-precision trajectory supervision, not as completely unsupervised learning.

## Consequences

- SPP, the identical forward ESKF, and pure-IMU propagation remain mandatory baselines.
- Copying the teacher cannot establish improvement over the teacher.
- The physics term must constrain a freely predicted state through an independent inertial equation. A penalty that is algebraically identical to the learned correction may be used as regularization but cannot be presented as independent PINN evidence.
- Tight coupling, full covariance modeling, complex delay replay, and extra estimator families are conditional extensions and do not block the primary phases.
- The information-isolation policy and sealed final-reference rule remain unchanged.
