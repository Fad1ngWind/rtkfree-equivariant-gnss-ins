# Project status

Last updated: 2026-08-28 (Asia/Shanghai)

## Current gate

- Current phase: Phase 5 **accepted**; Phase 6 has not started
- Infrastructure state: Phase 0 accepted by the controller on 2026-08-15
- Formal Phase 1 acceptance: **granted by the controller on 2026-08-24**
- Phase 2 authorization: **granted by the controller after Phase 1 acceptance**
- Formal Phase 2 acceptance: **granted by the controller on 2026-08-26**
- Formal Phase 3 acceptance: **granted by the controller on 2026-08-27**
- Formal Phase 4 acceptance: **granted by the controller on 2026-08-27**
- Formal Phase 5 acceptance: **granted by the controller on 2026-08-28**
- Formal data downloaded: yes, only to the approved repository-external WSL-native data root
- Model code implemented or migrated: deterministic baselines, the ordinary causal student, its rotation-augmented control, and one typed gravity-aware `SO(2)` student
- Training performed: yes, bounded reference-free Phase 4 and Phase 5 development training only; no high-precision reference used
- High-precision reference files downloaded, opened, or used: no; the sanitized controlled search-summary deviation is recorded in Phase 2 evidence
- Canonical project history: the accepted Phase 0–5 work is the current canonical project history
- GitHub publication: `https://github.com/Fad1ngWind/rtkfree-equivariant-gnss-ins` is public under `All rights reserved`; protected `main` is the default branch

## Frozen direction and claim status

`FROZEN_RESEARCH_CHARTER.md` is the normative scientific boundary and `ROADMAP_AND_GATES.md` is the normative phase order. The core hypothesis is pending verification and must not be described as outperforming SPP or conventional filters. After the mentor/controller reconciliation, the primary chain is standardized conventional WLS/SPP PVT plus low-cost IMU, a real-time forward ESKF as weak-label teacher/baseline, and a causal loose-coupled PINN producing the final navigation state. Tight coupling and full predictive covariance are conditional extensions rather than Phase 1–5 requirements.

## Phase 1 scope-reduction decision

On 2026-08-24 the controller found that the Phase 1 draft had prematurely frozen WLS constants, network dimensions, training budgets, a complete tight-coupled branch, a full covariance architecture, and an oversized ablation/statistics plan. These items were removed from the primary gate and returned to their owning later Phases or a non-blocking backlog. The concise method specification is now the only active Phase 1 architecture document.

## Claim vocabulary

- **Measured:** reproduced locally with indexed evidence.
- **Policy-frozen:** a constraint or decision, not an empirical result.
- **Assumption / pending verification:** plausible but not demonstrated.

Preferred terminology is “without high-precision trajectory supervision” or precisely defined “RTK-free learning.” “Self-supervised” or “weakly supervised” requires an exact signal definition. Directly fitting SPP is not purely unsupervised.

## Phase 4 result boundary

Phase 4 established a reproducible ordinary causal student and a matched no-physics/physics training comparison using the Phase 3 forward ESKF only as a weak pseudo-label teacher. It passed causality, stability, degeneracy, and artifact-reproduction checks. The physics variant did not improve the independent diagnostic physics residual in this bounded run, so no positive PINN-effect or navigation-performance claim is permitted. Phase 5 must preserve this negative result while testing the separately specified equivariant structure.

## Phase 5 result boundary

Phase 5 implemented capacity-controlled ordinary, rotation-augmented, and typed gravity-aware `SO(2)` students under the same no-physics training setting. The strict architecture passed the complete float64 recursive commutation audit, while its float32 rollout exceeded the preregistered tolerance by approximately `0.30–0.51 mm`; both facts must remain visible. Rotation augmentation had the lowest weak-label validation objective and the strict model did not beat ordinary on that objective, but these are not positioning-accuracy results. No Phase 6 winner has been selected.
