# Project status

Last updated: 2026-08-24 (Asia/Shanghai)

## Current gate

- Current phase: Phase 1 **accepted**; Phase 2 has not started
- Infrastructure state: Phase 0 accepted by the controller on 2026-08-15
- Formal Phase 1 acceptance: **granted by the controller on 2026-08-24**
- Phase 2 authorization: **not yet granted**
- Formal data downloaded: no
- Model code implemented or migrated: no
- Training performed: no
- High-precision reference viewed or used: no
- Canonical project history: the accepted Phase 0 foundation and Phase 1 theory/method specification are committed and synchronized
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
