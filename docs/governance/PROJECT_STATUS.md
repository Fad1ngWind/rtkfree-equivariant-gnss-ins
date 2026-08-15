# Project status

Last updated: 2026-08-15 (Asia/Shanghai)

## Current gate

- Current phase: Phase 0 accepted; Phase 1 started and is waiting for the user's first learning response
- Implementation state: independently reviewed through three correction cycles and accepted
- Formal acceptance: **granted by the controller on 2026-08-15**
- Phase 1 authorization: **granted for local, step-by-step theory/literature/method-specification work only**
- Formal data downloaded: no
- Model code implemented or migrated: no
- Training performed: no
- High-precision reference viewed or used: no
- Canonical project commit: `1c30708` created and verified after acceptance
- GitHub publication: owner-authorized for the reviewed Phase 0 foundation under `All rights reserved`; controller release checks and remote setup are in progress

## Frozen direction and claim status

`FROZEN_RESEARCH_CHARTER.md` is the normative scientific boundary and `ROADMAP_AND_GATES.md` is the normative phase order. The core hypothesis is pending verification and must not be described as outperforming SPP or conventional filters. The primary chain is standardized conventional WLS/SPP PVT plus low-cost IMU, with an explicit loosely coupled ESKF producing position, velocity, attitude, and covariance.

## Claim vocabulary

- **Measured:** reproduced locally with indexed evidence.
- **Policy-frozen:** a constraint or decision, not an empirical result.
- **Assumption / pending verification:** plausible but not demonstrated.

Preferred terminology is “without high-precision trajectory supervision” or precisely defined “RTK-free learning.” “Self-supervised” or “weakly supervised” requires an exact signal definition. Directly fitting SPP is not purely unsupervised.
