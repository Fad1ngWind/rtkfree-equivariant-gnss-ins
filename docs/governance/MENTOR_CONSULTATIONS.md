# Mentor consultation and Phase 1 study items

MC-001 through MC-005 are work to study, verify from first-party sources, teach to the user, and freeze during Phase 1. They are not answers the user must supply before entering Phase 1. They block the **Phase 1 gate and later implementation**, not Phase 1 start. Phase 1 itself still requires controller acceptance of Phase 0.

## MC-001 — Conventional WLS/SPP freeze definition

- Background: corrections, weighting, integrity checks, time/state fields, satellite count, DOP, and covariance/quality outputs vary by implementation.
- Options to study: auditable in-project solver; named/versioned official solver; cross-validated dual implementation.
- Evidence gap: no first-party comparison or approved observation contract exists here.
- Freeze impact: must be resolved in Phase 1 specification and implemented/verified in Phase 2 before model baselines.

## MC-002 — Dataset, sensor suite, and route/split protocol

- Background: licensing, clock behavior, route overlap, receiver identity, and urban-canyon diversity affect leakage and generalization.
- Options to study: route holdout within an official corpus; multi-corpus/domain holdout; new collection with a separately sealed final route.
- Evidence gap: no formal data was downloaded. UrbanNav version, license, scenario IDs, and official source remain unverified until Phase 2.
- Freeze impact: Phase 1 defines evaluation rules; Phase 2 confirms official facts and freezes manifests before implementation proceeds.

## MC-003 — State, group action, and equivariance convention

- Background: frames, gravity, group action, invariants, pseudovectors, and observability must be explicit.
- Preferred starting option: gravity-preserving `SO(2)` yaw equivariance.
- Conditional option: `O(2)` only if reflections correctly transform angular velocity as a pseudovector. Larger groups require explicit justification.
- Evidence gap: no first-party derivation or transformation tests yet.
- Freeze impact: resolve theory in Phase 1; blocks Phase 5 implementation/gate if unresolved.

## MC-004 — Genuinely independent physics-informed residual

- Background: a PINN residual must add separately stated physics rather than rename the ESKF mechanization identity.
- Options to study: independent strapdown/kinematic consistency, calibrated innovation physics, or a limited combination with explicit units and observability.
- Evidence gap: discretization, noise assumptions, loss scale, and non-reference selection protocol are unspecified.
- Freeze impact: resolve in Phase 1; blocks Phase 4 PINN baseline.

## MC-005 — Identification, covariance, and selection criteria

- Background: joint `Q/R/bias` learning can compensate; covariance inflation, all-GNSS rejection, SPP copying, and unobservable common bias are degenerate solutions.
- Sequence to study: fixed IMU assumptions and learned `R`/weights first; ordinary non-equivariant IMU plus independent PINN; gravity-aware equivariance; joint adaptation only after identifiability evidence.
- Evidence gap: no approved non-reference early-stopping or model-selection metric exists.
- Freeze impact: resolve in Phase 1; mandatory tests run in Phases 3, 4, and 6.

## MC-006 — Public license, institutional IP, and patent posture

- Background: no license choice, institutional policy, ownership, patent intent, dependency audit, or data-license audit is approved.
- Options belong to the owner/institution/mentor; Phase 0 does not select MIT, Apache-2.0, or another license.
- Freeze impact: public GitHub publication remains blocked. This does not block local Phase 1 after Phase 0 acceptance.
