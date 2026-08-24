# Phase 1 readiness report — theory, literature, and minimal method specification

Date: 2026-08-24 (Asia/Shanghai)

Executor status: **READY_FOR_CONTROLLER_REVIEW**

Formal phase status: **accepted by the controller on 2026-08-24**

## Controller decision

Phase 1 is accepted. The controller independently checked the active method specification, architecture ADR, mentor record, roadmap, teaching evidence, first-party evidence matrix, repository scope, and WSL health check. The method is coherent enough to enter Phase 2 without fixing data-dependent constants or optional architecture work in advance.

The open predictive-covariance question remains a non-blocking Phase 6 decision. A scientific dependency resolver and generated lock are likewise deferred until the first later Phase that actually introduces a third-party scientific dependency.

## Scope

Phase 1 was limited to theory, first-party literature, and the minimum specification needed for the first scientific baseline. It did not download formal data, implement or train a model, run a scientific experiment, or access a high-precision reference.

The controller-reduced scope governs this report. Earlier draft material that prematurely fixed WLS constants, network dimensions, training budgets, a full tight-coupled branch, a full covariance architecture, or a large ablation/statistics plan is not part of the Phase 1 gate.

## Mentor and controller alignment

The current documents preserve the confirmed mentor direction without expanding permission into a requirement:

- a real-time forward GNSS/INS ESKF supplies non-RTK weak pseudo-labels and remains the conventional baseline;
- the deployed causal PINN uses GNSS and IMU and produces the final position, velocity, and attitude;
- the first primary method is a loose-coupled mean-state model;
- 20-second and 30-second GNSS outages are compared with the same-state pure-IMU bridge;
- GNSS-available urban-degraded behavior is compared with the identical forward ESKF, with SPP also reported;
- ZUPT and NHC are not primary-method requirements;
- tight coupling is permission for later exploration, not a simultaneous deliverable; and
- the method is described as weak supervision without high-precision trajectory supervision, not as completely unsupervised.

Predictive covariance remains an explicit mentor question and a non-blocking Phase 6 conditional extension.

## Minimal method result

The active specification defines:

- a recursive position/velocity/attitude state in a fixed per-sequence local NED frame;
- causal standardized PVT/quality, low-cost IMU, inertial reference state, and previous student state as the primary information flow;
- a frozen forward ESKF teacher that is absent from deployment inputs;
- a freely predicted pre-GNSS student state and a separately computed inertial propagation reference;
- an independent physics residual between those two states, with current GNSS excluded from the pre-GNSS residual path;
- a correction-size penalty as regularization rather than independent PINN evidence whenever the residual is algebraically identical to a learned correction;
- gravity-preserving yaw `SO(2)` as the only primary symmetry;
- the minimum baselines and comparisons needed to test teacher copying, the independent physics term, and structural equivariance; and
- complete pre-freeze exclusion and later sealed use of high-precision reference information.

## First-party literature result

The evidence matrix records verified primary-source facts, admissible influence, rejected transfers, and permitted wording for the closest reviewed work, including EqNIO, PINK-GINS, AutoW, TLIO, RoNIN, RINS-W, AI-IMU Dead-Reckoning, RIO, IDOL, MoRPI-PINN, the 2026 Sensors GNSS/IMU PINN-EKF work, and the 2026 IEEE left-equivariant SINS/GNSS filter.

The matrix does not claim novelty from any single ingredient. It distinguishes neural coordinate equivariance from equivariant filter-error geometry, supervised trajectory learning from the candidate weak-supervision policy, and per-satellite tight coupling from PVT-level loose coupling. Its dated recency audit must be refreshed at the later novelty gate.

## Teaching-gate result

In the live Phase 1 teaching interaction, the user correctly explained:

- the forward ESKF teacher, deployed PINN, and independent inertial residual information flow;
- why copying the ESKF cannot demonstrate improvement over it;
- why a residual equal to the learned correction is only a regularizer;
- causal GNSS masking and the prohibition on future/smoothed labels;
- the sealed-reference rule and weak-supervision terminology;
- the primary outage and GNSS-available baselines; and
- the ordinary, rotation-augmented, and strict `SO(2)` comparison roles.

No private conversation transcript is copied into the repository. The controller should independently recheck the explanation at the gate if needed.

## Final consistency review

The accepted architecture ADR, frozen charter, roadmap, mentor record, and minimal method specification agree on the teacher/student roles and staged scope. Supporting README, risk-register, and evidence-matrix wording was reduced so predictive covariance, joint `Q/R/bias` learning, additional temporal backbones, and tight coupling are not presented as current hard requirements.

Document whitespace and residual-scope scans pass. The canonical WSL health check passes all 22 tests, and the public-release guard reports no recognized high-risk candidate.

## Known unknowns and later-phase ownership

- Phase 2 owns official dataset fields, licensing, calibration, timing, signal selection, WLS weights/corrections, validity behavior, and output covariance.
- Phase 3 owns the deterministic INS/ESKF implementation and its numerical validation.
- Phase 4 owns the simplest adequate ordinary student architecture and training budget.
- Phase 5 owns implementation and numerical testing of the `SO(2)` mapping.
- Phase 6 owns the final pre-reference experimental freeze and any separately approved predictive-covariance or tight-coupled extension.
- Performance, novelty, calibrated uncertainty, and superiority over any baseline remain unverified.

## Assessment

The reduced Phase 1 deliverables are internally consistent and satisfy the Phase 1 gate. Phase 2 is now eligible for a separate controller start decision under its existing data and information boundaries; this acceptance does not start Phase 2, authorize model training, or permit access to the sealed high-precision reference.
