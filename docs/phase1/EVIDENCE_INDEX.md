# Phase 1 evidence index

Phase 1 evidence is document-based. It contains no formal data, model output, training result, high-precision trajectory, or private conversation transcript.

| Review claim | Classification | Evidence | Controller recheck |
|---|---|---|---|
| Mentor-directed ESKF-teacher/PINN-final architecture is reconciled | policy decision | `../adr/0004-pinn-final-state-architecture-reconciliation.md`, `../governance/MENTOR_CONSULTATIONS.md` | compare confirmed mentor direction with ADR consequences |
| Loose-coupled mean-state learning is the only first primary method | policy-frozen scope | `../governance/FROZEN_RESEARCH_CHARTER.md`, `../governance/ROADMAP_AND_GATES.md` | confirm conditional extensions do not block the roadmap |
| High-precision information is excluded from development and sealed for final evaluation | policy-frozen boundary | `../governance/DATA_AND_INFORMATION_POLICY.md`, `../governance/FROZEN_RESEARCH_CHARTER.md` | inspect all prohibited influence and final-freeze conditions |
| State, frame, causality, teacher/student inputs, and final outputs are specified | method specification | `PHASE1_METHOD_SPEC.md` sections 3–5 and 7 | dependency and timing review |
| Physics residual acts on a freely predicted state and is not a correction identity | derivation/specification | `PHASE1_METHOD_SPEC.md` section 6 | check both equations and the prohibited tautological form |
| Gravity-preserving yaw `SO(2)` is the sole primary symmetry | mathematical specification | `PHASE1_METHOD_SPEC.md` section 8, `../governance/FROZEN_RESEARCH_CHARTER.md` | verify typed vector/scalar roles; implementation waits for Phase 5 |
| The two mentor-defined questions and minimum fair comparisons are frozen | evaluation specification | `PHASE1_METHOD_SPEC.md` sections 1, 9, and 10 | confirm pure-IMU and identical-ESKF baselines remain mandatory |
| Closest prior work is separated by verified supported and unsupported claims | first-party literature record | `FIRST_PARTY_EVIDENCE_MATRIX.md` | sample original-source links and refresh novelty search at its later gate |
| Predictive covariance and tight coupling are conditional rather than current blockers | scope control | `PHASE1_METHOD_SPEC.md` sections 11–12, `../governance/MENTOR_CONSULTATIONS.md` | confirm MC-001 remains open and non-blocking |
| Phase 1 has no scientific result or implementation claim | status/readiness | `PHASE1_ACCEPTANCE_REPORT.md`, `../governance/PROJECT_STATUS.md`, `PHASE_HANDOFF.md` | inspect worktree and formal-data/training/reference declarations |

Paths are relative to `docs/phase1/`. Historical Phase 0 acceptance files are not current scientific specifications; ADR-0004 and the controller-reduced Phase 1 documents supersede their earlier architecture wording.
