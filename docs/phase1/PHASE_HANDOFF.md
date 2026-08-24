# PHASE_HANDOFF — Phase 1 minimal theory and method specification

## Stage objective

Deliver the smallest coherent theory, first-party literature boundary, and method specification needed to implement and fairly test the mentor-directed loose-coupled GNSS/INS PINN baseline without high-precision trajectory supervision.

## Completed

- Reconciled the mentor-directed PINN-final architecture through accepted ADR-0004.
- Reduced the primary method to one loose-coupled mean-state student using conventional PVT and low-cost IMU.
- Fixed the real-time forward ESKF as training-only weak-label teacher and conventional baseline, not a deployment input.
- Specified causal recursive position, velocity, and attitude output around an explicit inertial propagation reference.
- Defined one independent inertial physics residual and explicitly excluded a correction-identity penalty from PINN physics evidence.
- Fixed gravity-preserving yaw `SO(2)` as the only primary symmetry and separated it from rotation augmentation and physical sensor remounting.
- Froze the two mentor-defined comparison questions, minimal baselines, and sealed-reference policy without premature numerical margins.
- Audited the closest first-party literature and recorded conservative permitted wording.
- Completed the user teaching gate for information flow, physics independence, causality, information isolation, baselines, and symmetry controls.
- Removed residual supporting-document language that made conditional covariance, joint noise learning, extra temporal backbones, or tight coupling look mandatory.

## Primary method handed forward

```text
deployable PVT + low-cost IMU -> causal recursive PINN -> final p/v/attitude

deployable PVT + low-cost IMU -> frozen forward ESKF -> training weak labels + baseline only

previous student posterior + IMU -> independent inertial propagation
freely predicted student pre-GNSS state minus inertial propagation -> physics residual
```

The current GNSS observation cannot enter the student pre-GNSS state used by the physics residual. The deployed student cannot read teacher states or covariance.

## Explicitly deferred

- per-satellite tight coupling;
- predictive covariance unless the mean model is stable and the mentor confirms it as core;
- learned `Q`, learned loose-coupling `R`, joint bias/noise/state adaptation, or learned GNSS rejection;
- ZUPT, NHC, and learned motion gates;
- additional GNSS temporal backbones or estimator families;
- `O(2)`, reflections, `SO(3)`, and alternative symmetry groups;
- WLS constants, network dimensions, optimizer steps, seed count, statistical thresholds, and large ablation grids.

Deferred work cannot block the primary roadmap and requires its owning later Phase or a separate controller scope decision.

## Unresolved but non-blocking

- MC-001: whether predictive covariance is a required main contribution or a Phase 6 secondary extension.
- Official dataset fields, license, timing, calibration, route hierarchy, and reference quality remain Phase 2 facts.
- Exact WLS behavior, architecture size, compute budget, seed policy, practical margins, and statistical procedure remain owned by later phases and must be frozen without high-precision-error influence.
- Scientific performance and novelty remain unverified.
- Institutional ownership, patent strategy, dependency licenses, and dataset publication rights must be reviewed before publishing data-derived artifacts or scientific results.

## Executor boundary confirmed

Before controller acceptance:

- No formal data was downloaded or processed.
- No model or scientific navigation method was implemented or trained.
- No RTK, PPK, post-processed, or other high-precision trajectory was viewed or used for development.
- No second runnable source tree was created.
- the Phase 1 executor performed no commit, remote change, push, or release-gate relaxation; and
- `科研手记.md` was not rewritten to claim Phase 1 completion.

## Suggested controller review

1. Compare ADR-0004, the frozen charter, roadmap, mentor record, and minimal method specification for teacher/student consistency.
2. Verify that `r_phys` constrains a freely predicted pre-GNSS state rather than reproducing a learned correction.
3. Sample EqNIO, PINK-GINS, AutoW, MoRPI-PINN, the 2026 Sensors paper, and the 2026 IEEE equivariant-filter entries against their first-party sources.
4. Confirm that tight coupling, covariance, additional backbones, exact WLS constants, and fixed training/statistical budgets are not current gates.
5. Confirm the sealed-reference prohibition and that no data, implementation, training, result, commit, or push occurred.

## Executor state

**READY_FOR_CONTROLLER_REVIEW**

## Controller outcome

**ACCEPTED on 2026-08-24**

The open predictive-covariance question remains non-blocking. Phase 2 is eligible for a separate controller start decision and may proceed only within its official-data, provenance, timing, calibration, split, isolation, and conventional WLS/SPP scope.
