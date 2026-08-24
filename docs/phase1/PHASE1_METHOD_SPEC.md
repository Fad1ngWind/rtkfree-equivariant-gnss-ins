# Phase 1 minimal method specification

Status: active Phase 1 specification after controller scope reduction on 2026-08-24. It defines the first implementable scientific question, not a completed phase or empirical result.

## 1. Research question

Can a causal GNSS/INS neural state estimator, trained without high-precision trajectory supervision from real-time forward-ESKF weak pseudo-labels and an independent inertial physics residual, improve:

1. 20-second and 30-second GNSS outages over pure-IMU propagation from the same causal starting state; and
2. urban GNSS-degraded positioning with GNSS available over the identical forward ESKF?

SPP remains reported in the second comparison. Both improvements are hypotheses, not promised results.

## 2. Information policy and terminology

RTK, PPK, post-processed high-precision trajectories, and their derivatives do not enter training, normalization, tuning, checkpoint or seed selection, data splits, initialization, or architecture decisions.

The method uses ESKF pseudo-labels and is described as weak supervision without high-precision trajectory supervision. It is not described as completely unsupervised.

The high-precision reference remains sealed until the complete final-evaluation freeze.

## 3. Minimal system architecture

The first primary method is loose coupled:

```text
low-cost RINEX -> conventional WLS/SPP PVT ----+
                                                 +-> causal PINN -> final navigation state
low-cost IMU -> deterministic INS reference ----+

PVT + IMU -> frozen forward ESKF -> training-only weak labels and baseline
```

The deployed PINN receives causal standardized PVT/quality information, low-cost IMU information, an explicit inertial reference state, and its own previous state. It does not receive the teacher ESKF state or covariance.

The PINN produces final position, velocity, and attitude. Internal IMU bias variables may be carried if required by the inertial equations, but they are not claimed as truth.

This is not direct absolute-position regression. The learned estimator is recursive, and absolute route coordinates, route/file/device identity, and future information are prohibited shortcuts.

## 4. State, frame, and timing

The internal navigation state at the IMU origin is

\[
x=(p^{n_0},v^{n_0},R_b^{n_0},b_a^b,b_g^b),
\]

where `n_0` is a per-sequence fixed local `NED` frame established from the first causally valid conventional PVT position. The exact initialization, lever arm, mounting rotation, time synchronization, and frame transformations are shared by every method.

The estimator uses only information that has arrived by the output time. If an official dataset lacks arrival-time records, the replay states the assumption that measurement and availability times are equal. Complex delay-buffer replay is not part of the first implementation.

Exact mechanization order, WLS models, covariance values, masks, and signal choices belong to Phases 2 and 3 after the official data interface is known. Phase 1 fixes their consistency across methods, not premature numerical constants.

## 5. Teacher and weak-label loss

One frozen real-time forward GNSS/INS ESKF is used both as teacher and conventional baseline. It uses the same deployable PVT and IMU stream as the student and never uses smoothing, future observations, or high-precision information.

The student state is compared with valid teacher position, velocity, and attitude using frame-correct state errors. Teacher covariance may supply fixed relative weighting only after Phase 3 shows that it is numerically meaningful. Ordinary robust normalized errors are sufficient; Phase 1 does not freeze a particular loss threshold.

The ESKF output is denoted as a weak pseudo-label, never ground truth. Copying the teacher may demonstrate distillation but cannot demonstrate improvement over the teacher.

## 6. Independent inertial physics residual

The network must produce a freely predicted pre-GNSS state `x^-_{theta,k}`. The deterministic inertial operator independently propagates the preceding posterior state with the measured IMU:

\[
\bar x_k^-=\Phi_{\Delta t_k}(x^+_{\theta,k-1},u_{k-1:k}),
\]

\[
r_{\mathrm{phys},k}=x^-_{\theta,k}\boxminus\bar x_k^-.
\]

The current GNSS measurement may then produce the posterior/final state `x^+_{theta,k}`. It cannot enter the pre-GNSS prediction used by the inertial residual.

This residual is valid only if `x^-_{theta,k}` is an independently predicted state. The specification must not define

\[
x^-_{\theta,k}=\bar x_k^-\boxplus\delta x_k
\]

and then present `r_phys=delta x_k` as independent physics evidence. Such a term is only a correction-size regularizer and must be named accordingly.

The final implementation records frames, units, discretization, valid IMU intervals, and the process assumptions. A no-physics version of the same student is mandatory.

## 7. GNSS masking and causality

Training includes controlled causal 20-second and 30-second GNSS masks so that the student experiences the mentor-defined outage conditions. During a mask, all contemporaneous GNSS-derived student inputs are hidden. The student may know elapsed time since the last available GNSS record but not the planned mask length or recovery time.

The forward ESKF teacher may retain the full causal GNSS stream for weak-label distillation, but this privileged training information is disclosed. Exact mask proportions and window lengths are selected in Phase 4 from route availability and compute limits before model comparison; they are not Phase 1 constants.

## 8. Gravity-preserving equivariance

The primary symmetry is a horizontal coordinate rotation about gravity:

\[
G_\psi=\operatorname{diag}(R_\psi,1).
\]

Navigation-frame position, velocity, attitude representation, innovations, and any corresponding covariance transform consistently under `G_psi`. Body-frame IMU measurements, time, masks, counts, and other physical scalars keep their declared invariant role.

Phase 5 must implement an exact numerical property test for the complete learned mapping. Ordinary, rotation-augmented, and `SO(2)` models are compared at similar capacity. Exact layer counts, channel counts, test-sample counts, and tolerances are implementation decisions owned by Phase 5.

`O(2)`, reflections, and larger symmetry groups are outside the primary project unless a later result establishes a concrete need.

## 9. Minimal baselines and ablations

Mandatory non-learning baselines:

- conventional SPP/WLS;
- the identical frozen forward ESKF;
- pure-IMU propagation from the same causal outage start.

Mandatory learned comparisons:

- ordinary student with and without the inertial physics residual;
- ordinary, rotation-augmented, and `SO(2)` students in Phase 5.

If artificial masking is used as a claimed contribution, one otherwise matched no-mask comparison is added. Other comparisons are diagnostic and do not block the main experiment.

## 10. Evaluation and selection

Development selection uses only deployable data and weak-label/physics criteria. It cannot use high-precision position error or a supervised high-precision oracle.

Final evaluation uses route/session-level separation, common initialization and masks, multiple fixed seeds, and identical validity accounting. It reports failures rather than deleting them.

The primary final endpoints are:

- horizontal relative-displacement error at 20-second and 30-second outage endpoints versus pure IMU;
- urban-degraded horizontal position error with GNSS available versus the forward ESKF, with SPP also reported.

Exact practical margins, seed count, confidence-interval method, and minimum sample support are frozen only after Phase 2 confirms the route hierarchy and Phase 6 confirms the available compute. They must be fixed before sealed-reference access and cannot be changed after results are viewed.

A constant common PVT bias may remain unobservable and is reported as a limitation.

## 11. Staged uncertainty objective

Credible predictive uncertainty remains a project objective, but it is not allowed to delay or obscure mean-state feasibility.

Phase 4 trains and validates mean states only. If that stage is stable, Phase 6 may add the simplest positive-definite covariance head that preserves the chosen `SO(2)` action. Its calibration, sharpness, and supported state subspace are then frozen before sealed evaluation.

Whether predictive covariance is a required main-paper contribution or a secondary extension remains the only immediate mentor architecture question.

## 12. Explicitly deferred work

The following are not Phase 1 gates and are not part of the first learned baseline:

- tight coupling and learned per-satellite processing;
- Set Transformers, graph encoders, and multiple GNSS temporal backbones;
- joint free learning of `Q`, `R`, bias, robust rejection, and state correction;
- `O(2)` or `SO(3)` equivariance;
- ZUPT, NHC, or learned motion-condition gates;
- a full `9 x 9` covariance architecture before mean-state feasibility;
- fixed network depth/width, optimizer steps, hard statistical thresholds, and large ablation grids;
- alternative estimators such as EqF, IEKF, or FGO.

Any deferred item requires a concrete observed need, a separate scope decision, and evidence that it will not delay the primary hypothesis.

## 13. Phase 1 exit

Phase 1 is ready for controller review when:

1. the user can explain the teacher/student/deployment information flow;
2. the user can explain why the selected physics residual is independent rather than a correction-size penalty;
3. the state, frame, causal inputs, `SO(2)` action, minimal baselines, and final-reference policy are clear;
4. the first-party evidence matrix establishes the relevant prior-art boundaries; and
5. the predictive-uncertainty role is resolved or explicitly left as a Phase 6 conditional extension.

No formal data download, model implementation, or training is authorized by this document.
