# Frozen research charter

Status: policy-frozen boundary after the 2026-08-24 mentor/controller architecture reconciliation. Scientific claims remain unverified.

## Research claim and information boundary

The project studies GNSS/INS learning without high-precision trajectory supervision. RTK, PPK, post-processed high-precision trajectories, and any derived proxy are prohibited from training, normalization, tuning, early stopping, architecture or seed selection, pseudo-label generation, filtering updates, and data partition decisions.

Preferred wording is “without high-precision trajectory supervision” or precisely defined “RTK-free learning.” The method uses real-time ESKF pseudo-labels and is therefore weakly supervised, not completely unsupervised.

RTK or another approved high-precision reference may be opened only after the method, code, configuration, splits, seeds, metrics, and claims are frozen. Any method change after viewing the result invalidates the affected route as a formal test route.

## Primary method

The deployable sources are low-cost IMU and a conventional, reproducible WLS/SPP PVT stream with time, validity, covariance, and available quality fields. Receiver-native PVT is a comparator. Raw per-satellite observations remain available for audit and a possible later extension.

A frozen real-time forward GNSS/INS ESKF generates weak pseudo-labels during training and serves as the conventional baseline. It cannot use smoothing, future data, or high-precision information. The deployed PINN does not read teacher states or covariance and directly produces the final causal navigation state.

The primary method is loose coupled. It uses standardized PVT and IMU, maintains a recursive position/velocity/attitude state around an explicit inertial propagation reference, and is not a black-box regression from IMU plus absolute coordinates to position. Route, file, device, timestamp identity, and other memory shortcuts are prohibited learned inputs.

Tight coupling is not a Phase 1–5 requirement. It requires verified raw observations, a stable loose-coupled result, a separate ADR, and a new fair comparison plan.

## Minimal learning sequence

1. Prove the deterministic WLS/SPP, INS, and ESKF chain first.
2. Build one ordinary non-equivariant loose-coupled mean-state student using the frozen forward ESKF teacher.
3. Add one genuinely independent inertial physics residual and compare with the same model without that residual.
4. Replace the ordinary IMU representation with a capacity-matched gravity-preserving `SO(2)` representation and compare with ordinary and rotation-augmented controls.
5. Only after mean-state behavior is stable may predictive covariance or another optional component be added.

Joint free learning of `Q`, `R`, bias, robust rejection, and state correction is outside the first implementation. Any later addition must have a specific identifiable role and a single-factor comparison.

## Physics boundary

The physics loss must act on a network-predicted state that is not algebraically defined as the same mechanization result. Its frame, units, discretization, validity conditions, and data dependencies must be explicit.

If a state is defined as mechanization plus a learned correction, then a residual equal to that correction is a correction prior, not independent evidence that a PINN learned inertial dynamics. GNSS innovation likelihood is a statistical loss and cannot be renamed as inertial physics.

## Symmetry boundary

Gravity-preserving yaw `SO(2)` is the only primary symmetry. Phase 1 defines the group action; Phase 5 implements and numerically tests it. `O(2)`, `SO(3)`, reflection handling, alternative invariant filters, and additional groups are deferred unless the primary comparison reveals a concrete need.

A mathematical coordinate-equivariance property test and a real sensor-remounting generalization experiment are different claims. The first is mandatory for an equivariance claim; the second is conditional on suitable data.

## Evaluation boundary

The two mentor-defined questions are:

1. Does the method improve 20-second and 30-second controlled GNSS outages over the same-state pure-IMU bridge?
2. With GNSS available, does it improve urban-degraded positioning over the identical forward ESKF, while SPP remains reported?

Route/session separation, common initialization, causal masking, multiple fixed seeds, and reporting failures are required. Exact numerical margins, sample counts, architecture sizes, training steps, and statistical procedures are not frozen in Phase 1 before the official data and compute budget are known. They must be fixed without reference-error influence before the relevant experiment.

A constant common SPP bias may be unobservable without independent absolute information. The project must state this limitation and cannot promise architecture-only absolute de-biasing.

## Scope control

An item enters a phase gate only if it directly tests the mentor-defined hypothesis, is necessary for mathematical/implementation correctness, or prevents leakage or an invalid conclusion. Otherwise it is placed in a non-blocking backlog.

Conditional extensions never block the primary method. A Phase may not freeze exact hyperparameters that depend on data fields, hardware, or a working baseline owned by a later Phase.

## Literature boundary

EqNIO, PINK-GINS, AutoW, and newer first-party work define prior-art boundaries. Equivariant inertial learning, PINN plus GNSS/INS, ESKF pseudo-label learning, or GNSS weighting cannot individually support a priority claim. Novelty remains a candidate combination claim until the first-party search and experiments are complete.
