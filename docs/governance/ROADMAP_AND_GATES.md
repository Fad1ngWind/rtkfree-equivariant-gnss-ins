# Phase 0–7 roadmap and gates

The phase order is fixed. Each phase has one primary question and a small mandatory gate. Conditional extensions are recorded but do not block progress.

## Phase 0 — infrastructure and governance

Establish one canonical repository, WSL execution, environment strategy, Git safety, information isolation, minimal package structure, tests, and evidence conventions.

Gate: accepted and complete. No formal data, model, or training was introduced.

## Phase 1 — theory, literature, and minimal method specification

Freeze only what is needed to implement the first scientific baseline: claim wording; state and frames; causal PVT/IMU inputs; real-time forward ESKF teacher role; PINN final-state role; one non-tautological inertial physics residual; `SO(2)` action; the two mentor-defined hypotheses; minimal baselines/ablations; and the sealed-reference rule.

Phase 1 does not freeze WLS constants before data inspection, network widths/depths, optimizer budgets, large statistical matrices, tight coupling, or a full covariance architecture.

Gate: the user can explain the information flow and physics residual; the first-party evidence matrix covers the closest prior work; the concise method specification has no unresolved architecture contradiction. No formal data or model implementation occurs.

## Phase 2 — official data and conventional WLS/SPP

Confirm official source, license, routes, sensor fields, calibration, timing, and reference availability. Create route/session-level splits, hash manifests, and physical separation of the high-precision reference. Implement and independently check the conventional WLS/SPP PVT stream.

This phase owns signal selection, WLS corrections/weights, validity/FDE behavior, timing assumptions, and output covariance because those choices depend on the actual data interface.

Gate: deterministic PVT and IMU records with verified time/frame/unit contracts; reproducible splits and provenance; no high-precision influence on development.

## Phase 3 — deterministic navigation baselines

Implement SPP-only, pure INS propagation, and one fixed loose-coupled forward ESKF. Validate initialization, lever arm, time synchronization, covariance propagation, and controlled GNSS masking on synthetic fixtures and deployable data. A simple rule-adaptive ESKF is optional and cannot delay the fixed baseline.

Gate: the fixed ESKF is numerically reliable and is frozen as both teacher and baseline; the pure-IMU outage bridge is reproducible.

## Phase 4 — ordinary RTK-free learned baseline

Build one ordinary non-equivariant loose-coupled causal student. Train state means from forward-ESKF weak labels plus one independent inertial residual, including causal 20/30-second GNSS masks. Start with the simplest adequate temporal encoder; architecture and training budget are chosen from deployable-only development behavior and compute constraints.

Mandatory comparisons: teacher ESKF, pure-IMU bridge, and the same student without the physics residual. Do not add tight coupling, joint `Q/R/bias` learning, or a predictive covariance head here.

Gate: the model runs recursively without teacher inputs; the physics residual is non-tautological; teacher copying, zero correction, route leakage, and numerical failure are reported.

## Phase 5 — gravity-aware equivariant IMU model

Implement the `SO(2)` representation and exact property tests. Compare the Phase 4 network with capacity-matched ordinary, rotation-augmented, and `SO(2)` variants while keeping inputs, data, training budget, and selection policy fixed.

Gate: the mathematical property test passes and any claimed benefit is separated from capacity and augmentation. `O(2)` remains deferred.

## Phase 6 — complete primary model and final freeze

Integrate only components that showed independent value. Run the minimum sufficient route/orientation/receiver/degradation experiments and multiple fixed seeds. Freeze the final method, checkpoints, splits, metrics, statistical unit, and claims.

Predictive covariance is added only if the mean-state model is stable and the mentor confirms it as a core deliverable. A causal GNSS time-series branch or tight-coupled extension requires separate controller approval and does not block the primary paper.

Gate: reproducibility and leakage audits pass; all final choices are recorded without viewing sealed reference error.

## Phase 7 — sealed evaluation and reporting

Open the sealed reference once under the frozen protocol. Evaluate the two mentor-defined hypotheses, report failures and limitations, complete the reproduction package, and write the paper.

Gate: no post-reference method change is presented as confirmation. Any changed method requires a new untouched route; otherwise the result is explicitly exploratory.
