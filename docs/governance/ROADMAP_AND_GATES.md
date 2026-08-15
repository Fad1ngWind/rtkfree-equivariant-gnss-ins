# Frozen Phase 0–7 roadmap and gates

This order is user-frozen. No phase starts until the controller accepts the preceding gate. Later descriptions are plans, not completed work. The scientific constraints are in `FROZEN_RESEARCH_CHARTER.md`.

## Phase 0 — infrastructure and governance

Deliver one canonical source tree; reproducible WSL entry; truthful environment/locking strategy; local Git safety; clean-room boundary; executable runtime and publication guards; frozen charter; evidence-backed handoff. During executor work and before controller acceptance: no formal data, model, training, commit, remote, or public release. After formal acceptance, the controller—not the Phase 0 executor—updates the research notes, creates the intentional initial commit, and synchronizes only through the license/IP and public-release gates.

## Phase 1 — theory, first-party literature, and method specification

With the user learning and judging decisions step by step, retrieve then-current first-party sources and freeze: state and observation definitions; frames/coordinates/units/time; group action; network inputs/outputs and causal boundaries; genuinely independent PINN residuals; self-/weak-supervised losses with exact signals; information policy; observable and degenerate solutions; pre-registered ablations; success/failure criteria; and development/final evaluation protocol.

Gate: MC-001 through MC-005 are resolved through documented Phase 1 study and decisions; no formal-data implementation or high-precision influence occurs. These items do not need answers before Phase 1 starts, but Phase 1 cannot pass without them.

## Phase 2 — official data and standardized WLS/SPP

Acquire official RINEX, IMU, and calibration material only after confirming dataset version, license, scenario/route IDs, sensor contents, and official download source. Record provenance, license, cryptographic hashes, route-level splits, and time/frame/unit contracts. Physically seal RTK or other high-precision reference away from deployable data and ordinary environments. Build the controlled raw-observation layer and independently reproducible conventional WLS/SPP standardized PVT stream including time, solution status/state, satellite count, DOP, covariance, and available quality statistics.

Gate: deterministic deployable pipeline, immutable split/provenance manifests, leakage review, and WLS/SPP reproducibility tests pass. UrbanNav facts are re-confirmed rather than inherited.

## Phase 3 — conventional baselines

Implement and freeze SPP-only, INS-only, fixed loosely coupled ESKF, and rule-adaptive ESKF baselines. The explicit ESKF outputs position, velocity, attitude, and covariance. Use both synthetic fixtures and approved real deployable data to demonstrate that the navigation chain is numerically and operationally reliable before learned methods.

Gate: state/error equations, numerical tests, timing, covariance and failure criteria are auditable; baselines are not selected using sealed reference information.

## Phase 4 — RTK-free ordinary non-equivariant PINN/self-supervised baseline

Build an ordinary non-equivariant encoder connected to a differentiable explicit ESKF, plus a truly independent PINN residual. First hold IMU noise/process assumptions fixed and learn `R`/measurement weights; then assess whether bias or `Q` can be identified. Execute the six mandatory degeneracy/leakage tests.

Gate: non-reference losses and selection rules are frozen; PINN is not a renamed ESKF identity; copying, covariance inflation, all-GNSS rejection, `Q/R/bias` compensation, route memory, and leakage are directly reported.

## Phase 5 — strictly equivariant IMU model and fair comparison

Introduce the gravity-aware equivariant IMU representation, beginning with `SO(2)`. `O(2)` requires correct reflection treatment of angular-velocity pseudovectors. Compare against both the ordinary non-equivariant network and rotation augmentation under matched capacity, inputs, latency, training budget, seeds, and selection policy.

Gate: group actions and transformation/property tests pass; comparison is capacity-matched; independent benefit is stable without sealed-reference selection.

## Phase 6 — complete model, causal GNSS branch, ablations, and freeze

Integrate only separately identifiable components. Compare rule-based GNSS handling, TCN/GRU, and a lightweight causal Transformer fairly; retain the Transformer only for stable independent benefit. Run registered ablations, cross-domain/receiver, degradation, multi-seed, covariance, and all mandatory degeneracy tests. Freeze method, code, configuration, splits, selection rule, seeds, and paper assumptions.

Gate: reproducibility and leakage audits pass and an immutable final-freeze manifest is approved. Only then may the separate final-reference procedure be prepared.

## Phase 7 — one sealed RTK evaluation and final reporting

Open the sealed reference once under the approved protocol. Produce statistics, covariance calibration, failure cases, reproducibility package, paper, and final repository review. If any method, code, configuration, split, seed choice, or paper assumption changes after viewing the result, that route becomes development data and a new untouched route is required.
