# Frozen research charter

Status: policy-frozen research boundary for Phase 0 review. Scientific claims remain unverified.

## Claim and information boundary

The core hypotheses are all pending verification: improved robustness under urban GNSS degradation; generalization across installation orientation, route, scene, and receiver; and calibrated uncertainty. None is a conclusion. The project must not claim superiority over SPP, conventional filtering, or any learned baseline until the pre-registered phase gates support that claim.

Training, tuning, early stopping, architecture choice, model/seed selection, pseudo-labels, filtering updates, or loss design must not use RTK, PPK, post-processed high-precision trajectories, or any derived proxy. Preferred terminology is “without high-precision trajectory supervision” or precisely defined “RTK-free learning.” Direct SPP fitting is not purely unsupervised.

## Frozen main information chain

The primary chain begins with low-cost-receiver RINEX and uses a frozen, reproducible, independently validated conventional WLS/SPP implementation to generate standardized PVT. Its contract includes time, solution status, satellite count, DOP, covariance, and all available reproducible quality statistics, together with the conventionally available position/velocity state. Low-cost IMU is the other primary source.

Receiver-native PVT is used for cross-checking, as an external comparator for the independently computed SPP, and for cross-receiver generalization tests. It is not the canonical main-chain target. Raw per-satellite observations remain in a controlled audit/future-extension layer. The main line does not pre-commit to tight coupling, a Set Transformer, or learned raw-observation processing.

The explicit loosely coupled ESKF remains responsible for the final position, velocity, attitude, and covariance. Direct absolute-position regression from IMU plus SPP is not the default architecture and cannot silently replace the ESKF.

## Candidate output boundaries and identification order

Candidate IMU-side learned outputs are limited to constrained bias corrections, motion increments, process-noise scales, or an inertial prior with its covariance; every output requires explicit frames, units, bounds, and an ESKF insertion point. Candidate GNSS-side learned outputs are limited to bounded measurement covariance `R`, robust weights, or anomaly probabilities. Learned PVT corrections, absolute-position corrections, and common-bias corrections are prohibited. Neither branch may expose a hidden absolute-trajectory target or bypass the explicit filter.

Identification proceeds in this order:

1. Hold IMU bias/process assumptions fixed and learn only measurement covariance `R` or equivalent weights.
2. Establish an ordinary non-equivariant IMU model plus a genuinely independent PINN residual.
3. Introduce gravity-aware equivariance and compare at matched capacity.
4. Only after the parts are separately identifiable may joint `Q`/`R`/bias adaptation be considered.

A PINN term must express a separately stated physical residual with units, discretization, and assumptions. It must not rename the same mechanization identity already executed by the ESKF.

## Symmetry boundary

Gravity-preserving `SO(2)` yaw symmetry is the preferred starting point. `O(2)` may be considered only after reflections are defined correctly, including the pseudovector transformation of angular velocity. `SO(3)` or any larger group requires an explicit gravity/frame justification and transformation tests; the word “equivariant” alone is not a method specification.

## Causal GNSS branch comparison

Rule-based quality handling, TCN/GRU, and a lightweight causal Transformer must be compared under the same deployable inputs, latency, capacity, and selection protocol. The Transformer is retained only if it gives stable independent benefit. It is not a default contribution.

## Observability and mandatory degeneracy tests

A constant common SPP bias may be unobservable without independent absolute information. The method and paper must state this limit rather than imply recovery by architecture alone.

Every applicable baseline and learned model must be directly tested for:

1. copying SPP rather than fusing information;
2. inflating covariance without improving calibrated uncertainty;
3. rejecting all GNSS updates;
4. compensating among `Q`, `R`, and bias so parameters lose identifiable meaning;
5. memorizing routes, receivers, timestamps, or environment identity; and
6. direct or derived high-precision/data-split leakage.

These are gate tests, not optional discussion points.

## Literature and novelty boundary

EqNIO, PINK-GINS, and AutoW are user-provided historical boundary markers only. Phase 1 must retrieve and read then-current first-party papers and official materials before fixing theory, comparisons, or novelty language. Weight learning, PINN constraints, or PINN plus GNSS/INS cannot by themselves support a priority or “first” claim.

## Dataset boundary

UrbanNav version, license, scenario/route identifiers, sensor contents, and official download source must be re-confirmed in Phase 2 from official sources. Phase 0 downloads nothing and makes no dataset-compatibility claim.

## Freeze consequence

Phase 1 turns this charter into a first-party-literature-backed method specification with the user learning and judging each decision step by step. Later implementation may not change these boundaries silently; any change requires an ADR, experiment-registry impact assessment, and controller gate review. After a sealed final result is viewed, a method change invalidates that route as a formal test route.
