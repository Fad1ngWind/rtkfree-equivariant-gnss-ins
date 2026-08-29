# Mentor decisions and open questions

Only decisions that materially change the research claim or deliverable should be sent to the mentor. Dataset fields, implementation constants, and routine engineering choices are resolved from official sources and experiments in their owning Phase.

## Confirmed mentor direction

- Use GNSS and IMU together in the deployed learned estimator.
- Use a real-time forward GNSS/INS ESKF to create non-RTK weak pseudo-labels.
- The PINN produces the final navigation result.
- First make the core GNSS/INS method reliable; do not make ZUPT, NHC, or other vehicle-motion constraints part of the primary method.
- Evaluate 20-second and 30-second GNSS outages against pure IMU and GNSS-available behavior against ESKF.
- Loose and tight coupling may both be explored, but the mentor did not require both as simultaneous primary methods.
- Avoid an unqualified “unsupervised” claim when ESKF pseudo-labels are used.

## Resolved mentor question

### MC-001 — Is predictive covariance a core paper contribution?

The original project vision includes credible uncertainty, but the later mentor messages explicitly confirmed final PINN navigation states without separately confirming a covariance deliverable.

The proposed minimal policy is to establish mean-state feasibility first and add predictive covariance in Phase 6 only if it is a required contribution and the mean method is stable.

Question for the mentor: should calibrated predictive covariance be a required main contribution, or may it remain a secondary extension after the position/velocity/attitude results are established?

Decision relayed by the user on 2026-08-28: the mentor said predictive covariance may be deferred. Phase 6 therefore freezes a causal position/velocity/attitude mean-state method only. It will not add a covariance head, covariance loss, calibration metric, or uncertainty claim. Predictive covariance is future work.

## Conditional later consultation

- Target deployment hardware and latency budget are needed only before Phase 6 runtime claims.
- Tight coupling is reconsidered only after the loose-coupled baseline is stable; mentor permission to try both is not a current implementation requirement.

## Publication posture

The repository may remain publicly visible for personal research management but is not open source and remains `All rights reserved`. Institutional ownership, patent strategy, dependency licenses, and dataset licenses are reviewed again before scientific implementation, data-derived material, or results are published.
