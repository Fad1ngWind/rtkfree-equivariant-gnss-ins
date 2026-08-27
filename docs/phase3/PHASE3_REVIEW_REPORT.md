# Phase 3 controller review report

Date: 2026-08-27 (Asia/Shanghai)

Executor status: **READY_FOR_CONTROLLER_REVIEW**

Formal phase status: **accepted by the controller on 2026-08-27**

This report includes the controller-requested bounded rework of the process-noise source, right-attitude finite-difference validation, and locked-environment entry. No architecture extension was made.

## Scope assessment

Phase 3 implemented only SPP-only output, causal pure-INS propagation, one fixed loose-coupled real-time forward ESKF, and reproducible 20/30-second GNSS outage scenarios. It consumed only the Phase 2 frozen standardized PVT/IMU and selected calibration boundary. The optional adaptive ESKF and every learned or tightly coupled extension were left unimplemented.

## Gate evidence

- Coordinate, lever-arm, mechanization, initialization, ESKF error-state, covariance, update, and time-causality behavior are covered by synthetic fixtures before deployable replay.
- Process noise now uses the four controller-preselected official Xsens average-axis values. Runtime verifies the frozen `xsens_imu_param.yaml` SHA-256, bias random walks are nonzero, and the `Qc` to interval `Qd` mapping is explicit.
- A physical nonzero-Earth-rate finite difference confirmed that right-attitude `F[att,att]` must receive bias-corrected inertial angular rate. The one incorrect extra Earth-rate subtraction was removed; nominal mechanization was not changed.
- README and `healthcheck.sh` now require the WSL-native locked Python 3.12/NumPy 2.5.2 environment and fail before discovery with one activation instruction when it is absent.
- The full synthetic replay runs twice deterministically and its validator rejects a one-byte output alteration.
- Two independent deployable Medium runs are byte-identical for all five baseline files.
- Real output states remain finite; rotations remain proper and orthonormal; covariances remain exactly symmetric and positive semidefinite; controlled outage masks and recovery endpoints match the fixed half-open definition.
- INS-only remains a causal propagation with zero position updates, and both 20/30-second pure-IMU bridge intervals are reproducible.
- The complete 71-test suite, locked-environment health check, public-release guard, and diff whitespace check pass.

These facts support numerical reliability and reproducibility. They do not demonstrate navigation accuracy or superiority.

## Required controller judgment

After the precommitted process-noise correction, the fixed ESKF still has elevated deployable-data NIS. Among 731 valid updates, the median is `7.258662982605188`, 352 (`48.1532%`) exceed the three-degree-of-freedom 95% threshold, 288 (`39.3981%`) exceed the 99% threshold, and the maximum is `305.1263011258299`. The largest innovation RMS component is vertical. A naturally invalid SPP epoch is followed by the largest valid-update correction of about `324.85` m; the state remains finite, time causal, and covariance PSD. No update was gated and no R scaling, adaptation, or post hoc parameter change was introduced. These are explicit limitations of the fixed traditional baseline/weak-label teacher under SPP covariance and non-Gaussian urban errors.

The three bounded controller rework items are complete. Remaining statistical inconsistency is reported rather than tuned away and can be reviewed from the frozen deployable interface, fixed assumptions, and diagnostics without opening a high-precision reference.

## Claim boundary

No high-precision reference was used. Receiver-native PVT was not treated as truth. Consequently, this report makes no accuracy, calibrated-uncertainty, teacher-quality, or performance claim. Phase 4 must not start until the controller formally accepts or returns this fixed baseline.

## Controller disposition

The controller independently reproduced the fixed baseline in a new external run directory, matched all five hashes, and accepted the documented statistical mismatch as a frozen conventional-baseline limitation. Phase 4 may use the states only as weak pseudo-labels and may not use the current teacher covariance for loss weighting.
