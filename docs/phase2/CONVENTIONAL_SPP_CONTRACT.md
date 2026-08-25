# Phase 2 conventional WLS/SPP contract

Status: frozen for the first conventional implementation; no receiver-native PVT or high-precision result was inspected to choose these settings.

Machine-readable profile: `../../config/phase2/gps_l1ca_spp_profile.json`.

## Minimal signal choice

The main chain uses only GPS L1 C/A pseudorange `G:C1C` with the matching `S1C` quality field and optional `D1C` retained for a later velocity-valid extension. The deployable scan found at least four `G:C1C` observations in 753 of 786 epochs. The other 33 epochs remain explicit invalid/no-solution PVT records. This is preferable for the minimum Phase 2 chain to introducing multiple constellation time systems, inter-system biases, and additional broadcast models merely to fill gaps.

RINEX interpretation follows IGS/RTCM RINEX 3.04. GPS signal, time, clock, group-delay, ionosphere, health, and broadcast-ephemeris equations follow IS-GPS-200N. The implementation is also compared structurally against RTKLIB standard positioning and broadcast ephemeris at pinned commit `71db0ffa0d9735697c6adfd06fdf766d0e5ce807`; RTKLIB is an independent conventional implementation reference, not a runtime dependency.

## Fixed model

- Output frame: broadcast WGS 84 Earth-centred Earth-fixed coordinates in metres. No unreported realization is invented.
- State: receiver ECEF position and GPS receiver clock bias in metres.
- Initialization: deployable RINEX approximate ECEF header position for the first epoch, then the preceding valid SPP state. The header value is never an evaluation target.
- Satellite state: causal healthy GPS broadcast ephemeris, at most 7,200 seconds from toe; transmit time is iterated from the pseudorange and broadcast clock.
- Applied corrections: broadcast clock polynomial, relativistic clock term, L1 TGD, Earth-rotation/Sagnac range, broadcast Klobuchar ionosphere, and Saastamoinen troposphere with fixed relative humidity 0.7.
- Excluded products: precise orbit/clock, augmentation, carrier smoothing, reference station data, receiver-native position as input, high-precision trajectory, and GNSS/IMU lever arm.
- Elevation mask: 5 degrees, fixed before any PVT cross-check. This is an operational project choice rather than a claim that the GPS specification recommends a receiver mask.
- Weight: fixed 3 m zenith code sigma mapped by elevation, plus broadcast URA, Klobuchar, and Saastamoinen model-variance terms. No truth- or residual-tuned weight is allowed.
- Solver: maximum 10 WLS iterations; position and clock update tolerances are 0.1 mm and 1 mm. No robust loss, RAIM/FDE, or residual threshold is added in this minimum chain.

## Validity and covariance

A valid record requires at least four finite positive `C1C` values, a causal healthy ephemeris for every used satellite, elevation at or above the fixed mask, a full-rank weighted normal matrix, finite state, and numerical convergence. Failure produces a timestamped record with a specific status and null state rather than dropping the epoch.

Position covariance is the position block of the inverse a-priori weighted normal matrix in ECEF metres squared. With only four observations there is no residual-degree-of-freedom scale estimate, so covariance is not silently rescaled. Each record includes covariance validity and construction method, used satellites, DOP when defined, post-fit residual RMS, correction flags, and velocity explicitly null/invalid until a separately checked Doppler solution exists.

The receiver NMEA solution is reserved for an independent non-high-precision cross-check after this profile and implementation are frozen. The mixed binary transport is scanned only for checksum-valid GGA sentences. Matching uses nearest integer UTC second because the two deployed formats differ by 0.006 seconds. Before comparison, the gross check is fixed at at least 100 matched valid epochs and median three-dimensional difference below 1 km; this can detect a time/frame/sign failure but is not an accuracy claim. It cannot change the signal choice, corrections, thresholds, or weights.
