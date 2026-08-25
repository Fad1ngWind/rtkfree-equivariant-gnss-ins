# Phase 2 deployable field contract

Status: frozen at Phase 2 acceptance for Phase 3 input. This contract uses no high-precision trajectory or reference error.

## Authoritative format sources

- RINEX observation and broadcast-navigation interpretation follows IGS/RTCM RINEX 3.04: `https://files.igs.org/pub/data/format/rinex304.pdf`.
- IMU field semantics follow the ROS Noetic `sensor_msgs/Imu` definition pinned at `https://raw.githubusercontent.com/ros/common_msgs/noetic-devel/sensor_msgs/msg/Imu.msg`.
- IMU header semantics follow the ROS Noetic `std_msgs/Header` definition pinned at `https://raw.githubusercontent.com/ros/std_msgs/noetic-devel/msg/Header.msg`.
- Dataset-specific IMU noise names and declared parent units come from the pinned official `xsens_imu_param.yaml` recorded in the source manifest.
- Only the positive-selected `ANTENNA_T_IMU` record from the official mixed extrinsic source is deployable. The complete mixed source is staging-only.

## Time contract

The common serialized time field is `timestamp_ns_utc`: signed 64-bit integer nanoseconds since the Unix epoch in UTC.

- RINEX native time is GPS. For this session, convert the RINEX calendar epochs to UTC by subtracting the `LEAP SECONDS` value 18 read from the acquired broadcast-navigation header. Do not hard-code a timeless GPS–UTC constant.
- IMU primary time is the integer nanosecond value in `field.header.stamp`. `%time` is retained only as bag/receipt timing metadata and never replaces sensor time.
- The ROS-epoch interpretation of `field.header.stamp` is corroborated by its alignment with the same session after the RINEX leap-second conversion. This is a dataset-specific evidence-backed interpretation, not a claim that every ROS clock is wall-clock UTC.
- GNSS spans UTC `2021-05-17T02:32:55.006000000` through `02:46:00.006000000`.
- IMU sensor stamps span UTC `2021-05-17T02:32:55.548635005` through `02:46:01.021894931`.
- Their continuous overlap is 784.457364995 seconds. Preserve all source records, but set `within_common_interval` deterministically. A Phase 3 fusion run starts at the first PVT epoch at or after the first IMU sensor stamp and ends at the last PVT epoch at or before the last IMU sensor stamp.

## RINEX observation contract

- Format: mixed-system RINEX 3.03 observation data.
- Native time system: GPS; standardized time: `timestamp_ns_utc` as above.
- Observation values remain in their RINEX-declared codes. Code observations begin with `C`, carrier phase with `L`, Doppler with `D`, and signal strength with `S`.
- Available systems and declared observations are recorded in `urbannav_medium_source_lock.json`; WLS may use only an explicitly documented supported subset.
- Empty receiver and antenna identity header fields remain null. They must not be inferred from the basename.
- Approximate header coordinates are deployable initialization metadata only, never truth or an evaluation target.
- Every standardized PVT record must state its terrestrial coordinate frame, units, solver status, used constellations/signals, satellite count, correction flags, residual quality, and covariance validity.

## IMU source and standardized record

The source is a 41-column flattened ROS `sensor_msgs/Imu` CSV in frame `/imu`.

Required standardized fields:

| Field | Type/unit | Source/meaning |
|---|---|---|
| `timestamp_ns_utc` | int64 ns | converted `field.header.stamp`; primary ordering key |
| `receipt_timestamp_ns` | int64 ns, nullable | `%time`; transport cross-check only |
| `sequence` | uint32 | `field.header.seq` |
| `frame_id` | string | normalized source `/imu` identifier |
| `orientation_xyzw` | float64[4] | source quaternion in ROS x/y/z/w order; auxiliary, not an INS state estimate |
| `angular_velocity_radps` | float64[3] | ROS x/y/z angular velocity, rad/s |
| `linear_acceleration_mps2` | float64[3] | ROS x/y/z linear acceleration, m/s² |
| three covariance fields | float64[9] each | source row-major covariance about x/y/z |
| `valid` | bool | required numeric fields finite and timestamp strictly increasing |
| `timing_gap` | bool | sensor interval greater than twice the session median; flag only, no deletion/interpolation |
| `within_common_interval` | bool | timestamp lies in the deterministic GNSS/IMU overlap |

The repository-external standardized representation is CSV with vector components suffixed `_x`, `_y`, `_z` (and orientation `_w`) and each row-major covariance flattened with suffixes `_0` through `_8`. This is a serialization detail only; the units and frame semantics above are unchanged.

Full-file evidence: 314,194 rows; sequences 666379–980572 with no discontinuity or missing ID; no missing or non-finite required vector value; median sensor interval 2,497,911 ns; nominal rate approximately 400.335 Hz; no nonpositive interval; 29 intervals above twice the median; maximum quaternion norm error about `7.1e-13`. Sequence continuity means the 29 timing flags are not classified as missing samples.

## Calibration contract

- Body frame is the IMU frame, with source-declared axes x right, y forward, z up.
- The deployable GNSS/IMU extrinsic record is `gnss_imu_extrinsic.json`, selected only from official key `ANTENNA_T_IMU`. Its matrix direction is retained exactly as labeled by the source: GNSS antennas to IMU.
- The translation unit is `m`. This is an evidence-backed interpretation: the same official calibration file passes translation columns from matrices of the same format into ROS static-transform examples, while ROS REP-103 defines metre as the SI length unit. The unit is not explicitly annotated beside `ANTENNA_T_IMU`, so the record preserves both the basis and that qualification.
- Phase 2 WLS/SPP does not apply this lever arm and continues to output the GNSS antenna phase-center position. Before Phase 3 first uses it, one minimal coordinate-transform/sign composition test must confirm the mapping direction and sign; the label alone is not sufficient evidence.
- The Xsens calibration declares gyroscope parent unit `rad/s` and accelerometer parent unit `m/s^2`. Preserve the official scalar names `gyr_n`, `gyr_w`, `acc_n`, and `acc_w` and their axis/average grouping. Their more specific stochastic dimensions are not inferred in Phase 2 and they are not used by WLS/SPP.

## Standardized PVT interface required from Phase 2 WLS/SPP

Each record must include at minimum:

- `timestamp_ns_utc`, native GPS epoch metadata, and `within_common_interval`;
- ECEF position in metres with an explicit terrestrial frame identifier;
- ECEF velocity in m/s when valid, otherwise null with a separate velocity-valid flag;
- receiver clock bias in metres and, when solved, clock drift in m/s;
- position/velocity covariance with frame, units, construction method, and validity;
- solution status, number of used satellites, used systems/signals, inter-system bias states if any, correction flags, DOP values when defined, and post-fit residual RMS;
- no receiver-native position and no high-precision reference value as a WLS input.

The frozen first implementation uses GPS `G:C1C` with broadcast WGS 84 ECEF output, as specified in `CONVENTIONAL_SPP_CONTRACT.md` and `gps_l1ca_spp_profile.json`. Later extensions must remain separate experiments and cannot silently change this Phase 2 baseline.
