# PHASE_HANDOFF — Phase 2 deployable data and conventional SPP

## Stage objective

Establish one official, auditable, reference-isolated development session and produce reproducible conventional WLS/SPP PVT plus standardized low-cost IMU records for Phase 3, without using high-precision trajectory information for any development choice.

## Completed

- Fixed UrbanNav official repository commit `075f96b6a6d9252b37486ecb175b4ae690c56f54` and recorded the official open/public access wording without inventing a license name.
- Prepared only `UrbanNav-HK-Medium-Urban-1` as the development session; raw, deployable, derived, run, and sealed-reference storage remain outside the public repository.
- Recorded source URLs, retrieval dates, sizes, and SHA-256 for the acquired GNSS transport, low-cost IMU, IMU parameters, mixed official calibration source, selected GNSS/IMU extrinsic, and daily mixed broadcast navigation.
- Positively selected only the confirmed F9P splitter observation and NMEA members from the GNSS ZIP. Unselected members remain unextracted and unread; the known faulty non-splitter F9P observation is not a main input.
- Retained ordinary path traversal, absolute-path, duplicate-basename, directory, and symbolic-link checks on selected archive members without introducing a general data platform.
- Confirmed RINEX identity, GPS time, epoch coverage, 1 Hz continuity, observation types, per-signal satellite coverage, and broadcast-navigation coverage from deployable data.
- Confirmed the full 314,194-row IMU field, unit, frame, time, continuity, finite-value, and common-interval contract.
- Reduced the deployable mixed calibration to the selected `ANTENNA_T_IMU` record. Translation unit is `m`, based on the official file's internal ROS transform examples plus ROS REP-103, while preserving that the unit was not annotated beside that key.
- Froze Deep as the primary sealed holdout by deterministic deployable metadata before development results. No Deep byte has been downloaded, read, or hashed.
- Implemented the frozen standard-library GPS L1 C/A `C1C` broadcast WLS/SPP profile with causal healthy ephemeris, clock/relativity/TGD, Sagnac, Klobuchar, Saastamoinen, fixed elevation weighting, explicit invalid epochs, DOP, residual quality, and a-priori covariance.
- Checked the WLS algebra on a synthetic fixture and compared the frozen output against checksum-valid receiver-native GGA without using it as truth or changing configuration.
- Generated standardized repository-external PVT and IMU records twice with unchanged byte hashes.

## Data and split boundary

- Development: `UrbanNav-HK-Medium-Urban-1`.
- Related TST lineage: `UrbanNav-HK-Data20190428` cannot serve as independent cross-route generalization evidence while Medium is development data.
- Primary sealed holdout: `UrbanNav-HK-Deep-Urban-1` route identity, upstream commit, and acquisition-entry category only. Content hashes do not exist because no file has been obtained.
- High-precision reference: none downloaded, opened, parsed, trained on, tuned on, or used for validation. The root-only sealed-reference directory remains empty.
- Archive allowlist: `UrbanNav-HK-Medium-Urban-1.ublox.f9p.splitter.obs` and `.nmea` only. No reference, post-processed navigation, ROS bag, or unselected receiver member was extracted.
- Phase 2 direct sources contain only GNSS observations, receiver-native NMEA cross-check data, low-cost IMU, selected calibration inputs, and conventional broadcast navigation. Precise orbit/clock products are excluded.

## Frozen interfaces and measured outputs

### PVT

- Profile: `gps_l1ca_broadcast_spp_v1`.
- Frame/unit: broadcast WGS 84 ECEF metres; receiver clock bias metres; velocity null with an explicit invalid flag.
- Records: 786 total, 753 valid, 33 explicit `insufficient_c1c_observations`.
- Output: repository-external `pvt.jsonl`, 1,640,270 bytes, SHA-256 `c0ee6282191a3fc08676e6be32554a9bf3ef0c25b35c22f097e7b93a1b5d0723`.
- Independent receiver cross-check: 778 checksum-valid GGA records, 744 matched valid epochs; median 3D difference 30.626 m and median horizontal difference 12.725 m. The pre-registered gross check passed. These are cross-check differences, not reference errors or performance claims.

### IMU

- Contract: `imu_contract_v1`; UTC Unix nanoseconds from `field.header.stamp`, `%time` retained as receipt metadata, `/imu` frame, ROS units and row-major covariances.
- Records: 314,194 total and valid; 29 timing-gap flags; 313,787 within the PVT/IMU common interval; no interpolation.
- Output: repository-external `imu.csv`, 114,764,045 bytes, SHA-256 `c2538e409091452c549c7bd380a6865e125b1adc4fdbd5e3c579dd3ad0c8e516`.

Phase 3 may consume only the standardized interfaces. WLS/SPP remains at the GNSS antenna phase center. Before first lever-arm use, Phase 3 must perform one minimal coordinate-transform/sign composition test; it must not infer direction solely from the source key name.

## Boundary records

- `INC-P2-20260825-001`: controller-reviewed controlled sealed-procedure deviation from a generic search-result summary; no evidence of experimental influence. The event record is retained without values, original summary, screenshot, or sensitive URL.
- `INC-P2-20260825-002`: controller-reviewed ordinary mixed-calibration data-minimization note, not an information-security incident. Other static-device transforms were not used.

Neither record changes Medium's development role or Deep's independently frozen holdout role.

## Verification

- Full standard-library suite: 42 tests pass with warnings treated as errors.
- `git diff --check`: pass.
- Public-release guard: pass after keeping parser/test RINEX labels as runtime-constructed strings; final controller should independently rerun it.
- Canonical `HEAD` and `origin/main` remain `3a66f40eeb27536be9e36f2b06664344e86ee963`.
- No second runnable source tree, model training, INS/ESKF implementation, learned model, commit, push, merge, or research-notebook completion claim was created by the Phase 2 executor.

## Controller review checklist

1. Recheck the source lock, exact selected archive members, external unified acquisition manifest, and observed hashes.
2. Reapply the Deep holdout eligibility rule to the pinned official metadata and confirm no Deep content exists locally.
3. Review both sanitized boundary records against their controller rulings.
4. Compare the frozen SPP profile with RINEX 3.04, IS-GPS-200N, and pinned RTKLIB structural evidence; run the synthetic tests.
5. Reproduce Medium PVT/IMU and compare the recorded output hashes and receiver-native cross-check report.
6. Run the full tests, `git diff --check`, and the public-release guard; inspect that external data and outputs are absent from Git candidates.
7. Confirm that Phase 3 receives only standardized PVT/IMU and the recorded calibration boundary.

## Executor state

**READY_FOR_CONTROLLER_REVIEW**

## Controller outcome

**ACCEPTED — 2026-08-26**

The controller independently reproduced the recorded PVT and IMU hashes, reran the receiver-native cross-check and all 42 tests in WSL Python 3.12, and found no Phase 2 gate failure or unnecessary expansion into later phases.
