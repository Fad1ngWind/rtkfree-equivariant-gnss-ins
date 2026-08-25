# Phase 2 data start and isolation plan

Status: frozen at Phase 2 acceptance. The selected Medium deployable inputs, daily broadcast navigation, unified provenance manifest, RINEX diagnostics, and IMU/time contract have been prepared and checked outside the repository.

Machine-readable source record: `../../config/phase2/urbannav_medium_source_lock.json`.

## Minimal development start

- Official source: IPNL-POLYU UrbanNav repository and its linked public downloads.
- Access record: the authors describe the dataset as open-source/open-sourcing and publicly available, but do not list a specific license text. This project may use the official download for non-commercial research and local development, will cite the official source, will not invent a license name, and will not redistribute raw or derived data through the public repository.
- Development session: `UrbanNav-HK-Medium-Urban-1`, TST collection on 2021-05-17, with official RINEX v3.02, Xsens MTi-10 IMU, and calibration entries.
- Official repository version: `refs/heads/master` resolved by `git ls-remote` on 2026-08-25 to `075f96b6a6d9252b37486ecb175b4ae690c56f54`.
- Route role: Medium is the normal development start chosen before incident `INC-P2-20260825-001`; the incident review found no evidence that reference information influenced an experimental choice.
- Related route: `UrbanNav-HK-Data20190428` shares the TST route/version lineage, so it cannot demonstrate independent cross-route generalization while Medium is development data. This is a route-correlation constraint, not an incident downgrade.

Repository-hosted calibration files are pinned to the recorded commit. A later upstream revision requires an explicit source-record update.

The official extrinsic source contains multiple static-device calibrations. The deployable interface keeps only the positive-selected `ANTENNA_T_IMU` record as a data-minimization choice; the complete official source remains in external staging for provenance. The selected translation unit is interpreted as metres from the official file's internal ROS static-transform examples plus ROS REP-103, with the qualification that the unit is not annotated beside this key. Phase 2 WLS/SPP does not apply the lever arm. Record `INC-P2-20260825-002` is retained for audit continuity but is controller-classified as an ordinary boundary note, not an information-security incident.

## External data boundary

- Deployable data: `/home/fadingwind/rtkfree-data`, WSL-native ext4, mode `0700`.
- Run output: `/home/fadingwind/rtkfree-runs`, WSL-native ext4, mode `0700`.
- Sealed reference: `/var/lib/rtkfree-sealed-reference`, WSL-native ext4, root-owned mode `0700`.

Raw data, prepared records, outputs, and reference bytes remain outside Git. The sealed-reference directory is empty and unreadable by the development user. Phase 2 does not download high-precision reference data.

## Acquisition and provenance

The source record contains the exact official source URL, source-check date, repository commit, observed retrieval date, byte size, and one SHA-256 value for each Medium input already obtained. The GNSS ZIP observed on 2026-08-25 is 22,526,745 bytes with SHA-256 `b265d9983533b2c038ba2ec46c90a4c078bef5119fef128e6b843af3a1bc5fe2`.

That ZIP digest records the bytes actually obtained. It is used once to identify the same existing local ZIP; it is not a claim that a future dynamically generated transport container must have identical bytes. The ZIP will not be downloaded again.

Preparation writes one repository-external `provenance_manifest.json`. It records source, retrieval date, size, SHA-256, source version, and the size and SHA-256 of every selected output. Per-file receipt layers and repository copies of raw-data manifests are not required.

The read-only local check found the recorded ZIP bytes in both existing staging directories. The main staging path was reused because it also held the direct inputs; both staging directories remain unchanged. No preservation copy, move, rename, or deletion was performed.

## Positive extraction boundary

The positive list contains only:

- `UrbanNav-HK-Medium-Urban-1.ublox.f9p.splitter.obs` as the confirmed main RINEX observation input; and
- `UrbanNav-HK-Medium-Urban-1.ublox.f9p.splitter.nmea` for receiver-native PVT cross-checking only.

The known faulty `UrbanNav-HK-Medium-Urban-1.ublox.f9p.obs` cannot be the main input, based on the official missing-observation issue. Other receiver records are not permanently rejected: they remain unread in the external ZIP and may later support an independent deployable-data check.

Only positive-listed basenames are located and opened. Selected entries must be regular ZIP members with relative, traversal-free paths; absolute, drive-like, `..`, backslash, duplicate-basename, directory, and symbolic-link selections are rejected. Unselected members are not extracted or parsed. High-precision reference, post-processed navigation, ROS bag, and unrelated sensor products are outside the positive list.

Deployable diagnostics confirmed the main observation input independently of its filename. It is mixed-system RINEX 3.03 in GPS time with 786 consecutive 1 Hz epochs from 02:33:13.006 to 02:46:18.006, no nonpositive intervals or gaps above 1.5 seconds, and 21–34 satellites carrying pseudorange per epoch (median 27). Declared and parsed satellite counts agree at every epoch. The observed systems are BeiDou, Galileo, GPS, QZSS, GLONASS, and SBAS. The RINEX receiver and antenna identity fields are blank; this is recorded as unavailable metadata and is not inferred. Other receiver records remain available in the external ZIP for later independent checks.

The GNSS transport contains no selected broadcast-navigation input. The fixed source record therefore uses BKG IGS Data Center's anonymous-HTTPS entry for DLR's daily RINEX 3 mixed broadcast-navigation file `BRDM00DLR_S_20211370000_01D_MN.rnx.gz`, covering UTC 2021-05-17 (day 137). It was acquired once outside the repository: 1,244,184 bytes with SHA-256 `904e70cc2b0fe61e2751f216196b7a6f9769d932e8d7868ebffcc3789b012955`. The same unified external manifest records the source and output. No precise orbit or clock product is used.

For data freezing, use pinned first-party repository metadata and listed official acquisition entries, and do not open or use high-precision reference content. This does not prohibit ordinary official-metadata or literature searches that do not expose or use reference results.

## Primary sealed holdout

`UrbanNav-HK-Deep-Urban-1` is independently approved as the Phase 7 primary sealed holdout. It was selected before development results using this deterministic rule:

1. exclude development routes, the same route/version lineage, and any route whose reference content influenced development;
2. require separate conventional RINEX, low-cost IMU, required non-reference calibration, and reference entry categories;
3. require official reference coverage without a known partial-validation limitation;
4. require a scene directly matching the urban-canyon/GNSS-degradation question and at least 600 seconds of official duration as an operational block-formation threshold, not a statistical-power claim; and
5. if multiple routes qualify, choose the shortest, then break a tie by route ID.

At the pinned official commit, Deep is the only eligible route. Harsh has a partial-reference limitation, Tunnel is shorter than the operational threshold and is a special stress condition, and Tokyo lacks the needed available extrinsics and has a more complex package boundary.

No Deep file is downloaded, read, hashed, trained on, tuned on, or used for normalization, thresholds, seeds, early stopping, or model selection during development. The present freeze covers only the route identity, official source commit, and acquisition-entry category. No Deep content hash exists because no Deep file has been obtained. Phase 6 must audit the complete method freeze before any Phase 7 controlled retrieval.
