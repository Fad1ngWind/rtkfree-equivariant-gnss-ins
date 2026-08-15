# Data and information policy

This policy is subordinate to and interpreted with `FROZEN_RESEARCH_CHARTER.md`.

## Allowed deployable information

Only the following may enter ordinary preprocessing, training, tuning, model selection, filtering, and deployment:

1. Low-cost IMU measurements and their non-reference calibration metadata.
2. A frozen, independently reproducible conventional WLS/SPP PVT stream and its deployable quality information.

Raw per-satellite observations may exist in a controlled data layer in later phases, subject to provenance and licensing review. Phase 0 downloads and processes no formal data.

## Prohibited pre-freeze influence

RTK/PPK/post-processed or otherwise high-precision trajectories must not enter network inputs, losses, pseudo-labels, filter updates, preprocessing choices, training or tuning, early stopping, model selection, architecture selection, random-seed selection, split decisions, or paper-assumption changes.

Derived values or metadata that encode such a reference are prohibited to the same extent as the original reference. Renaming or aggregating reference information does not make it deployable.

## Physical isolation

- The Git repository contains no formal data, raw observations, derived datasets, weights, or run artifacts.
- `RTKFREE_DATA_ROOT` and `RTKFREE_RUN_ROOT`, when later approved, must be absolute WSL-native paths outside the repository.
- An eventual sealed reference root must be a third, separately permissioned location outside both deployable data and run roots.
- `RTKFREE_SEALED_REFERENCE_ROOT` is absent from ordinary development/training environments. Phase 0 deliberately provides no code path that can open it.
- Because `/mnt/e` ignores POSIX chmod in the measured configuration, it must not store secrets or sealed reference material. Windows ACLs and/or a WSL-native sealed store must be reviewed before Phase 7.
- A repository-local ignored dotenv may contain non-secret machine convenience settings only. Secrets must be injected from an approved external store or controlled process environment.

## Logical isolation

- Runtime paths and JSON configurations pass through `policy.py`, which rejects obvious high-precision/reference markers.
- Repository candidates and exact staged blobs pass through `public_release_check.py`.
- `.gitignore` blocks common observation, trajectory, HDF5, model, archive, secret, private-chat, and runtime-output forms.
- The Git hook is defense in depth; human review, provenance manifests, and later dedicated secret/data scans remain mandatory.

## Final sealed evaluation

Access is permitted only after the method, code, configuration, data splits, random-seed selection rule, and paper assumptions are recorded in an immutable freeze manifest and independently approved. Phase 0 does not define an executable bypass. Phase 6 must add a reviewed, separate final-evaluation interface before Phase 7.
