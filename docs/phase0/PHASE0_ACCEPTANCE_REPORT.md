# Phase 0 acceptance report — controller-requested rework

Date: 2026-08-15 (Asia/Shanghai)

Executor status: **READY_FOR_REVIEW**

Formal phase status: **not accepted; controller decision required**

## Scope and unchanged prohibitions

This rework repairs Phase 0 safety and governance findings. It does not implement a model, download formal data, train/tune, open a high-precision reference, write an old project, create a canonical commit/remote, push, enter Phase 1, or select a public license.

## Controller findings repaired

### Publication and runtime policy

- Repository paths are exact forward-slash relative paths; root dotfiles are preserved. Absolute, dot, empty, and traversal components fail closed.
- Any normalized component containing an explicit RTK or PPK marker is rejected across config keys, string values, and paths. Only exact reviewed `rtkfree` names are exempt; compound dangerous names are not.
- Ignore, scanner and tests cover environment/cache, RINEX including compressed/compact forms, HDF/HDF5, trajectory/reference aliases, weights/serialization, logs/backups/archives/secrets/chats/unreviewed material.
- Content signatures detect RINEX headers and HDF5 magic as defense in depth.
- Staged enumeration includes type changes. Index modes reject symlinks, gitlinks, and unknown modes; enumeration/read/parse failures become findings.
- Every candidate is also checked in one NUL-safe batch against the repository's actual `.gitignore`, including tracked/force-added paths; ignore-query failure blocks the scan.
- Runtime config/path policy handles camel/compact markers, supported recursive containers and PathLike; unsupported leaves/cycles/non-finite values fail closed. Raw/resolved paths and symlink chains are checked. No sealed-reference override exists.

### Health and reproducibility

- Health gates use explicit checks under `python -O`, not `assert`.
- They verify schema version, phase, pending status, both disabled flags, exact ordered deployable sources, empty dependency closure, Python range, and package version.
- Windows Python 3.9 produces a clear non-normative-runtime error before importing `tomllib`.
- Health, hook, and direct entry points disable bytecode writes; final review requires zero ignored cache residue.
- Python 3.12/empty Phase 0 lock and the one E: physical tree plus `/mnt/e` execution ADR remain intact.

### Frozen governance

- The user-frozen Phase 0–7 roadmap has been restored.
- `FROZEN_RESEARCH_CHARTER.md` records the unverified hypothesis, canonical WLS/SPP PVT plus IMU chain, receiver/raw-observation roles, explicit ESKF output, learned-output boundaries, identification order, independent-PINN rule, SO(2)/conditional-O(2) rule, causal GNSS comparison, observability limit, six mandatory degeneracy tests, literature/novelty limits, and UrbanNav Phase 2 verification requirement.
- MC-001 through MC-005 are Phase 1 learning/research/freeze tasks. They do not require pre-Phase-1 answers, but they block the Phase 1 gate and later implementation.
- License, institutional IP, ownership, patent intent, and data-license questions remain unresolved; public release is fail-closed.

## Verification results

- Ubuntu-24.04 Python 3.12.3 optimized health gate: pass.
- Unit and temporary-Git integration tests: 22 passed, 0 skipped, 0 failed.
- Full tracked/unignored candidate scan: pass.
- Root dotenv, explicit RTK/PPK, ignored bytecode/private-note, category force-add, staged type-change/symlink, gitlink, read-failure, and ignore-query-failure cases: blocked as intended.
- Safe staged candidate: no finding.
- Windows Python 3.9 negative runtime check: expected explicit failure.
- Final exact staged hook, whitespace, modes, sizes, secrets, ignored residue, commits, remotes and counts are recorded in evidence 07.

## Local object-store nuance

The earlier 51-byte synthetic HDF5-path probe is absent from the worktree/index and all reachable commits/proposed history, but remains as an unreachable local blob. It contains no formal data or secret. No destructive GC was used. Unreachable blobs do not travel in a normal explicit branch/tag push; mirror/all-ref pushes are prohibited. Evidence 09 records the audit.

## Measured platform conclusion retained

`E:\rtkfree-equivariant-gnss-ins` remains the single physical repository, executed at `/mnt/e/rtkfree-equivariant-gnss-ins`. DrvFS is case-insensitive in the probe and did not enforce requested chmod narrowing. The 8 MiB result is a microprobe, not a training benchmark. Secrets, reference material, environments, formal data and run outputs remain external; representative data I/O waits for Phase 2 approval.

## Unverified items and residual risk

Scientific libraries, GPU/CUDA, resolver-generated scientific lock, editable installation, formal datasets, official UrbanNav facts, WLS/SPP implementation, learned method, and dedicated third-party history/secret tooling remain unverified. Name/content guards are defense in depth and cannot prove absence of semantic leakage; later provenance, lineage, split manifests and independent review are mandatory.

## Assessment

All controller-identified Phase 0 blockers now have code, tests, policy, and indexed evidence. The implementation is ready for another independent controller review. This report does not declare Phase 0 formally complete.
