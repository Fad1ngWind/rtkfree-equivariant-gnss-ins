# Naming, configuration, logs, tests, and evidence traceability

## Repository records

- ADRs: `docs/adr/NNNN-short-decision.md`
- Future phase evidence: `docs/phaseN/evidence/YYYYMMDDTHHMMSSZ_short-name.txt`
- Handoff: `docs/phaseN/PHASE_HANDOFF.md`
- Research notes: `科研手记.md`
- Experiment registry: `docs/governance/EXPERIMENT_REGISTRY.md`

The existing `docs/phase0/evidence/00_...` through numbered files are an explicit Phase 0 bootstrap exception created before the timestamp convention was frozen. They remain stable, ordered review records; future evidence uses UTC timestamps rather than mechanically renaming the bootstrap set.

## Configuration schema

Every executable config must contain `schema_version`, phase/status, and only documented fields. Phase 0 schema version is integer `1`; phase is integer `0`; status is one of `awaiting_controller_review`, `returned_for_rework`, `ready_for_controller_review`, or `accepted`; only the controller may set `accepted`. The current configuration is `accepted`. Formal data and training flags must be exactly `false`; deployable sources must exactly match the frozen ordered list. A semantic field change requires a schema bump, migration note, policy validation, and tests. Unknown runtime leaves/types fail closed.

## Future run identity and manifest

After experimentation is authorized:

`YYYYMMDDTHHMMSSZ_pN_<experiment-id>_<git-short>_<config-sha8>_s<seed>`

The external run manifest records UTC time, phase, registered experiment ID, full commit, clean/dirty state, config/schema hash, seed, environment-lock hash, data/split-manifest hash, exact command, host class without personal identity, status, and evidence links.

## Log location, redaction, and retention

Runtime logs live only under external `RTKFREE_RUN_ROOT`. They must not include raw observations, high-precision/reference values, tokens, credentials, private chat, usernames, hostnames, or personal absolute paths. Paths and identifiers are replaced with stable non-reversible labels before a reviewed excerpt becomes evidence. Every run declares retention owner/period and deletion rule; absent an approved rule, logs are not retained as public artifacts. Small Phase evidence is retained with the repository only after redaction and review.

## Test levels

1. Unit classification/policy tests: pure synthetic values and content.
2. Temporary-Git integration tests: force-add, exact staged blobs, type change, symlink, gitlink, read failure, and safe candidates; temporary repos are created only under ignored `.phase0_tmp` and removed.
3. Optimized health gate: `python -O` validates runtime/config invariants without `assert`.
4. Full candidate scan: current tracked and unignored worktree candidates.
5. Real hook/staged scan: exact index blobs and modes.
6. Final repository/history review: whitespace, sizes/types, ignored residue, secrets, object reachability, commits, and remotes.

## Evidence refresh rule

Refresh evidence whenever guard behavior, configuration schema, environment facts, test count, candidate scope, Git state, or a claimed result changes. During an unaccepted Phase 0 rework, replace the corresponding bootstrap evidence and retain the rejected-run failure summary. After phase acceptance, do not rewrite measured history; append a timestamped superseding record and update the index.

## Status semantics

- `planned`: registered, never started
- `running`: active and incomplete
- `failed`: technical execution criteria not met
- `completed_unreviewed`: outputs exist, not scientifically accepted
- `accepted`: explicitly approved at its gate
- `retired`: historical only and not a current baseline

Phase 0 registers no scientific performance experiment.
