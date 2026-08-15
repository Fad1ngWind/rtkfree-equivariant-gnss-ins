# PHASE_HANDOFF — Phase 0 rework

## Stage objective

Deliver a clean, auditable, WSL-runnable Phase 0 foundation with fail-closed information/publication controls and the user-frozen scientific route. No scientific implementation or experiment is in scope.

## Completed rework

- Replaced permissive path normalization with exact relative-path validation.
- Unified and expanded ignore/scanner/test risk classes, including content signatures.
- Added staged type-change enumeration and fail-closed Git mode/read/parse handling.
- Strengthened nested runtime configuration and raw/resolved/symlink path policy without a sealed override.
- Replaced `assert` health gates with explicit optimized-mode checks and an early non-Python-3.12 rejection.
- Added bytecode suppression and cleaned repository cache residue.
- Added 22 tests, including eight real self-cleaning temporary-Git integrations.
- Made explicit RTK/PPK markers default-deny with only exact reviewed `rtkfree` component exemptions.
- Added one NUL-safe batch check against the actual `.gitignore` for all candidates, including tracked/force-added paths.
- Restored the exact Phase 0–7 route and added the frozen scientific charter.
- Corrected MC-001–005 to Phase 1 study/freeze work and expanded direct degeneracy risks/tests.
- Documented config schemas, log redaction/retention, test levels, evidence refresh and bootstrap names.
- Audited unreachable local objects without destructive GC.

## Change list

- Safety metadata: `.gitignore`, `.gitattributes`, `.env.example`, `.githooks/pre-commit`
- Runtime guards: `policy.py`, `release_guard.py`, `public_release_check.py`
- Health: `healthcheck.py`, `healthcheck.sh`, Phase 0 config
- Tests: policy/classification tests plus `test_release_guard_git.py`
- Governance: frozen charter, restored roadmap, status/data/environment/naming/risk/mentor/release documents and ADR clarifications
- Delivery: revised acceptance report, evidence 05–09, index, research-note rework entry, and this handoff

## Verification commands

From Ubuntu-24.04:

```bash
cd /mnt/e/rtkfree-equivariant-gnss-ins
bash scripts/healthcheck.sh
sh .githooks/pre-commit
python3 -B scripts/public_release_check.py --staged
git diff --cached --check
git fsck --full --unreachable --no-reflogs
git status --short --branch
git remote -v
```

Recorded result: optimized health gates pass; 22 tests pass; full-candidate and exact staged scans pass. Final scope/count/mode/cache/size/history results are in evidence 07; adversarial categories are in evidence 06; controller remediation is in evidence 08; historical and current object reachability snapshots are in evidence 09 and 10.

## Canonical engineering location

Keep `E:\rtkfree-equivariant-gnss-ins` as the only physical source repository and `/mnt/e/rtkfree-equivariant-gnss-ins` as its WSL execution path. The source/test tree is usable. Case-insensitive naming and ineffective chmod make DrvFS unsuitable for secrets or sealed references; the 8 MiB microprobe is not a training benchmark. Approved environments/data/runs are future external WSL-native roots. No second runnable source tree exists.

## Information isolation

The repository has no final-reference loader or override. Runtime policy rejects explicit high-precision aliases across raw/resolved paths and nested configs and rejects symlink chains. Release safety checks exact staged blobs and modes and blocks risky aliases/types/content, symlink/gitlink, and read/parse failures. Semantic leakage still requires later provenance/lineage and independent review.

## Known risks and non-blocking unknowns

- No unresolved Phase 0 implementation blocker is known.
- MC-001–005 are Phase 1 work and must be resolved by the Phase 1 gate before later implementation; they are not pre-entry user homework.
- Scientific lock/GPU/CUDA/datasets/WLS/method remain unverified by design.
- The unreachable local synthetic blob is not publishable history; no destructive GC was attempted.
- License, institutional IP/patent posture and dedicated release tooling remain unresolved; public release remains blocked.

## Prohibited actions confirmed

- No modification of old projects and no copied old code/config/model/artifact/result.
- No formal data, training, model implementation, or high-precision result access.
- No canonical commit, remote, GitHub repository, or push.
- No Phase 1 entry and no claim that Phase 0 is formally accepted.

## Suggested controller review

1. Review `FROZEN_RESEARCH_CHARTER.md` and the restored roadmap against the user-frozen order.
2. Run the verification commands above from the exact WSL path.
3. Inspect the exact staged modes/blobs and evidence 05–09.
4. Confirm canonical commit count/remotes are zero and distinguish the unreachable local blob from reachable publish history.
5. Confirm `.env.example` and DrvFS policy prohibit repository-local secrets.
6. If Phase 0 is accepted, authorize a separate initial commit. Do not authorize public publication until license/IP and release gates pass.

## Phase 1 prerequisites

- Controller formally accepts Phase 0.
- Phase 1 preserves the charter, clean-room boundary and no-high-precision-supervision policy.
- Phase 1 retrieves first-party sources and guides the user through MC-001–005; answers are outputs of Phase 1, not prerequisites for starting it.
- Any third-party scientific import waits for an approved resolver ADR and verified generated lock.

## Executor state

**READY_FOR_REVIEW**
