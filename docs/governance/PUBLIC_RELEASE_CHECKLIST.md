# Public GitHub maintenance and release checklist

The owner authorized public visibility on 2026-08-15 under `All rights reserved`. The repository is not open source. Passing the automated scanner alone does not authorize an update.

Before the first public push and every release:

- [x] Controller has accepted Phase 0 and the owner has explicitly authorized public visibility for its reviewed foundation.
- [x] `LICENSE.md` states `All rights reserved` and makes clear that public visibility is not an open-source grant; Phase 0 has no third-party runtime dependency or dataset.
- [x] `git status --short`, staged diff, and full tracked-file list have been reviewed.
- [x] `bash scripts/healthcheck.sh` passes from the canonical WSL path with 22 tests.
- [x] Exact staged blobs pass `python3 scripts/public_release_check.py --staged`.
- [x] Every staged index mode is reviewed; only `100644` and `100755` occur.
- [x] Gitleaks 8.30.1 passes for the worktree and `HEAD` history after its official Linux x64 archive hash is verified; the largest tracked file is about 12 KB and no file over 1 MB is present outside `.git`.
- [x] No raw/formal data, RINEX, HDF5, weights, environment, cache, logs, artifacts, archives, personal paths, private chats, or unreviewed notes are tracked or present in the published branch history.
- [x] No legacy code, model, configuration, or result has been copied.
- [x] Documentation distinguishes measured facts, frozen policy, and assumptions.
- [x] No scientific result is included; result-to-experiment mapping is not applicable to this Phase 0 publication.
- [x] Remote `https://github.com/Fad1ngWind/rtkfree-equivariant-gnss-ins` is verified empty and `PUBLIC` before the first push.
- [ ] Configure and verify `main` branch protection immediately after the initial synchronization creates the branch; an empty repository has no branch to protect before that push.
- [x] Only the explicitly reviewed `main` branch will be pushed; `--mirror`, `--all`, and internal `refs/codex/*` are prohibited.
- [x] This is the initial repository publication, not a numbered scientific release; the accepted Phase 0 commit and empty dependency lock are recorded, while changelog, release tag, and data-manifest hash are not applicable.

Maintenance policy: small reviewed commits, protected main branch, no force-push to protected history, dependency updates through reviewed lock regeneration, and immediate credential rotation/history remediation if a secret is exposed.
