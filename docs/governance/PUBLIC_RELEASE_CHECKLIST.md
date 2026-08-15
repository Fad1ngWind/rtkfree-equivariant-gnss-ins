# Public GitHub maintenance and release checklist

Public publication is currently blocked. Passing the automated scanner alone does not authorize publication.

Before the first public push and every release:

- [ ] Controller has accepted the current phase and explicitly authorized publication.
- [ ] Approved license text replaces `LICENSE.md`; dependency and dataset licenses are compatible.
- [ ] `git status --short`, staged diff, and full tracked-file list have been human-reviewed.
- [ ] `bash scripts/healthcheck.sh` passes from the canonical WSL path.
- [ ] Exact staged blobs pass `python3 scripts/public_release_check.py --staged`.
- [ ] Every staged index mode is reviewed; symlinks (`120000`), gitlinks/submodules (`160000`), and unsupported modes are absent.
- [ ] Dedicated secret scanning and large-file/history scanning pass (tool not selected in Phase 0).
- [ ] No raw/formal data, RINEX, HDF5, weights, environment, cache, logs, artifacts, archives, personal paths, private chats, or unreviewed notes are tracked or present in history.
- [ ] No legacy code, model, configuration, or result has been copied.
- [ ] Documentation distinguishes measured facts, frozen policy, and assumptions.
- [ ] Any published result maps to an accepted experiment and immutable evidence.
- [ ] Remote URL and visibility are verified before push; branch protection/review settings are configured.
- [ ] Only an explicitly reviewed branch/tag is pushed; `--mirror`, `--all`, and internal `refs/codex/*` are prohibited for public publication.
- [ ] Release tag, changelog, commit hash, environment-lock hash, and signed data-manifest hash are recorded.

Maintenance policy: small reviewed commits, protected main branch, no force-push to protected history, dependency updates through reviewed lock regeneration, and immediate credential rotation/history remediation if a secret is exposed.
