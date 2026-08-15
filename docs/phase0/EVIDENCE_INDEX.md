# Phase 0 evidence index

Evidence is small, text-only, sanitized, and independent of formal data or high-precision results. Numbered names are the documented Phase 0 bootstrap exception in `NAMING_AND_TRACEABILITY.md`.

| Acceptance claim | Classification | Evidence | Recheck |
|---|---|---|---|
| New project began empty/outside Git | measured | `evidence/00_scope_and_initial_state.txt` | historical transcript |
| Windows toolchain inventoried | measured | `evidence/01_windows_toolchain.txt` | `python --version`; `git --version` |
| Ubuntu-24.04 WSL2, Python 3.12 and mapped path usable | measured | `evidence/02_wsl_toolchain_and_mount.txt` | WSL status/tool/path checks |
| Legacy projects contributed facts only, no copied code/result | measured boundary | `evidence/03_legacy_governance_facts.txt` | read-only filename/policy audit only |
| DrvFS case/chmod/symlink/micro-write behavior | measured with limitation | `evidence/04_wsl_mount_probe.txt`, ADR-0001 | `python3 scripts/probe_wsl_mount.py` |
| Explicit optimized health gate and 22 tests pass | measured | `evidence/05_healthcheck_and_tests.txt` | `bash scripts/healthcheck.sh` |
| Required guard categories and failure modes are covered | measured | `evidence/06_negative_guard_and_ignore.txt` | unit/integration tests |
| Final candidate/index/file/history review is clean | measured at handoff | `evidence/07_final_repository_review.txt` | final commands listed there |
| Controller A–C findings have explicit remediations | implementation map | `evidence/08_controller_rework_matrix.txt` | code/docs/tests review |
| Old synthetic probe is unreachable and not publishable history | measured | `evidence/09_git_object_reachability.txt` | `git fsck`; refs/commit/remote review |
| Current post-restaging unreachable blob/tree/commit counts | measured snapshot | `evidence/10_third_rework_object_snapshot.txt` | rerun `git fsck`; counts may grow with superseded/app snapshots |
| Controller independently re-ran the final gates and formally accepted Phase 0 | measured decision | `evidence/20260815T061939Z_controller_acceptance.txt` | controller-only gate decision |
| Scientific charter and exact Phase 0–7 route are frozen | policy | `../governance/FROZEN_RESEARCH_CHARTER.md`, `../governance/ROADMAP_AND_GATES.md` | document review |
| MC-001–005 belong inside Phase 1 and license/IP remains unresolved | policy/pending | `../governance/MENTOR_CONSULTATIONS.md` | controller/mentor review |
| Degeneracy, leakage and observability risks map to direct gates | policy/pending tests | `../governance/RISK_REGISTER.md` | Phase 1 specification review |

Evidence paths are relative to `docs/phase0/`; governance documents are under `docs/governance/` and ADRs under `docs/adr/`.
