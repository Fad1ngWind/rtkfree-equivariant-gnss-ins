# Phase 5 evidence index

Status: **ACCEPTED by the controller on 2026-08-28**.

All data, logs, checkpoints, and diagnostic JSON files remain outside the repository. No sealed/high-precision reference, teacher covariance, route/file/device identity, or future information was used.

## Frozen repository artifacts

- Contract: `docs/phase5/PHASE5_IMPLEMENTATION_CONTRACT.md`
- Fair-comparison profile: `config/phase5/gravity_aware_so2_v1.json`
- Group action and strict model: `src/rtkfree_equivariant_gnss_ins/phase5_group.py`, `phase5_layers.py`, `phase5_student.py`
- Phase 4 reuse adapters: `phase5_augmentation.py`, `phase5_learning.py`, `phase5_config.py`
- Bounded runners and diagnostics: `scripts/phase5/`
- Repository-external artifact audit: `phase5_validate.py` and `verify_medium_artifacts.py`

## Repository-external evidence

| Purpose | Absolute WSL path | Key SHA-256 |
|---|---|---|
| Primary smoke | `/home/fadingwind/rtkfree-runs/phase5/smoke_user_20260828` | `summary.json` `7ac890f761a58753bb7b3a1ecb6dce19f1fdfbf5b53627f5d4efbd9591e23897` |
| Independent smoke repeat | `/home/fadingwind/rtkfree-runs/phase5/smoke_user_repeat1_20260828` | exact match across 8 audited byte/tensor artifacts |
| Interrupted development attempt | `/home/fadingwind/rtkfree-runs/phase5/development_user_20260828` | only `normalizer.json`; not a scientific run |
| Primary complete development | `/home/fadingwind/rtkfree-runs/phase5/development_user_retry1_20260828` | `summary.json` `24a49c18469be633d8bb236387074002c19f93b99291a713e10f9006a0df2553` |
| Independent complete development repeat | `/home/fadingwind/rtkfree-runs/phase5/development_user_repeat1_20260828` | exact match across 11 audited byte/tensor artifacts |
| Initial float32-only diagnostic audit | primary development directory, `diagnostics.json` | `6be5b44585b917886b6c912576b313c029e951d2a1b4d5f1ae2841abff6d3388` |
| Retained dual-dtype diagnostic audit | primary development directory, `diagnostics_dtype_audit.json` | `d81d74303a5b792059c6b43b42af3074fe9b512bd576019076d800e29843a4cc` |

The common train-only normalizer hash is `f08d10ab35c1bd909296fe0bdca6169bf068b9cc3168730e03de1a6c019ace0b`.

## Development training hashes

| Variant | Training JSONL SHA-256 | Best checkpoint tensor SHA-256 | Final checkpoint tensor SHA-256 |
|---|---|---|---|
| ordinary | `a50d9d2af861849e24e114b2b556c1206f459777d9c36facee71a5740d923d9e` | `a33ccb0fadc1369fc8bdf8732c95e933f841f13ba0f62dae894dd43c101b3079` | same as best |
| rotation-augmented | `793e903f3e0b859787ebae02141c3ecf8e3898ae885161e45bab33ee3999fcdd` | `b1808a9b6d90d824b1f2b62d98b3888a66827d6aa800081e8d400c1da3a1e53d` | same as best |
| strict SO(2) | `d2509d8c6b453703a8965f277761229e74ecc84ffbd16f526bbd4729e4273496` | `b0b9f2b23b6f09365fcc902106800410849fd4decdf9374112f029189de310c7` | `1c231108bcd212166b3ca7e70cad21d8874ae0d4667c7cf678df1ec233b37903` |

## Verification record

- User-run `bash scripts/healthcheck.sh`: 146 tests passed in the locked CPU environment; public-release guard passed.
- Existing Phase 5 semantic/validator subset after diagnostic wording correction: 8 tests passed with warnings treated as errors.
- Artifact comparison: primary/repeat development summaries, logs, normalizer, selection results, and checkpoint tensors were exact; the repeat was not diagnosed again because tensor identity makes that redundant.
- No third smoke, third development, or repeated diagnostic was performed.

The legacy schema-1 field `diagnostics_passed=true` in `diagnostics_dtype_audit.json` represented reference-free safety plus float64 structural classification only; the same file explicitly records `float32_complete_mapping_gate_passed=false`. It must not be cited as an overall numerical-equivalence pass.

The controller independently reproduced the 146-test health check and public-release guard, then reran the repository-external verifier. The development pair matched across all 11 audited training artifacts and the smoke pair across all eight. Formal acceptance covers implementation, structural float64 equivariance, fair comparison, and reproducibility; it does not convert the retained float32 failure or weak-label metrics into a benefit claim.
