# RTK-Free Equivariant Physics-Informed GNSS/INS Fusion

This repository is the clean-room research workspace for a GNSS/INS fusion system that learns **without high-precision trajectory supervision**. Phase 0 contains governance, reproducibility scaffolding, and executable information-isolation guards only. It contains no research model, formal dataset, trained weight, or result.

## Canonical workspace

- Physical repository: `E:\rtkfree-equivariant-gnss-ins`
- WSL execution path: `/mnt/e/rtkfree-equivariant-gnss-ins`
- Required runtime: Ubuntu-24.04 on WSL2, Python 3.12
- One source tree only. Do not clone or mirror a second runnable source tree into WSL.

Use WSL for all reproducible commands:

```bash
cd /mnt/e/rtkfree-equivariant-gnss-ins
bash scripts/healthcheck.sh
```

## Phase boundary

The current status is **Phase 0 formally accepted by the controller**. Local Phase 1 work is authorized under its step-by-step user-learning gate; no formal data, scientific implementation, or training is authorized yet. The detailed gate is in `docs/governance/PROJECT_STATUS.md`, the frozen scientific boundary is in `docs/governance/FROZEN_RESEARCH_CHARTER.md`, and the controller acceptance record is indexed in `docs/phase0/EVIDENCE_INDEX.md`.

## Information policy

Only reproducible conventional WLS/SPP PVT and low-cost IMU are deployable information sources. RTK, PPK, post-processed high-precision trajectories, or equivalent references must never influence inputs, losses, pseudo-labels, filter updates, training choices, early stopping, model selection, seed selection, or any pre-freeze decision. A sealed reference may be opened only for one final evaluation after the method, code, configuration, splits, and paper assumptions are frozen.

See `docs/governance/DATA_AND_INFORMATION_POLICY.md` before adding any data-related code or configuration.

## Repository safety

The repository deliberately excludes raw observations, RINEX, HDF5, weights, caches, environments, run logs, secrets, private chats, and unreviewed notes. Before any commit, run:

```bash
python3 scripts/public_release_check.py --staged
```

The configured pre-commit hook runs the same check. Public publication remains blocked until the license and release checklist receive explicit approval.
