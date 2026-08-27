# Environment and dependency policy

## Measured Phase 0 environment

- Ubuntu 24.04.4 LTS under WSL2
- Python 3.12.3; pip 24.0; Git 2.43.0
- Python `venv` and `tomllib` modules import successfully
- `uv` and Poetry were not found
- Windows Python is 3.9.13 and is not the normative runtime

## Frozen strategy

Python is constrained to `>=3.12,<3.13` and `.python-version` records 3.12. Phase 0 has no third-party runtime dependencies, so `requirements/phase0.lock` truthfully records an empty closure and all checks use the standard library.

No Phase should hand-edit a pretend scientific lock. Before the first later Phase adds a third-party import, it must:

1. choose and record one resolver/lock format in a new ADR;
2. resolve from an approved index with network access;
3. capture exact versions, hashes, Python/platform scope, resolver version, and generation command;
4. verify environment creation in Ubuntu-24.04; and
5. document GPU/CUDA packages separately because they are hardware/platform-specific.

A virtual environment should live on WSL-native storage outside the repository. It must never be copied into or committed from the canonical tree.

## Verified Phase 3 numerical environment

- WSL-native virtual environment: `~/rtkfree-venvs/phase3`.
- Runtime: CPython 3.12 on Ubuntu-24.04 WSL x86_64.
- Only scientific dependency: NumPy 2.5.2.
- Normative installation input: `requirements/phase3.lock`, requiring the measured binary-wheel SHA-256.
- Per-session check: activate the environment and run `bash scripts/healthcheck.sh`; the script fails before test discovery unless Python 3.12 and NumPy 2.5.2 are active.

The dependency decision, platform scope, wheel filename, and hash are recorded in `docs/adr/0005-phase3-numpy-lock.md`.

## Not yet verified

No deep-learning library, CUDA stack, GPU package, editable package installation, or offline wheelhouse has been validated. These remain outside Phase 3.
