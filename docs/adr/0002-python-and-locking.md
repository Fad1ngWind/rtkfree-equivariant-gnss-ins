# ADR-0002: Python 3.12 standard-library-only Phase 0

- Status: accepted for Phase 0
- Date: 2026-08-15

## Decision

Use Ubuntu-24.04 Python 3.12 as the only normative runtime. Phase 0 uses no third-party runtime package. The empty `requirements/phase0.lock` is the complete verified Phase 0 dependency closure, not a forecast of scientific dependencies.

The Windows Python 3.9 installation is not supported for reproducible project execution. No online resolution is needed for Phase 0. Phase 1 must select a resolver and create a real platform-scoped, hash-bearing lock before any third-party scientific import.

## Rationale

The available WSL runtime already provides Python 3.12, pip, `venv`, `tomllib`, and Git. uv and Poetry are absent. Choosing an unverified complex tool or fabricating versions would reduce rather than improve reproducibility.
