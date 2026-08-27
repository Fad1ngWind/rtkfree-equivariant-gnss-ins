# ADR-0005: NumPy and a platform-scoped hashed pip lock for Phase 3

- Status: accepted for Phase 3 implementation
- Date: 2026-08-26

## Decision

Use NumPy 2.5.2 as the only Phase 3 third-party runtime dependency. Install it
in a WSL-native virtual environment from `requirements/phase3.lock`, which
requires a binary wheel and its measured SHA-256.

The lock is scoped to CPython 3.12 on Ubuntu-24.04 WSL x86_64. It was resolved
with system pip 24.0 from the official Python package index. The selected wheel
is
`numpy-2.5.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`,
with SHA-256
`3cdec01fa790a186d430433fdd4d4ffb70eed6f0eeb4bf05c8dbe2dce0a9bcb8`.

The project's mandatory dependency list remains empty so the accepted Phase 0
standard-library health check remains truthful. NumPy is declared under the
`phase3` optional dependency group, while the hash-bearing requirements file is
the normative Phase 3 installation input.

## Rationale

The fixed 15-state ESKF needs reliable dense matrix multiplication, linear
solves, covariance symmetrization, and eigenvalue checks. NumPy supplies this
minimum numerical substrate. Hand-writing a matrix package would add numerical
risk and implementation scope; SciPy and larger navigation frameworks are not
needed for the Phase 3 gate.

NumPy 2.5.2 was the current stable release reported by the official package
index on the decision date, and its official metadata includes Python 3.12.
The wheel was downloaded without installation before its hash was recorded.

## Reproduction

Create the environment outside the repository and install only from the lock:

```bash
python3.12 -m venv ~/rtkfree-venvs/phase3
~/rtkfree-venvs/phase3/bin/python -m pip install -r requirements/phase3.lock
```

Official sources:

- `https://pypi.org/project/numpy/`
- `https://numpy.org/doc/stable/release.html`
