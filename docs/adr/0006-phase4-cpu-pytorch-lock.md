# ADR-0006: Phase 4 CPU-only PyTorch lock

- Status: accepted for Phase 4 implementation
- Date: 2026-08-27
- Scope: ordinary mean-state student development environment

## Context

Phase 4 needs automatic differentiation and a minimal recurrent network. The measured WSL host has 20 logical CPUs, 7.6 GiB RAM, and an RTX 4060 Laptop GPU with 8 GiB VRAM. The only development sequence has 764 one-hertz student steps, so a CUDA runtime is not necessary for the bounded smoke and development runs.

## Decision

Use CPython 3.12 in a new WSL-native environment at `~/rtkfree-venvs/phase4`. Lock NumPy 2.5.2 and the official Linux x86-64 CPU wheel for PyTorch 2.13.0, including every transitive wheel and SHA-256 in `requirements/phase4.lock`. Resolution used pip 24.0 and the official PyTorch CPU index documented by the PyTorch local-install guide.

Install only with:

```bash
python3.12 -m venv ~/rtkfree-venvs/phase4
~/rtkfree-venvs/phase4/bin/python -m pip install -r requirements/phase4.lock
```

## Consequences

- Phase 4 training is CPU-only and cannot silently depend on a host CUDA version.
- No torchvision, torchaudio, Lightning, navigation framework, experiment platform, or notebook stack is added.
- Model weights, package caches, and run outputs remain outside the repository.
- A later GPU environment would require a separate measured lock and a demonstrated compute need; it is not part of Phase 4.
