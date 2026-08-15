# ADR-0001: One Windows physical repository, executed through WSL

- Status: accepted for Phase 0; re-evaluate only with representative Phase 2 data-path evidence
- Date: 2026-08-15

## Context

The requested physical workspace is `E:\rtkfree-equivariant-gnss-ins`, available inside Ubuntu-24.04 as `/mnt/e/rtkfree-equivariant-gnss-ins`. Two runnable source trees would create drift and ambiguous provenance.

## Decision

Maintain exactly one source repository at the Windows path and execute reproducible commands from its WSL mapping. No runnable WSL-home clone or synchronized source mirror is allowed. Approved future data, virtual environments, caches, and run outputs must be outside Git and should use WSL-native storage.

## Evidence

Measured on 2026-08-15: WSL2 can read/write the mapped path; the mount is 9p/DrvFS with no `metadata` option; case aliases resolve; chmod from 0777 to requested 0600 remains 0777; symlink creation/resolution works; one self-cleaning 8 MiB sequential write probe measured 78.34 MiB/s. That microprobe is not a training benchmark.

## Consequences

The location is adequate for source, small configuration, tests, and reviewed evidence. It is not trusted for POSIX permissions, case-sensitive naming, secrets, sealed reference material, or high-volume training I/O. All names must avoid case-only distinctions. Phase 2 must benchmark representative data loading from approved external WSL-native roots before data-intensive work. If source-tree behavior itself proves unsuitable, a later ADR may migrate the *single* canonical repository to WSL-native storage and document a Windows `\\wsl.localhost\Ubuntu-24.04\...` mapping; migration must replace, not duplicate, the runnable tree.
