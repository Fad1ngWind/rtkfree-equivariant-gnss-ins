# ADR-0003: Fail-closed separation of deployable and sealed information

- Status: accepted as policy; final-access mechanism intentionally absent
- Date: 2026-08-15

## Decision

Ordinary runtime code may load only conventional frozen WLS/SPP PVT and low-cost IMU information. High-precision trajectory references are isolated physically outside the repository and deployable roots, omitted from ordinary environments, rejected across raw/resolved paths and nested configuration values, and blocked from Git candidates by ignore, content-signature, exact staged-blob, and index-mode checks. Symlink chains have no Phase 0 runtime exception.

Phase 0 implements no override or final-reference loader. Phase 6 must design one behind an immutable freeze manifest and independent approval. Phase 7 may perform one sealed evaluation. Any post-view method change permanently downgrades that test route to development status.

## Limitations

Name-based guards cannot prove semantic non-leakage. Later phases require source provenance, transformation lineage, manifests, access logging, and independent review. This ADR records a defense-in-depth baseline, not a mathematical proof.
