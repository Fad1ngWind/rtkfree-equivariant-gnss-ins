# Phase 6 evidence index

Status: **ACCEPTED by the controller on 2026-08-29**. Paths under `~/rtkfree-runs` are repository-external and contain no reference truth.

| Frozen statement | Evidence | Controller check |
|---|---|---|
| Phase 6 rules were registered before results | `../governance/EXPERIMENT_REGISTRY.md`; `PHASE6_DECISION_AND_EXPERIMENT_CONTRACT.md`; config SHA-256 `7d6ec6c2995c63d31a57544d2b9baff59d586807dd5fbddb66772d66fda79de0` | compare registry, config, and `FREEZE_MANIFEST.json` |
| Exactly three fixed seeds, 200 steps, cadence 10, three no-physics structures and the minimum 2x2 were used | external `summary.json`, SHA-256 `54dec97a895772df171c31d2dcee9de178853c31158c9f620e27bb8750112718` | run `scripts/phase6/verify_medium_freeze.py` on the external run |
| All 15 required arm/seed runs passed qualification; no rotation-physics arm was added | external per-run `summary.json` and `diagnostics.json`; root summary | verify all arm/seed sets and qualification flags |
| Ordinary was selected by the prospective stability/simplicity rule | root `structure_decision`: rotation-augmented median `1.498722831` but one seed win; ordinary median `1.563463370` and two seed wins | independently recompute with `phase6_selection.select_structure` |
| Physics 0.1 was selected only for the ordinary deployment primary | root `physics_decision`: physics-Huber medians `0.117669680` to `0.105250198`, two-of-three directions; validation medians `1.563463370` to `1.546205799` | independently recompute with `phase6_selection.select_physics` |
| The minimum 2x2 gives mixed RTK-free interaction directions | validation interaction median `-0.103235364` with 2/3 negative; physics-Huber interaction median `+0.019837602` with 3/3 positive | inspect `interaction_report_only`; confirm it never enters selection |
| Strict SO(2) remains a confirmatory arm with the numerical limitation retained | every strict run: float64 structural classification passed; float32 complete deployed rollout failed | inspect each strict `diagnostics.json`; do not relabel float32 as pass |
| The low-quality metadata stratum is report-only and finite | 53/108 union; designated checkpoint weak-label Huber `1.547270674`, physics Huber `0.106839470`, gate passed | inspect designated checkpoint diagnostics and `enters_model_selection=false` |
| The 20/30-second masks remain causal and finite | designated diagnostics: exact 20/30 valid-PVT masks, bit-identical prefixes, finite rollouts, GNSS recovery | inspect `controlled_outages`; do not interpret proxy distances as accuracy |
| Only the final designated configuration was reproduced | final and reproduction tensor SHA-256 `bf4fc42ca6eb5b8a7acb90c7e3d59eda6b65e27868e8e74cec257ef00dc8fb44`; identical file and training-log hashes | compare paths recorded in `FREEZE_MANIFEST.json` |
| Final checkpoint is a bounded-training freeze | designated `best_optimizer_step=200`, equal to the common frozen budget; comparison remains matched but convergence is not established | confirm the budget is not extended and retain this limitation in Phase 7 interpretation |
| The read-only artifact verifier passed | validator output: `PASS`, selected ordinary, physics `0.1`, 81 artifacts | rerun verifier without data or reference arguments |
| Locked repository checks pass | 162 tests with warnings as errors; public-release guard pass; `git diff --check` pass | rerun `bash scripts/healthcheck.sh` and `git diff --check` |
| Phase 7 is frozen before sealed access | `PHASE7_SEALED_EVALUATION_PROTOCOL.md`: one reference-bearing invocation, exact endpoints/support/claim rules, pre-access checklist, immutable boundary | review and approve the evaluator and exact command before acquiring Deep/reference bytes |
| Deep/reference and new receiver/route data remained unused | root/run flags, frozen input hashes, no reference argument in Phase 6 entries | inspect source, config, external summaries, and repository candidates |

The external root is `~/rtkfree-runs/phase6/medium_final_freeze_user_20260828`. Model weights, normalizer, logs, diagnostics, and run summaries remain outside Git.
