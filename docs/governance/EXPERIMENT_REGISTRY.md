# Experiment registry

No scientific experiment was run in Phases 0 or 1. Entries for completed Phases 2–5 below are retrospective provenance only; they were added after their results existed and are not pre-registration.

## Retrospective provenance

| ID | Phase | Role | Status | Evidence |
|---|---:|---|---|---|
| P0-INFRA-001 | 0 | Infrastructure safety checks on synthetic strings/files | accepted; original entry | `docs/phase0/EVIDENCE_INDEX.md` |
| P2-DATA-SPP-RETRO-001 | 2 | Medium deployable-interface and conventional SPP provenance | accepted; retrospective | `docs/phase2/PHASE_HANDOFF.md` |
| P3-BASELINES-RETRO-001 | 3 | Deterministic INS/ESKF baselines and weak-label teacher | accepted; retrospective | `docs/phase3/PHASE_HANDOFF.md` |
| P4-PHYSICS-RETRO-001 | 4 | Matched ordinary physics 0/0.1 comparison; independent physics Huber worsened from `0.374354878` to `0.418493815` | accepted negative result; retrospective | `docs/phase4/PHASE4_REVIEW_REPORT.md` |
| P5-STRUCTURE-RETRO-001 | 5 | One-seed ordinary / rotation-augmented / strict SO(2) no-physics comparison | accepted mixed result; retrospective | `docs/phase5/PHASE5_REVIEW_REPORT.md` |

The Phase 4/5 entries must never be described as prospective evidence. Their runs may inform the stated prior limitation, but they do not replace the Phase 6 result-before freeze below.

## Phase 6 prospective entries

### P6-STRUCTURE-001 — multiple-seed no-physics structure selection

- **Status:** executed once through the frozen main entry on 2026-08-29; all nine required runs passed qualification.
- **Question:** which eligible no-physics structure is supported as the deployment primary by the frozen RTK-free development rule?
- **Allowed inputs:** only the accepted Medium deployable PVT/IMU/extrinsic, Phase 3 forward-ESKF weak-label means, and existing Phase 4/5 code and configuration. Teacher covariance, receiver-native PVT as truth, route/device/time identity, future information, Deep, and every high-precision reference are prohibited.
- **Route and splits:** `UrbanNav-HK-Medium-Urban-1`; unchanged Phase 4 chronological train `[0,458)`, validation `[488,626)`, and diagnostic `[656,764)` with both 30-step guards.
- **Arms:** ordinary, rotation-augmented, and strict SO(2), all with `physics_weight=0.0`; no new architecture or hyperparameter search.
- **Budget:** CPU-only 200 optimizer steps and checkpoint cadence 10; one run per arm and seed. All other accepted Phase 5 training settings remain unchanged. This result-before increase is frozen because the Phase 5 ordinary and rotation-augmented checkpoints both selected the previous 40-step budget boundary; Phase 6 must not extend 200 steps after seeing results.
- **Seeds:** `163736869`, `188003056`, `1899326961`. They are the nonnegative 31-bit big-endian prefix of SHA-256 over UTF-8 `P6-STRUCTURE-001:seed:{0,1,2}` and cannot be replaced after a failure.
- **Checkpoint rule:** minimum finite common unrotated validation weak-label state loss within each run.
- **Qualification gate:** every reported run must retain causality/prefix isolation, finite states, proper rotations, the frozen 60 m/s numerical bound, exact controlled-mask endpoints and recovery, GNSS-use/degeneracy checks, prohibited-input guards, and repository-external artifacts. Strict SO(2) must retain the float64 structural result and separately report the preregistered float32 failure; the latter is not relabeled as a pass.
- **Selection metric:** the existing common validation composite is the only structure-selection proxy; seed is the statistical unit. Independent diagnostic physics Huber is reported separately as a secondary diagnostic and is not a structure-selection gate. Neither metric is accuracy truth.
- **Selection rule:** after all qualification/safety gates pass, a structure may replace the ordinary default only if it has the unique lowest three-seed median validation composite and is the per-seed validation winner in at least two of the three matched seeds. Otherwise select the simpler ordinary model. No tuning follows an unstable or unsupported difference.
- **Abort/failure:** any nonfinite run, information-boundary violation, missing fixed-seed result, config/hash mismatch, or sealed-reference presence stops selection; the failed seed remains reported and is not replaced.
- **Configuration:** `config/phase6/final_freeze_v1.json`, SHA-256 `7d6ec6c2995c63d31a57544d2b9baff59d586807dd5fbddb66772d66fda79de0`.
- **Recorded result:** three-seed median validation was ordinary `1.563463370`, rotation-augmented `1.498722831`, and strict SO(2) `1.758293748`. Per-seed winners were rotation-augmented once and ordinary twice. Rotation-augmented had the unique lowest median but failed the required two-of-three seed-win condition, so the frozen rule selected ordinary by simplicity. Physics Huber remained report-only for this decision. These are weak-label proxies, not accuracy.

### P6-PHYSICS-001 — conditional matched physics test

- **Status:** executed for the ordinary structure selected by `P6-STRUCTURE-001`; the conditional rotation-augmented physics arm was not run.
- **Arms:** selected structure with `physics_weight=0.0` versus `0.1`, using the same three seeds, initialization, inputs, masks, optimizer order, 200-step budget, cadence 10, checkpoint rule, and qualification gates. The no-physics checkpoints from `P6-STRUCTURE-001` are reused rather than retrained. If ordinary or strict SO(2) is selected, reuse its `0.1` runs from `P6-INTERACTION-001`; if rotation-augmented is selected, run only its matched `0.1` three-seed arm and do not add it to the factorial comparison.
- **Selection rule:** physics `0.1` enters the deployment primary only if its paired three-seed physics-Huber median is lower than `0.0`, the improvement direction is consistent in at least two of the three paired seeds, and its three-seed median common validation composite is no higher than `0.0`. Otherwise freeze no-physics as the deployment primary. Negative results are retained; no weight search follows.
- **Confirmatory role:** regardless of deployment-primary selection, physics and strict SO(2) remain frozen Phase 7 confirmatory hypotheses through `P6-INTERACTION-001`.
- **Abort/failure and configuration:** same boundary and exact config/hash as `P6-STRUCTURE-001`.
- **Recorded result:** ordinary physics Huber median decreased from `0.117669680` to `0.105250198`, with improvement in two of three paired seeds; common validation median decreased from `1.563463370` to `1.546205799`. The frozen rule therefore selected `physics_weight=0.1` for the deployment primary. One seed's physics Huber worsened, and the accepted Phase 4 negative result remains unchanged; no general physics-benefit or accuracy claim follows.

### P6-INTERACTION-001 — controller-approved minimum 2x2 confirmatory freeze

- **Status:** executed once for all six missing physics runs; both no-physics arms were reused from `P6-STRUCTURE-001` as required.
- **Factors:** exactly `structure={ordinary, strict_so2}` x `physics_weight={0.0, 0.1}`. Rotation-augmented is excluded from this factorial.
- **Matched boundary:** use the same three fixed seeds, Medium inputs and splits, initialization policy, masks, optimizer settings and order, 200 optimizer steps, cadence 10, checkpoint rule, qualification gates, and prohibited-information boundary as `P6-STRUCTURE-001` and `P6-PHYSICS-001`.
- **Reuse:** ordinary/no-physics and strict-SO(2)/no-physics come directly from `P6-STRUCTURE-001` and are not retrained. The only new factorial runs are ordinary/physics and strict-SO(2)/physics, one run for each of the three seeds: six runs total.
- **Role:** preserve the core strict-SO(2)+physics hypothesis and identify the direction of the structure-by-physics interaction. For each seed and separately for common validation composite and independent physics Huber, report `(strict_physics - strict_no_physics) - (ordinary_physics - ordinary_no_physics)`, its three-seed median, and sign count. These are RTK-free proxy directions, not truth-referenced accuracy effects.
- **Non-selection rule:** this 2x2 does not participate in the three-structure selection, cannot replace the deployment primary, and triggers no physics-weight, architecture, seed, budget, or checkpoint tuning. Phase 7 evaluates only these already frozen arms with the sealed reference and reports results whether positive or negative.
- **Rotation-augmented boundary:** it receives a physics `0.1` arm only if selected by `P6-STRUCTURE-001`, solely for the matched deployment-physics decision in `P6-PHYSICS-001`; that arm is not added to this factorial.
- **Abort/failure and configuration:** same boundary and exact config/hash as `P6-STRUCTURE-001`.
- **Recorded result:** the validation difference-of-differences median was `-0.103235364` with two negative and one positive seed direction. The independent physics-Huber difference-of-differences median was `+0.019837602` with all three directions positive. The RTK-free interaction evidence is therefore mixed and does not support a combined strict-SO(2)+physics benefit claim. It did not enter deployment selection.

### P6-DEGRADATION-001 — minimum reference-free robustness checks

- **Status:** executed as report-only diagnostics; it did not enter model selection.
- **Checks:** existing causal 20-second and 30-second outages plus the single GNSS-available condition below. No scenario grid is authorized.
- **Deployable-metadata low-quality stratum:** use only valid-PVT epochs in the fixed diagnostic split `[656,764)`. An epoch is included when its deployable `dop.hdop` is at or above the empirical 75th percentile **or** its deployable `postfit_residual_rms_m` is at or above the empirical 75th percentile. The empirical percentile is the element at one-based rank `ceil(0.75 n)` after ascending sort, with inclusive threshold comparison.
- **Frozen Medium audit:** the accepted PVT file hash `c0ee6282191a3fc08676e6be32554a9bf3ef0c25b35c22f097e7b93a1b5d0723` has 108 valid diagnostic epochs. The frozen thresholds are HDOP `1.827707154949175` and residual RMS `32.6901790767606 m`. The **union of those two worst-quartile conditions** contains 53 epochs; the 53/108 union is not itself described as one worst quartile.
- **Failure rule:** for each required seed/checkpoint, the selected epochs and both outage rollouts must remain causal, finite, proper-rotation, below the frozen 60 m/s numerical bound, and must not collapse to reject-all-GNSS behavior. Any failure is retained and stops deployment-primary eligibility.
- **Reported proxies:** report the weak-label state loss and independent physics Huber on the fixed stratum, plus the gate outcomes. These are deployable-only diagnostics, not truth-referenced accuracy, and do not enter the `P6-STRUCTURE-001` selection rule.
- **Claims:** this is only a `deployable-metadata low-quality stratum` for robustness reporting. It does not enter model selection and is not a complete definition of urban severity or truth-referenced positioning accuracy.
- **Recorded result:** every required arm/seed passed the finite, causal, nondegenerate robustness gate. For the designated deployment checkpoint, all 53 selected epochs were finite, GNSS corrections were not all zero, weak-label Huber was `1.547270674`, and independent physics Huber was `0.106839470`. These are proxy diagnostics only.

Future entries must be created before execution and state the question, allowed information, route/split IDs, primary comparison, metric, seed policy, configuration hash, and abort condition. Add fields only when the experiment actually needs them. A retrospective entry must be labeled retrospective and cannot substitute for pre-registration.

Legacy smoke results are not imported or eligible as a Phase 6 prospective baseline.
