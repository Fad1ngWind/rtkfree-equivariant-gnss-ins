# Phase 7 sealed-evaluation protocol

Status: **FROZEN_PENDING_CONTROLLER_REVIEW**. This protocol does not authorize Deep or reference access.

## One-time interface

The Phase 7 evaluator has one allowed scientific interface:

```text
frozen Phase 6 run root
+ sealed Deep deployable PVT/IMU/extrinsic root
+ sealed Deep reference root (scoring only)
-> one new repository-external evaluation directory
```

It may expose paths and an output directory as command-line arguments. It must not expose training, seed, checkpoint, architecture, physics-weight, mask-schedule, metric, threshold, or arm-selection arguments. Before the one permitted reference-bearing invocation, it may be exercised only on synthetic fixtures without Deep or reference bytes.

The evaluator must load the Phase 6 config and every arm/seed checkpoint fixed by `FREEZE_MANIFEST.json` and the external Phase 6 `summary.json`. The deployment primary remains `ordinary_p01`, with seed `163736869` as the designated deployed checkpoint. Ordinary/no-physics, ordinary/physics, strict-SO2/no-physics, strict-SO2/physics, and rotation-augmented/no-physics are all reported; reference ranking cannot select another primary.

## Frozen scoring questions

All estimators use identical deployable inputs, initialization boundaries, validity accounting, and causal masks. High-precision reference samples are available only to the scoring layer after an estimator output has been written.

1. **Controlled outages.** Form eight paired start targets at 120, 180, 240, 300, 360, 420, 480, and 540 seconds after the first standardized Deep epoch. For each target, use the first valid PVT epoch at or after the target and before target plus 10 seconds. Apply left-closed, right-open 20-second and 30-second masks from that same causal start. A target is eligible only if initialization exists and the complete 30-second interval plus its endpoint are covered. The primary endpoint is horizontal relative-displacement error: the norm of estimated minus reference horizontal displacement from the common start to the masked endpoint. The mandatory comparator is the frozen pure-IMU bridge initialized from the same causal start state. Report every block and duration; do not replace an unavailable block.
2. **GNSS-available urban degradation.** On all reference-alignable epochs where standardized PVT is valid and no artificial mask is active, compute horizontal position RMSE for the deployment primary and the identical frozen forward ESKF. Report conventional SPP on the same validity set. The primary comparison is deployment primary versus ESKF; SPP is a reported comparator, not a selection oracle.

Horizontal error is the Euclidean norm of north/east error in one common local tangent frame. Reference interpolation is scoring-only, bracketed in time, and never extrapolated. The frozen Phase 3 time, frame, and lever-arm conventions are reused without re-estimation. Any format adapter needed after controlled retrieval may only implement these conventions; it cannot alter estimator outputs, masks, validity sets, or metrics.

## Units, support, and decision rules

- The field-data statistical unit is the sealed route/session (`n=1`). The three fixed seeds are algorithmic replicates for training variability. Outage blocks and epochs are repeated measurements, not independent routes.
- No inferential confidence interval or route-generalization claim is made from one route. Report per-seed values, the three-seed median and range, and all per-block outage values.
- Each outage duration requires at least six of the eight frozen targets. The GNSS-available comparison requires at least 300 reference-alignable valid-PVT epochs spanning at least 300 seconds. Failure to meet support makes that question unevaluable; it does not permit a new route, schedule, threshold, or metric.
- The directional comparison threshold is `0 m`: ties do not count as a lower-error direction. This is only a sign threshold for a within-session descriptive comparison and is not a practical-significance margin. A within-session descriptive improvement direction for either question requires a lower error for the designated checkpoint, a lower three-seed median, and the same lower-error direction in at least two of three seeds. The outage direction must satisfy this separately at both 20 and 30 seconds.
- Even when that descriptive direction is satisfied, the single sealed route/session and absence of an inferential interval do not support practical significance, route generalization, or broad superiority.
- A nonfinite output, improper rotation, missing required artifact, hash mismatch, information-boundary violation, or insufficient support is retained as a failure or unevaluable result. Seeds, epochs, or blocks are never silently dropped or replaced.

Physics and strict-SO2 effects are reported as matched per-seed differences using the frozen arms. Their direction may be summarized by the paired three-seed median and count, but they cannot change the deployment primary or trigger tuning. Every positive and negative result is reported.

## Pre-access checklist and immutable boundary

Before acquiring or opening any Deep or reference byte, the controller must record that:

1. Phase 6 has been formally accepted and `FREEZE_MANIFEST.json` plus all method/config hashes match;
2. the external Phase 6 root summary, normalizer, all required checkpoints, and the designated reproduction match their frozen hashes, with no failure artifact;
3. the evaluator passes synthetic, no-reference tests and has no training or selection path;
4. the exact reference-bearing command, sealed input locations, new output location, and access time are written to an access ledger; and
5. Deep deployable data and reference are acquired only under the separately approved Phase 2 isolation plan.

The reference-bearing evaluator is invoked once. After the first Deep or reference byte is opened, no method, parser semantics, code, config, checkpoint, seed, mask schedule, metric, support threshold, statistical unit, arm role, failure rule, or claim-selection rule may change. An interface incompatibility or failed run is reported to the controller; it is not repaired by viewing results and rerunning.
