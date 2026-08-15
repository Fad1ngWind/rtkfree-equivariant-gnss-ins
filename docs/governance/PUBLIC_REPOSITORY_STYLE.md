# Public repository presentation style

Status: owner-approved maintenance rule from 2026-08-15.

## README purpose

The public README is a project introduction, not an internal governance report. It should help a reader understand what the project studies, how the proposed system is organized, what has actually been completed, and how to inspect or run the current code.

Use Chinese as the primary language unless the owner requests otherwise. Write short, natural sentences and prefer the style already used in `maixcam-steel-ball-tracker`, `PINN-GINav`, and `DNN-RDOP-GNSS`.

## Preferred order

1. Project name and one-paragraph introduction.
2. Research goal.
3. Method or system overview.
4. Current progress and verified result status.
5. Short roadmap.
6. Repository layout.
7. Basic environment or usage instructions.
8. Research notes and license.

Not every update needs every section. Keep only what helps a public reader understand the current project.

## Keep out of the README

- detailed RTK/reference isolation mechanisms;
- release-guard commands and publication checklists;
- evidence naming, indexing, reachability, or audit procedures;
- controller handoffs, acceptance mechanics, and internal review cycles;
- private consultation records or internal operational notes.

These records may remain under `docs/` when needed. The README may accurately state the scientific condition that learning does not use high-precision trajectory supervision, but it should not reproduce the internal enforcement process.

## Future updates

- Update the README when the scientific method, current phase, runnable workflow, or verified results materially change.
- Report only results supported by an accepted experiment; otherwise say that validation is still pending.
- Keep phase summaries short. Do not turn the README into a changelog or evidence index.
- Keep the repository root and landing page simple even when internal governance grows.
- Write `科研手记.md` as a short first-person diary entry with natural wording and no formal acceptance language.
