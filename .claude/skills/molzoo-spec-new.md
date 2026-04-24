---
name: molzoo-spec-new
description: Scaffold a new molzoo encoder spec from an arXiv reference. Creates <encoder>.md (paper-aligned spec), <encoder>_walkthrough.md (code↔spec↔paper audit), and <encoder>_experiments.csv (append-only run log) under src/molzoo/specs/, and adds the encoder to src/molzoo/README.md. Use when introducing a new descriptor/encoder before any code is written.
---

# molzoo-spec-new

Scaffold the three sibling artifacts for a new encoder:

- `src/molzoo/specs/<encoder>.md` — paper-aligned **spec**
- `src/molzoo/specs/<encoder>_walkthrough.md` — code↔spec↔paper **note** (verdict log)
- `src/molzoo/specs/<encoder>_experiments.csv` — append-only **log** (run history)

## Arguments

The user invokes `/molzoo-spec-new <encoder_name> <arxiv_url_or_paper_title> [optional one-line description]`.

- `<encoder_name>` — lowercase identifier (e.g. `painn`, `nequip`). Must be a valid Python module name (matches `^[a-z][a-z0-9_]*$`).
- `<arxiv_url_or_paper_title>` — either `https://arxiv.org/abs/XXXX.XXXXX` or a quoted paper title. Resolve via `WebFetch` on the arXiv abstract page.

If the user omits either argument, ask once via `AskUserQuestion`. Do NOT guess.

## Preconditions (refuse with a clear error if violated)

1. None of `src/molzoo/specs/<encoder>.md`, `<encoder>_walkthrough.md`, `<encoder>_experiments.csv` exist. If any do, stop and tell the user which. Never overwrite.
2. `<encoder>` is a valid identifier. Reject names like `PaiNN` (uppercase) or `my encoder` (space).

## Step 1 — Resolve the paper

If given an arXiv URL, `WebFetch` the abstract page and extract: `authors`, `title`, `venue`, `year`, `arxiv_id`, `doi` (if present). If given only a title, `WebSearch` arXiv first, present the top hit to the user with `AskUserQuestion`, and only proceed once confirmed.

On resolution failure: ask the user to paste the fields. Do not invent citations.

## Step 2 — Write `<encoder>.md`

Copy the **section structure verbatim** from `src/molzoo/specs/allegro.md`. Preserve section numbers (§1 Scope, §2 I/O Contract, §3 Notation, §4 Architecture Overview, §5 Module Specifications, §6 Mathematical Properties, §7 Implementation Notes & Deviations from the Paper, §8 Configuration, §9 Complexity, §10 Validation, §11 Changelog Anchors).

Populate only the **Reference** header at the top (authors, title, venue, year, arXiv, DOI if any). Replace every other section body with a single TODO line in the form:

```
<!-- TODO(spec): fill in §N.M once the implementation begins -->
```

**Do not embed an Experiment Log section or table.** The experiment log is the sibling CSV — keep the spec free of run rows.

## Step 3 — Write `<encoder>_walkthrough.md`

Copy the structure of `src/molzoo/specs/allegro_walkthrough.md` header (title, three-way audit table of Code/Spec/Paper/Ref-impl paths, verdict legend). Leave the "Executive Summary" table empty (header row only). After the legend, add:

```markdown
## Run-linked investigations

_Headings under this section are created by `molzoo-auditor` when a run logged in `<encoder>_experiments.csv` triggers an investigation. Each heading is `#run-<id>-<slug>` and is referenced back from the CSV row's `note_ref` column._
```

## Step 4 — Write `<encoder>_experiments.csv`

Create with a single header row, exactly:

```
run_id,date,commit,dirty,dataset,config_label,steps,train_mae,val_mae,fwd_ms,bwd_ms,compiled,note_ref
```

No data rows. No trailing blank line beyond the final `\n` after the header.

## Step 5 — Update `src/molzoo/README.md`

Find the `| Model | Spec | Paper |` table and append one row:

```
| <Encoder> | [`specs/<encoder>.md`](specs/<encoder>.md) · [walkthrough](specs/<encoder>_walkthrough.md) · [experiments](specs/<encoder>_experiments.csv) | <Authors et al.>, <Venue Year> ([arXiv](<URL>)) |
```

Preserve existing rows. Match the casing style already in the table (e.g. `Allegro`, `MACE`).

## Step 6 — Close the loop

Print exactly these next steps (do not embellish):

```
Created:
  - src/molzoo/specs/<encoder>.md
  - src/molzoo/specs/<encoder>_walkthrough.md
  - src/molzoo/specs/<encoder>_experiments.csv

Next:
  1. Implement src/molzoo/<encoder>.py following src/molzoo/allegro.py.
  2. Fill in the TODO sections of <encoder>.md as the implementation firms up.
  3. After the first benchmark/training run, use /molzoo-spec-log <encoder> to record results.
```

## Non-goals

- Do NOT write any Python code in this skill. Scaffolding is strictly for the spec/note/log trio.
- Do NOT backfill prior experiments into the new CSV.
- Do NOT touch any other encoder's specs.
