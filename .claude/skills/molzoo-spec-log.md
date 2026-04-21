---
name: molzoo-spec-log
description: Append one row to src/molzoo/specs/<encoder>_experiments.csv after a benchmark or training run, and close the loop by prompting a scientist investigation on MAE regression or dirty git tree. Use after running pytest-benchmark, a training job, or any parameter sweep whose result should be captured.
---

# molzoo-spec-log

Append exactly one row to `src/molzoo/specs/<encoder>_experiments.csv`. Never rewrite existing rows (except to backfill `note_ref` when the user opens an investigation right after logging).

## Arguments

`/molzoo-spec-log <encoder_name> [--from <bench_json_path>] [--note-ref <slug>]`

- `<encoder_name>` — matches the filename stem under `src/molzoo/specs/`. Required.
- `--from <path>` — optional `pytest-benchmark` JSON output (produced by `pytest --benchmark-json=<path>`). If provided, parse `fwd_ms`, `bwd_ms`, `compiled` from it.
- `--note-ref <slug>` — optional pre-assigned walkthrough heading slug (uncommon; usually filled in by the loop, not the user).

## Preconditions

1. `src/molzoo/specs/<encoder>_experiments.csv` exists. If not, refuse: "No experiment log for `<encoder>`. Run `/molzoo-spec-new <encoder> <arxiv>` first."
2. Header row of the CSV is exactly `run_id,date,commit,dirty,dataset,config_label,steps,train_mae,val_mae,fwd_ms,bwd_ms,compiled,note_ref`. If it differs, refuse with a clear message — do not reformat the user's log.

## Step 1 — Gather row values

| Field | How to fill |
|-------|-------------|
| `run_id` | `wc -l < <csv>` minus 1 (header), then +1. Integer. |
| `date` | `date +%Y-%m-%d` |
| `commit` | `git -C <repo_root> rev-parse --short HEAD` |
| `dirty` | `1` if `git status --porcelain` non-empty else `0` |
| `dataset` | Ask user — short label such as `qm9_u0`, `md17_aspirin`. |
| `config_label` | Ask user — short human tag such as `l2_f64_layers3`. Not the full Pydantic dump. |
| `steps` | Parse from training log path or ask user. Integer. |
| `train_mae`, `val_mae` | Parse from training log path or ask user. Float, same units across rows — ask what the units are on the first row of a fresh CSV and remind the user to keep them consistent. |
| `fwd_ms`, `bwd_ms` | From `--from <bench.json>` if given (`min` or `mean`, be explicit to the user about which); else ask. |
| `compiled` | `1` if run used `torch.compile`, else `0`. Ask if unclear. |
| `note_ref` | Empty unless `--note-ref` was passed. |

Escape any commas / quotes inside text fields with CSV quoting rules (`"..."`). Keep values short; long notes belong in the walkthrough, referenced via `note_ref`.

## Step 2 — Append the row

Use `Bash` with `printf '%s\n' "<row>" >> <csv>`. Single `>>`, never rewrite. Do not sort, do not deduplicate.

After appending, show the user the new row with its `run_id`.

## Step 3 — Close the loop (MANDATORY)

Read the **previous row** from the CSV (the row with `run_id = new_run_id - 1`, if any). Compute:

- `val_mae_delta = new_val_mae - prev_val_mae`
- `val_mae_ratio = new_val_mae / prev_val_mae` (guard against divide-by-zero)
- `fwd_delta_pct`, `bwd_delta_pct` (ignore if either row has empty perf fields)

Then decide:

### Case A — MAE regression (`val_mae_ratio > 1.10`) OR `dirty == 1`

Print a clearly-marked block:

```
⚠ Anomaly at run_id=<N>:
  - val_mae regressed <X%> vs run_id=<N-1>   (only if regressed)
  - working tree was dirty at commit time     (only if dirty=1)
  - perf delta: fwd <P%>, bwd <Q%>            (only if available)

Open an investigation with molnex-scientist? [y/N]
```

On `y`:
1. Build slug `run-<N>-<short-topic>`. Topic defaults to `mae-regression` (MAE) or `dirty-tree` (dirty) or `investigation` (other). If both, prefer `mae-regression`.
2. Append a stub heading to `<encoder>_walkthrough.md` under the `## Run-linked investigations` section:
   ```markdown
   ### run-<N>-<topic>
   _Stubbed by `/molzoo-spec-log`. To be filled by `molnex-scientist`._
   - Trigger: run_id=<N>, val_mae=<v>, prev val_mae=<p>, delta=<d%>
   - Commit: <sha> (dirty=<0/1>)
   - Question: _TBD_
   ```
3. Backfill the CSV row: replace the just-appended row's `note_ref` with `#run-<N>-<topic>`. Do this with a targeted `sed` or by reading the full file and rewriting only that one line via `Edit` — never disturb other rows. Verify byte-level that every row other than the last is unchanged.
4. Delegate to the `molnex-scientist` agent with a prompt that includes: the triggering run_id, the stub slug, and the CSV row values. Run the agent in the foreground so the user can see the verdict.

On `N` (the default): print a one-liner "Skipped investigation. You can open one later with `molnex-scientist`." Do **not** touch the walkthrough.

### Case B — Clean progress

No prompt, no walkthrough edit. Print a single confirmation line: `Logged run_id=<N> — val_mae <v> (<delta_str>).`

## Invariants

- The CSV is **append-only**. The only permitted post-append edit is backfilling the just-written row's `note_ref` in the scientist-handoff path. Every earlier row must be byte-identical before and after this skill runs. Verify this by reading the previous tail before and after.
- The skill never edits `<encoder>.md` or the walkthrough body (only appends a run heading stub when handing off to scientist).
- The skill never runs tests or training itself — it only logs already-produced results.

## Failure modes

- If `git rev-parse HEAD` fails (detached repo state), ask the user to confirm the commit manually rather than writing `unknown`.
- If the CSV's previous row is malformed, log the new row anyway but skip the regression check and print a warning.
- If the user passes numeric values in wildly different units than previous rows (sanity check: `> 100×` difference), warn and confirm before appending.
