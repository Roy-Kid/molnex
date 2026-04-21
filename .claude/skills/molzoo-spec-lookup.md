---
name: molzoo-spec-lookup
description: Surface paper/equation/notation details from a molzoo encoder's spec + walkthrough + recent experiments when debugging or answering a conceptual question. Reads only — does not edit. Refuses to fabricate content when the topic is not covered; suggests molnex-scientist instead.
---

# molzoo-spec-lookup

Retrieve what the existing artifacts already say about a topic for a given encoder. Never summarise the paper from model memory — only surface what is physically written in the files.

## Arguments

`/molzoo-spec-lookup <encoder_name> <topic_or_question>`

- `<encoder_name>` — filename stem under `src/molzoo/specs/`.
- `<topic_or_question>` — free-form. Multi-word is fine. Single-quote or double-quote not required; the skill takes everything after `<encoder_name>` as the topic.

## Preconditions

1. `src/molzoo/specs/<encoder>.md` exists. If not, refuse: "No spec for `<encoder>` yet. Scaffold with `/molzoo-spec-new <encoder> <arxiv>`."
2. `<encoder>_walkthrough.md` and `<encoder>_experiments.csv` may or may not exist — absence is fine, just skip those sections in the response.

## Step 1 — Read all three artifacts

- Read `<encoder>.md` in full.
- Read `<encoder>_walkthrough.md` in full (if present).
- Read the **last 10 data rows** of `<encoder>_experiments.csv` (if present).

Do not `Grep` the paper or the code — the point is to surface what is already captured.

## Step 2 — Find matches

For each artifact, find section(s) that match the topic. Matching is keyword-based and loose:
- Split the topic into words, lowercase, strip punctuation.
- A section "matches" if its heading or body contains ≥ 1 of the keywords (ignoring stopwords: `the`, `a`, `and`, `of`, `is`, `what`, `does`, `how`, `why`).
- A walkthrough row matches if its Topic column or body contains ≥ 1 keyword.
- A CSV row matches if its `note_ref` column, when followed into the walkthrough, lands on a matching heading.

## Step 3 — Present results

Output structure (fixed order):

### A. Drift alerts (always first if any exist)

If the walkthrough has ⚠️ or 🆚 verdict rows matching the topic, list them at the very top under the heading **"Known drift on this topic:"** with:
- verdict emoji
- one-line description
- link to the walkthrough section (`<encoder>_walkthrough.md#<slug>`)

If there are none, skip this block — do not print "No drift" for visual noise.

### B. Spec sections

Print the matching sections from `<encoder>.md` verbatim (not summarised). Use their existing section numbers (e.g. `§5.2 Step 3 — Equivariant tensor product`). If a matched section is > 40 lines, print the first 40 lines and a pointer `(see src/molzoo/specs/<encoder>.md:<line>-<end>)`.

### C. Walkthrough verdicts (excluding drift alerts already printed)

List ✅/ℹ️ rows matching the topic, with verdict + one-line description + slug link. Condensed, not verbatim.

### D. Recent runs with linked investigations

If the last-10 CSV tail has rows whose `note_ref` points into a matched walkthrough heading, print them as a mini-table:

```
run_id | date       | commit  | val_mae | note_ref
-------|------------|---------|---------|---------
  17   | 2026-04-10 | a1b2c3d |  0.023  | #run-17-mae-regression
```

## Step 4 — Close the loop

### Miss path (MANDATORY)

If Steps 2–3 produce **zero matches across all three artifacts**, do not try to answer the question yourself. Print:

```
No coverage of "<topic>" in src/molzoo/specs/<encoder>.md, its walkthrough, or recent experiments.

Options:
  1. Invoke `molnex-scientist` to research the paper and extend the walkthrough.
  2. Rephrase the topic with different keywords (the matcher is keyword-based).
```

**Do not fabricate an answer from model memory.** The whole point of the lookup loop is to force paper details to be written down once, then surfaced from disk, so every user hits the same curated content.

### Hit path

After presenting matches, add a one-line footer:

```
Read 3 artifact(s). To deepen or correct the walkthrough on this topic, invoke `molnex-scientist`.
```

## Non-goals

- Do NOT edit any file.
- Do NOT read the encoder's Python source (`src/molzoo/<encoder>.py`) unless a matched section explicitly references a line range and the user asks a follow-up. This skill's job is artifact retrieval.
- Do NOT call `WebFetch` on the paper. If the spec is incomplete on the topic, that is the signal to invoke `molnex-scientist`, not a reason to browse arXiv inside this skill.
