---
name: mn-note
description: Capture evolving decisions into .claude/NOTES.md, detect conflicts, and promote stable entries into CLAUDE.md. Writes to NOTES.md and (on promotion) CLAUDE.md.
argument-hint: <note text | "sweep" | "promote <heading>">
user-invocable: true
---

# MolNex Note

Read `.claude/NOTES.md` and `CLAUDE.md` before adding or promoting.

## Procedure

### Capture mode — when `$ARGUMENTS` is free text

1. Read the free-text note.
2. **Conflict check.** Grep `CLAUDE.md` and existing `NOTES.md` entries
   for the same topic. If the new note contradicts an existing rule:
   - State the conflict explicitly back to the user.
   - Ask which one wins before writing.
3. Append a new entry to `NOTES.md` with the template:

   ```
   ## <short title>  (YYYY-MM-DD)
   **Context.** <why this came up>.
   **Decision.** <what we chose>.
   **Status.** provisional.
   ```

### Sweep mode — when `$ARGUMENTS` == `"sweep"`

1. Walk every entry in `NOTES.md`.
2. For each, check whether the behaviour it describes is still present in
   the repo (grep keyword, verify file path).
3. Report stale entries (behaviour removed, file gone, decision reverted)
   and ask the user to delete or archive each one.

### Promotion mode — when `$ARGUMENTS` == `"promote <heading>"`

1. Locate `<heading>` in `NOTES.md`.
2. Propose a 1–3 line addition to the relevant section of `CLAUDE.md`
   (user approves the exact wording).
3. On approval, edit `CLAUDE.md` and delete the note from `NOTES.md`.

## Output

- Which file(s) changed and how many lines.
- Any detected conflicts with prior rules.
- The new or promoted entry in full.

## Rules

- Never write a note that duplicates an existing CLAUDE.md rule.
- Always convert relative dates (`yesterday`, `last week`) to absolute
  dates when writing.
