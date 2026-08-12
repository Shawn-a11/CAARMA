# Research Idea Review Workflow

This directory stores only ideas that have been explicitly reviewed and
accepted by the project owner.

## Workflow

1. Codex researches and develops a candidate idea.
2. Codex presents the complete candidate in the conversation first.
3. The project owner responds with one of these decisions:
   - `accept`: record the reviewed idea in this directory;
   - `revise`: update the proposal and present it again;
   - `reject`: do not write the candidate into this directory;
   - `hold`: keep discussing it without recording it.
4. Only after `accept`, Codex creates an individual Markdown file and adds it
   to `accepted_ideas.md`.

An idea discussed in chat is not considered part of the project method until
it appears in `accepted_ideas.md`.

## Required Review Card

Before requesting acceptance, every candidate must show:

1. **Problem:** the unresolved limitation supported by current experiments.
2. **Core mechanism:** the single load-bearing change.
3. **Derivation:** why the mechanism should affect the target objective.
4. **Difference:** how it differs from completed and pending experiments.
5. **Prior-art collision:** the closest related methods and novelty boundary.
6. **Minimal experiment:** the cheapest decisive test.
7. **Controls:** positive, negative, shuffled, or compute-matched controls.
8. **Success criterion:** the evidence required to retain the method.
9. **Kill criterion:** the outcome that terminates the direction.
10. **Cost:** implementation risk, expected runtime, and required hardware.

## File Naming

Accepted ideas use:

```text
YYYY-MM-DD_short_descriptive_name.md
```

Use ASCII lowercase words separated by underscores. Do not overwrite an
accepted idea when its mechanism changes substantially; create a new version
and link the records.

## Evidence Rule

- Exploratory single-run results must be labeled as exploratory.
- Paper-level claims require runtime-matched controls and multiple seeds.
- Best-observed results may be recorded, but must not replace mean and
  variability reporting.
- A hyperparameter sweep is supporting analysis, not a standalone innovation.

