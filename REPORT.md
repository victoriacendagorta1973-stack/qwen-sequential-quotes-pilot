# Recorded results and interpretation

The training run finished on 6 November 2026. The separate manual evaluation finished on 7 November. The same frozen final checkpoint was used, verified by SHA256 `6b73b78fdd648975387bb921c9786b58727a7b71887edd96f4d6de27fbc62a34`.

## What was learned and what was retained

N2 development exact match improved from 0/7 to 6/7, then 7/7 after learning N3. N3 development exact match rose from 0/12 to 2/12, then 6/12. Training N2 remained 17/17 across the two trained stages. The result supports retention in this small setting, with replay. Since the old data were replayed, this is not a zero-rehearsal result.

The no-quote development regression is explicit: `PAR_02886:CONTROL` changes “Sin embargo, la evolución no es eso.” to “Sin embargo, la evolución no es así.” after both stages. This is a content-preservation failure. Controls are task-specific; they are not a measure of general reasoning or instruction-following.

## New manual evaluation

Base exact: 2/12; final exact: 7/12. Paired changes: six improvements, one regression, five unchanged. No output hit its token limit. No examples were placed in either model's prompt.

Final failures:

- `MANUAL_01:N2`: drops an outer layer and returns an N1-style sentence.
- `MANUAL_01:N3`: drops/misassigns delimiters and leaves an unmatched outer closer.
- `MANUAL_02:N1`: uses English quotes for the second separate outer citation; this is the base-correct → adapter-wrong regression.
- `MANUAL_02:N2`: assigns third-level single quotes to a sibling second-level citation.
- `MANUAL_03:N3`: inserts/repositions quotes around the parenthetical enumeration and fails nesting.

All three Dorotea-family texts are exact after training, versus none for the base. “Dodotea” is intentionally preserved as written in the author's reference: the task is delimiter reconstruction, not general editing.

## What cannot be concluded

The small new manual set has four related families. It cannot support broad statistical claims. The layer range was a design choice, not a demonstrated modular separation of syntax, memory and alignment. Neither subspace projection nor gating was implemented. There is no KL baseline, no unrestricted-LoRA baseline and no multi-seed robustness result. No book knowledge was tested.

Closed-model assistance was used for programming, experiment organization and documentation. The training text route is human templates with Qwen lexical generation and deterministic processing; see PROVENANCE.md. This repository contains the actual final-run data, not the abandoned closed-model synthetic exploration sets.
