# RLIE Datasets

This directory contains the raw dataset files bundled with the RLIE release.

The current release includes data for these tasks:

- `real/headline_binary`
- `real/deceptive_reviews`
- `real/dreaddit`
- `real/llamagc_detect`
- `real/persuasive_pairs`
- `real/retweet`
- `real/vast` (raw CSVs; build the JSON with `python -m rlie.datasets.prepare_vast`)
- `real/vast_short` (500 train / 200 dev rows from that same command)

Task definitions, prompt templates, and task metadata live under
`rlie/task_specs/`.

Dataset-specific licensing and redistribution terms may differ from the RLIE
code license. See `data/THIRD_PARTY.md` and the original dataset sources
before redistributing or modifying these files.
