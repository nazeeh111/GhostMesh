# Contributing

Use Python3.12 and `uv sync --locked --extra test`. Run `uv run pytest -q` and
`node --test tests/test_replay.mjs` for relevant changes. Core commands are
listed in the README. Existing experiment/data output paths are intentionally
refused; use a new destination when rebuilding.

Include a reproducible example and explain its user-visible or scientific
effect. Keep new dependencies proportionate and freely available. Never
commit credentials, private recordings, temporary environments or raw dataset
archives. Licensed examples need provenance, attribution and modification
notes.

For research changes, freeze the proposed protocol before final testing.
Validation selects checkpoints; test results evaluate a frozen choice. Report
negative results, coverage, original coordinate units and the dependence
between people, rooms and nearby frames. Changes to model/data/preprocessing
require new experiments, not relabeled older scores.

The browser UI uses native JavaScript and CSS without a build service. Inspect
the current source at desktop and mobile widths when changing visual behavior.
Preserve keyboard controls, reduced motion, upload errors, invalid-signal
withholding and uncalibrated confidence wording.

Hardware work must use consenting participants and authorized test sites.
Record signal quality and synchronization evidence before pose claims.
