# Experiment artifacts

Run directories are generated locally and excluded from Git. Each successful
training run saves its frozen config, history, validation-selected checkpoint,
source/data/config fingerprints, environment and coverage. Held-out evaluation
is a separate command. A frozen suite writes `preregistration.json` before
training and `suite_report.json` after all requested evaluations finish.

Final release results and selected small model/replay artifacts will be
published separately from raw datasets and unfinished experiments. A directory,
config or running process is not a completed experiment.
