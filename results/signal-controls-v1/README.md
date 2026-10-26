# Supplemental signal controls

Supplemental descriptive diagnostics of sample-specific CSI-target correspondence beyond static room pose distributions. Oracle room metadata; person/action/recording cues may remain. No significance tests, causal motion, localization, unseen-action or cross-dataset claim.

One fixed permutation; descriptive changes only, without p-values or model selection.

| Protocol | Model | Mode | NJE | PCK@0.2 | Valid frames |
|---|---|---|---:|---:|---:|
| cross_subject | cohort20-baseline | aligned | 0.132753 | 0.829731 | 31320 |
| cross_subject | cohort20-baseline | permuted | 0.136717 | 0.824572 | 31320 |
| cross_subject | cohort20-temporal | aligned | 0.134285 | 0.827380 | 31320 |
| cross_subject | cohort20-temporal | permuted | 0.137597 | 0.821941 | 31320 |
| cross_environment | cohort20-environment-baseline | aligned | 0.218111 | 0.467736 | 39150 |
| cross_environment | cohort20-environment-baseline | permuted | 0.219282 | 0.475046 | 39150 |
| cross_environment | cohort20-environment-temporal | aligned | 0.214399 | 0.492235 | 39150 |
| cross_environment | cohort20-environment-temporal | permuted | 0.217255 | 0.485516 | 39150 |
| cross_subject | source-room-prior-oracle-metadata | prior | 0.130122 | 0.853788 | 31320 |
| cross_environment | source-room-prior-oracle-metadata | prior | 0.233341 | 0.387047 | 39150 |

Full metrics, fitted priors, coverage, permutation hashes/fixed points and artifact hashes: results.json.
