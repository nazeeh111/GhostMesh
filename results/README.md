# Measured findings

GhostMesh's release is an original implementation and completed, bounded
MM-Fi study. The proposed temporal-plus-teacher method has mixed or negative
findings. These results do not establish robust dynamic pose recovery or
reliable unseen-room generalization.

The cohort contains 27 action categories, 20 people, four environments,
540 sequences and 160,380 frames. Every category appears in development.
Cross-subject testing holds out four people; E04 testing holds out five people
and their environment together. Supplied 2D teacher labels provide the reference.

## Read the evidence together

- [Primary study](cohort20-v1/results.md): all 12 neural arms, both seeds,
  static global priors, three PCK thresholds, normalized joint error, coverage,
  calibration, latency and paired sequence/subject bootstrap intervals.
- [Supplemental controls](signal-controls-v1/README.md): source-room priors and
  one fixed within-room permutation, frozen before the first original test.
- [Qualitative cases](qualitative-v1/qualitative.png): the first recorded
  minimum/maximum NJE endpoints for the primary temporal model in each protocol.
  These four extremes are not typical or representative frames.

## Conclusions

The primary seed-2027 temporal model obtains PCK@.2 0.82738 / NJE 0.13429 in
cross-subject testing and 0.49223 / 0.21440 on E04. Its stricter PCK@.05 is only
0.13198 and 0.02359 respectively. PCK@.2 tolerates an error equal to 20% of the
reference body's bounding-box diagonal; it is not a general probability that
a reconstructed skeleton is correct.

Temporal-plus-teacher does not consistently beat the last-frame encoder.
Cross-subject seed 2027 worsens error. E04 seed 2027 improves versus its matched
baseline, while seed 2028 reverses: paired NJE increases by about 0.0107 and
PCK@.2 decreases by about 0.0689. Two seeds are sensitivity checks, not a stable
estimate of training variance. The original paired intervals remain intact.

A source-trained room mean with oracle environment metadata reaches NJE
0.130122 and PCK@.2 0.853788, better than both primary seed-2027 cross-subject
neural models. This limits claims that their performance establishes useful
motion recovery beyond static room information. The E04 room prior falls back
to the global training prior because E04 has no source labels.

Within-room permutation changes primary temporal NJE from 0.134285 to 0.137597
in cross-subject testing and from 0.214399 to 0.217255 on E04. These are small,
descriptive changes from one predeclared permutation. E04 baseline PCK improves
when permuted. Person, action and recording cues can remain, so this diagnostic
does not establish a causal motion or localization mechanism.

Primary confidence ECE is 0.1406 in cross-subject testing and 0.4183 on E04 for
the event of joint error within .05 annotation-plane units. This event differs
from bounding-box-normalized PCK@.05. Confidence is not reliably calibrated.
Signal rejection and metadata domain warnings cannot detect every wrong pose.

## Scope and artifact identity

The full 40-person expansion stopped at an observed provider quota during
collection. Room/person changes are confounded in E04. No unseen-action,
cross-device, multi-person, 3D or physical ESP32 pose result is claimed.
Coordinates use a declared isotropic 640-square annotation plane. Timing uses
reconstructed nominal 10 Hz frame IDs; original packet synchronization was not
independently measured. About .7 seconds of context precedes the first prediction.

Forward-only timing excludes preprocessing, radio acquisition, I/O and browser
rendering. A high processing FPS does not raise the recording's temporal resolution.

The primary renderer manifest binds 72 input/output hashes. Independent review
matched all 12 embedded neural reports to original artifacts and reconstructed
all 26 stored sequence/subject paired-bootstrap result objects exactly. These
checks establish artifact consistency, not a stronger scientific conclusion.
Raw CSV/JSON, frozen plans, checkpoints and data manifests are retained with
the release. [Reproduction guide](../docs/REPRODUCE.md).

MM-Fi-derived data and figures retain [CC BY-NC 4.0](../data/LICENSE.md)
attribution. References are model-generated annotations, not independently
audited manual ground truth. [Dataset credit](../NOTICE.md).
