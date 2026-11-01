# Research map

**Unseen-room pose recovery remains unresolved.** The runnable replay and local
motion diagnostic are verified software paths. They do not establish consistent
zero-shot or shared-room calibration superiority. Failed comparisons remain
available, and no failed prototype replaces the application models.

| Question | Evidence and reproduction | Finding |
|---|---|---|
| Can the application reconstruct its shipped predictions from CSI alone? | [Reproduction](REPRODUCE.md), [model card](MODEL_CARD.md) | Yes, including real held-time and unseen-room replays; unseen-room accuracy is weak |
| Is useful motion observable within one recording? | [Local observability](LOCAL_OBSERVABILITY.md) | Later-block motion R² 0.917, with dense same-person/room/action references |
| Does temporal neural learning transfer to held people and rooms? | [Original study](../results/README.md), [domain protocol](DOMAIN_RESEARCH.md) | Inconsistent improvements; protected final protocol remains open |
| Do corrected CSI units and receiver covariance solve the domain shift? | [Covariance](COVARIANCE_RESEARCH.md), [packet time](PACKET_TIME_RESEARCH.md) | Some restricted gains reverse across people/actions; full-action transfer fails |
| Can reference poses correct an unfamiliar environment? | [Calibration](CALIBRATION_RESEARCH.md), [residual correction](RESIDUAL_CALIBRATION.md) | Candidates fail their matched gates |
| Does masked CSI representation learning help? | [Masked study](MASKED_CSI_RESEARCH.md) | All four body/full × zero/calibrated development modes fail; 89 prediction sets audited |
| Can additional geometric teacher supervision improve RF representations? | [Geometry](GEOMETRY_RESEARCH.md) | Source-person gate fails; 40 readouts and four neural arms retained |
| Does training through adaptation or retaining full magnitudes help? | [Reference learning](REFERENCE_LEARNING.md) | Both source pilots fail, with independently checked equations and metrics |
| Does correcting an empirical time offset and supervising across source rooms help? | [Timing and transfer](TIMING_AND_TRANSFER.md) | Reproducible lag pattern; neither pose-prototype study passes its source gate |

The latest timing and transfer report contains all 42 candidate checkpoints
across its two studies, both calibrated and zero-shot selections, per-person,
per-action and available per-room metrics, shuffled-signal controls and audit
summaries. It does not substitute a favorable scalar for the frozen gates.

Source validation people have been reused during development. E02/E03 also have
historical project exposure. They are development evidence. The original
[exposure ledger and protected final protocol](DOMAIN_RESEARCH.md) preserve that
distinction. Personal enrollment with consented poses is a separately approved
research direction, not evidence that the two original modes have succeeded.
No personal prototype is promoted to application inference in this revision.

All reported poses are **2D**, based on supplied teacher annotations. Confidence
is not reliable on unfamiliar domains. Data retain their noncommercial license;
code is independently implemented. [Data and permissions](DATA_ACCESS.md),
[references](REFERENCES.md), [model limitations](MODEL_CARD.md).
