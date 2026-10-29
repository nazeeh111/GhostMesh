# Verification evidence

Software checks and sensing accuracy answer different questions. These checks
verify implementation behavior; the [experiment results](../results/cohort20-v1/results.md)
report accuracy on actual held-out recordings. Synthetic fixtures are never
used as evidence that Wi-Fi predicts human pose.

## Verified research snapshot

Commit `eb35455b43f8257eb9f19b0e0124f6fcd11752a0` passes the
[Ubuntu, macOS and reporting workflow](https://github.com/nazeeh111/GhostMesh/actions/runs/40834787654).
That published checkout has 330 core Python tests, 22 isolated reporting tests
and nine browser tests. Both operating systems also reproduce the real CSI-only
examples, build the wheel and check it outside the source checkout.

An earlier commit exposed a test import-path dependency under the console
`pytest` launcher. It was reproduced in a clean checkout and fixed by declaring
the repository path in pytest configuration. Passing `python -m pytest` alone
had not caught that difference. These software checks do not establish an
unseen-room accuracy improvement.

## Frozen v0.1 software checks

| Check | Observed result | Reproduce |
|---|---|---|
| Numerical, data, training, inference, parser and real HTTP tests | 168 passed | `uv run pytest -q --ignore=tests/test_report_results.py` |
| Scientific reporting and artifact-binding checks | 22 passed | Follow [reporting setup](RESULTS_FORMAT.md), then `.report-venv/bin/python -m pytest -q tests/test_report_results.py` |
| Browser replay contracts | 8 passed | `node --test tests/test_replay.mjs` |
| Locked dependency consistency | Passed | `uv lock --check` |
| Python lint | Ruff 0.16.2 passed | `uvx ruff==0.16.2 check .` |

These were separate runs, without skipped tests in the named suites. Core tests
exercise temporary HTTP servers and need permission to bind loopback in a
sandbox. Reporting tests use explicitly labeled fabricated fixtures only to
test the renderer's refusal and formatting behavior.

The causal temporal implementation is checked against native convolutions for
outputs, gradients, reload and future-frame independence. Data tests cover
exact frame pairing, integrity, recording boundaries and split leakage.
Training tests check validation-only selection and exact completed-epoch
recovery, including optimizer, teacher and random states. Replay tests check
annotation-free inputs, timestamp gaps, invalid-signal withholding and full
sensor-layout matching. The ESP32 parser is tested against official-header
fixtures with startup logs, clock wraps and unsupported payloads; no board was
opened for these tests.

## Reproduction and timing boundaries

The release's primary arm is fixed before testing. Configs, data manifests,
runtime source and runner are bound by preregistration. Checkpoints use
validation error for epoch selection. Independent training processes keep
their own random states; every arm finishes before final evaluation begins.

[Host and scheduling record](verification-host.json) identifies the Apple M4,
32 GiB workstation and runtime. Training included serial and parallel phases
under shared workstation load. Training wall times therefore do not establish
a fair speed comparison. Reported model latency is separately measured using
batch-one forward calls and excludes acquisition, preprocessing, file access,
network transport and browser rendering. The source recording remains 10 Hz.

The [study bundle](STUDY_BUNDLE.md) preserves exact selected model and report
bytes for inspecting and re-rendering stored findings. Re-rendering tables is
different from independently re-evaluating the models or retraining them.
The v0.1 collection stopped at a provider quota. Normal access later returned;
the full 40-person acquisition now passes the [v2 byte audit](domain-v2-full40-byte-audit.json).
The released study and its original cohort remain frozen. ESP32 pose accuracy
and original packet-level synchronization remain unverified.


## Release artifacts checked locally

Both CSI-only examples independently reproduce all 290 valid frames with
maximum pose and confidence differences of 0.0 under the locked CPU environment.
The first seven frames supply causal context. Both NPZ inputs contain only
`csi` and `timestamps`. No pose annotations or camera images enter inference.

The actual application capture contains 80 frames at 10 FPS, eight seconds,
H.264 video and a matching GIF. Current-source desktop and mobile screenshots
were inspected. Exact backward seeking reproduced the whole viewport, mobile
390-pixel layout had no overflow, and the capture recorded no JavaScript
errors. The encoded video was decoded and its midpoint inspected. Capture
metadata binds the replay, web sources and all media hashes.

The final scientific renderer binds 72 input/output hashes. Independent review
matched all twelve neural test reports to original artifacts and reconstructed
all 26 sequence/subject paired-bootstrap result objects exactly. Qualitative
cases are explicitly labeled minimum/maximum-error extremes, not typical
accuracy. These checks establish consistency, not successful dynamic sensing.

The release retains mixed and negative outcomes: the temporal method does not
consistently improve its baseline, the source-room pose prior is strong, and
unseen-room accuracy and confidence remain weak. Read the
[integrated findings](../results/README.md) with the supplemental controls.


The installed wheel was imported from a separate installation directory and
served its packaged real replay over loopback HTTP from a temporary working
directory. Its command entry point, metadata, all license files and static web
asset hashes passed. The prepared cohort archive has 549 verified members,
including exactly 540 NPZ files, and is 1,531,558,056 bytes. Every member size and
SHA-256 matches its internal manifest; no window cache or RGB images are included.
External archive SHA-256:
`485efadd0c819493bed83e3eaa891455b475c1767e084d46f7ef7cf39846bf69`.
