# Actual held-out application capture

These screenshots, GIF and H.264 video show the actual GhostMesh application
using E01/S05/A02, the test recording selected before test evaluation. Teal is
CSI-only model output; dashed orange is the supplied model-generated reference
annotation attached after inference. The eight-second recording seeks to real
source timestamps at 10 Hz; no pose interpolation or invented motion is used.
The display is not evidence of typical accuracy. See the measured findings.

`capture.json` binds the replay, current web sources, timestamps and media hashes.
`node scripts/capture_demo.mjs --help` documents capture options.

Original application code is MIT licensed. These MM-Fi-derived visual artifacts
retain CC BY-NC 4.0. Dataset: Jianfei Yang, He Huang, Yunjiao Zhou, Xinyan Chen,
Yuecong Xu, Shenghai Yuan, Han Zou, Chris Xiaoxuan Lu and Lihua Xie, MM-Fi,
NeurIPS 2023. Source: https://ntu-aiot-lab.github.io/mm-fi . Changes: original
CSI model inference, skeleton rendering, signal maps and screen capture.
No endorsement is implied. See ../../../NOTICE.md and ../../../data/LICENSE.md.
