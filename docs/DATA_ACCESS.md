# Data, access and evaluation boundaries

GhostMesh's primary available source cohort contains **all 27 MM-Fi actions from 20 people across four environments**, totaling **540 sequences and 160,380 paired frames**. This is a balanced cohort, not the complete 40-person benchmark. Its array validation, split assignment and trained results are established by the corresponding prepared manifest and experiment reports.

An earlier pilot uses **A02, horizontal chest expansion, and A06, mark time**, from all 40 people: 80 sequences and 23,760 frames. The pilot's exact original source fingerprints are in [data-provenance.json](data-provenance.json). Pilot results remain separate from the primary cohort.

## Access limitation during collection

On October 25, 2026, expansion to all 27 actions stopped at a verified upstream Google Drive quota. The first missing E01 range, bytes `12565793522–12579265953`, returned **HTTP 200, `text/html`, no Content-Range**, with a page titled **Google Drive - Quota exceeded**. It said the file had too many recent downloads and might be unavailable for up to 24 hours. The downloader rejected the HTML response; it did not treat it as CSI. All four download jobs terminated and all validated local bytes were preserved. A cached range near the archive start still returned 206, so a tiny successful probe alone would not prove missing data was accessible. No alternate host, query variation, credential or access bypass was attempted. [Compact response and coverage record](access-status.json).

The preserved source inventory contains **641 complete paired sequences, 190,377 paired frames**, and seven incomplete sequence directories excluded from any complete-cohort claim. All 27 actions have complete paired files for **20 people**, S01–S05, S11–S15, S21–S25 and S31–S35, five people per environment. Only A02/A06 cover all 40 people. This permits a separately labeled 20-person, 27-action experiment, while the intended 40-person full-action acquisition remains blocked. Complete file coverage is distinct from successful array validation and training; the prepared experiment manifest and evaluation reports prove those later stages.

## Rights and consent

MM-Fi is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), according to its [official project site](https://ntu-aiot-lab.github.io/mm-fi). Credit belongs to Jianfei Yang, He Huang, Yunjiao Zhou, Xinyan Chen, Yuecong Xu, Shenghai Yuan, Han Zou, Chris Xiaoxuan Lu and Lihua Xie, NeurIPS 2023. Its [paper, section 4.1 and appendix A.1](https://arxiv.org/html/2305.10345v2) reports signed participant consent and approval by Nanyang Technological University's institutional review board, IRB-2022-1067. This is the original authors' consent record, not an independent participant audit by GhostMesh.

Use these data and derived replay assets for attributed, noncommercial research. [Data attribution and modifications](../data/LICENSE.md) are separate from the implementation license. Only public CSI arrays and supplied de-identified 2D keypoints were acquired. No RGB images, participant contact details or new human recordings are required. Camera-free inference still estimates behavior and requires consent for new deployments; it does not make covert monitoring acceptable.

## What each file means

The [official toolbox documentation](https://github.com/ybhbingo/MMFi_dataset/tree/c9c1e3a8d5099779a618eae1bb351e253daa52d0) describes the layout. The [dataset owner clarifies the 2D label format](https://github.com/ybhbingo/MMFi_dataset/issues/2#issuecomment-1683569153).

| Source | Verified content | GhostMesh use |
| --- | --- | --- |
| `E*/S*/A*/wifi-csi/frameNNN.mat` | `CSIamp`, shape `[3,114,10]`; phase is also present | Amplitude input only |
| `E*/S*/A*/rgb/frameNNN.npy` | 17 two-dimensional keypoints, shape `[17,2]` | Supplied teacher annotations; never camera input |
| `ground_truth.npy` | Separate 3D annotation convention | Not loaded or projected into a claimed 2D truth |

The RGB-folder joint order is COCO-17: nose; left/right eyes, ears, shoulders, elbows, wrists, hips, knees and ankles. Source annotations are model-generated labels, not manually verified ground truth for this subset. No per-joint visibility/confidence field is supplied, so validity means a finite coordinate inside the declared reference plane, not verified visibility.

## Coordinate and time conventions

All **160,380** primary-cohort keypoint arrays were inspected. Observed ranges are x **53.193497–624.076294**, y **12.148865–498.916016**. There are zero nonfinite coordinates, negative joints or joints outside the 640 square. The pilot ranges were x157.926697–556.998779 and y58.923553–480.466492, also finite.

The annotation image canvas was not documented sufficiently to recover its dimensions. Both cohorts declare a **640 × 640 annotation reference plane** and divide both coordinates by 640. This is a normalization convention, not a claim that camera images were square or had that resolution. Equal-axis scaling preserves pixel-space distances and bounding-box-normalized PCK/NJE. Dividing x by640 and y by480 would change metric geometry. Future expanded cohorts must inspect their own ranges before reusing that plane.

PCK is the proportion of joints inside a stated error threshold; NJE is mean joint distance divided by the target skeleton's bounding-box diagonal. Exact thresholds and exclusions are defined in the [model card](MODEL_CARD.md). Neither metric measures distance in centimeters or establishes 3D reconstruction.

The paper describes a synchronized 10 Hz stream and CSI blocks spanning about 100 ms. Replay timestamps are reconstructed from frame numbers at 10 Hz; original ROS timestamps were not recovered. GhostMesh's frame-ID pairing and gap checks verify file alignment, not the original authors' synchronization error. A temporal window ends at its label frame; preprocessing never reads later frames. Eight frames require seven frame intervals, about 0.7 seconds of startup context.

Per-packet timestamps are not included in this prepared path. Algorithmic causality is therefore guaranteed at the released-frame level, not independently verified inside each upstream 100 ms CSI block. Model-forward latency excludes that block's acquisition time, nominal frame sampling, initial context and rendering; a high processing FPS does not increase the recorded stream's 10 Hz temporal resolution.

## Fetch only the required streams

Run from the repository root. Python's standard library is sufficient for acquisition:

```bash
uv run python scripts/fetch_mmfi.py --env E01 --actions A02 A06 --inventory
```

This reads bounded ZIP directory metadata and prints selected compressed size before extraction. Each environment archive is roughly 19–22 GB; the selected modalities/actions need only a fraction. Directory metadata across four archives totals about 266 MB. The verified full 27-action CSI/keypoint inventory contains 641,520 files, **14,628,683,489 compressed bytes** and **17,767,537,920 original bytes**, before filesystem overhead. Inventory establishes expected coverage, not completed downloads. Fetch the pilot with:

```bash
for environment in E01 E02 E03 E04; do
  uv run python scripts/fetch_mmfi.py --env "$environment" \
    --actions A02 A06 --out data/mmfi-raw \
    --inventory-dir data/mmfi-raw/provenance --max-payload-mb 400
done
```

For all 27 actions, replace `--actions A02 A06` with `--actions A01 A02 A03 A04 A05 A06 A07 A08 A09 A10 A11 A12 A13 A14 A15 A16 A17 A18 A19 A20 A21 A22 A23 A24 A25 A26 A27` and set `--max-payload-mb 4000` per environment after checking its inventory. Use separate prepared output and experiment directories; pilot results cannot be relabeled as full-action results. Reusing the raw directory preserves identical files and refuses different existing bytes.

The downloader uses public files linked by the [official repository](https://github.com/ybhbingo/MMFi_dataset). It requires HTTP 206 with exact Content-Range, checks ZIP CRC and SHA-256, limits ranges and decompressed members, rejects unsafe paths and preserves different existing file bytes. No credentials or permission bypass are involved. If range access stops working, it fails rather than downloading an entire archive; use the official route or retry after access is restored. The separately hosted anonymized image archive returned HTTP 403 during review and was not used.

Once the provider restores access, a manually invoked `--resume` verifies existing files against archive length/CRC, records their SHA-256 and skips complete groups. It does not schedule retries or bypass a denial. Different existing bytes cause a failure instead of replacement. CRC provides the archive's corruption check; the recorded SHA-256 fingerprints bind the bytes actually used by each experiment.

Preparation requires a provenance JSON with `dataset`, `source`, `license` and `consent_basis`; [mmfi-source.json](mmfi-source.json) supplies the factual record:

```bash
uv run ghostmesh prepare-mmfi \
  --source data/mmfi-raw --out data/mmfi-subset \
  --width 640 --height 640 --fps 10 --labels rgb \
  --provenance docs/mmfi-source.json

uv run ghostmesh audit --manifest data/mmfi-subset/manifest.json
```

Existing output directories are refused. CSI and keypoints join by exact frame number, not filesystem position. Prepared NPZ files retain complete sequences, raw-file hashes and manifest content hashes. Use a fresh directory when rebuilding.

## Subject and environment splits

The official toolbox holds out S05, S10, S15, S20, S25, S30, S35 and S40. The primary 20-person cohort uses the intersection **S05, S15, S25 and S35** as final cross-subject test data; **S04, S14, S24 and S34** provide validation; the remaining 12 people train. The earlier 40-person pilot uses all eight official test subjects and eight validation subjects ending in4 or9, leaving24 training people. Every action and adjacent window for a person stays in that person's split.

Cross-environment experiments hold out E04, following the [toolbox configuration](https://github.com/ybhbingo/MMFi_dataset/blob/c9c1e3a8d5099779a618eae1bb351e253daa52d0/config.yaml). The primary cohort uses12 people from E01–E03 for training, three for validation and five new E04 people for testing. The pilot uses24 training people, six for validation and ten E04 people for testing. **Environment and person changes are confounded**, because different rooms contain different people. These results cannot isolate a room-only causal effect.

The two manifests define distinct experiments. E04 may appear in the cross-subject model's source data, but a newly initialized cross-environment model never sees E04 during development. Checkpoints bind to the corresponding manifest hash. Test data never selects an epoch; best checkpoints use validation NJE.

## Scope of a valid conclusion

Report cohort scores and compare identical held-out windows with the last-frame encoder and no-signal mean-pose baselines. Show both good and bad sequences. The two-action pilot does not establish general action coverage. The primary cohort includes all27 categories in development and test, so it measures new people/environments within those categories, not unseen action categories. Neither establishes 3D pose, multi-person tracking or cross-device transfer. Published MM-Fi tables use different tasks/protocols; GhostMesh's 2D scores are not a direct reproduction of those tables.
