# Research and implementation references

GhostMesh's model, loader, preprocessing, experiment runner and viewer are independently implemented. These references explain the problem, data contracts and comparison context. Their published results are not GhostMesh measurements.

| Reference | Role in GhostMesh | Source boundary |
| --- | --- | --- |
| [MM-Fi, NeurIPS 2023](https://arxiv.org/abs/2305.10345) | Paired CSI and supplied 2D annotations; subject/environment protocols | Data CC BY-NC 4.0; no toolbox code copied |
| [MM-Fi toolbox snapshot](https://github.com/ybhbingo/MMFi_dataset/tree/c9c1e3a8d5099779a618eae1bb351e253daa52d0) | MAT/keypoint format, frame naming and subject assignments | Format reference; no assumed toolbox redistribution grant |
| [MM-Fi owner clarification, issue 2](https://github.com/ybhbingo/MMFi_dataset/issues/2#issuecomment-1683569153) | RGB/infra NPY arrays are 17×2 keypoints in COCO order | Factual schema information, not an image source |
| [CSI-Pose](https://github.com/sel00000/csi-pose) | ESP32 capture/replay architecture reference | No model code, weights, research figures or demos copied |
| [ESP-CSI](https://github.com/espressif/esp-csi) | Separate hardware acquisition route and CSI format context | Upstream Apache-2.0; serial/CSV adapter original |
| [Person-in-WiFi, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Person-in-WiFi_Fine-Grained_Person_Perception_Using_WiFi_ICCV_2019_paper.html) | Foundational fine-grained human perception research | Bibliographic reference; no implementation or results reused |

## Established methodological building blocks

Temporal convolutions and weight-averaged teachers predate GhostMesh. The original work here is their CSI-specific implementation, normalization and evaluation design, whose value must be demonstrated by the matched experiments, not asserted from component names.

- Shaojie Bai, J. Zico Kolter and Vladlen Koltun. [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271), 2018. Background for causal temporal convolution; GhostMesh does not reproduce that paper's architecture or benchmarks.
- Antti Tarvainen and Harri Valpola. [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/abs/1703.01780), NeurIPS 2017. Background for exponential moving average teacher consistency. GhostMesh's consistency uses source training data only; no claim is made that this supervised source-only variant reproduces the original semi-supervised results.

## Benchmark compatibility

[MM-Fi Table 3](https://arxiv.org/html/2305.10345v2#S5) evaluates 3D targets in
millimeters. Its all-action Wi-Fi MPJPE rises from 197.1 mm in the random setting
to 369.5 mm across environments. Those numbers cannot be compared directly with
GhostMesh's dimensionless 2D NJE/PCK. Similarity-aligned 3D error uses the test
reference to fit a transformation; it is not a deployed correction.
[Pinned evaluation implementation](https://github.com/ybhbingo/MMFi_dataset/blob/c9c1e3a8d5099779a618eae1bb351e253daa52d0/mmfi_lib/evaluate.py).

In the [pinned toolbox split implementation](https://github.com/ybhbingo/MMFi_dataset/blob/c9c1e3a8d5099779a618eae1bb351e253daa52d0/mmfi_lib/mmfi.py#L24),
`random_split` assigns complete subject/action recordings, selecting subjects
separately for each action. A person can appear on both sides through different
actions. This differs from GhostMesh's person-disjoint folds and does not justify
calling that implementation an adjacent-frame random split.

The [inspected CSI loader](https://github.com/ybhbingo/MMFi_dataset/blob/c9c1e3a8d5099779a618eae1bb351e253daa52d0/mmfi_lib/mmfi.py#L256)
also min-max normalizes each magnitude frame and fills missing entries using
their packet-column mean. GhostMesh's transforms
and targets are documented independently. The references establish neither an
exact reproduced paper baseline nor a demonstrated solution to the stricter
unseen-room requirement. Any future 3D auxiliary target needs a verified
coordinate/joint contract before supporting a camera-independent claim.

## Relationship to Poseborne

The inspected local Poseborne reference was clean at commit `f73195dd4daf957c0c97cf5edf6a42e196b02e57`. Its notice retains CSI-Pose snapshot `02b1001590beea8e7d6daac57a4d8b4a345eb50f` and identifies WiSPPN-derived model code without a separately established redistribution grant. GhostMesh avoids that code entirely, introducing an original compact temporal model and matched baselines. The original researchers' figures and recorded pose results are not GhostMesh assets.

## Cite the dataset

```bibtex
@inproceedings{yang2023mmfi,
  title={MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing},
  author={Yang, Jianfei and Huang, He and Zhou, Yunjiao and Chen, Xinyan and
          Xu, Yuecong and Yuan, Shenghai and Zou, Han and Lu, Chris Xiaoxuan and Xie, Lihua},
  booktitle={Advances in Neural Information Processing Systems, Datasets and Benchmarks Track},
  year={2023},
  url={https://openreview.net/forum?id=1uAsASS1th}
}
```

[DATA_ACCESS.md](DATA_ACCESS.md) gives the actual data selection, annotation limits, coordinate plane, consent basis and reproduction commands. Measured scores must come from GhostMesh experiment reports, never these references.
