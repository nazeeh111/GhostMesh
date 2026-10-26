# MM-Fi data attribution

MM-Fi data, supplied 2D labels, and their redistributed processed excerpts retain the original **[Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)** terms. This includes dataset-backed CSI feature maps and supplied keypoint overlays in replay files. The GhostMesh implementation license does not relicense this material.

**Credit:** Jianfei Yang, He Huang, Yunjiao Zhou, Xinyan Chen, Yuecong Xu, Shenghai Yuan, Han Zou, Chris Xiaoxuan Lu and Lihua Xie. *MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing.* NeurIPS 2023 Datasets and Benchmarks Track.

[Original project and license declaration](https://ntu-aiot-lab.github.io/mm-fi) · [Official dataset and download links](https://github.com/ybhbingo/MMFi_dataset) · [Paper](https://arxiv.org/abs/2305.10345) · [License legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)

**Changes in GhostMesh:** select the stated action subset or full action set; pair CSI amplitude and supplied 2D keypoints by frame ID; convert arrays to float32; normalize coordinates by a declared isotropic reference plane; reconstruct nominal 10 Hz timestamps; create causal windows, signal maps, model predictions and replay overlays. The reference plane is not a recovered camera calibration. Exact selected actions, source hashes and counts are recorded in experiment manifests, the [primary cohort receipt](../docs/cohort-provenance.json) and the [earlier pilot receipt](../docs/data-provenance.json).

Give appropriate attribution, retain the license link, identify modifications, and use the material for noncommercial purposes. The MM-Fi authors do not endorse GhostMesh. New collection requires informed participant consent and a separately validated acquisition/training setup.

Generated software fixtures are explicitly labeled synthetic and are not MM-Fi measurements.
