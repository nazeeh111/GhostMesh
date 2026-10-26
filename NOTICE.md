# Source, data and attribution

GhostMesh is an independent implementation. It does not copy the CSI-Pose,
WiSPPN or MM-Fi toolbox model/loader source or upstream model weights and
research figures. It uses licensed MM-Fi CSI and keypoint data, including the
small derived excerpts and visualizations distributed with this repository.
Poseborne is the user's existing upstream-derived project and a reference for
the workflow. No quantitative superiority over Poseborne has been established.

The original Python and browser software uses the MIT license in LICENSE.
MM-Fi data, supplied 2D teacher annotations, derived replay visualizations and
trained model artifacts retain the dataset's CC BY-NC 4.0 noncommercial terms.
See data/LICENSE.md for citation, source and transformation details. These
data-derived artifacts are not relicensed as MIT.

MM-Fi: Jianfei Yang, He Huang, Yunjiao Zhou, Xinyan Chen, Yuecong Xu, Shenghai
Yuan, Han Zou, Chris Xiaoxuan Lu and Lihua Xie. *MM-Fi: Multi-Modal Non-Intrusive
4D Human Dataset for Versatile Wireless Sensing*. NeurIPS 2023 Datasets and
Benchmarks. https://openreview.net/forum?id=1uAsASS1th

ESP-CSI is a separately licensed Apache-2.0 firmware reference. GhostMesh's
serial parser is independently implemented; firmware is not bundled. NumPy,
SciPy, PyTorch and the Python/web toolchains retain their respective licenses.

Use only recordings from participants and sites with permission to collect
and analyze them. The app does not provide covert collection, tracking,
medical assessment, fall-safety decisions or multi-person identification.
