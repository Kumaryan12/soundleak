# Recorded Wi-Fi to 3D point clouds

This experiment runs a published CSI2PointCloud checkpoint on **six actual MM-Fi
Wi-Fi CSI frames** from `E01/S05/A05` (frames 001, 051, 101, 151, 201 and 251).
The point clouds are model predictions, not hand-authored room geometry. A paired
LiDAR scan is loaded after each model forward pass for comparison. No camera,
LiDAR, reference position or reference normalization is passed into inference.

The hosted **Wi-Fi 3D replay** displays these precomputed results. It does not
collect signals from a visitor's router or execute the large model on each visit.
Its **Evaluation lab** adds negative controls, temporal surface-change metrics,
error-colored clouds and playback; the original replay remains available under
**Original six-frame replay**. See [the evaluation guide](wifi3d-evaluation.md)
for current measured results and the blocked multi-environment expansion.
The **Wi-Fi lab** remains a separate Mac RSSI diagnostic tool; RSSI captures from
that tool are not compatible with this model.

## Reproduce locally

Use a separate Python environment. The first run here used Python 3.14 on Apple
Silicon, CPU inference with four Torch threads, and the versions in
`requirements-reconstruction.txt`. CUDA is not required. For example:

```sh
python3 -m venv outputs/research/wifi3d/venv
outputs/research/wifi3d/venv/bin/python -m pip install -r requirements-reconstruction.txt
outputs/research/wifi3d/venv/bin/python scripts/fetch_wifi3d.py
outputs/research/wifi3d/venv/bin/python scripts/run_wifi3d.py
outputs/research/wifi3d/venv/bin/python scripts/export_wifi3d.py
```

Fetch downloads the approximately 365 MB checkpoint and uses HTTP ranges to read
the E01 archive's approximately 67 MB central directory plus only twelve paired
CSI/LiDAR members. It refuses a server response that would download the full
22 GB archive. Subsequent runs can reuse the index and verified member files.
The inference output directory must not already exist; choose `--output` to keep
multiple runs. Pass the same directory to the export script with `--folder`.

The checkpoint is pinned by SHA256:

```text
7a624013000a34f7f24a939e4b69e305bb4c6dd0d3bdc8623ed869170dc128cb
```

It is loaded on CPU with `weights_only=True`, with only the specific NumPy
scalar/dtype constructors needed for its saved loss additionally allowed. The
architecture is vendored from upstream commit
`a180c3b6b55d00d142166dc0735856d7c5772b55`; its MIT notice is in
`soundleak/_vendor/CSI2POINTCLOUD_LICENSE.txt`.

## What the model receives

Each `.mat` frame contains `CSIamp` and `CSIphase`, each shaped 3 antennas × 114
subcarriers × 10 time slices. As in upstream preprocessing, missing values are
filled from the same time slice, amplitude is scaled to [0, 1], and amplitude and
phase are stacked. All-missing slices and constant amplitude are rejected.
The resulting tensor is z-scored and passed to the model in a batch of one.

The architecture uses embedding dimension 512, four encoder and decoder layers,
four attention heads and 1,200 output points. Input hashes, source commit, package
versions and inference times are recorded in `reconstruction.json`.

## Coordinates and reference

The upstream training and testing scripts standardize target coordinates with a
single scalar mean and standard deviation. The output therefore has **standardized
model coordinates**, not verified meters. We do not infer physical room size or
use test LiDAR statistics to rescale predictions into meters.

For comparison, actual LiDAR points with first coordinate >= 0 are retained and
deterministically subsampled to at most 1,200 points. Unlike the upstream loader,
we do not fabricate padding points when fewer points are available. Reference
coordinates are independently standardized for comparison after inference. The
two clouds are not aligned using ICP. Labels and exports preserve this coordinate
convention.

## Results and controls

Mean bidirectional Euclidean nearest-point distance on this selected sample:

| Input or baseline | Distance in standardized units |
| --- | ---: |
| Each frame's Wi-Fi CSI | 0.06327 |
| All-zero tensor through the same model | 0.06682 |
| Reuse frame 001's prediction for every reference | 0.06271 |

Lower is better. The zero-input baseline shows that a trained model can produce
a scene-shaped cloud from its learned prior. More importantly, the fixed first
prediction is slightly better than updating predictions from each CSI frame.
**This experiment does not establish reliable reconstruction of changing geometry.**
It is also not a reproduction of the paper's complete benchmark: the sample is
small, selected from one recording, uses batch size one, and checkpoint training
membership is unverified. No held-out or cross-room accuracy claim is made.

## Outputs

`outputs/wifi3d/mmfi-reference/` contains:

- `viewer.html`: self-contained interactive orbit viewer with frame selection,
  synchronized prediction/reference views, overlay and zero-input control.
- `comparison.png`: three prediction/reference comparisons with consistent axes.
- `frameNNN-prediction.ply`: 1,200 predicted points per frame.
- `frameNNN-reference.ply`: actual measured, standardized comparison points.
- `reconstruction.json`: all point arrays, metrics, limitations and provenance.

The result is a learned point-cloud estimate of a recorded dataset scene. It is
not a textured mesh, surveyed floor plan, or reconstruction of the user's home.

## Attribution and use

**MM-Fi**: Jianfei Yang, He Huang and colleagues, *MM-Fi: Multi-Modal Non-Intrusive
4D Human Dataset for Versatile Wireless Sensing*, NeurIPS 2023. The dataset is
released under **CC BY-NC 4.0**. The included reference-derived replay is for
noncommercial research and retains this attribution and restriction.

**CSI2PointCloud**: Tuomas Määttä, Sasan Sharifipour, Miguel Bordallo López and
Constantino Álvarez Casado, *Spatio-Temporal 3D Point Clouds from Wi-Fi-CSI Data via
Transformer Networks*, IEEE JC&S 2025. Architecture code: MIT, copyright 2024
Constantino Álvarez. The large checkpoint and raw recordings are not included in
the deployment checkout; source links and hashes are provided instead.

- [Official CSI2PointCloud implementation](https://github.com/Arritmic/csi2pointcloud)
- [Official MM-Fi dataset and license](https://ntu-aiot-lab.github.io/mm-fi/)
- [MM-Fi download links and acquisition details](https://github.com/ybhbingo/MMFi_dataset)
