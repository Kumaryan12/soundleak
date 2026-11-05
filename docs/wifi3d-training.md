# Train a small Wi-Fi point-cloud model

Soundleak now includes an offline, supervised training pipeline and a small
**170,832-parameter model** trained from random initialization. It accepts the
recorded MM-Fi CSI amplitude/phase representation and outputs **256 XYZ points**.
No pretrained CSI2PointCloud checkpoint is used by this model.

## What has actually run

`outputs/wifi3d/training-sanity-v1/` contains a completed 100-step CPU optimization
check on the **six cached E01/S05/A05 frames**. All six frames are training data.
There is no validation or test set in this run.

- Initial training squared-Chamfer loss: **2.086983**.
- Final training squared-Chamfer loss: **0.001263**.
- Final training Euclidean nearest-point distance: **0.029413**.
- Reusing the first prediction for every training target: **0.028455**.

The loss reduction verifies that forward passes, gradients, updates, checkpoint
saving and reloading work. It does **not** establish accurate reconstruction from
unseen signals. In fact, the fixed-prediction control is still slightly better
than matching predictions to each training frame. The model can learn the common
scene geometry without reliably learning its changes.

These numbers are not directly comparable to the original model's replay scores:
this smaller architecture produces fewer points and was scored on its own
training data. Squared Chamfer is also distinct from the evaluation lab's
Euclidean nearest-point metric.

The checkpoint carries `diagnostic_only: true` and is **local only**. The hosted
site continues showing the earlier evaluation; this diagnostic has not replaced
its model or results.

## Architecture and targets

`SmallCSICloud` in `soundleak/reconstruction_train.py` uses two compact convolution
layers, adaptive pooling, a 128-dimensional hidden representation and a learned
point-set output. The six receiver channels are antenna × amplitude/phase; the
other axes are subcarrier and time. Initialization is seeded and training uses
AdamW with gradient clipping and bounded CLI budgets.

CSI preprocessing follows the verified recording loader. Actual measured LiDAR
points are filtered/subsampled without synthetic padding, then independently
standardized as targets. The point-set loss averages both directed squared
nearest-point distances and accepts differing measured point counts.
Coordinates are standardized, **not verified meters**. Per-frame target
normalization removes some physical translation/scale information.

## Reproduce a diagnostic

Use the optional environment described in [the reconstruction guide](wifi3d.md).
The main website environment does not need Torch.

```sh
outputs/research/wifi3d/venv/bin/python scripts/train_wifi3d.py \
  --sanity-check --steps 100 \
  --output outputs/wifi3d/training-another-diagnostic
```

The output directory must be new. Each run saves:

- `model.pt`: state dictionary, model configuration, seed, split identities and
  diagnostic status; load with `torch.load(..., weights_only=True)`.
- `training.json`: provenance, hashes, training configuration, losses, split
  identities, measured training controls and limitations.
- `loss.csv` and `loss.png`: optimization history with explicit split labels.

CPU is the tested path. Optional MPS/CUDA device paths are exposed but were not
validated in this run; operator support and memory limits must be checked before
using them for a larger experiment.

## Train with real separation

Normal training requires **separate training, validation and final-test
environments**. The runner refuses to split adjacent frames of one recording into
an alleged independent test. It checks input SHA256 hashes, disjoint environments,
and duplicate CSI bytes across partitions.

When the expanded download is available:

```sh
outputs/research/wifi3d/venv/bin/python scripts/fetch_wifi3d_suite.py
outputs/research/wifi3d/venv/bin/python scripts/train_wifi3d.py \
  --validation-environments E03 --test-environments E04 \
  --steps 1000 --points 256 --batch-size 4 \
  --output outputs/wifi3d/training-environment-pilot
```

In that fixed pilot, E01/E02 are training environments, E03 selects the checkpoint,
and E04 is scored after selection. The downloaded suite is still only a pilot:
96 frames, four participants and two actions. More recordings and participants
are needed before treating a result as robust. The official archive-index download
was quota-blocked during setup, so **this multi-environment training has not run**.

A custom verified manifest can be passed with `--manifest`. Normal training does
not use an unknown pretrained checkpoint, so its own training membership is
recorded explicitly. Validation loss selects the saved checkpoint; final-test
labels do not contribute to gradients or that selection. Repeatedly adjusting the
model after seeing test scores would require reserving a new final test set.

Before promoting a trained model into the live reconstruction lab, evaluate it
against fixed, shuffled, zero-input and different-environment controls using the
same point count and coordinate convention. Neither low training loss nor an
attractive point cloud alone demonstrates signal-dependent reconstruction.

## Tests and attribution

Training tests run in the optional environment:

```sh
outputs/research/wifi3d/venv/bin/python -m pip install 'pytest>=8,<10'
outputs/research/wifi3d/venv/bin/python -m pytest tests/test_reconstruction_train.py -q
```

The checks cover environment isolation, duplicate-input leakage, three-way
partitioning, unordered/variable-size targets, gradient updates, and exact
checkpoint reloading. The base test environment skips the Torch-only module.

Data: [MM-Fi](https://ntu-aiot-lab.github.io/mm-fi/), Jianfei Yang and colleagues,
NeurIPS 2023, **CC BY-NC 4.0**, noncommercial research. The diagnostic weights and
raw recordings are not included in the public deployment bundle. The larger
[CSI2PointCloud model](https://github.com/Arritmic/csi2pointcloud) remains a separate
upstream architecture and recorded replay, with its MIT notice retained.
