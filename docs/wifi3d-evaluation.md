# Wi-Fi reconstruction evaluation lab

The **Evaluation lab** inside **Wi-Fi 3D replay** compares recorded CSI predictions
against deliberately mismatched inputs and static baselines. It includes sampled
frame playback, prediction/reference overlays, nearest-surface error colors,
shape-error curves, temporal change-field scores, a training-overlap audit, and
CSV/JSON/PLY exports. The hosted app replays offline inference; it does not collect
visitor Wi-Fi signals or run Torch on every visit.

## Current measured result

Only the original **six verified E01/S05/A05 frames** have been scored. Expansion
to eight recordings / 96 paired frames across four environments was attempted,
but the official Google Drive E01 archive returned quota-exceeded HTML instead
of its requested index bytes. Small range probes succeeded intermittently; no
additional complete paired recordings were obtained. The app marks the
**different-environment control unavailable**. It never substitutes generated data,
a second segment of the same clip, or a score of zero for missing measurements.

| Input | Mean shape distance | Mean temporal change-field MAE |
| --- | ---: | ---: |
| Correct CSI | 0.063273 | 0.016585 |
| Zero tensor | 0.066816 | 0.014878 |
| Fixed first prediction | 0.062714 | 0.014878 |
| Shuffled CSI frames | 0.063389 | 0.016506 |
| Different environment | Unavailable | Unavailable |

Both columns use standardized coordinate units; **lower is better**. The shuffled
control is close to correctly matched CSI, while the fixed prediction is better
on both metrics. This experiment does not establish reliable signal-dependent
reconstruction or temporal tracking. Six correlated frames of one recording are
not independent trials or a generalization benchmark.

## Controls and source pairing

- **Correct CSI:** matching recorded amplitude/phase, with the measured LiDAR
  introduced only after inference.
- **Zero input:** all-zero post-normalization tensor through the same model.
  This is a learned-prior diagnostic outside the recorded input distribution,
  not a physical zero-signal condition.
- **Fixed first frame:** the recording's first prediction reused at every frame.
- **Shuffled CSI:** a deterministic, seeded derangement of the recording's frame
  indices. Every input appears once and none is paired with its own reference.
  For an evaluation-mode deterministic model, reusing the saved wrong-frame
  prediction is equivalent to rerunning that wrong CSI input. The exact source
  identity appears in the UI and report.
- **Different environment:** when the expanded sample is available, use the same
  action and frame index from the next environment and a different participant.
  The mapping is predetermined, never chosen by reference fit. This has not yet
  been measured in the current run.

All controls use the same model, coordinate convention and nearest-point metric.
There is no alignment, fine-tuning, target-driven prediction rescaling, or selection
of recordings based on score. Aggregates weight recordings equally rather than
counting correlated frames as independent experiments.

## Shape and temporal metrics

Shape error is the average of the two directed mean Euclidean nearest-point
distances between predicted and measured clouds. Pointwise colors display both
directions separately: a reference point far from all predictions can reveal a
measured surface missing from the prediction. Colors saturate at a visible,
user-selected distance limit; they represent surface error, not uncertainty.

Point clouds are unordered, so temporal comparison does not subtract arbitrary
point indices. Instead, a **fixed 12×12×12 grid (1,728 probes)** spans all scored
clouds for a recording. At each frame and probe, compute distance to the nearest
surface. For each consecutive sampled pair, take the signed difference in this
field, then measure mean absolute error between predicted and measured field
changes. A static prediction has exactly zero predicted change; its error equals
the measured field-change magnitude. The grid is derived after inference and
shared across all methods, including controls.

This metric describes local surface changes in the replay's standardized
coordinates. The reference is independently standardized per frame, as in the
existing replay; that removes some physical translation and scale changes.
The result is **not displacement in meters, a motion-direction estimate or a
person's trajectory**. The grid is coarse and includes static background.
Playback loops sampled indices at 1.25 seconds per step, not the recording's
original real-time speed.

## Training overlap audit

The downloaded checkpoint contains only `epoch`, `loss`, `model_state_dict` and
`optimizer_state_dict`. It has no training sample list or split identifier.
The pinned repository configuration selects `manual_split`; it also contains
cross-subject, cross-scene and random alternatives. Our selected subjects would
be validation subjects under the cross-subject option, but that does not prove
which configuration produced these weights. The UI shows conditional membership
for each option and keeps checkpoint membership **unverified**.

The exact configuration snapshot is in
[`research/csi2pointcloud-config.yaml`](research/csi2pointcloud-config.yaml).
Its SHA256 is recorded alongside the official config, loader and checkpoint-writer
links in `evaluation.json`. This snapshot is MIT code from CSI2PointCloud, pinned
at `a180c3b6b55d00d142166dc0735856d7c5772b55`; the upstream MIT notice is retained
in `soundleak/_vendor/CSI2POINTCLOUD_LICENSE.txt`.

## Reproduce the current cached evaluation

Use the isolated environment from [the reconstruction guide](wifi3d.md):

```sh
outputs/research/wifi3d/venv/bin/python scripts/evaluate_wifi3d.py --cached-only
outputs/research/wifi3d/venv/bin/python scripts/export_wifi3d_evaluation.py
```

The default output is `outputs/wifi3d/evaluation`. It must not already exist.
Choose a different `--output` to preserve an earlier run and pass its directory
as `--folder` to the export script. Raw CSI and the large pinned checkpoint remain
local; only derived replay clouds and reports are deployed.

## Run the expanded sample when downloads are available

```sh
outputs/research/wifi3d/venv/bin/python scripts/fetch_wifi3d_suite.py
outputs/research/wifi3d/venv/bin/python scripts/evaluate_wifi3d.py --output outputs/wifi3d/evaluation-expanded
outputs/research/wifi3d/venv/bin/python scripts/export_wifi3d_evaluation.py --folder outputs/wifi3d/evaluation-expanded
```

The selection is fixed before scoring: `E01/S05`, `E02/S15`, `E03/S25`, `E04/S35`;
for each, actions `A05` and `A27`; frames 1 through 276 in steps of 25. This makes
96 paired frames from eight recordings. Participant and environment are confounded
in this small design, so the suite alone cannot establish independent room
robustness. It also does not resolve checkpoint overlap.

The downloader reads only official ZIP central directories and selected members
using HTTP ranges, with CRC/SHA256 verification and three bounded retries for
intermittent quota pages. It refuses full archive responses. Existing checked
members and indexes can be reused; a failed fetch does not write a successful
suite manifest or silently change the selection. A successful expanded run must
be reviewed and deliberately promoted into the hosted `evaluation` directory.

## Files and attribution

`outputs/wifi3d/evaluation/` contains the evaluation index, one derived-cloud JSON
per completed recording, all frame/method scores as CSV, and a standalone
scientific `evaluation.png`. The app also exports the displayed method's PLY.
The JSON index includes input hashes, control source identities, temporal scores,
checkpoint identity, training-audit evidence and explicit limitations.

[Official CSI2PointCloud implementation](https://github.com/Arritmic/csi2pointcloud):
Määttä, Sharifipour, Bordallo López and Álvarez Casado (2025), MIT architecture code.
[Official MM-Fi dataset](https://ntu-aiot-lab.github.io/mm-fi/): Jianfei Yang and
colleagues (NeurIPS 2023), **CC BY-NC 4.0**, noncommercial research. These dataset
attributions and restrictions apply to the derived replay. No raw archive or
model checkpoint is included in the hosted deployment.
