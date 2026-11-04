---
license: apache-2.0
tags:
  - audio
  - audio-classification
  - audio-event-detection
  - yamnet
  - audioset
  - onnx
library_name: onnx
pipeline_tag: audio-classification
---

# YAMNet (ONNX)

An ONNX export of **Google's YAMNet**, an audio event classifier trained on
[AudioSet](https://research.google.com/audioset/). It predicts 521 audio event
classes and also exposes a 1024-dimensional embedding that is well suited to
transfer learning on small, custom sound datasets.

This repository exists so that [Audio Magic](https://audiomagic.studio)'s
"Detect Sounds" tool can fetch the model from a host we control. It is a
straight format conversion — **no weights were retrained, fine-tuned, pruned or
quantized.**

## Provenance

| | |
|---|---|
| Upstream model | [`google/yamnet`](https://www.kaggle.com/models/google/yamnet) version 1 (formerly `tfhub.dev/google/yamnet/1`) |
| Upstream code | [`tensorflow/models` → `research/audioset/yamnet`](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) |
| Upstream licence | Apache 2.0 |
| Source archive sha256 | `b80da2a1a56926fb0767205051a200dd7b3beaf3ea1ea126c42a53943996e5e0` |

## Changes made to the original work

As required by Apache 2.0 §4(b), the modifications are:

1. The TensorFlow SavedModel was converted to ONNX with `tf2onnx`
   (`--saved-model`, opset 15). No weight values were altered.
2. `yamnet_class_map.csv` is copied verbatim from the SavedModel's `assets/`
   directory for convenience.

Nothing else was changed. The original `NOTICE` file is not reproduced because
upstream does not ship one.

## Inputs and outputs

The mel-spectrogram frontend is **baked into the graph**, so the model consumes
a raw waveform directly — there is no feature extraction to reimplement (and so
no opportunity to get it subtly wrong).

**Input** — `waveform`, `float32`, shape `[num_samples]`: mono PCM at
**16 kHz**, nominally in `[-1.0, 1.0]`. Length is dynamic.

**Outputs**, one row per frame:

| Name | Shape | Meaning |
|---|---|---|
| `output_0` | `[frames, 521]` | Per-class scores over the AudioSet ontology |
| `output_1` | `[frames, 1024]` | Embeddings (the useful part for transfer learning) |
| `output_2` | `[frames × 96, 64]` | Log-mel spectrogram patches |

Each frame covers **0.96 s** of audio and the window advances **0.48 s**, so
frame `i` spans `[i × 0.48, i × 0.48 + 0.96]` seconds. Input shorter than one
patch is zero-padded and still yields one frame. Empirically:

```
frames = 1 + ceil(max(0, num_samples - 15360) / 7680)
```

## Usage

```python
import numpy as np, onnxruntime as ort

session = ort.InferenceSession("yamnet.onnx")
waveform = np.zeros(16000 * 3, dtype=np.float32)  # 3 s of 16 kHz mono
scores, embeddings, log_mel = session.run(None, {"waveform": waveform})
```

## Verification

The conversion was checked against YAMNet's documented reference behaviour
rather than assumed correct:

| Input | Top class |
|---|---|
| 1 kHz sine wave | `Beep, bleep` |
| Digital silence | `Silence` (1.00) |
| White noise | `Static` / `White noise` |

Frame counts match the published 0.96 s / 0.48 s framing, and outputs were
compared numerically against an independent existing conversion across sine,
silence, noise, chirp and sub-second inputs.

## Licence

Apache 2.0, inherited from the upstream model and code.
Copyright 2022 Google LLC. The full licence text is in [`LICENSE`](./LICENSE).

The AudioSet ontology and labels are released by Google under CC BY 4.0.

## Citation

YAMNet implements the architecture described in:

```bibtex
@inproceedings{hershey2017cnn,
  title     = {{CNN} Architectures for Large-Scale Audio Classification},
  author    = {Hershey, Shawn and Chaudhuri, Sourish and Ellis, Daniel P. W. and
               Gemmeke, Jort F. and Jansen, Aren and Moore, R. Channing and
               Plakal, Manoj and Platt, Devin and Saurous, Rif A. and
               Seybold, Bryan and Slaney, Malcolm and Weiss, Ron J. and
               Wilson, Kevin},
  booktitle = {2017 IEEE International Conference on Acoustics, Speech and
               Signal Processing (ICASSP)},
  year      = {2017}
}

@inproceedings{gemmeke2017audioset,
  title     = {Audio Set: An ontology and human-labeled dataset for audio events},
  author    = {Gemmeke, Jort F. and Ellis, Daniel P. W. and Freedman, Dylan and
               Jansen, Aren and Lawrence, Wade and Moore, R. Channing and
               Plakal, Manoj and Ritter, Marvin},
  booktitle = {Proc. IEEE ICASSP 2017},
  address   = {New Orleans, LA},
  year      = {2017}
}
```

It uses the MobileNetV1 depthwise-separable convolution architecture:

```bibtex
@article{howard2017mobilenets,
  title   = {{MobileNets}: Efficient Convolutional Neural Networks for Mobile
             Vision Applications},
  author  = {Howard, Andrew G. and Zhu, Menglong and Chen, Bo and
             Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and
             Andreetto, Marco and Adam, Hartwig},
  journal = {arXiv preprint arXiv:1704.04861},
  year    = {2017}
}
```
