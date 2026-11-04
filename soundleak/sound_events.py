"""Local multi-label sound suggestions, separate from Doppler localization.

Google YAMNet, Apache-2.0, via a pinned community ONNX conversion. The model
consumes mono 16 kHz waveform and produces scores for 521 sound-event labels.
"""
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .archive import file_sha256

MODEL_DIRECTORY = Path('data/real/models/yamnet-onnx')
MODEL_REVISION = 'f25b741c2f0bdc6d7e6db24b5fddda23347dbafd'
MODEL_SOURCE = 'https://huggingface.co/audiomagic/yamnet-onnx'
MODEL_SHA256 = 'd3835ffbbd4a1bb3e777f0ca217b5007907f5171dd5d17c4236b95b2af8f908e'
ASSET_SHA256 = {'yamnet.onnx': MODEL_SHA256,
                'yamnet_class_map.csv': 'cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2',
                'LICENSE': 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',
                'README.md': 'fcf25e5a0a1cbf3a9a0bfd97de6aabfa0efdbe4a4ff8cc5db0df5584bc01707f'}


def fetch_sound_model(directory=MODEL_DIRECTORY):
    import requests
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = f'{MODEL_SOURCE}/resolve/{MODEL_REVISION}/'
    files = []
    for name in ('yamnet.onnx', 'yamnet_class_map.csv', 'LICENSE', 'README.md'):
        path = directory/name
        if not path.exists() or file_sha256(path) != ASSET_SHA256[name]:
            temporary = path.with_suffix(path.suffix+'.part')
            with requests.get(base+name, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                size = 0
                with temporary.open('wb') as output:
                    for block in response.iter_content(1024*1024):
                        size += len(block)
                        if size > 25_000_000:
                            raise ValueError('Unexpected sound-model download size.')
                        output.write(block)
            if file_sha256(temporary) != ASSET_SHA256[name]:
                raise ValueError('Sound model asset failed its pinned SHA-256 check.')
            temporary.replace(path)
        files.append({'file': name, 'sha256': file_sha256(path), 'url': base+name})
    manifest = {'schema_version': 1, 'upstream': 'https://www.tensorflow.org/hub/tutorials/yamnet',
                'conversion_source': MODEL_SOURCE, 'revision': MODEL_REVISION, 'license': 'Apache-2.0',
                'files': files, 'interpretation': 'Pretrained sound-event scores, not proof of objects or calibrated probabilities. No source-location estimates.'}
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def load_sound_model(directory=MODEL_DIRECTORY):
    import onnxruntime as ort
    directory = Path(directory)
    if file_sha256(directory/'yamnet.onnx') != MODEL_SHA256:
        raise ValueError('Sound model failed its pinned SHA-256 check.')
    manifest = json.loads((directory/'manifest.json').read_text())
    for entry in manifest['files']:
        if (entry['file'] not in ASSET_SHA256 or entry['sha256'] != ASSET_SHA256[entry['file']]
            or file_sha256(directory/entry['file']) != ASSET_SHA256[entry['file']]):
            raise ValueError('Sound model assets failed their integrity check.')
    if {a['file'] for a in manifest['files']} != set(ASSET_SHA256):
        raise ValueError('Incomplete sound-model asset manifest.')
    rows = list(csv.DictReader((directory/'yamnet_class_map.csv').read_text().splitlines()))
    if len(rows) != 521 or [int(r['index']) for r in rows] != list(range(521)):
        raise ValueError('Unexpected sound-label mapping.')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(directory/'yamnet.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    if session.get_inputs()[0].name != 'waveform':
        raise ValueError('Unexpected sound-model input.')
    return session, [r['display_name'] for r in rows]


def summarize_sound_scores(scores, labels, duration_s, threshold=.25):
    scores = np.asarray(scores)
    if (scores.ndim != 2 or scores.shape[1] != len(labels) or not len(scores) or not np.isfinite(scores).all()
        or np.any((scores < -1e-5) | (scores > 1+1e-5)) or not 0 <= threshold <= 1
        or not np.isfinite(duration_s) or duration_s <= 0):
        raise ValueError('Invalid sound scores or display threshold.')
    order = np.argsort(-np.mean(scores, axis=0), kind='stable')[:5]
    overall = [{'label': labels[i], 'score': float(np.mean(scores[:, i]))} for i in order]
    peak_scores = np.max(scores, axis=0)
    peak_order = np.argsort(-peak_scores, kind='stable')[:5]
    peaks = [{'label': labels[i], 'score': float(peak_scores[i]), 'mean_score': float(np.mean(scores[:, i])),
              'start_s': min(float(np.argmax(scores[:, i])*.48), duration_s)} for i in peak_order]
    intervals = []
    for j, label in enumerate(labels):
        if label == 'Silence':
            continue
        active = (scores[:, j] >= threshold) & (np.arange(len(scores))*.48 < duration_s)
        edges = np.diff(np.r_[False, active, False].astype(int))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            intervals.append({'label': label, 'start_s': float(start*.48),
                              'end_s': min(float((end-1)*.48+.975), duration_s),
                              'peak_score': float(np.max(scores[start:end, j]))})
    intervals.sort(key=lambda item: (item['start_s'], -item['peak_score'], item['label']))
    frames = []
    for i, row in enumerate(scores):
        start = i*.48
        if start >= duration_s:
            break
        best = np.argsort(-row, kind='stable')[:3]
        frames.append({'start_s': start, 'end_s': min(start+.975, duration_s),
                       'label': labels[best[0]] if row[best[0]] >= threshold else 'No strong sound match',
                       'top': [{'label': labels[j], 'score': float(row[j])} for j in best]})
    return {'overall': overall, 'peaks': peaks, 'label_intervals': intervals, 'frames': frames, 'display_threshold': threshold,
            'interpretation': 'Multiple labels may apply simultaneously. Scores are uncalibrated model outputs; silence, masking and unseen sound sources can cause errors. No object identity or location is established.'}


def classify_sound(audio, sample_rate, model, threshold=.25):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or not 1000 <= sample_rate <= 192000:
        raise ValueError('Need finite mono audio and a supported sample rate.')
    if len(audio)/sample_rate > 120.001:
        raise ValueError('Sound suggestions are limited to two minutes per sample.')
    if sample_rate != 16000:
        from math import gcd
        divisor = gcd(int(sample_rate), 16000)
        audio = resample_poly(audio, 16000//divisor, int(sample_rate)//divisor).astype(np.float32)
    audio = np.clip(audio, -1., 1.)
    session, labels = model
    output_name = session.get_outputs()[0].name
    scores = session.run([output_name], {'waveform': audio})[0]
    result = summarize_sound_scores(scores, labels, len(audio)/16000, threshold)
    result.update({'model': 'YAMNet', 'model_revision': MODEL_REVISION, 'model_sha256': MODEL_SHA256,
                   'sample_rate': 16000, 'duration_s': len(audio)/16000,
                   'decoded_audio_sha256': sha256(audio.tobytes()).hexdigest(), 'local_inference': True})
    return result
