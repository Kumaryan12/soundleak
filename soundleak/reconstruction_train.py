"""Small CSI-to-point-set model and bounded supervised training utilities.

Torch is optional and this module is never imported by the hosted replay.
"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .reconstruction import prepare_csi, measured_reference, standardize_reference


class SmallCSICloud(nn.Module):
    """A compact model initialized from scratch; no pretrained model is used."""
    def __init__(self, points=256):
        super().__init__()
        if not 16 <= points <= 1200:
            raise ValueError('Output point count must be between 16 and 1200.')
        self.points = points
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 16, (5, 3), stride=(2, 1), padding=(2, 1)), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 2)), nn.Flatten(),
            nn.Linear(512, 128), nn.GELU())
        self.decoder = nn.Linear(128, points*3)

    def forward(self, csi):
        # (B, antenna, subcarrier, amp/phase, time) -> six receiver channels.
        value = csi.permute(0, 1, 3, 2, 4).reshape(-1, 6, 114, 10)
        return self.decoder(self.encoder(value)).reshape(-1, self.points, 3)


def chamfer_loss(predicted, targets):
    """Mean bidirectional squared nearest-point loss; no fabricated padding.

    Targets are a list because measured point counts can differ by frame.
    The loss is squared distance, distinct from the evaluation lab's Euclidean
    surface distance. Targets and predictions must already share a convention.
    """
    if predicted.ndim != 3 or predicted.shape[-1] != 3 or len(targets) != len(predicted):
        raise ValueError('Expected a batch of XYZ predictions with one target per frame.')
    losses = []
    for prediction, target in zip(predicted, targets):
        if target.ndim != 2 or target.shape[1] != 3 or len(target) < 3:
            raise ValueError('Targets must contain actual measured XYZ points.')
        distances = torch.cdist(prediction.unsqueeze(0), target.unsqueeze(0)).square()[0]
        losses.append((distances.min(dim=1).values.mean()+distances.min(dim=0).values.mean())/2)
    return torch.stack(losses).mean()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_training_data(root, manifest_path):
    """Read verified recorded inputs and measured targets; never pad geometry."""
    root = Path(root)
    manifest = json.loads(Path(manifest_path).read_text())
    entries = {entry['path']: entry for entry in manifest['files']}
    records = manifest.get('recordings') or [{'recording': manifest['recording'], 'frame_ids': manifest['frame_ids']}]
    examples = []
    for record in records:
        recording = record['recording']
        parts = recording.split('/')
        if len(parts) != 3 or parts[0] not in ('E01', 'E02', 'E03', 'E04') or any('..' in p for p in parts):
            raise ValueError('Invalid dataset recording identity.')
        for number in record['frame_ids']:
            paths = [f'{recording}/wifi-csi/frame{number:03d}.mat', f'{recording}/lidar/frame{number:03d}.bin']
            for name in paths:
                if name not in entries or digest(root/name) != entries[name]['sha256']:
                    raise ValueError('Input integrity check failed: '+name)
            features = torch.from_numpy(prepare_csi((root/paths[0]).read_bytes()))
            features = (features-features.mean())/features.std()
            target, normalization = standardize_reference(measured_reference((root/paths[1]).read_bytes(), seed=number))
            examples.append({'id': f'{recording}/frame{number:03d}', 'recording': recording,
                             'features': features, 'target': torch.from_numpy(target),
                             'reference_normalization': normalization,
                             'csi_sha256': entries[paths[0]]['sha256'], 'reference_sha256': entries[paths[1]]['sha256']})
    if len({e['id'] for e in examples}) != len(examples):
        raise ValueError('Duplicate frame identity in training manifest.')
    return examples


def split_examples(examples, held_out_environments):
    """Enforce environment-level separation; adjacent frames are never held out."""
    held_out = set(held_out_environments)
    if not held_out or not held_out <= {'E01', 'E02', 'E03', 'E04'}:
        raise ValueError('Choose at least one valid held-out environment.')
    training = [e for e in examples if e['recording'].split('/')[0] not in held_out]
    validation = [e for e in examples if e['recording'].split('/')[0] in held_out]
    if not training or not validation:
        raise ValueError('Training and held-out environments must both have recorded examples. Six frames from one recording cannot provide this split.')
    training_hashes = {e['csi_sha256'] for e in training}
    if training_hashes & {e['csi_sha256'] for e in validation}:
        raise ValueError('Identical CSI bytes appear in both partitions.')
    return training, validation


def model_batch(model, examples, device):
    return model(torch.stack([e['features'] for e in examples]).to(device))


def partition_examples(examples, validation_environments, test_environments):
    """Three disjoint environment groups, with test reserved until selection ends."""
    validation_envs, test_envs = set(validation_environments), set(test_environments)
    if not validation_envs or not test_envs or validation_envs & test_envs:
        raise ValueError('Validation and test environments must be nonempty and disjoint.')
    training, held_out = split_examples(examples, validation_envs | test_envs)
    validation, test = split_examples(held_out, test_envs)
    return training, validation, test
