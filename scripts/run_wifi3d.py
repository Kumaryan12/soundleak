"""Run the pinned CSI2PointCloud model on six real MM-Fi frames, offline."""
import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from soundleak._vendor.csi2pointcloud import CSI2PointCloudModel
from soundleak.reconstruction import (prepare_csi, measured_reference, standardize_reference,
                                      symmetric_nn_distance, ply_bytes, read_reconstruction)

CHECKPOINT_SHA256 = '7a624013000a34f7f24a939e4b69e305bb4c6dd0d3bdc8623ed869170dc128cb'


def digest(path):
    h = sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_model(checkpoint):
    """Load only the pinned model with restricted deserialization."""
    if digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint does not match the pinned research model.')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    with torch.serialization.safe_globals([(np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
                                           np.dtype, np.dtypes.Float64DType]):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = CSI2PointCloudModel(512, 4, 4, 4, 1200)
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.eval()
    return model, sorted(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/real/wifi3d')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/wifi3d/mmfi-reference')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists. Choose a new directory to preserve the earlier run.')
    manifest = json.loads((args.data/'manifest.json').read_text())
    checkpoint = args.data/'checkpoint_epoch_41_Arch01_v1.pth'
    if digest(checkpoint) != CHECKPOINT_SHA256:
        parser.error('Checkpoint does not match the pinned research model.')
    for entry in manifest['files']:
        path = args.data/entry['path']
        if digest(path) != entry['sha256']:
            parser.error('Recorded input integrity check failed: ' + entry['path'])
    model, _ = load_model(checkpoint)
    args.output.mkdir(parents=True)
    frames = []
    with torch.inference_mode():
        zero = model(torch.zeros((1, 3, 114, 2, 10)))[0][0].numpy()
        for number in manifest['frame_ids']:
            folder = args.data/manifest['recording']
            csi = folder/'wifi-csi'/f'frame{number:03d}.mat'
            lidar = folder/'lidar'/f'frame{number:03d}.bin'
            features = torch.from_numpy(prepare_csi(csi.read_bytes())).unsqueeze(0)
            features = (features - features.mean()) / features.std()
            start = time.perf_counter()
            predicted = model(features)[0][0].numpy()
            elapsed = time.perf_counter() - start
            # Geometry is introduced only after the forward pass, for comparison.
            reference = measured_reference(lidar.read_bytes(), seed=number)
            reference, normalization = standardize_reference(reference)
            frame = {'id': f'{manifest["recording"]}/frame{number:03d}', 'frame_number': number,
                     'predicted': predicted.round(6).tolist(), 'reference': reference.round(6).tolist(),
                     'inference_seconds': round(elapsed, 4),
                     'symmetric_nn_distance': symmetric_nn_distance(predicted, reference),
                     'zero_input_nn_distance': symmetric_nn_distance(zero, reference),
                     'reference_normalization': normalization,
                     'csi_sha256': digest(csi), 'reference_sha256': digest(lidar)}
            frames.append(frame)
            (args.output/f'frame{number:03d}-prediction.ply').write_bytes(ply_bytes(predicted))
            (args.output/f'frame{number:03d}-reference.ply').write_bytes(ply_bytes(reference, 'Measured LiDAR; standardized comparison coordinates'))
            print(frame['id'], 'NN distance:', round(frame['symmetric_nn_distance'], 4),
                  'zero-input:', round(frame['zero_input_nn_distance'], 4), 'seconds:', round(elapsed, 2), flush=True)
    for frame in frames:
        frame['fixed_first_frame_nn_distance'] = symmetric_nn_distance(frames[0]['predicted'], frame['reference'])
    report = {'schema_version': 1, 'source_kind': 'recorded_wifi_csi',
              'coordinate_units': 'standardized_model_coordinates', 'dataset': 'MM-Fi',
              'recording': manifest['recording'], 'dataset_license': 'CC BY-NC 4.0',
              'dataset_home': manifest['dataset_home'], 'model_repository': manifest['model_repository'],
              'model_commit': manifest['model_commit'], 'checkpoint_sha256': CHECKPOINT_SHA256,
              'checkpoint_epoch': int(41), 'architecture_sha256': digest(ROOT/'soundleak/_vendor/csi2pointcloud.py'),
              'input_manifest': manifest, 'inference_device': 'cpu', 'inference_batch_size': 1,
              'package_versions': {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'scipy']},
              'metric': 'Mean bidirectional Euclidean nearest-point distance, standardized units, no ICP alignment.',
              'mean_nn_distance': float(np.mean([f['symmetric_nn_distance'] for f in frames])),
              'mean_zero_input_nn_distance': float(np.mean([f['zero_input_nn_distance'] for f in frames])),
              'mean_fixed_frame_nn_distance': float(np.mean([f['fixed_first_frame_nn_distance'] for f in frames])),
              'zero_input_control': zero.round(6).tolist(), 'frames': frames,
              'limitations': [
                  'Six selected frames of one public recording; not an independent reconstruction benchmark.',
                  'Checkpoint training membership is unverified; no held-out accuracy claim.',
                  'Predictions use only CSI amplitude and phase; LiDAR is used afterwards for comparison.',
                  'Upstream preprocessing and a batch size of one are used; published benchmark scores are not reproduced.',
                  'The model predicts standardized coordinates. Room dimensions in meters cannot be recovered from this output.',
                  'Reference points are measured LiDAR, forward-filtered and deterministically subsampled. No synthetic padding.',
                  'Learned point-cloud prediction is not a surveyed floor plan, textured room mesh or reconstruction of the user’s home.'
              ]}
    path = args.output/'reconstruction.json'
    path.write_text(json.dumps(report, separators=(',', ':'), allow_nan=False))
    read_reconstruction(path)
    print('Saved recorded CSI reconstruction:', path, flush=True)


if __name__ == '__main__':
    main()
