"""Run a fixed multi-environment CSI inference and negative-control evaluation."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from run_wifi3d import digest, load_model, CHECKPOINT_SHA256
from soundleak.reconstruction import prepare_csi, measured_reference, standardize_reference
from soundleak.reconstruction_eval import evaluate_recording, summarize, read_evaluation, METHODS


def training_audit(records, checkpoint_fields):
    config = ROOT/'docs/research/csi2pointcloud-config.yaml'
    commit = 'a180c3b6b55d00d142166dc0735856d7c5772b55'
    rows = []
    manual_subjects = {'S01', 'S02', 'S03', 'S04', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11'}
    for item in records:
        scene, subject, action = item['recording'].split('/')
        # This matches the pinned loader's action-specific random subject split.
        order = np.random.RandomState(int(action[1:])-1).permutation(40)
        random_train = int(subject[1:])-1 in order[:32]
        rows.append({'recording': item['recording'], 'checkpoint_membership': 'unverified',
                     'if_current_manual_split': 'train' if subject in manual_subjects and int(action[1:]) <= 7 else 'not in train',
                     'if_current_cross_subject_split': 'validation subject' if int(subject[1:]) % 5 == 0 else 'train subject',
                     'if_current_cross_scene_split': 'train' if scene != 'E04' else 'validation',
                     'if_current_random_split': 'train' if random_train else 'validation'})
    return {'status': 'unverified', 'checkpoint_fields': checkpoint_fields,
            'reason': 'Checkpoint contains no training sample manifest or split identifier. Repository configurations are conditional evidence, not proof of checkpoint membership.',
            'config_source': f'https://github.com/Arritmic/csi2pointcloud/blob/{commit}/data/configurations/config_mmfi_csi2pointcloud.yaml',
            'config_sha256': digest(config),
            'loader_source': f'https://github.com/Arritmic/csi2pointcloud/blob/{commit}/src/mmfi/mmfi_lib/mmfi.py',
            'checkpoint_writer_source': f'https://github.com/Arritmic/csi2pointcloud/blob/{commit}/scripts/train_model.py',
            'config_split_to_use': 'manual_split', 'records': rows,
            'note': 'The sampled subjects are validation subjects under the repository cross-subject configuration. This does not establish which configuration produced this checkpoint.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/real/wifi3d')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/wifi3d/evaluation')
    parser.add_argument('--cached-only', action='store_true',
                        help='Explicitly evaluate only the original downloaded recording; different-environment control stays unavailable.')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists. Choose another --output to preserve earlier results.')
    manifest = json.loads((args.data/('manifest.json' if args.cached_only else 'evaluation-manifest.json')).read_text())
    if args.cached_only:
        manifest['recordings'] = [{'recording': manifest['recording'], 'frame_ids': manifest['frame_ids']}]
        manifest['selection'] = 'Only the original cached E01/S05/A05 recording, frames 1,51,101,151,201,251. Additional archives are unavailable; no sample substitution.'
    for entry in manifest['files']:
        if digest(args.data/entry['path']) != entry['sha256']:
            parser.error('Input integrity check failed: '+entry['path'])
    model, checkpoint_fields = load_model(args.data/'checkpoint_epoch_41_Arch01_v1.pth')
    args.output.mkdir(parents=True)
    clouds = {}
    with torch.inference_mode():
        zero = model(torch.zeros((1, 3, 114, 2, 10)))[0][0].numpy().round(6).tolist()
        for item in manifest['recordings']:
            recording = item['recording']
            frames = []
            for number in item['frame_ids']:
                folder = args.data/recording
                csi = folder/'wifi-csi'/f'frame{number:03d}.mat'
                lidar = folder/'lidar'/f'frame{number:03d}.bin'
                features = torch.from_numpy(prepare_csi(csi.read_bytes())).unsqueeze(0)
                features = (features-features.mean())/features.std()
                start = time.perf_counter()
                predicted = model(features)[0][0].numpy()
                elapsed = time.perf_counter()-start
                # Load the target only after inference. No target-driven rescaling.
                reference, normalization = standardize_reference(measured_reference(lidar.read_bytes(), seed=number))
                frames.append({'id': f'{recording}/frame{number:03d}', 'frame_number': number,
                               'predicted': predicted.round(6).tolist(), 'reference': reference.round(6).tolist(),
                               'reference_normalization': normalization, 'inference_seconds': round(elapsed, 4),
                               'csi_sha256': digest(csi), 'reference_sha256': digest(lidar)})
            clouds[recording] = {'schema_version': 1, 'source_kind': 'recorded_wifi_csi',
                'coordinate_units': 'standardized_model_coordinates', 'dataset': 'MM-Fi',
                'dataset_license': 'CC BY-NC 4.0', 'recording': recording,
                'model_commit': manifest['model_commit'], 'checkpoint_sha256': CHECKPOINT_SHA256,
                'zero_input_control': zero, 'frames': frames}
            print('Inferred', recording, len(frames), 'frames', flush=True)
    records = []
    names = list(clouds)
    for i, name in enumerate(names):
        # Same action in the next environment; frame numbers remain matched.
        other_name = names[(i+2) % len(names)] if len(names) > 1 else None
        record = evaluate_recording(clouds[name]['frames'], zero, clouds[other_name]['frames'] if other_name else None)
        filename = name.replace('/', '-')+'.json'
        path = args.output/filename
        path.write_text(json.dumps(clouds[name], separators=(',', ':'), allow_nan=False))
        record.update(cloud_file=filename, cloud_sha256=digest(path), unrelated_recording=other_name)
        records.append(record)
        print('Evaluated', name, record['mean_distance'], flush=True)
    report = {'schema_version': 1, 'source_kind': 'recorded_wifi_csi_evaluation',
              'coordinate_units': 'standardized_model_coordinates', 'dataset': 'MM-Fi',
              'dataset_license': 'CC BY-NC 4.0', 'input_manifest': manifest,
              'checkpoint_sha256': CHECKPOINT_SHA256, 'model_commit': manifest['model_commit'],
              'architecture_sha256': digest(ROOT/'soundleak/_vendor/csi2pointcloud.py'),
              'package_versions': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scipy')},
              'inference_device': 'cpu', 'inference_batch_size': 1,
              'training_overlap': training_audit(manifest['recordings'], checkpoint_fields),
              'metric': 'Mean bidirectional Euclidean nearest-point distance; lower is better; no fitted alignment.',
              'temporal_metric': 'MAE of signed nearest-surface distance changes at a shared fixed 12x12x12 probe grid per recording; no point correspondence or alignment.',
              'control_design': {
                  'correct': 'Prediction from the matching recorded CSI frame.',
                  'zero': 'Same model with all-zero post-normalization input; learned-prior diagnostic, not a physical zero signal.',
                  'fixed': 'Reuse the first prediction in this recording for every frame.',
                  'shuffled': 'Seeded derangement within each recording: no CSI frame paired with its own reference. Reuses deterministic model outputs, equivalent to rerunning the wrong input.',
                  'unrelated': 'Same action and frame number from the next environment and a different participant; deterministic source mapping, no target selection.'},
              'records': records, 'summary': summarize(records),
              'sample_mode': 'cached_single_recording' if args.cached_only else 'four_environment_suite',
              'expansion_status': {
                  'status': 'download_blocked' if args.cached_only else 'complete',
                  'planned_frames': 96, 'planned_recordings': 8, 'planned_environments': 4,
                  'reason': 'Official Google Drive returned quota-exceeded HTML for the E01 archive index. The expanded suite stopped before obtaining additional recordings.' if args.cached_only else '',
                  'next_run': 'python scripts/fetch_wifi3d_suite.py && python scripts/evaluate_wifi3d.py --output outputs/wifi3d/evaluation-expanded'},
              'limitations': [
                  'Original cached sample only: six frames of one recording. Multi-environment controls were not run because official downloads were blocked.' if args.cached_only else 'Fixed exploratory sample: four environments, one participant per environment, two actions and twelve frames per recording. Not the full dataset benchmark.',
                  'Participant and environment effects are confounded. Frames in a recording are correlated; aggregate scores weight recordings equally.',
                  'Checkpoint training overlap is unverified. No independent, held-out, cross-room-generalization or calibrated accuracy claim.',
                  'Prediction and reference use the existing standardized coordinate convention. Per-frame reference normalization removes some physical translation/scale changes; temporal scores are not displacement in meters.',
                  'All-zero input is outside the recorded input distribution. It diagnoses model priors, not a realistic sensor condition.',
                  'Shuffled and different-environment controls use a predetermined mapping; their outcomes can depend on the chosen sample and pairing.',
                  'Distance-field error measures local surface change at a coarse grid. It does not identify people, object identities or reliable motion trajectories.',
                  'The shared probe grid bounds all scored clouds and is used only after inference. Pointwise colors measure surface distance, not uncertainty.',
                  'No training, fine-tuning, target-based alignment or model/recording selection by score was performed.'
              ]}
    path = args.output/'evaluation.json'
    path.write_text(json.dumps(report, separators=(',', ':'), allow_nan=False))
    read_evaluation(path)
    with (args.output/'frame-scores.csv').open('w', newline='') as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(['recording', 'frame', 'method', 'source', 'nearest_point_distance_standardized'])
        for record in records:
            for row in record['frames']:
                for method in row['scores']:
                    source = (row['shuffled_source'] if method == 'shuffled' else row['unrelated_source'] if method == 'unrelated'
                              else record['frames'][0]['id'] if method == 'fixed' else 'all-zero tensor' if method == 'zero' else row['id'])
                    writer.writerow([record['recording'], row['frame_number'], method, source, row['scores'][method]])
    print('Summary:', json.dumps(report['summary'], indent=2), flush=True)
    print('Saved', path, flush=True)


if __name__ == '__main__':
    main()
