"""Signal-dependence controls and temporal evaluation of unordered point clouds.

All comparisons use standardized coordinates and never fit an alignment. This
module accepts predictions, not a model, so reference geometry cannot enter its
forward pass. Temporal fields use one fixed probe grid for every method.
"""
import json
from hashlib import sha256
from pathlib import Path
import re

import numpy as np
from scipy.spatial import cKDTree

from .reconstruction import validate_cloud, symmetric_nn_distance, read_reconstruction

METHODS = {
    'correct': 'Correct CSI', 'zero': 'Zero input', 'fixed': 'Fixed first frame',
    'shuffled': 'Shuffled CSI frames', 'unrelated': 'Different environment CSI',
}


def derangement(count, seed=20261105):
    if count < 2:
        raise ValueError('Shuffled control needs at least two frames.')
    order = np.random.default_rng(seed).permutation(count)
    mapping = np.empty(count, dtype=int)
    mapping[order] = np.roll(order, 1)
    return mapping.tolist()


def point_errors(predicted, reference):
    predicted, reference = validate_cloud(predicted), validate_cloud(reference)
    return cKDTree(reference).query(predicted)[0], cKDTree(predicted).query(reference)[0]


def probe_grid(clouds, resolution=12):
    """One shared grid bounds all scored geometry; evaluation only, not inference."""
    points = np.concatenate([validate_cloud(p) for p in clouds])
    low, high = points.min(axis=0), points.max(axis=0)
    pad = np.maximum((high-low)*.05, .01)
    axes = [np.linspace(low[i]-pad[i], high[i]+pad[i], resolution) for i in range(3)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)


def temporal_scores(predictions, references, probes):
    """Compare signed changes of nearest-surface distances at fixed probes.

    Clouds have no point correspondence. A constant prediction has exactly zero
    field change. No per-frame alignment is applied and the comparison is in the
    existing standardized coordinate convention, not physical displacement.
    """
    if len(predictions) != len(references) or len(predictions) < 2:
        raise ValueError('Temporal comparison requires paired multi-frame clouds.')
    predicted_fields = np.array([cKDTree(validate_cloud(p)).query(probes)[0] for p in predictions])
    reference_fields = np.array([cKDTree(validate_cloud(p)).query(probes)[0] for p in references])
    pred_delta, ref_delta = np.diff(predicted_fields, axis=0), np.diff(reference_fields, axis=0)
    return [{'predicted_change': float(np.abs(p).mean()), 'reference_change': float(np.abs(r).mean()),
             'change_field_mae': float(np.abs(p-r).mean())} for p, r in zip(pred_delta, ref_delta)]


def evaluate_recording(frames, zero, other_frames=None, seed=20261105):
    if len(frames) < 2 or (other_frames is not None and len(frames) != len(other_frames)):
        raise ValueError('Controls require equally sized recordings with multiple frames.')
    own_ids, other_ids = {f['id'] for f in frames}, {f['id'] for f in other_frames or []}
    if own_ids & other_ids or any(a['id'].split('/')[0] == b['id'].split('/')[0]
                                 for a, b in zip(frames, other_frames or [])):
        raise ValueError('Different-environment control must come from another environment.')
    mapping = derangement(len(frames), seed)
    clouds = {
        'correct': [f['predicted'] for f in frames], 'zero': [zero]*len(frames),
        'fixed': [frames[0]['predicted']]*len(frames),
        'shuffled': [frames[i]['predicted'] for i in mapping],
    }
    if other_frames is not None:
        clouds['unrelated'] = [f['predicted'] for f in other_frames]
    references = [f['reference'] for f in frames]
    probes = probe_grid(references + [p for series in clouds.values() for p in series])
    temporal = {key: temporal_scores(value, references, probes) for key, value in clouds.items()}
    rows = []
    for i, frame in enumerate(frames):
        rows.append({'id': frame['id'], 'frame_number': frame['frame_number'],
                     'scores': {key: symmetric_nn_distance(value[i], references[i]) for key, value in clouds.items()},
                     'shuffled_source': frames[mapping[i]]['id'],
                     'unrelated_source': other_frames[i]['id'] if other_frames is not None else None})
    transitions = []
    for i in range(len(frames)-1):
        transitions.append({'from_frame': frames[i]['frame_number'], 'to_frame': frames[i+1]['frame_number'],
                            'reference_change': temporal['correct'][i]['reference_change'],
                            'methods': {key: temporal[key][i] for key in clouds}})
    means = {key: float(np.mean([r['scores'][key] for r in rows])) for key in clouds}
    temporal_means = {key: float(np.mean([t['change_field_mae'] for t in temporal[key]])) for key in clouds}
    return {'recording': frames[0]['id'].rsplit('/', 1)[0], 'sample_count': len(frames),
            'frames': rows, 'mean_distance': means, 'mean_change_field_mae': temporal_means,
            'transitions': transitions, 'probe_count': len(probes), 'shuffle_seed': seed,
            'wins_against_control': {key: sum(r['scores']['correct'] < r['scores'][key] for r in rows)
                                     for key in clouds if key != 'correct'}}


def summarize(records):
    if not records:
        raise ValueError('No evaluated recordings.')
    # Recordings, rather than correlated frames, receive equal aggregate weight.
    methods = [key for key in METHODS if all(key in r['mean_distance'] for r in records)]
    means = {key: float(np.mean([r['mean_distance'][key] for r in records])) for key in methods}
    temporal = {key: float(np.mean([r['mean_change_field_mae'][key] for r in records])) for key in methods}
    return {'recordings': len(records), 'frames': sum(r['sample_count'] for r in records),
            'environments': len({r['recording'].split('/')[0] for r in records}),
            'available_methods': methods, 'missing_methods': [key for key in METHODS if key not in methods],
            'mean_distance': means, 'mean_change_field_mae': temporal,
            'records_beating_all_controls': sum(all(r['mean_distance']['correct'] < r['mean_distance'][k]
                for k in METHODS if k != 'correct') for r in records) if len(methods) == len(METHODS) else 0,
            'records_beating_fixed_temporal': sum(r['mean_change_field_mae']['correct'] <
                                                  r['mean_change_field_mae']['fixed'] for r in records),
            'correct_beats_all_controls': len(methods) == len(METHODS) and all(
                means['correct'] < means[k] for k in METHODS if k != 'correct')}


def read_evaluation(path):
    path = Path(path)
    if path.stat().st_size > 5_000_000:
        raise ValueError('Evaluation index exceeds the display budget.')
    report = json.loads(path.read_text())
    if (report.get('schema_version') != 1 or report.get('source_kind') != 'recorded_wifi_csi_evaluation'
            or report.get('coordinate_units') != 'standardized_model_coordinates'
            or report.get('training_overlap', {}).get('status') != 'unverified'):
        raise ValueError('Unsupported evaluation provenance or coordinate convention.')
    records = report['records']
    if not 1 <= len(records) <= 32 or len({r['recording'] for r in records}) != len(records):
        raise ValueError('Invalid recording list.')
    clouds = {}
    for record in records:
        filename = record['cloud_file']
        if not re.fullmatch(r'E0[1-4]-S\d{2}-A\d{2}\.json', filename):
            raise ValueError('Invalid cloud file path.')
        cloud_path = path.parent/filename
        if sha256(cloud_path.read_bytes()).hexdigest() != record['cloud_sha256']:
            raise ValueError('Cloud report integrity check failed.')
        cloud = read_reconstruction(cloud_path)
        if cloud['recording'] != record['recording'] or len(cloud['frames']) != record['sample_count']:
            raise ValueError('Recording identity or frame count mismatch.')
        for metric in ('mean_distance', 'mean_change_field_mae'):
            if (set(record[metric]) not in (set(METHODS), set(METHODS)-{'unrelated'}) or not all(
                    np.isfinite(v) and v >= 0 for v in record[metric].values())):
                raise ValueError('Invalid evaluation metric.')
        if [f['id'] for f in record['frames']] != [f['id'] for f in cloud['frames']]:
            raise ValueError('Frame identities mismatch.')
        clouds[record['recording']] = cloud
    all_frames = {f['id']: f for cloud in clouds.values() for f in cloud['frames']}
    for record in records:
        for row in record['frames']:
            if row['shuffled_source'] not in all_frames or (row['unrelated_source'] is not None and row['unrelated_source'] not in all_frames):
                raise ValueError('Unknown control source.')
            if (row['id'] == row['shuffled_source'] or
                row['id'].rsplit('/', 1)[0] != row['shuffled_source'].rsplit('/', 1)[0] or
                (row['unrelated_source'] is not None and row['id'].split('/')[0] == row['unrelated_source'].split('/')[0])):
                raise ValueError('Invalid control pairing.')
    if report['summary'] != summarize(records):
        raise ValueError('Evaluation aggregate disagrees with recording scores.')
    return report, clouds
