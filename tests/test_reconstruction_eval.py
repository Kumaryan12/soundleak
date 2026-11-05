import copy
from hashlib import sha256
import json

import numpy as np
import pytest

from soundleak.reconstruction_eval import (METHODS, derangement, evaluate_recording,
    point_errors, probe_grid, read_evaluation, summarize, temporal_scores)


def cloud(offset=0):
    return (np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., .2], [.3, .5, 1.]])+[offset, 0, 0]).tolist()


def frames(recording, offsets):
    return [{'id': f'{recording}/frame{i+1:03d}', 'frame_number': i+1,
             'predicted': cloud(offset), 'reference': cloud(offset)} for i, offset in enumerate(offsets)]


def test_shuffle_has_no_correct_pairs_and_preserves_every_input_once():
    for count in (2, 3, 12):
        mapping = derangement(count)
        assert sorted(mapping) == list(range(count))
        assert all(i != j for i, j in enumerate(mapping))
        assert mapping == derangement(count)
    with pytest.raises(ValueError):
        derangement(1)


def test_perfect_changing_predictions_beat_wrong_inputs_and_fixed_geometry():
    own = frames('E01/S05/A05', [0, .4, 1.1])
    other = frames('E02/S15/A05', [3, 4, 5])
    result = evaluate_recording(own, cloud(8), other)
    assert result['mean_distance']['correct'] == 0
    assert result['mean_change_field_mae']['correct'] == 0
    for method in METHODS:
        if method != 'correct':
            assert result['mean_distance'][method] > 0
    assert result['mean_change_field_mae']['fixed'] > 0
    assert [r['unrelated_source'] for r in result['frames']] == [f['id'] for f in other]
    assert all(r['id'] != r['shuffled_source'] for r in result['frames'])
    assert all(t['methods']['fixed']['predicted_change'] == 0 for t in result['transitions'])


def test_different_environment_control_rejects_same_environment_even_different_subject():
    with pytest.raises(ValueError, match='another environment'):
        evaluate_recording(frames('E01/S05/A05', [0, 1]), cloud(), frames('E01/S01/A27', [2, 3]))


def test_temporal_metric_retains_signed_field_change_and_is_point_order_invariant():
    reference = [cloud(), cloud(.7)]
    probes = probe_grid(reference+[cloud(-.7)])
    exact = temporal_scores([cloud()[::-1], cloud(.7)[::-1]], reference, probes)[0]
    wrong_direction = temporal_scores([cloud(), cloud(-.7)], reference, probes)[0]
    assert exact['change_field_mae'] == 0
    assert wrong_direction['change_field_mae'] > .1
    fixed = temporal_scores([cloud(), cloud()], reference, probes)[0]
    assert fixed['predicted_change'] == 0
    assert fixed['change_field_mae'] == pytest.approx(fixed['reference_change'])


def test_surface_errors_include_reference_surfaces_missing_from_prediction():
    pred = cloud()
    reference = pred+[[9, 9, 9]]
    forward, reverse = point_errors(pred, reference)
    assert np.max(forward) == 0
    assert reverse[-1] > 10


def test_summary_weights_recordings_equally_not_correlated_frame_counts():
    one = {'recording': 'E01/S05/A05', 'sample_count': 2,
           'mean_distance': {k: 1. for k in METHODS}, 'mean_change_field_mae': {k: 2. for k in METHODS}}
    two = {'recording': 'E02/S15/A05', 'sample_count': 20,
           'mean_distance': {k: 3. for k in METHODS}, 'mean_change_field_mae': {k: 4. for k in METHODS}}
    result = summarize([one, two])
    assert result['mean_distance']['correct'] == 2
    assert result['mean_change_field_mae']['correct'] == 3
    assert result['frames'] == 22 and not result['correct_beats_all_controls']


def evaluation_fixture(folder):
    own, other = frames('E01/S05/A05', [0, .3]), frames('E02/S15/A05', [2, 3])
    records = []
    for a, b in ((own, other), (other, own)):
        record = evaluate_recording(a, cloud(5), b)
        filename = record['recording'].replace('/', '-')+'.json'
        cloud_report = {'schema_version': 1, 'source_kind': 'recorded_wifi_csi',
                        'coordinate_units': 'standardized_model_coordinates', 'recording': record['recording'],
                        'frames': a, 'zero_input_control': cloud(5)}
        raw = json.dumps(cloud_report).encode()
        (folder/filename).write_bytes(raw)
        record.update(cloud_file=filename, cloud_sha256=sha256(raw).hexdigest())
        records.append(record)
    report = {'schema_version': 1, 'source_kind': 'recorded_wifi_csi_evaluation',
              'coordinate_units': 'standardized_model_coordinates', 'training_overlap': {'status': 'unverified'},
              'records': records, 'summary': summarize(records)}
    path = folder/'evaluation.json'
    path.write_text(json.dumps(report))
    return path, report


def test_evaluation_loader_checks_hashes_control_identity_aggregate_and_overlap(tmp_path):
    path, report = evaluation_fixture(tmp_path)
    loaded, clouds = read_evaluation(path)
    assert len(clouds) == 2 and loaded['summary']['frames'] == 4
    changed = copy.deepcopy(report)
    changed['records'][0]['cloud_file'] = '../secret.json'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='path'):
        read_evaluation(path)
    changed = copy.deepcopy(report)
    changed['records'][0]['frames'][0]['shuffled_source'] = changed['records'][0]['frames'][0]['id']
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='pairing'):
        read_evaluation(path)
    changed = copy.deepcopy(report)
    changed['summary']['frames'] = 99
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='aggregate'):
        read_evaluation(path)
    changed = copy.deepcopy(report)
    changed['training_overlap']['status'] = 'held out'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='provenance'):
        read_evaluation(path)
    path.write_text(json.dumps(report))
    (tmp_path/report['records'][0]['cloud_file']).write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        read_evaluation(path)


def test_missing_different_environment_control_stays_unavailable_not_zero_or_success():
    result = evaluate_recording(frames('E01/S05/A05', [0, 1]), cloud(5))
    assert 'unrelated' not in result['mean_distance']
    assert all(row['unrelated_source'] is None for row in result['frames'])
    summary = summarize([result])
    assert summary['missing_methods'] == ['unrelated']
    assert not summary['correct_beats_all_controls']
    assert summary['records_beating_all_controls'] == 0
