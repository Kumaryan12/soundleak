import io
import json

import numpy as np
import pytest
from scipy.io import savemat

from soundleak.reconstruction import (prepare_csi, measured_reference, standardize_reference,
                                      symmetric_nn_distance, ply_bytes, read_reconstruction)


def mat_bytes(amplitude, phase):
    stream = io.BytesIO()
    savemat(stream, {'CSIamp': amplitude, 'CSIphase': phase})
    return stream.getvalue()


def test_csi_preprocessing_preserves_phase_and_normalizes_amplitude_without_geometry():
    amplitude = np.arange(3420, dtype=float).reshape(3, 114, 10)
    phase = np.full_like(amplitude, -2.)
    value = prepare_csi(mat_bytes(amplitude, phase))
    assert value.shape == (3, 114, 2, 10) and value.dtype == np.float32
    np.testing.assert_allclose(value[:, :, 0], amplitude / amplitude.max(), atol=1e-7)
    np.testing.assert_array_equal(value[:, :, 1], phase)


def test_missing_values_use_the_same_time_slice_only():
    amplitude = np.arange(3420, dtype=float).reshape(3, 114, 10)
    phase = np.zeros_like(amplitude)
    phase[:, :, 0] = 4
    phase[0, 0, 0] = np.nan
    value = prepare_csi(mat_bytes(amplitude, phase))
    assert value[0, 0, 1, 0] == 4
    np.testing.assert_array_equal(value[:, :, 1, 1:], 0)


@pytest.mark.parametrize('amplitude,phase', [
    (np.zeros((3, 114, 10)), np.zeros((3, 114, 10))),
    (np.ones((3, 114, 9)), np.ones((3, 114, 9))),
    (np.full((3, 114, 10), np.nan), np.zeros((3, 114, 10))),
])
def test_invalid_csi_cannot_generate_a_plausible_cloud(amplitude, phase):
    with pytest.raises(ValueError):
        prepare_csi(mat_bytes(amplitude, phase))


def test_measured_reference_does_not_invent_padding_points():
    points = np.array([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.], [-1., 0., 0.]], dtype='<f8')
    reference = measured_reference(points.tobytes())
    np.testing.assert_array_equal(reference, points[:3])
    normalized, info = standardize_reference(reference)
    np.testing.assert_allclose(normalized * info['scalar_std'] + info['scalar_mean'], reference, atol=1e-6)


def test_nearest_point_metric_is_order_invariant_and_never_aligns_away_translation():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    assert symmetric_nn_distance(points, points[::-1]) == 0
    assert symmetric_nn_distance(points, points + [0, 0, 2]) == 2
    ply = ply_bytes(points).decode()
    assert 'element vertex 3' in ply and 'standardized' in ply and 'not meters' in ply
    with pytest.raises(ValueError, match='nonfinite'):
        ply_bytes([[0, 0, 0], [1, 0, 0], [np.nan, 0, 0]])


def test_replay_rejects_missing_provenance_and_unrecognized_units(tmp_path):
    path = tmp_path/'report.json'
    path.write_text(json.dumps({'schema_version': 1, 'source_kind': 'synthetic'}))
    with pytest.raises(ValueError, match='provenance'):
        read_reconstruction(path)
    path.write_text(json.dumps({'schema_version': 1, 'source_kind': 'recorded_wifi_csi', 'coordinate_units': 'meters'}))
    with pytest.raises(ValueError, match='coordinate'):
        read_reconstruction(path)
