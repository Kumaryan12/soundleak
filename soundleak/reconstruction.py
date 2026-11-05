"""Recorded CSI preprocessing, point-cloud comparison and portable exports."""
import io
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree


def prepare_csi(raw):
    """Match the upstream amplitude/phase preprocessing before Torch z-scoring.

    Input consists of 3 antennas × 114 subcarriers × 10 time slices for each of
    amplitude and phase. No reference geometry is accepted by this function.
    """
    if len(raw) > 4_000_000:
        raise ValueError('CSI frame exceeds the 4 MB budget.')
    values = loadmat(io.BytesIO(raw))
    arrays = []
    for key in ('CSIamp', 'CSIphase'):
        if key not in values:
            raise ValueError('CSI frame requires CSIamp and CSIphase arrays.')
        value = np.asarray(values[key])
        if value.shape != (3, 114, 10) or value.dtype.kind not in 'fiu':
            raise ValueError('Expected real numeric CSI arrays shaped (3, 114, 10).')
        value = value.astype(np.float64, copy=True)
        value[~np.isfinite(value)] = np.nan
        for i in range(10):
            section = value[:, :, i]
            valid = np.isfinite(section)
            if not valid.any():
                raise ValueError('CSI time slice has no finite measurements.')
            section[~valid] = section[valid].mean()
        arrays.append(value)
    amplitude, phase = arrays
    span = np.ptp(amplitude)
    if not np.isfinite(span) or span <= 0:
        raise ValueError('CSI amplitude has no dynamic range.')
    amplitude = (amplitude - amplitude.min()) / span
    return np.stack((amplitude, phase), axis=2).astype(np.float32)


def measured_reference(raw, seed=0, max_points=1200):
    """Keep actual measured LiDAR points only; never fabricate padding points."""
    if len(raw) == 0 or len(raw) % 24 or len(raw) > 4_000_000:
        raise ValueError('Reference must contain bounded float64 XYZ triples.')
    points = np.frombuffer(raw, dtype='<f8').reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError('Reference contains nonfinite coordinates.')
    points = points[points[:, 0] >= 0]
    if len(points) < 3:
        raise ValueError('Reference has too few measured forward-facing points.')
    if len(points) > max_points:
        indices = np.random.default_rng(seed).choice(len(points), max_points, replace=False)
        points = points[np.sort(indices)]
    return points.copy()


def standardize_reference(points):
    """Reference-only comparison coordinates; never fed into model inference."""
    points = np.asarray(points, dtype=np.float32)
    scale = float(points.std(ddof=1))
    center = float(points.mean())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Reference has no coordinate variation.')
    return (points - center) / scale, {'scalar_mean': center, 'scalar_std': scale}


def validate_cloud(points):
    value = np.asarray(points, dtype=float)
    if value.ndim != 2 or value.shape[1] != 3 or not 3 <= len(value) <= 20_000:
        raise ValueError('Expected 3–20,000 XYZ points.')
    if not np.isfinite(value).all():
        raise ValueError('Point cloud contains nonfinite coordinates.')
    return value


def symmetric_nn_distance(predicted, reference):
    """Average bidirectional Euclidean nearest-neighbor distance, without ICP."""
    predicted, reference = validate_cloud(predicted), validate_cloud(reference)
    p_to_r = cKDTree(reference).query(predicted)[0]
    r_to_p = cKDTree(predicted).query(reference)[0]
    return float((p_to_r.mean() + r_to_p.mean()) / 2)


def ply_bytes(points, label='Wi-Fi prediction; standardized coordinates, not meters'):
    points = validate_cloud(points)
    label = str(label).replace('\n', ' ').replace('\r', ' ')[:200]
    header = ('ply\nformat ascii 1.0\ncomment ' + label + '\nelement vertex ' + str(len(points))
              + '\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
    return (header + '\n'.join(' '.join(f'{v:.7f}' for v in row) for row in points) + '\n').encode()


def read_reconstruction(path):
    path = Path(path)
    if path.stat().st_size > 10_000_000:
        raise ValueError('Reconstruction report exceeds the display budget.')
    report = json.loads(path.read_text())
    if report.get('schema_version') != 1 or report.get('source_kind') != 'recorded_wifi_csi':
        raise ValueError('Unsupported reconstruction or missing recorded-data provenance.')
    if report.get('coordinate_units') != 'standardized_model_coordinates':
        raise ValueError('Unrecognized coordinate convention.')
    if not 1 <= len(report['frames']) <= 30:
        raise ValueError('Unsupported number of recorded frames.')
    for frame in report['frames']:
        validate_cloud(frame['predicted'])
        validate_cloud(frame['reference'])
    validate_cloud(report['zero_input_control'])
    return report
