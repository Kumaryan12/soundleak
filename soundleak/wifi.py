"""RSSI diagnostics for a stationary Wi-Fi link; no human-presence inference."""
import csv
from dataclasses import dataclass
import io

import numpy as np

MAX_BYTES = 2_000_000
MAX_ROWS = 10_000


@dataclass
class WifiCapture:
    time_s: np.ndarray
    rssi_dbm: np.ndarray
    noise_dbm: np.ndarray
    channel: np.ndarray
    declared_source: str


def read_wifi_csv(raw):
    if isinstance(raw, str):
        raw = raw.encode()
    if len(raw) > MAX_BYTES:
        raise ValueError('Wi-Fi captures are limited to 2 MB.')
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('Upload a UTF-8 CSV from the Wi-Fi capture tool.') from exc
    source = 'unverified-upload'
    lines = []
    for line in text.splitlines():
        if line.startswith('# source='):
            source = line.split('=', 1)[1].strip()[:80]
        elif line.strip() and not line.startswith('#'):
            lines.append(line)
    reader = csv.DictReader(io.StringIO('\n'.join(lines)))
    if not {'time_s', 'rssi_dbm'}.issubset(reader.fieldnames or []):
        raise ValueError('CSV needs time_s and rssi_dbm columns.')
    rows = []
    try:
        for row in reader:
            if len(rows) >= MAX_ROWS:
                raise ValueError('Wi-Fi captures are limited to 10,000 samples.')
            if None in row:
                raise ValueError('CSV contains an unexpected number of fields.')
            timestamp = float(row['time_s'])
            status = row.get('status', 'connected')
            if status not in ('connected', 'unavailable'):
                raise ValueError('Unknown Wi-Fi sample status.')
            rssi = float(row['rssi_dbm']) if row['rssi_dbm'].strip() else np.nan
            if status == 'unavailable':
                rssi = np.nan
            elif not np.isfinite(rssi) or not -127 <= rssi <= -1:
                raise ValueError('Connected RSSI must be between -127 and -1 dBm; use unavailable for missing data.')
            noise = float(row['noise_dbm']) if row.get('noise_dbm', '').strip() else np.nan
            if not np.isnan(noise) and (not np.isfinite(noise) or not -127 <= noise <= -1):
                raise ValueError('Noise must be between -127 and -1 dBm or blank.')
            channel = float(row['channel']) if row.get('channel', '').strip() else 0
            if not np.isfinite(channel) or channel != int(channel) or not 0 <= channel <= 233:
                raise ValueError('Invalid Wi-Fi channel.')
            rows.append((timestamp, rssi, noise, channel))
    except (TypeError, AttributeError, csv.Error) as exc:
        raise ValueError('Malformed Wi-Fi CSV.') from exc
    if len(rows) < 12:
        raise ValueError('Need at least 12 timestamped Wi-Fi samples.')
    data = np.asarray(rows, dtype=float)
    times = data[:, 0]
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('Sample times must be finite and strictly increasing; do not concatenate captures.')
    if times[-1] - times[0] > 1200:
        raise ValueError('Analyze at most 20 minutes per capture.')
    times -= times[0]
    return WifiCapture(times, data[:, 1], data[:, 2], data[:, 3], source)


def analyze_wifi(capture, baseline_s=10., window_s=3., minimum_event_s=1.):
    """Compare rolling RSSI variability with the declared static reference period.

    Uses trailing windows only. Gaps invalidate windows, and channel changes
    invalidate the capture. Exceeding a threshold is RF variation, not motion or
    person classification. The first baseline_s seconds must be a static reference.
    """
    if not 5 <= baseline_s <= 60 or not 1 <= window_s <= baseline_s - 1:
        raise ValueError('Use a 5–60 second reference and a window at least one second shorter than it.')
    if not np.isfinite(minimum_event_s) or minimum_event_s <= 0:
        raise ValueError('Minimum event duration must be positive.')
    t, rssi = capture.time_s, capture.rssi_dbm
    if t[-1] < baseline_s + window_s + minimum_event_s:
        raise ValueError('Capture must continue beyond the reference by at least one window and event duration.')
    valid = np.isfinite(rssi)
    channels = np.unique(capture.channel[valid & (capture.channel > 0)])
    if len(channels) > 1:
        raise ValueError('The Wi-Fi channel changed. Recapture on a stable link; a channel switch is not human activity.')
    step = float(np.median(np.diff(t)))
    if step > window_s / 3:
        raise ValueError('Too few samples per analysis window. Capture at least once per second.')
    gap_limit = max(1.5, 3 * step)
    bad_gap = np.r_[False, np.diff(t) > gap_limit]
    reference = t <= baseline_s
    if (np.count_nonzero(valid & reference) < 10 or valid[reference].mean() < .9
            or bad_gap[reference].any() or t[reference][-1] < baseline_s - step * 1.5):
        raise ValueError('Reference period has too few valid measurements or a data gap. Record a new static reference.')

    starts = np.searchsorted(t, t - window_s, side='left')
    ends = np.arange(len(t)) + 1
    def window_sum(values):
        cumulative = np.r_[0., np.cumsum(values)]
        return cumulative[ends] - cumulative[starts]
    count = window_sum(valid)
    values = np.where(valid, rssi, 0.)
    average = window_sum(values) / np.maximum(count, 1)
    variance = np.maximum(0, window_sum(values ** 2) / np.maximum(count, 1) - average ** 2)
    spread = np.sqrt(variance)
    usable = (valid & (count >= max(3, np.ceil(.6 * window_s / step)))
              & (t - t[starts] >= .8 * window_s)
              & (window_sum(bad_gap) == 0))
    spread[~usable] = np.nan
    calibration = spread[(t <= baseline_s) & np.isfinite(spread)]
    if len(calibration) < 3:
        raise ValueError('Too few complete reference windows to calibrate variability.')
    center = float(np.median(calibration))
    mad = float(np.median(np.abs(calibration - center)))
    threshold = center + max(2., 6 * 1.4826 * mad)
    flags = usable & (t > baseline_s) & (spread > threshold)
    events = []
    run = None
    for i in range(len(t) + 1):
        active = i < len(t) and flags[i]
        if active and run is None:
            run = i
        if not active and run is not None:
            end = min(t[i - 1] + step, t[-1])
            if end - t[run] >= minimum_event_s:
                events.append({'start_s': float(t[run]), 'end_s': float(end),
                               'peak_variability_db': float(np.max(spread[run:i]))})
            run = None
    def optional(value):
        return float(value) if np.isfinite(value) else None
    states = [('reference' if x <= baseline_s else 'no_measurement' if not good
               else 'elevated_rf_variability' if flag else 'within_reference_variability')
              for x, good, flag in zip(t, usable, flags)]
    warnings = ['RSSI changes do not establish human presence, movement, identity or position.',
                'Keep the Mac and router stationary. Traffic, radio power, interference and roaming can change RSSI.',
                'The reference must be representative. Sampling frequency may exceed the Wi-Fi driver update rate.']
    if not len(channels):
        warnings.append('Channel information is unavailable; channel changes cannot be screened.')
    if not valid.all() or bad_gap.any():
        warnings.append('Missing samples or gaps are shown as unavailable, not quiet or empty.')
    if center > 2:
        warnings.append('Reference variability is already high; this can hide subsequent changes. Inspect the reference conditions.')
    return {'schema_version': 1, 'measurement': 'rssi', 'declared_source': capture.declared_source,
            'human_presence_status': 'not_estimated', 'location_status': 'not_estimated',
            'duration_s': float(t[-1]), 'samples': len(t), 'valid_samples': int(valid.sum()),
            'reference_s': float(baseline_s), 'window_s': float(window_s),
            'reference_median_rssi_dbm': float(np.median(rssi[valid & reference])),
            'reference_variability_db': center, 'threshold_db': threshold,
            'channel': int(channels[0]) if len(channels) else None,
            'latest_status': states[-1], 'events': events, 'warnings': warnings,
            'frames': [{'time_s': float(x), 'rssi_dbm': optional(r), 'noise_dbm': optional(n),
                        'variability_db': optional(s), 'status': status}
                       for x, r, n, s, status in zip(t, rssi, capture.noise_dbm, spread, states)]}


def synthetic_wifi_csv():
    """Explicitly synthetic link variation, without human labels or accuracy claims."""
    rng = np.random.default_rng(8)
    t = np.arange(0, 60, .5)
    rssi = -48 + rng.normal(0, .35, len(t))
    event = (t >= 25) & (t < 36)
    rssi[event] += 8 * np.sin(t[event] * 4.2)
    output = io.StringIO()
    output.write('# source=synthetic-rssi\n')
    writer = csv.writer(output)
    writer.writerow(['time_s', 'rssi_dbm', 'noise_dbm', 'channel', 'status'])
    writer.writerows((f'{x:.3f}', f'{r:.3f}', '-95', '6', 'connected') for x, r in zip(t, rssi))
    return output.getvalue().encode()
