"""Recorded playback with cumulative Doppler fits and an explicit shared clock.

The playback reveals previously analyzed complete proposals. It is an offline
replay, not a causal streaming detector or a fresh independent location test.
"""
import io
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import spectrogram
from matplotlib import colormaps, image

from .association import Hypothesis


def doppler_grid(track, receivers, receive_times, sound_speed=343.):
    """Vectorized counterpart of physics.doppler_factor, without extrapolation."""
    receivers = np.asarray(receivers, dtype=float)
    times = np.asarray(receive_times, dtype=float)
    if receivers.ndim != 2 or receivers.shape[1] != 3 or not np.isfinite(receivers).all():
        raise ValueError('Receivers must be finite XYZ rows.')
    velocity = track.velocity_mps
    if not np.isfinite(sound_speed) or sound_speed <= 0 or np.any(np.linalg.norm(velocity, axis=1) >= sound_speed):
        raise ValueError('Doppler replay requires a positive sound speed and subsonic tracks.')
    distance = np.linalg.norm(track.xyz_m[None, :, :]-receivers[:, None, :], axis=2)
    arrivals = track.time_s[None, :]+distance/sound_speed
    idx = np.stack([np.searchsorted(row, times, side='right')-1 for row in arrivals])
    idx = np.clip(idx, 0, len(velocity)-1)
    r0 = track.xyz_m[idx]-receivers[:, None, :]
    v = velocity[idx]
    delta = times[None, :]-track.time_s[idx]
    a = sound_speed**2-np.sum(v*v, axis=2)
    b = sound_speed**2*delta+np.sum(r0*v, axis=2)
    c = sound_speed**2*delta**2-np.sum(r0*r0, axis=2)
    discriminant = np.sqrt(np.maximum(b*b-a*c, 0.))
    u = (b-discriminant)/a
    np.divide(c, b+discriminant, out=u, where=(b >= 0) & (b+discriminant > 0))
    u = np.clip(u, 0, track.time_s[idx+1]-track.time_s[idx])
    displacement = r0+v*u[:, :, None]
    radial = np.sum(v*displacement, axis=2)/np.maximum(np.linalg.norm(displacement, axis=2), 1e-9)
    factor = sound_speed/(sound_speed+radial)
    factor[(times[None, :] < arrivals[:, :1]) | (times[None, :] > arrivals[:, -1:]) | ~np.isfinite(times[None, :])] = np.nan
    return factor


def _eligible(proposal, tracks, east, north, offsets, altitude, speed):
    keys = []
    for key, track in tracks.items():
        if np.any(np.linalg.norm(track.velocity_mps, axis=1) >= speed):
            raise ValueError('Replay requires subsonic tracks.')
        first, last = track.xyz_m[0], track.xyz_m[-1]
        closest = (np.clip(first[0], east.min(), east.max()), np.clip(first[1], north.min(), north.max()), altitude)
        first_delay = np.linalg.norm(first-np.asarray(closest))/speed
        last_delay = max(np.linalg.norm(last-np.array((x, y, altitude))) for x in (east.min(), east.max()) for y in (north.min(), north.max()))/speed
        if track.time_s[0]+first_delay <= proposal.ridge.time_s[0]+offsets.max() and track.time_s[-1]+last_delay >= proposal.ridge.time_s[-1]+offsets.min():
            keys.append(key)
    return keys


def _rank_prefix(proposals, keys, east, north, offsets, losses, mse, winner, tones, origin, threshold_hz):
    objective = np.sqrt(np.mean(losses, axis=0))
    clocks = np.argmin(objective, axis=1)
    surface = objective[np.arange(len(clocks)), clocks].reshape(len(north), len(east))
    candidates = np.argsort(surface.ravel(), kind='stable')
    spacing = max(np.ptp(east)/max(1, len(east)-1), np.ptp(north)/max(1, len(north)-1))
    retained = []
    for cell in candidates:
        iy, ix = divmod(int(cell), len(east))
        oi = clocks[cell]
        matches, unmatched, unavailable = [], [], []
        for pi, proposal in enumerate(proposals):
            variance = max(float(np.var(proposal.ridge.frequency_hz[::4])), 1e-12)
            explained = 1-mse[pi, cell, oi]/variance
            if winner[pi, cell, oi] >= 0 and mse[pi, cell, oi] <= threshold_hz**2 and explained >= .6:
                matches.append({'event_id': proposal.id, 'candidate': keys[int(winner[pi, cell, oi])],
                                'source_tone_hz': float(tones[pi, cell, oi]), 'rmse_hz': float(np.sqrt(mse[pi, cell, oi])),
                                'trend_explained_fraction': float(explained)})
            else:
                unmatched.append(proposal.id)
                if winner[pi, cell, oi] < 0:
                    unavailable.append(proposal.id)
        if not matches:
            continue
        h = Hypothesis(float(east[ix]), float(north[iy]), float(offsets[oi]), float(surface[iy, ix]), matches, unmatched, unavailable)
        if all(np.hypot(h.east_m-r.east_m, h.north_m-r.north_m) >= max(500., spacing*1.5) for r in retained):
            retained.append(h)
        if len(retained) == 8:
            break
    reasons = []
    status = 'candidate_fit'
    if not retained or len({m['candidate'] for m in retained[0].matches}) < 2:
        status = 'insufficient_evidence'
        reasons.append('Fewer than two distinct flight segments support these observations under one shared clock.')
    else:
        best = retained[0]
        if best.unmatched:
            status = 'partial_fit'
            reasons.append(f'{len(best.unmatched)} acoustic groups remain unmatched.')
        if best.east_m in (east[0], east[-1]) or best.north_m in (north[0], north[-1]):
            status = 'ambiguous'; reasons.append('The best point lies on the spatial search boundary.')
        if len(offsets) > 1 and best.clock_offset_s in (offsets[0], offsets[-1]):
            status = 'ambiguous'; reasons.append('The best clock lies on the searched time boundary.')
        if len(retained) > 1 and retained[1].objective_hz-best.objective_hz < .1:
            status = 'ambiguous'; reasons.append('A separated location has a similar score.')
    return {'status': status, 'reasons': reasons, 'surface_hz': surface.round(7).tolist(),
            'hypotheses': [h.metadata(origin) for h in retained]}


def cumulative_replay(proposals, tracks, east_m, north_m, clock_offsets_s=(0.,), receiver_altitude_m=0.,
                      sound_speed=343., threshold_hz=3., origin=(0., 0.), max_models=25_000_000, progress=None):
    """Evaluate each proposal once, then accumulate event costs before profiling clocks."""
    started = time.monotonic()
    east, north, offsets = (np.asarray(a, dtype=float) for a in (east_m, north_m, clock_offsets_s))
    if (not proposals or not tracks or any(a.ndim != 1 or not len(a) or not np.isfinite(a).all() for a in (east, north, offsets))
        or min(len(east), len(north)) < 2 or not np.isfinite(receiver_altitude_m)
        or np.any(np.diff(east) <= 0) or np.any(np.diff(north) <= 0)):
        raise ValueError('Need proposals, flights and finite increasing spatial axes.')
    if max(len(east), len(north)) > 31 or len(offsets) > 21 or not 0 < threshold_hz <= 20 or not 300 <= sound_speed <= 370:
        raise ValueError('Invalid replay grid, clock count or model assumptions.')
    proposals = sorted(proposals, key=lambda p: (float(p.ridge.time_s[-1]), p.id))
    if len({p.id for p in proposals}) != len(proposals):
        raise ValueError('Replay requires unique acoustic-group identifiers.')
    keys = list(tracks)
    key_index = {key: i for i, key in enumerate(keys)}
    xx, yy = np.meshgrid(east, north)
    receivers = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, receiver_altitude_m)))
    eligible = [_eligible(p, tracks, east, north, offsets, receiver_altitude_m, sound_speed) for p in proposals]
    models = len(receivers)*len(offsets)*sum(map(len, eligible))
    if models > max_models:
        raise ValueError(f'Replay needs {models:,} models; budget is {max_models:,}.')
    shape = (len(proposals), len(receivers), len(offsets))
    mse = np.full(shape, np.inf)
    tones = np.zeros(shape)
    winner = np.full(shape, -1, dtype=np.int32)
    for pi, proposal in enumerate(proposals):
        observed = proposal.ridge.frequency_hz[::4]
        receive = proposal.ridge.time_s[::4][None, :]+offsets[:, None]
        for key in eligible[pi]:
            for start in range(0, len(receivers), 32):
                stop = min(start+32, len(receivers))
                q = doppler_grid(tracks[key], receivers[start:stop], receive.ravel(), sound_speed).reshape(stop-start, len(offsets), -1)
                valid = np.isfinite(q).all(axis=2)
                f0 = np.sum(q*observed, axis=2)/np.sum(q*q, axis=2)
                errors = np.mean((q*f0[:, :, None]-observed)**2, axis=2)
                better = valid & (errors < mse[pi, start:stop])
                mse[pi, start:stop][better] = errors[better]
                tones[pi, start:stop][better] = f0[better]
                winner[pi, start:stop][better] = key_index[key]
        if progress:
            progress((pi+1)/len(proposals))
    losses = np.full(shape, threshold_hz**2)
    for pi, p in enumerate(proposals):
        explained = 1-mse[pi]/max(float(np.var(p.ridge.frequency_hz[::4])), 1e-12)
        accepted = np.isfinite(mse[pi]) & (mse[pi] <= threshold_hz**2) & (explained >= .6)
        losses[pi, accepted] = mse[pi, accepted]
    states = [{'at_s': 0., 'seen_ids': [], 'status': 'waiting', 'reasons': ['Waiting for a complete acoustic proposal.'],
               'surface_hz': None, 'hypotheses': []}]
    for count in range(1, len(proposals)+1):
        at = float(proposals[count-1].ridge.time_s[-1])
        if count < len(proposals) and proposals[count].ridge.time_s[-1] == at:
            continue
        state = _rank_prefix(proposals[:count], keys, east, north, offsets, losses[:count], mse[:count], winner[:count], tones[:count], origin, threshold_hz)
        states.append({'at_s': at, 'seen_ids': [p.id for p in proposals[:count]], **state})
    return {'states': states, 'east_m': east.tolist(), 'north_m': north.tolist(), 'clock_offsets_s': offsets.tolist(),
            'evaluated_models': models, 'elapsed_s': time.monotonic()-started,
            'settings': {'threshold_hz': threshold_hz, 'receiver_altitude_m': receiver_altitude_m, 'sound_speed_mps': sound_speed,
                         'scoring_frame_stride': 4, 'grid_spacing_m': float(east[1]-east[0]) if len(east)>1 else None,
                         'common_clock': True, 'objective': 'Equal-weight capped event MSE, accumulated before profiling one shared clock.',
                         'replay_kind': 'Offline analyzed observations revealed at their last measured sample; not causal streaming detection.'}}


def display_assets(audio, sample_rate):
    """Bounded, downsampled spectrogram for client-side playback only."""
    f, t, power = spectrogram(audio, sample_rate, nperseg=2048, noverlap=0, mode='psd')
    keep = (f >= 40) & (f <= 800)
    power = power[keep]
    stride = max(1, int(np.ceil(len(t)/1400)))
    columns = int(np.ceil(power.shape[1]/stride))
    padded = np.pad(power, ((0, 0), (0, columns*stride-power.shape[1])), constant_values=0)
    pooled = padded.reshape(len(power), columns, stride).max(axis=2)
    db = 10*np.log10(np.maximum(pooled, 1e-20))
    low, high = np.percentile(db, (10, 99.7))
    pixels = colormaps['magma'](np.clip((db-low)/max(high-low, 1), 0, 1))
    output = io.BytesIO()
    image.imsave(output, pixels, format='png', origin='lower')
    import base64
    seconds = len(audio)//sample_rate
    rms = np.sqrt(np.mean(audio[:seconds*sample_rate].reshape(seconds, sample_rate)**2, axis=1)) if seconds else np.array([0.])
    return {'spectrogram_png': 'data:image/png;base64,'+base64.b64encode(output.getvalue()).decode(),
            'spectrogram_frequency_hz': [float(f[keep][0]), float(f[keep][-1])],
            'rms_db': (20*np.log10(np.maximum(rms, 1e-10))).round(2).tolist()}


def replay_payload(audio, sample_rate, discovery, tracks, origin, replay):
    proposals = [{'id': p.id, 'start_s': float(p.ridge.time_s[0]), 'end_s': float(p.ridge.time_s[-1]),
                  'time_s': p.ridge.time_s.round(3).tolist(), 'frequency_hz': p.ridge.frequency_hz.round(3).tolist()}
                 for p in discovery.proposals]
    # Geometry selection is independent of acoustic fit quality. Keep every track
    # with any reported point near the declared displayed search region.
    margin = max(np.max(np.abs(replay['east_m'])), np.max(np.abs(replay['north_m'])))+5000
    paths = []
    for name, track in tracks.items():
        if not np.any((np.abs(track.xyz_m[:, 0]) <= margin) & (np.abs(track.xyz_m[:, 1]) <= margin)):
            continue
        indices = np.unique(np.r_[np.arange(0, len(track.time_s), max(1, len(track.time_s)//300)), len(track.time_s)-1])
        paths.append({'name': name, 'time_s': track.time_s[indices].round(3).tolist(), 'xyz_m': track.xyz_m[indices].round(2).tolist()})
    return {'schema_version': 1, 'duration_s': len(audio)/sample_rate, 'origin': list(origin),
            'proposals': proposals, 'tracks': paths, 'candidate_segments': len(tracks),
            **replay, **display_assets(audio, sample_rate)}


def prepare_case_replay(case_directory, output, grid_points=11, clock_count=5, progress=None):
    from .archive import file_sha256
    from .cases import load_case
    case_directory, output = Path(case_directory), Path(output)
    if not 2 <= grid_points <= 31 or not 1 <= clock_count <= 21:
        raise ValueError('Use 2–31 spatial points and 1–21 clock candidates.')
    if output.exists():
        raise ValueError('Replay output already exists. Choose a different output directory.')
    audio, sr, discovery, tracks, origin, provenance, review, result = load_case(case_directory)
    if result is None:
        raise ValueError('A saved flight-comparison case is required for recorded replay.')
    settings = result.settings
    selected = set(review.get('selected_event_ids', [p.id for p in discovery.proposals]))
    east = np.linspace(settings['east_m'][0], settings['east_m'][-1], grid_points)
    north = np.linspace(settings['north_m'][0], settings['north_m'][-1], grid_points)
    offsets = (np.linspace(settings['clock_offsets_s'][0], settings['clock_offsets_s'][-1], clock_count)
               if clock_count > 1 else np.array([np.median(settings['clock_offsets_s'])]))
    replay = cumulative_replay([p for p in discovery.proposals if p.id in selected], tracks, east, north, offsets,
                               settings['receiver_altitude_m'], settings['sound_speed_mps'], settings['max_event_rmse_hz'],
                               origin, progress=progress)
    payload = replay_payload(audio, sr, discovery, tracks, origin, replay)
    payload.update({'name': 'Oyster Bay · regional recording', 'source_kind': 'real_recording',
                    'source_case': str(case_directory), 'source_report_sha256': file_sha256(case_directory/'report.json'),
                    'source_audio_npz_sha256': file_sha256(case_directory/'audio.npz'),
                    'provenance': provenance, 'recording_start_unix_s': provenance.get('recording_start_unix_s'),
                    'scope': 'Recorded offline replay. Source-informed regional bounds and clock. Coarser replay grid; no blind location accuracy claim.'})
    output.mkdir(parents=True, exist_ok=False)
    (output/'replay.json').write_text(json.dumps(payload, allow_nan=False))
    return payload
