"""Deterministic synthetic data; no real recording or flight data implied."""
from dataclasses import dataclass
import io
import csv
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.io import wavfile

from .physics import Track, doppler_factor, latlon


@dataclass
class Demo:
    audio: np.ndarray
    sample_rate: int
    tracks: dict
    windows: list
    truth_m: tuple
    origin: tuple


def make_demo(seed=7):
    rng = np.random.default_rng(seed)
    sr, duration = 4096, 160
    times = np.arange(duration * sr) / sr
    emission = np.arange(-120., 281., 2.)
    tracks = {
        "SYNTH-A": Track("SYNTH-A", emission, np.column_stack((95 * (emission - 47), np.full_like(emission, -900), np.full_like(emission, 1300)))),
        "SYNTH-B": Track("SYNTH-B", emission, np.column_stack((np.full_like(emission, 1600), 82 * (emission - 104), np.full_like(emission, 1700)))),
    }
    truth = (650., -450., 40.)
    audio = rng.normal(0, .16, len(times)) + .035 * np.sin(2 * np.pi * 60 * times)
    windows = [("SYNTH-A", 15., 86., 92., 172.), ("SYNTH-B", 77., 151., 170., 305.)]
    for (name, start, end, _, _), f0 in zip(windows, (125., 230.)):
        q = doppler_factor(tracks[name], truth, times)
        frequency = f0 * q
        phase = 2 * np.pi * cumulative_trapezoid(frequency, times, initial=0)
        envelope = np.minimum(np.clip((times - start + 8) / 10, 0, 1), np.clip((end + 8 - times) / 10, 0, 1))
        audio += .11 * envelope * np.sin(phase)
    audio = (audio / max(1., np.max(np.abs(audio)) * 1.01)).astype(np.float32)
    return Demo(audio, sr, tracks, windows, truth, (50.85, -1.4))


def wav_bytes(demo):
    buffer = io.BytesIO()
    wavfile.write(buffer, demo.sample_rate, demo.audio)
    return buffer.getvalue()


def tracks_csv(demo):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["aircraft_id", "time_s", "latitude", "longitude", "altitude_m"])
    for name, track in demo.tracks.items():
        lat, lon = latlon(track.xyz_m[:, 0], track.xyz_m[:, 1], *demo.origin)
        for t, la, lo, z in zip(track.time_s, lat, lon, track.xyz_m[:, 2]):
            writer.writerow([name, t, la, lo, z])
    return buffer.getvalue()

