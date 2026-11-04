from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np
from scipy.signal import ShortTimeFFT, windows


@dataclass
class Ridge:
    time_s: np.ndarray
    frequency_hz: np.ndarray
    prominence_db: np.ndarray
    frequency_bins_hz: np.ndarray
    power_db: np.ndarray

    @property
    def span_hz(self):
        return float(np.percentile(self.frequency_hz, 95) - np.percentile(self.frequency_hz, 5))

    @property
    def usable(self):
        # A quality screen, not an aircraft classifier or calibrated significance test.
        return len(self.time_s) >= 20 and self.span_hz >= 3 and np.median(self.prominence_db) >= 8


def decode_audio(path: str | Path, sample_rate=4096, max_duration_s=180, offset_s=0):
    """Decode locally; never invoke a shell or pass URLs to FFmpeg."""
    path = Path(path).resolve(strict=True)
    if not np.isfinite(offset_s) or offset_s < 0 or not 4 <= max_duration_s <= 7200:
        raise ValueError("Choose a nonnegative media offset and a duration between 4 and 7200 seconds.")
    command = ["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
               "-ss", str(offset_s), "-i", str(path), "-t", str(max_duration_s), "-vn", "-ac", "1", "-ar", str(sample_rate),
               "-f", "f32le", "pipe:1"]
    try:
        result = subprocess.run(command, capture_output=True, check=True, timeout=60)
    except FileNotFoundError as exc:
        raise ValueError("FFmpeg is required for audio/video uploads. Install it and retry.") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError("Could not decode this media file: " + exc.stderr.decode(errors="replace")[-500:]) from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Audio decoding exceeded 60 seconds.") from exc
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(audio) < sample_rate * 4 or not np.isfinite(audio).all():
        raise ValueError("Need at least four seconds of finite audio samples.")
    return audio, sample_rate


def spectrogram_data(audio, sample_rate, start_s, end_s, low_hz, high_hz):
    """A bounded spectrogram window; callers can reuse it across search bands."""
    audio = np.asarray(audio)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples.")
    if not 0 <= start_s < end_s <= len(audio) / sample_rate:
        raise ValueError("Choose an event window inside the recording.")
    if end_s - start_s < 4:
        raise ValueError("Event windows must be at least four seconds.")
    if not 0 < low_hz < high_hz < sample_rate / 2:
        raise ValueError("Invalid frequency band.")
    clip = audio[int(start_s * sample_rate):int(end_s * sample_rate)]
    n = int(sample_rate * 2)
    hop = int(sample_rate / 4)
    stft = ShortTimeFFT(windows.hann(n, sym=False), hop, sample_rate, mfft=n * 2, scale_to="psd")
    power = stft.spectrogram(clip)
    times = stft.t(len(clip))
    good_t = (times >= 1) & (times <= len(clip) / sample_rate - 1)
    good_f = (stft.f >= low_hz) & (stft.f <= high_hz)
    frequencies = stft.f[good_f]
    if len(frequencies) < 4:
        raise ValueError("Frequency band is too narrow.")
    power_db = 10 * np.log10(np.maximum(power[np.ix_(good_f, good_t)], 1e-20))
    return times[good_t] + start_s, frequencies, power_db.astype(np.float32)


def trace_spectrogram(time_s, frequencies, power_db, max_slope_hz_s=12, max_up_slope_hz_s=None):
    """Trace a tonal ridge without recomputing the spectrogram."""
    if max_slope_hz_s <= 0:
        raise ValueError("Maximum ridge slope must be positive.")
    # Reduce stationary tones and normalize each frame's background.
    evidence = power_db - .7 * np.median(power_db, axis=1, keepdims=True)
    evidence -= np.median(evidence, axis=0, keepdims=True)
    evidence = np.clip(evidence, -15, 30)
    bins, frames = evidence.shape
    step = frequencies[1] - frequencies[0]
    hop_s = float(np.median(np.diff(time_s)))
    max_jump = max(1, int(np.floor(max_slope_hz_s * hop_s / step)))
    offsets = np.arange(-max_jump, max_jump + 1)
    if max_up_slope_hz_s is not None:
        offsets = offsets[offsets >= -int(np.floor(max_up_slope_hz_s * hop_s / step))]
    previous = evidence[:, 0].copy()
    back = np.zeros((frames, bins), dtype=np.int32)
    targets = np.arange(bins)
    for frame in range(1, frames):
        source = targets[None, :] + offsets[:, None]
        allowed = (source >= 0) & (source < bins)
        options = previous[np.clip(source, 0, bins - 1)] - .2 * (offsets[:, None] * step) ** 2
        options[~allowed] = -np.inf
        best = np.argmax(options, axis=0)
        back[frame] = source[best, targets]
        previous = evidence[:, frame] + options[best, targets]
    path = np.empty(frames, dtype=int)
    path[-1] = np.argmax(previous)
    for frame in range(frames - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]
    # Sub-bin interpolation avoids quantized frequency curves.
    center = np.clip(path, 1, bins - 2)
    a, b, c = (power_db[center + offset, np.arange(frames)] for offset in (-1, 0, 1))
    denominator = a - 2 * b + c
    fraction = np.zeros(frames)
    np.divide(.5 * (a - c), denominator, out=fraction, where=np.abs(denominator) > 1e-8)
    fraction = np.clip(fraction, -.5, .5)
    fraction[(path == 0) | (path == bins - 1)] = 0
    ridge_frequency = frequencies[path] + fraction * step
    prominence = power_db[path, np.arange(frames)] - np.median(power_db, axis=0)
    return Ridge(np.asarray(time_s), ridge_frequency, prominence, frequencies, power_db)


def extract_ridge(audio, sample_rate, start_s, end_s, low_hz, high_hz, max_slope_hz_s=12):
    """Trace one smooth narrowband ridge with dynamic programming."""
    time_s, frequencies, power_db = spectrogram_data(audio, sample_rate, start_s, end_s, low_hz, high_hz)
    return trace_spectrogram(time_s, frequencies, power_db, max_slope_hz_s)

