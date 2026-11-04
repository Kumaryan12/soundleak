"""Whole-recording acoustic proposals, independent of flight data and location.

These are moving-tone proposals, not certified aircraft detections. Thresholds
are explicit heuristics. The input is scanned in overlapping bounded windows.
"""
from dataclasses import dataclass, asdict
from hashlib import sha256
import time

import numpy as np
from scipy.ndimage import median_filter, binary_closing, label
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from .audio import Ridge, spectrogram_data, trace_spectrogram


@dataclass(frozen=True)
class DiscoveryConfig:
    window_s: float = 90.
    stride_s: float = 60.
    min_duration_s: float = 20.
    min_prominence_db: float = 12.
    min_drop_fraction: float = .08
    max_fit_rmse_hz: float = 2.5
    bands: tuple = ((40., 100.), (70., 170.), (120., 280.), (200., 450.), (350., 800.))


@dataclass
class Proposal:
    id: str
    ridge: Ridge
    tone_fit: dict
    members: int = 1

    def metadata(self):
        r = self.ridge
        return {"id": self.id, "start_s": float(r.time_s[0]), "end_s": float(r.time_s[-1]),
                "duration_s": float(r.time_s[-1]-r.time_s[0]),
                "frequency_start_hz": float(r.frequency_hz[0]), "frequency_end_hz": float(r.frequency_hz[-1]),
                "prominence_db": float(np.median(r.prominence_db)), "grouped_proposals": self.members,
                **self.tone_fit}


@dataclass
class DiscoveryResult:
    proposals: list[Proposal]
    duration_s: float
    scanned_windows: int
    raw_proposals: int
    config: dict
    elapsed_s: float

    def metadata(self):
        return {"duration_s": self.duration_s, "scanned_windows": self.scanned_windows,
                "raw_proposals": self.raw_proposals, "elapsed_s": self.elapsed_s, "config": self.config,
                "events": [p.metadata() for p in self.proposals],
                "interpretation": "Unconfirmed moving-tone proposals; heuristic thresholds, not an aircraft classifier."}


def straight_pass_frequency(time_s, f0, speed, slant, cpa_receive_s, sound_speed=343.):
    """Analytic constant-speed pass, parameterized by closest-arrival time."""
    delta = np.asarray(time_s) - (cpa_receive_s-slant/sound_speed)
    a = sound_speed**2-speed**2
    emission = (sound_speed**2*delta-np.sqrt(sound_speed**2*speed**2*delta**2+a*slant**2))/a
    radial = speed**2*emission/np.hypot(speed*emission, slant)
    return f0*sound_speed/(sound_speed+radial)


def fit_tone(ridge):
    t = ridge.time_s-ridge.time_s[0]
    f = ridge.frequency_hz
    # Thin only the nonlinear diagnostic fit, retaining the original measurements.
    keep = np.arange(0, len(t), max(1, len(t)//100))
    t, f = t[keep], f[keep]
    duration = float(t[-1])
    guess = [float(np.median(f)), 100., 2500., duration/2]
    bounds = ([float(np.min(f)*.5), 20., 100., -duration],
              [float(np.max(f)*1.5), 300., 20000., duration*2])
    result = least_squares(lambda p: straight_pass_frequency(t, *p)-f, guess,
                           bounds=bounds, loss="soft_l1", f_scale=1., max_nfev=100)
    residual = straight_pass_frequency(t, *result.x)-f
    rmse = float(np.sqrt(np.mean(residual**2)))
    baseline = float(np.mean((f-np.mean(f))**2))
    return {"tone_fit_rmse_hz": rmse, "trend_explained_fraction": float(1-np.mean(residual**2)/max(baseline, 1e-12)),
            "fitted_tone_hz": float(result.x[0]),
            "approx_cpa_receive_s": float(ridge.time_s[0]+result.x[3]),
            "fit_converged": bool(result.success)}


def _trim(ridge, indices):
    return Ridge(ridge.time_s[indices], ridge.frequency_hz[indices], ridge.prominence_db[indices],
                 ridge.frequency_bins_hz.copy(), ridge.power_db[:, indices].copy())


def _same_event(a, b):
    lo = max(a.ridge.time_s[0], b.ridge.time_s[0])
    hi = min(a.ridge.time_s[-1], b.ridge.time_s[-1])
    if hi-lo < max(8., .3*min(a.metadata()["duration_s"], b.metadata()["duration_s"])):
        return False
    t = np.linspace(lo, hi, 25)
    fa = np.interp(t, a.ridge.time_s, a.ridge.frequency_hz)
    fb = np.interp(t, b.ridge.time_s, b.ridge.frequency_hz)
    ratios = fa/fb
    ratio = float(np.median(ratios))
    # Same ridge or harmonics. Harmonics form one event, never extra flyovers.
    harmonic = min(abs(ratio-p/q)/(p/q) for p in range(1, 5) for q in range(1, 5))
    return harmonic < .025 and np.median(np.abs(ratios-ratio))/ratio < .012


def group_proposals(proposals):
    groups = []
    for proposal in sorted(proposals, key=lambda p: p.ridge.time_s[0]):
        related = [i for i, members in enumerate(groups) if any(_same_event(proposal, other) for other in members)]
        merged = [proposal]
        for i in reversed(related):
            merged.extend(groups.pop(i))
        groups.append(merged)
    result = []
    for members in groups:
        # Preserve one actual harmonic's measurements, not a mixture of different harmonics.
        best = max(members, key=lambda p: (p.ridge.time_s[-1]-p.ridge.time_s[0]) /
                   (1+p.tone_fit["tone_fit_rmse_hz"]))
        best.members = len(members)
        result.append(best)
    return sorted(result, key=lambda p: p.ridge.time_s[0])


def discover(audio, sample_rate, config=DiscoveryConfig(), progress=None):
    started = time.monotonic()
    duration = len(audio)/sample_rate
    if config.stride_s <= 0 or config.window_s < config.stride_s or config.min_duration_s < 4:
        raise ValueError("Invalid discovery window, stride or minimum duration.")
    if duration < config.min_duration_s+2:
        return DiscoveryResult([], duration, 0, 0, asdict(config), time.monotonic()-started)
    starts = np.arange(0, duration-config.min_duration_s-1, config.stride_s)
    raw = []
    for index, start in enumerate(starts):
        end = min(start+config.window_s, duration)
        times, frequencies, power = spectrogram_data(audio, sample_rate, start, end,
                                                      config.bands[0][0], config.bands[-1][1])
        for low, high in config.bands:
            band = (frequencies >= low) & (frequencies <= high)
            ridge = trace_spectrogram(times, frequencies[band], power[band], 6., .75)
            strength = median_filter(ridge.prominence_db, size=9) >= config.min_prominence_db
            strength = binary_closing(strength, structure=np.ones(9))
            regions, count = label(strength)
            for region in range(1, count+1):
                indices = np.flatnonzero(regions == region)
                if len(indices) < 2 or ridge.time_s[indices[-1]]-ridge.time_s[indices[0]] < config.min_duration_s:
                    continue
                candidate = _trim(ridge, indices)
                smoothed = median_filter(candidate.frequency_hz, size=9)
                drop = float(np.median(smoothed[:5])-np.median(smoothed[-5:]))
                if drop < max(4., config.min_drop_fraction*np.median(smoothed)):
                    continue
                if spearmanr(candidate.time_s, smoothed).statistic > -.75:
                    continue
                fit = fit_tone(candidate)
                if (not fit["fit_converged"] or fit["tone_fit_rmse_hz"] > config.max_fit_rmse_hz
                    or fit["trend_explained_fraction"] < .75
                    or not candidate.time_s[0] <= fit["approx_cpa_receive_s"] <= candidate.time_s[-1]):
                    continue
                identity = sha256(f"{candidate.time_s[0]:.2f}:{candidate.time_s[-1]:.2f}:{np.median(smoothed):.2f}".encode()).hexdigest()[:8]
                raw.append(Proposal("event-"+identity, candidate, fit))
        if progress:
            progress((index+1)/len(starts))
    return DiscoveryResult(group_proposals(raw), duration, len(starts), len(raw), asdict(config), time.monotonic()-started)
