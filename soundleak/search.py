from dataclasses import dataclass
import numpy as np

from .audio import Ridge
from .physics import Track, doppler_factor


@dataclass
class Event:
    track: Track
    ridge: Ridge


@dataclass
class SearchResult:
    east_m: np.ndarray
    north_m: np.ndarray
    rmse_hz: np.ndarray
    best_x_m: float
    best_y_m: float
    clock_offset_s: float
    source_frequencies_hz: list[float]
    predictions_hz: list[np.ndarray]
    sound_speed_mps: float

    @property
    def best_rmse_hz(self):
        return float(np.nanmin(self.rmse_hz))


def search(events, east_m, north_m, receiver_altitude_m=0, sound_speed=343, clock_offsets_s=(0,), progress=None):
    """Equal-sample least squares with one constant source tone per event.

    The shared clock offset is added to measured audio times. Every candidate
    must cover every selected frame: missing track data cannot improve a score.
    Surface stores the best RMSE over the requested clock-offset grid.
    """
    east_m, north_m = np.asarray(east_m), np.asarray(north_m)
    offsets = np.asarray(clock_offsets_s, dtype=float)
    if not events or not east_m.size or not north_m.size or not offsets.size:
        raise ValueError("Need events, a spatial grid, and clock offsets.")
    if not all(np.isfinite(a).all() for a in (east_m, north_m, offsets)):
        raise ValueError("Search grids must be finite.")
    if not np.isfinite(receiver_altitude_m):
        raise ValueError("Receiver altitude must be finite.")
    if max(len(east_m), len(north_m)) > 101 or len(offsets) > 41:
        raise ValueError("Limit searches to 101×101 points and 41 clock offsets.")
    for event in events:
        if not event.ridge.usable:
            raise ValueError("Selected ridge lacks sufficient tone strength or frequency movement. Choose another window/band.")
    costs = np.full((len(north_m), len(east_m)), np.nan)
    best_cost, best = np.inf, None
    total_samples = sum(len(event.ridge.time_s) for event in events)
    for iy, y in enumerate(north_m):
        for ix, x in enumerate(east_m):
            receiver = (x, y, receiver_altitude_m)
            for offset in offsets:
                sse, tones, predictions = 0., [], []
                for event in events:
                    q = doppler_factor(event.track, receiver, event.ridge.time_s + offset, sound_speed)
                    if not np.isfinite(q).all():
                        sse = np.inf
                        break
                    observed = event.ridge.frequency_hz
                    f0 = float(np.dot(q, observed) / np.dot(q, q))
                    prediction = f0 * q
                    sse += float(np.sum((observed - prediction) ** 2))
                    tones.append(f0)
                    predictions.append(prediction)
                cost = np.sqrt(sse / total_samples)
                if np.isfinite(cost) and (np.isnan(costs[iy, ix]) or cost < costs[iy, ix]):
                    costs[iy, ix] = cost
                if cost < best_cost:
                    best_cost, best = cost, (float(x), float(y), float(offset), tones, predictions)
        if progress:
            progress((iy + 1) / len(north_m))
    if best is None:
        raise ValueError("No candidate has full flight-track coverage. Include earlier emission samples and later track samples, or shorten the event.")
    return SearchResult(east_m, north_m, costs, *best, float(sound_speed))

