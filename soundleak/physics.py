"""Local tangent-plane acoustics. Positive radial velocity means receding."""
from dataclasses import dataclass
import csv
import io

import numpy as np
from numpy.typing import NDArray

EARTH_RADIUS_M = 6_371_000.0


def local_xy(latitude, longitude, origin_lat, origin_lon):
    """Equirectangular coordinates; intended for search regions <40 km across."""
    x = EARTH_RADIUS_M * np.cos(np.deg2rad(origin_lat)) * np.deg2rad(np.asarray(longitude) - origin_lon)
    y = EARTH_RADIUS_M * np.deg2rad(np.asarray(latitude) - origin_lat)
    return x, y


def latlon(x, y, origin_lat, origin_lon):
    return (origin_lat + np.rad2deg(np.asarray(y) / EARTH_RADIUS_M),
            origin_lon + np.rad2deg(np.asarray(x) / (EARTH_RADIUS_M * np.cos(np.deg2rad(origin_lat)))))


@dataclass
class Track:
    name: str
    time_s: NDArray
    xyz_m: NDArray

    def __post_init__(self):
        self.time_s = np.asarray(self.time_s, dtype=float)
        self.xyz_m = np.asarray(self.xyz_m, dtype=float)
        if self.time_s.ndim != 1 or len(self.time_s) < 3 or self.xyz_m.shape != (len(self.time_s), 3):
            raise ValueError("Each aircraft needs at least three time/position samples.")
        if not np.isfinite(self.time_s).all() or not np.isfinite(self.xyz_m).all():
            raise ValueError("Track values must be finite.")
        if np.any(np.diff(self.time_s) <= 0):
            raise ValueError("Track times must be unique and strictly increasing per aircraft.")

    @property
    def velocity_mps(self):
        return np.diff(self.xyz_m, axis=0) / np.diff(self.time_s)[:, None]


def read_tracks(contents: str, origin_lat: float, origin_lon: float) -> dict[str, Track]:
    """CSV times are emission times, seconds relative to audio sample zero."""
    rows = csv.DictReader(io.StringIO(contents))
    required = {"aircraft_id", "time_s", "latitude", "longitude", "altitude_m"}
    if not required.issubset(rows.fieldnames or []):
        raise ValueError("CSV needs: " + ", ".join(sorted(required)))
    grouped = {}
    for row in rows:
        key = row["aircraft_id"].strip()
        if not key:
            raise ValueError("aircraft_id cannot be empty.")
        t, lat, lon, alt = (float(row[k]) for k in ("time_s", "latitude", "longitude", "altitude_m"))
        if not -90 < lat < 90 or not -180 <= lon <= 180:
            raise ValueError("Invalid track latitude/longitude.")
        x, y = local_xy(lat, lon, origin_lat, origin_lon)
        if np.hypot(x, y) > 100_000:
            raise ValueError("Track is >100 km from the search center; use a closer origin.")
        grouped.setdefault(key, []).append((t, x, y, alt))
    if not grouped:
        raise ValueError("No aircraft rows found.")
    tracks = {}
    for key, points in grouped.items():
        data = np.asarray(sorted(points))
        tracks[key] = Track(key, data[:, 0], data[:, 1:])
    return tracks


def doppler_factor(track: Track, receiver_m, receive_time_s, sound_speed=343.0):
    """Return f_received/f_emitted, including propagation delay.

    Invert t_receive = t_emit + |position(t_emit)-receiver|/c.
    Solve exactly within each piecewise-linear, subsonic track segment.
    Out-of-coverage times return NaN rather than extrapolated positions.
    """
    if not np.isfinite(sound_speed) or sound_speed <= 0:
        raise ValueError("Sound speed must be positive.")
    velocity = track.velocity_mps
    if np.any(np.linalg.norm(velocity, axis=1) >= sound_speed):
        raise ValueError(f"{track.name}: track contains sonic/supersonic speeds or bad timestamps.")
    receiver = np.asarray(receiver_m, dtype=float)
    times = np.atleast_1d(np.asarray(receive_time_s, dtype=float))
    distance = np.linalg.norm(track.xyz_m - receiver, axis=1)
    arrivals = track.time_s + distance / sound_speed
    idx = np.clip(np.searchsorted(arrivals, times, side="right") - 1, 0, len(velocity) - 1)
    # Solve the retarded-time quadratic within the selected linear segment.
    # c²(delta-u)² = |r0 + v*u|²; the smaller root arrives after emission.
    # Rationalize that root when subtraction would lose precision.
    r0 = track.xyz_m[idx] - receiver
    v = velocity[idx]
    delta = times - track.time_s[idx]
    a = sound_speed**2 - np.sum(v*v, axis=1)
    b = sound_speed**2 * delta + np.sum(r0*v, axis=1)
    c = sound_speed**2 * delta**2 - np.sum(r0*r0, axis=1)
    discriminant = np.sqrt(np.maximum(b*b - a*c, 0.))
    u = (b - discriminant) / a
    np.divide(c, b + discriminant, out=u, where=(b >= 0) & (b + discriminant > 0))
    u = np.clip(u, 0, track.time_s[idx+1] - track.time_s[idx])
    displacement = r0 + v * u[:, None]
    radial = np.sum(velocity[idx] * displacement, axis=1) / np.maximum(np.linalg.norm(displacement, axis=1), 1e-9)
    factor = sound_speed / (sound_speed + radial)
    factor[(times < arrivals[0]) | (times > arrivals[-1]) | ~np.isfinite(times)] = np.nan
    return factor

