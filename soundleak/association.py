"""Rank locations, shared clocks and aircraft associations on a finite grid.

Every selected acoustic group contributes to a capped event-level loss, including
unmatched groups. This supports review, not calibrated localization confidence.
"""
from dataclasses import dataclass
import time

import numpy as np

from .physics import doppler_factor, latlon


@dataclass
class Hypothesis:
    east_m: float
    north_m: float
    clock_offset_s: float
    objective_hz: float
    matches: list[dict]
    unmatched: list[str]
    unavailable: list[str]

    @property
    def rmse_hz(self):
        return float(np.sqrt(np.mean([m["rmse_hz"]**2 for m in self.matches]))) if self.matches else None

    def metadata(self, origin):
        lat, lon = latlon(self.east_m, self.north_m, *origin)
        return {"latitude": float(lat), "longitude": float(lon), "east_m": self.east_m, "north_m": self.north_m,
                "clock_offset_s": self.clock_offset_s, "capped_objective_hz": self.objective_hz,
                "matched_event_rmse_hz": self.rmse_hz, "matched_groups": len(self.matches),
                "matches": self.matches, "unmatched_groups": self.unmatched,
                "groups_without_complete_track_coverage": self.unavailable}


@dataclass
class AssociationResult:
    hypotheses: list[Hypothesis]
    east_m: np.ndarray
    north_m: np.ndarray
    surface_hz: np.ndarray
    status: str
    reasons: list[str]
    settings: dict
    evaluated_models: int
    elapsed_s: float


def _batched_point(proposals, tracks, eligible, samples, x, y, offsets, altitude, sound_speed, threshold):
    """Evaluate identical scalar hypotheses with clock times in one NumPy batch."""
    n = len(offsets)
    losses = np.full((len(proposals), n), threshold**2)
    matches, unmatched, unavailable = [[] for _ in offsets], [[] for _ in offsets], [[] for _ in offsets]
    evaluated = 0
    for pi, p in enumerate(proposals):
        indices = samples[p.id]
        observed = p.ridge.frequency_hz[indices]
        receive = p.ridge.time_s[indices][None, :]+offsets[:, None]
        best_mse = np.full(n, np.inf)
        best_f0 = np.zeros(n)
        best_key = np.full(n, -1, dtype=int)
        keys = eligible[p.id]
        for ki, key in enumerate(keys):
            q = doppler_factor(tracks[key], (x, y, altitude), receive.ravel(), sound_speed).reshape(receive.shape)
            evaluated += n
            valid = np.flatnonzero(np.isfinite(q).all(axis=1))
            if not len(valid):
                continue
            good = q[valid]
            f0 = np.sum(good*observed, axis=1)/np.sum(good*good, axis=1)
            mse = np.mean((good*f0[:, None]-observed)**2, axis=1)
            better = mse < best_mse[valid]
            selected = valid[better]
            best_mse[selected], best_f0[selected], best_key[selected] = mse[better], f0[better], ki
        explained = 1-best_mse/max(float(np.var(observed)), 1e-12)
        accepted = np.isfinite(best_mse) & (best_mse <= threshold**2) & (explained >= .6)
        losses[pi, accepted] = best_mse[accepted]
        for oi in range(n):
            if accepted[oi]:
                matches[oi].append({"event_id": p.id, "candidate": keys[best_key[oi]],
                                    "source_tone_hz": float(best_f0[oi]), "rmse_hz": float(np.sqrt(best_mse[oi])),
                                    "trend_explained_fraction": float(explained[oi])})
            else:
                unmatched[oi].append(p.id)
                if best_key[oi] < 0:
                    unavailable[oi].append(p.id)
    objectives = np.sqrt(np.mean(losses, axis=0))
    oi = int(np.argmin(objectives))
    return Hypothesis(float(x), float(y), float(offsets[oi]), float(objectives[oi]), matches[oi], unmatched[oi], unavailable[oi]), evaluated


def ranked_search(proposals, tracks, east_m, north_m, clock_offsets_s=(0.,), receiver_altitude_m=0.,
                  sound_speed=343., max_event_rmse_hz=3., max_models=300_000, progress=None, backend="scalar"):
    started = time.monotonic()
    east, north, offsets = (np.asarray(x, dtype=float) for x in (east_m, north_m, clock_offsets_s))
    if (not proposals or not tracks or not east.size or not north.size or not offsets.size
        or not all(np.isfinite(a).all() for a in (east, north, offsets))):
        raise ValueError("Need acoustic groups, flight candidates and finite search grids.")
    if not 0 < max_event_rmse_hz <= 20 or not 300 <= sound_speed <= 370 or not np.isfinite(receiver_altitude_m):
        raise ValueError("Invalid frequency threshold, sound speed or receiver elevation.")
    if max(len(east), len(north)) > 61 or len(offsets) > 41:
        raise ValueError("Use at most 61 points per spatial axis and 41 clock offsets.")
    if np.any(np.diff(east) <= 0) or np.any(np.diff(north) <= 0):
        raise ValueError("Spatial grids must be strictly increasing.")
    if backend not in ("scalar", "batched"):
        raise ValueError("Unknown search backend.")
    # Conservative time coverage sieve; exact coverage is still checked at each point.
    eligible = {}
    for p in proposals:
        options = []
        for key, track in tracks.items():
            if np.any(np.linalg.norm(track.velocity_mps, axis=1) >= sound_speed):
                raise ValueError(f"{key}: invalid subsonic flight trajectory.")
            first, last = track.xyz_m[0], track.xyz_m[-1]
            closest = (np.clip(first[0], east.min(), east.max()), np.clip(first[1], north.min(), north.max()), receiver_altitude_m)
            first_min_delay = np.linalg.norm(first-np.asarray(closest))/sound_speed
            last_max_delay = max(np.linalg.norm(last-np.array((x, y, receiver_altitude_m)))
                                 for x in (east.min(), east.max()) for y in (north.min(), north.max()))/sound_speed
            if (track.time_s[0]+first_min_delay <= p.ridge.time_s[0]+offsets.max()
                and track.time_s[-1]+last_max_delay >= p.ridge.time_s[-1]+offsets.min()):
                options.append(key)
        eligible[p.id] = options
    estimate = len(east)*len(north)*len(offsets)*sum(map(len, eligible.values()))
    if estimate > max_models:
        raise ValueError(f"Search requires up to {estimate:,} model evaluations; limit is {max_models:,}. "
                         "Reduce the grid, clock range, selected events or candidate set.")
    samples = {p.id: np.arange(0, len(p.ridge.time_s), 4) for p in proposals}
    surface = np.full((len(north), len(east)), np.nan)
    points, evaluated = [], 0
    for iy, y in enumerate(north):
        for ix, x in enumerate(east):
            if backend == "batched":
                best, count = _batched_point(proposals, tracks, eligible, samples, x, y, offsets,
                                             receiver_altitude_m, sound_speed, max_event_rmse_hz)
                evaluated += count
                points.append(best)
                surface[iy, ix] = best.objective_hz
                continue
            best = None
            for offset in offsets:
                losses, matches, unmatched, unavailable = [], [], [], []
                for p in proposals:
                    indices = samples[p.id]
                    observed = p.ridge.frequency_hz[indices]
                    found = None
                    for key in eligible[p.id]:
                        q = doppler_factor(tracks[key], (x, y, receiver_altitude_m), p.ridge.time_s[indices]+offset, sound_speed)
                        evaluated += 1
                        if not np.isfinite(q).all():
                            continue
                        f0 = float(np.dot(q, observed)/np.dot(q, q))
                        mse = float(np.mean((f0*q-observed)**2))
                        if found is None or mse < found[0]:
                            found = (mse, key, f0)
                    losses.append(min(found[0], max_event_rmse_hz**2) if found else max_event_rmse_hz**2)
                    explained = 1-found[0]/max(float(np.var(observed)), 1e-12) if found else -np.inf
                    if found and found[0] <= max_event_rmse_hz**2 and explained >= .6:
                        matches.append({"event_id": p.id, "candidate": found[1],
                                        "source_tone_hz": found[2], "rmse_hz": float(np.sqrt(found[0])),
                                        "trend_explained_fraction": float(explained)})
                    else:
                        losses[-1] = max_event_rmse_hz**2
                        unmatched.append(p.id)
                        if found is None:
                            unavailable.append(p.id)
                objective = float(np.sqrt(np.mean(losses)))
                h = Hypothesis(float(x), float(y), float(offset), objective, matches, unmatched, unavailable)
                if best is None or objective < best.objective_hz:
                    best = h
            points.append(best)
            surface[iy, ix] = best.objective_hz
        if progress:
            progress((iy+1)/len(north))
    # Keep spatial alternatives rather than presenting a single winning coordinate.
    diverse = []
    spacing = max(float(np.ptp(east)/max(1, len(east)-1)), float(np.ptp(north)/max(1, len(north)-1)))
    for h in sorted(points, key=lambda point: point.objective_hz):
        if not h.matches:
            continue
        if all(np.hypot(h.east_m-other.east_m, h.north_m-other.north_m) >= max(500., spacing*1.5) for other in diverse):
            diverse.append(h)
        if len(diverse) == 8:
            break
    status, reasons = "candidate_fit", []
    if not diverse or len({m["candidate"] for m in diverse[0].matches}) < 2:
        status = "insufficient_evidence"
        reasons.append("Fewer than two distinct candidate flight segments support the selected acoustic groups under one shared clock.")
    else:
        best = diverse[0]
        if best.unmatched:
            status = "partial_fit"
            reasons.append(f"{len(best.unmatched)} selected acoustic groups remain unmatched; some may be non-aircraft sounds or missing flights.")
        if best.unavailable:
            reasons.append(f"{len(best.unavailable)} groups lack complete candidate-track coverage at the best fit.")
        if best.east_m in (east[0], east[-1]) or best.north_m in (north[0], north[-1]):
            status = "ambiguous"
            reasons.append("The best point is on the spatial search boundary.")
        if len(offsets) > 1 and best.clock_offset_s in (offsets[0], offsets[-1]):
            status = "ambiguous"
            reasons.append("The best shared clock is on the searched time boundary.")
        if len(diverse) > 1 and diverse[1].objective_hz-best.objective_hz < .1:
            status = "ambiguous"
            reasons.append("Another separated location has a similar capped objective (gap <0.1 Hz).")
    reasons.append("Statuses use transparent review heuristics; no calibrated probability or blind accuracy claim.")
    settings = {"max_event_rmse_hz": max_event_rmse_hz, "receiver_altitude_m": receiver_altitude_m,
                "sound_speed_mps": sound_speed, "clock_offsets_s": offsets.tolist(),
                "east_m": east.tolist(), "north_m": north.tolist(),
                "scoring_frame_stride": 4, "estimated_max_model_evaluations": estimate,
                "backend": backend, "max_model_evaluations": max_models,
                "minimum_event_trend_explained_fraction": .6,
                "eligible_candidates_per_group": eligible,
                "objective": "sqrt(mean(min(event frequency MSE, rejection threshold squared))); equal weight per selected acoustic group"}
    return AssociationResult(diverse, east, north, surface, status, reasons, settings, evaluated, time.monotonic()-started)


def fit_at_hypothesis(proposal, track, hypothesis, receiver_altitude_m=0., sound_speed=343.):
    q = doppler_factor(track, (hypothesis.east_m, hypothesis.north_m, receiver_altitude_m),
                       proposal.ridge.time_s+hypothesis.clock_offset_s, sound_speed)
    if not np.isfinite(q).all():
        return None
    return float(np.dot(q, proposal.ridge.frequency_hz)/np.dot(q, q))*q
