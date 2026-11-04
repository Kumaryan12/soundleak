"""Source-informed replay of colsto's public raw recording.

The clock, aircraft associations, event windows and regional search bounds are
published research inputs, not discoveries made by this implementation.
"""
from pathlib import Path
import json

import numpy as np

from .audio import decode_audio, extract_ridge
from .physics import Track, local_xy, latlon
from .search import Event, search
from .report import event_figure, heatmap_figure, png_bytes, summary, json_text

REFERENCE_SOURCE = "https://www.youtube.com/watch?v=4CxVxuMVkMc"
UPSTREAM = "https://github.com/1337isnot1337/DOPPLERGEIST"
RECORDING_START_UNIX_S = 1788898649.6620934
ORIGIN = (40.875, -73.53)
PUBLIC_REFERENCE = (40.8766, -73.5247)
# A constant approximation around Oyster Bay, not a terrain/geoid raster model.
GEOID_UNDULATION_M = -31.26
WINDOWS = [("a1dfeb", 2100., 2280., 75., 190.), ("ace391", 3040., 3125., 95., 185.)]


def readsb_track(contents, recording_start_unix_s, origin, window_s, geoid_undulation_m=0., max_gap_s=30.):
    """Read genuine readsb trace points, retaining geometric altitude only.

    Trace altitude is feet; timestamps are base Unix seconds + row[0].
    Geometric altitude is WGS84 ellipsoidal. Subtract geoid undulation to obtain
    approximate MSL altitude, compatible with the receiver's MSL elevation.
    Missing geometric heights may be interpolated only between samples <=30s
    apart. Barometric and geometric altitude are never silently interchanged.
    """
    data = json.loads(contents) if isinstance(contents, (str, bytes)) else contents
    values = {}
    for row in data["trace"]:
        if len(row) < 8 or row[3] == "ground" or (int(row[6]) & 1):
            continue
        t = float(data["timestamp"]) + float(row[0]) - recording_start_unix_s
        if not window_s[0] - 150 <= t <= window_s[1] + 60:
            continue
        geom = row[3] if int(row[6]) & 8 else (row[10] if len(row) > 10 else None)
        if geom is None and len(row) > 8 and isinstance(row[8], dict):
            geom = row[8].get("alt_geom")
        if row[1] is None or row[2] is None:
            continue
        values[t] = (float(row[1]), float(row[2]), float(geom)*.3048-geoid_undulation_m if geom is not None else np.nan)
    if len(values) < 3:
        raise ValueError(f"{data.get('icao')}: not enough track points in the selected time window.")
    times = np.array(sorted(values))
    rows = np.asarray([values[t] for t in times])
    known = np.flatnonzero(np.isfinite(rows[:, 2]))
    for i in np.flatnonzero(~np.isfinite(rows[:, 2])):
        before, after = known[known < i], known[known > i]
        if len(before) and len(after):
            a, b = before[-1], after[0]
            if times[b] - times[a] <= max_gap_s:
                rows[i, 2] = np.interp(times[i], times[[a, b]], rows[[a, b], 2])
    valid = np.isfinite(rows).all(axis=1)
    times, rows = times[valid], rows[valid]
    if len(times) < 3 or np.any(np.diff(times) > max_gap_s):
        raise ValueError(f"{data.get('icao')}: insufficient geometric-altitude coverage or track gap >{max_gap_s:g}s.")
    x, y = local_xy(rows[:, 0], rows[:, 1], *origin)
    return Track(f"{data['icao']} · {data.get('t', 'aircraft')}", times, np.column_stack((x, y, rows[:, 2])))


def reference_events(data_dir=Path("data/real/reference")):
    root = Path(data_dir)
    events = []
    for aircraft, start, end, low, high in WINDOWS:
        audio, sr = decode_audio(root / "original.webm", offset_s=start, max_duration_s=end-start)
        ridge = extract_ridge(audio, sr, 0, len(audio)/sr, low, high, max_slope_hz_s=4)
        ridge.time_s += start
        track = readsb_track((root/f"trace_full_{aircraft}.json").read_text(), RECORDING_START_UNIX_S,
                             ORIGIN, (start, end), GEOID_UNDULATION_M)
        events.append(Event(track, ridge))
    return events


def run_reference(data_dir=Path("data/real/reference"), output=Path("outputs/real/reference"), progress=None):
    events = reference_events(data_dir)
    west, south = local_xy(40.83, -73.59, *ORIGIN)
    east, north = local_xy(40.92, -73.47, *ORIGIN)
    coarse = search(events, np.linspace(west, east, 41), np.linspace(south, north, 41),
                    receiver_altitude_m=0., clock_offsets_s=np.arange(-5, 5.01, .5), progress=progress)
    # Refine around the coarse winner using the same preselected observations.
    result = search(events, np.arange(coarse.best_x_m-400, coarse.best_x_m+401, 25),
                    np.arange(coarse.best_y_m-400, coarse.best_y_m+401, 25), receiver_altitude_m=0.,
                    clock_offsets_s=np.arange(coarse.clock_offset_s-1, coarse.clock_offset_s+1.01, .25))
    report = summary(result, events, ORIGIN, 0.)
    ref_x, ref_y = local_xy(*PUBLIC_REFERENCE, *ORIGIN)
    report.update({
        "mode": "source_informed_real_replay",
        "source_audio": REFERENCE_SOURCE,
        "source_reproduction": UPSTREAM,
        "recording_start_unix_s_prior": RECORDING_START_UNIX_S,
        "clock_provenance": "Inherited fitted clock from published reproduction; not independently measured recorder UTC.",
        "reference_coordinate": {"latitude": PUBLIC_REFERENCE[0], "longitude": PUBLIC_REFERENCE[1]},
        "reference_source": "https://app.osprey-watch.org/nest/3916",
        "reference_kind": "Published public-platform map marker; physical microphone uncertainty unquantified.",
        "distance_to_public_reference_m": float(np.hypot(result.best_x_m-ref_x, result.best_y_m-ref_y)),
        "initial_search_bounds_wsen": [-73.59, 40.83, -73.47, 40.92],
        "initial_search_grid_points_per_axis": 41,
        "initial_clock_offset_grid_s": [-5, 5, .5],
        "geoid_undulation_m_approximation": GEOID_UNDULATION_M,
        "source_informed_inputs": ["Event time windows and frequency bands", "Aircraft identities and date",
                                   "Regional search bounds", "Recording-start clock prior"],
        "independent_blind_localization": False,
    })
    report["limitations"] = [value for value in report["limitations"] if not value.startswith("Synthetic validation")]
    report["limitations"] += ["Search inputs were informed by a published reproduction.",
                              "Distance to a rounded public marker is not surveyed location accuracy.",
                              "No independent audio-sample-to-UTC binding or wind model."]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"report.json").write_text(json_text(report))
    (output/"heatmap.png").write_bytes(png_bytes(heatmap_figure(result, events, (ref_x, ref_y), reference_label="Public camera reference")))
    (output/"regional-heatmap.png").write_bytes(png_bytes(heatmap_figure(coarse, events, (ref_x, ref_y), reference_label="Public camera reference")))
    for i, (event, prediction) in enumerate(zip(events, result.predictions_hz), 1):
        (output/f"event-{i}.png").write_bytes(png_bytes(event_figure(event, prediction)))
        # Preserve the exact real clip used for each displayed acoustic event.
        from scipy.io import wavfile
        _, start, end, _, _ = WINDOWS[i-1]
        audio, sr = decode_audio(Path(data_dir)/"original.webm", offset_s=start, max_duration_s=end-start)
        wavfile.write(output/f"event-{i}.wav", sr, audio)
        np.savetxt(output/f"ridge-{i}.csv",
                   np.column_stack((event.ridge.time_s, event.ridge.frequency_hz, prediction, event.ridge.prominence_db)),
                   delimiter=",", header="recording_time_s,measured_hz,predicted_hz,prominence_db", comments="")
    return events, coarse, result, report

