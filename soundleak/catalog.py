"""Candidate flight import with explicit clocks, altitude units and data gaps."""
import csv
from datetime import datetime
import io
import json

import numpy as np

from .physics import Track, read_tracks
from .real import readsb_track


def utc_timestamp(value):
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Enter an ISO UTC time, for example 2026-09-08T20:17:30Z.") from exc
    if dt.tzinfo is None:
        raise ValueError("Recording timestamps must include Z or an explicit UTC offset.")
    return dt.timestamp()


def split_tracks(tracks, max_gap_s=30., sound_speed=343.):
    segments, rejected = {}, []
    for aircraft, track in tracks.items():
        cuts = np.flatnonzero(np.diff(track.time_s) > max_gap_s)+1
        for number, indices in enumerate(np.split(np.arange(len(track.time_s)), cuts), 1):
            identity = f"{aircraft} / segment {number}"
            if len(indices) < 3:
                rejected.append({"id": identity, "reason": "Fewer than three samples in a contiguous segment."})
                continue
            segment = Track(identity, track.time_s[indices], track.xyz_m[indices])
            if np.any(np.linalg.norm(segment.velocity_mps, axis=1) >= sound_speed):
                rejected.append({"id": identity, "reason": "Sonic/supersonic segment or inconsistent timestamps/positions."})
                continue
            segments[identity] = segment
    return segments, rejected


def csv_candidates(contents, origin, recording_start_unix_s=None, max_gap_s=30., sound_speed=343.):
    reader = csv.DictReader(io.StringIO(contents))
    fields = reader.fieldnames or []
    if "time_s" not in fields:
        if "time_utc" not in fields or recording_start_unix_s is None:
            raise ValueError("Flight CSV needs time_s relative to recording sample zero, or time_utc plus a recording-start UTC.")
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[*fields, "time_s"])
        writer.writeheader()
        for row in reader:
            row["time_s"] = utc_timestamp(row["time_utc"])-recording_start_unix_s
            writer.writerow(row)
        contents = output.getvalue()
    return split_tracks(read_tracks(contents, *origin), max_gap_s, sound_speed)


def readsb_candidates(contents, origin, recording_start_unix_s, duration_s,
                      geoid_undulation_m, max_gap_s=30., sound_speed=343.):
    data = json.loads(contents) if isinstance(contents, (str, bytes)) else contents
    # Segment before filling geometric altitude, so interpolation cannot bridge flights.
    rows = [row for row in data["trace"] if -200 <= float(data["timestamp"])+row[0]-recording_start_unix_s <= duration_s+60]
    rows.sort(key=lambda row: row[0])
    cuts = np.flatnonzero(np.diff([r[0] for r in rows]) > max_gap_s)+1 if rows else []
    result, rejected = {}, []
    for n, indices in enumerate(np.split(np.arange(len(rows)), cuts), 1):
        identity = f"{data.get('icao', 'unknown')} / segment {n}"
        subset = {**data, "trace": [rows[i] for i in indices]}
        try:
            track = readsb_track(subset, recording_start_unix_s, origin, (0, duration_s),
                                 geoid_undulation_m, max_gap_s)
            track.name = f"{identity} · {data.get('t', 'aircraft')}"
            if np.any(np.linalg.norm(track.velocity_mps, axis=1) >= sound_speed):
                raise ValueError("Sonic/supersonic segment or inconsistent timestamps/positions.")
            result[track.name] = track
        except ValueError as exc:
            rejected.append({"id": identity, "reason": str(exc)})
    return result, rejected


def candidate_table(tracks):
    return [{"candidate": key, "samples": len(t.time_s), "start_s": float(t.time_s[0]),
             "end_s": float(t.time_s[-1]), "max_gap_s": float(np.max(np.diff(t.time_s)))} for key, t in tracks.items()]
