"""Reproducible workbench runs without the Streamlit interface."""
from hashlib import sha256
from pathlib import Path

import numpy as np

from .association import ranked_search
from .audio import decode_audio
from .catalog import csv_candidates, readsb_candidates
from .cases import new_case_directory, save_case
from .discovery import DiscoveryConfig, discover


def analyze(recording, origin=(0., 0.), flight_csv=None, readsb_files=(), recording_start_unix_s=None,
            clock_evidence="Operator supplied; independently verify timing.", geoid_undulation_m=0.,
            max_duration_s=3600, config=DiscoveryConfig(), half_width_m=5000., grid_points=21,
            clock_sweep_s=5., receiver_altitude_m=0., sound_speed=343., max_event_rmse_hz=3.,
            output=None, provenance=None, progress=None):
    if not 0 < half_width_m <= 15_000 or not 1 <= grid_points <= 61 or not 0 <= clock_sweep_s <= 120:
        raise ValueError("Use a positive search half-width up to 15 km, 1–61 grid points, and a 0–120 s clock sweep.")
    recording = Path(recording)
    audio, sr = decode_audio(recording, max_duration_s=max_duration_s)
    discovery = discover(audio, sr, config, progress=(lambda p: progress("discovery", p)) if progress else None)
    tracks, rejected = {}, []
    if flight_csv is not None:
        tracks, rejected = csv_candidates(Path(flight_csv).read_text(), origin, recording_start_unix_s, sound_speed=sound_speed)
    for file in readsb_files:
        if recording_start_unix_s is None:
            raise ValueError("Native readsb files need a recording-start UTC.")
        candidates, errors = readsb_candidates(Path(file).read_text(), origin, recording_start_unix_s,
                                               len(audio)/sr, geoid_undulation_m, sound_speed=sound_speed)
        tracks.update(candidates)
        rejected.extend(errors)
    result = None
    if discovery.proposals and tracks:
        grid = np.linspace(-half_width_m, half_width_m, grid_points)
        offsets = np.linspace(-clock_sweep_s, clock_sweep_s, 21) if clock_sweep_s else [0.]
        result = ranked_search(discovery.proposals, tracks, grid, grid, offsets, receiver_altitude_m,
                               sound_speed, max_event_rmse_hz,
                               progress=(lambda p: progress("association", p)) if progress else None)
    metadata = dict(provenance or {})
    metadata.update({"recording_name": recording.name, "input_sha256": sha256(recording.read_bytes()).hexdigest(),
                     "sample_rate": sr, "decode_limit_s": max_duration_s, "decoded_duration_s": len(audio)/sr,
                     "recording_start_unix_s": recording_start_unix_s, "clock_source": clock_evidence,
                     "rejected_track_segments": rejected,
                     "discovery_inputs": "Audio samples and fixed acoustic screen only; no flight IDs, location or event hints."})
    if flight_csv is not None:
        metadata["flight_csv_sha256"] = sha256(Path(flight_csv).read_bytes()).hexdigest()
    if readsb_files:
        metadata["native_track_sha256"] = {Path(p).name: sha256(Path(p).read_bytes()).hexdigest() for p in readsb_files}
    review = {"selected_event_ids": [p.id for p in discovery.proposals], "excluded_event_ids": [],
              "notes": {}, "selection": "All automatically proposed groups; no manual event windows supplied."}
    destination = save_case(output or new_case_directory(), audio, sr, discovery, tracks, origin, metadata, review, result)
    return destination, discovery, result

