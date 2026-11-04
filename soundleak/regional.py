"""Expand a frozen case's candidate catalog and compare the same observations."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .archive import decode_trace, file_sha256
from .association import ranked_search
from .cases import load_case, save_case
from .catalog import readsb_candidates


def proposal_fingerprint(discovery):
    digest = sha256(json.dumps(discovery.config, sort_keys=True).encode())
    for proposal in discovery.proposals:
        digest.update(proposal.id.encode())
        for name in ("time_s", "frequency_hz", "prominence_db", "frequency_bins_hz", "power_db"):
            data = np.ascontiguousarray(getattr(proposal.ridge, name))
            digest.update(str((data.dtype.str, data.shape)).encode())
            digest.update(data.tobytes())
    return digest.hexdigest()


def load_regional_catalog(directory, origin, recording_start_unix_s, duration_s,
                          geoid_undulation_m, sound_speed=343., progress=None):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    if not manifest.get("complete_archive_scan"):
        raise ValueError("Regional catalog extraction is incomplete.")
    tracks, rejected, supported = {}, [], []
    entries = manifest["selected_aircraft"]
    root = directory.resolve()
    for i, entry in enumerate(entries):
        path = (directory/entry["file"]).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != entry["sha256"]:
            raise ValueError("Regional trace integrity check failed.")
        data = decode_trace(path.read_bytes())
        try:
            found, errors = readsb_candidates(data, origin, recording_start_unix_s, duration_s,
                                               geoid_undulation_m, sound_speed=sound_speed)
            tracks.update(found)
            rejected.extend(errors)
            if found:
                supported.append(entry["icao"])
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({"id": entry["icao"], "reason": str(exc)})
        if progress and (i+1) % 50 == 0:
            progress(i+1, len(entries))
    summary = {"source_release": manifest["source_release"], "license": manifest["license"],
               "trace_members_scanned": manifest["trace_members_scanned"],
               "selected_aircraft_records": len(entries), "usable_aircraft": len(set(supported)),
               "usable_flight_segments": len(tracks), "usable_position_samples": sum(len(t.time_s) for t in tracks.values()),
               "rejected_segments": rejected, "archive_scan_errors": manifest["errors"],
               "selection_complete_without_errors": manifest["selection_complete_without_errors"],
               "selection_origin_latitude_longitude": manifest["origin_latitude_longitude"],
               "selection_radius_m": manifest["radius_m"],
               "selection_time_start_unix_s": manifest["time_start_unix_s"],
               "selection_time_end_unix_s": manifest["time_end_unix_s"],
               "manifest_sha256": file_sha256(directory/"manifest.json"),
               "limitations": ["Coverage is limited to the provider's reported positions, time window and declared regional selection.",
                               "Geometric-altitude and trajectory-quality checks may reject aircraft or segments; no silent barometric-height substitution.",
                               "A lower residual with more candidates does not establish greater location accuracy."]}
    return tracks, summary


def _search_summary(result):
    if not result or not result.hypotheses:
        return {"status": result.status if result else "not_searched", "matched_groups": 0,
                "unmatched_groups": None, "groups_without_complete_coverage": None}
    best = result.hypotheses[0]
    return {"status": result.status, "matched_groups": len(best.matches), "unmatched_groups": len(best.unmatched),
            "groups_without_complete_coverage": len(best.unavailable),
            "distinct_matched_flight_segments": len({m["candidate"] for m in best.matches}),
            "objective_hz": best.objective_hz, "matched_event_rmse_hz": best.rmse_hz,
            "east_m": best.east_m, "north_m": best.north_m, "clock_offset_s": best.clock_offset_s}


def expand_frozen_case(case_directory, catalog_directory, output, geoid_undulation_m=-31.26,
                       max_models=100_000_000, progress=None):
    source = Path(case_directory)
    audio, sr, discovery, baseline_tracks, origin, provenance, review, baseline = load_case(source)
    if baseline is None:
        raise ValueError("The comparison requires a previously searched frozen case.")
    clock = provenance.get("recording_start_unix_s")
    if clock is None:
        raise ValueError("The frozen case needs recording-start UTC evidence to align the archive.")
    settings = baseline.settings
    fingerprint = proposal_fingerprint(discovery)
    tracks, catalog = load_regional_catalog(catalog_directory, origin, clock, discovery.duration_s,
                                             geoid_undulation_m, settings["sound_speed_mps"])
    if not tracks:
        raise ValueError("No regional flight segments passed the documented import checks.")
    retained, changed = [], []
    for key, baseline_track in baseline_tracks.items():
        if key in tracks and np.array_equal(tracks[key].time_s, baseline_track.time_s) and np.array_equal(tracks[key].xyz_m, baseline_track.xyz_m):
            retained.append(key)
        else:
            changed.append(key)
    chosen = set(review["selected_event_ids"])
    selected = [p for p in discovery.proposals if p.id in chosen]
    result = ranked_search(selected, tracks, settings["east_m"], settings["north_m"], settings["clock_offsets_s"],
                           settings["receiver_altitude_m"], settings["sound_speed_mps"], settings["max_event_rmse_hz"],
                           max_models=max_models, backend="batched", progress=progress)
    if proposal_fingerprint(discovery) != fingerprint:
        raise RuntimeError("Frozen acoustic measurements changed during catalog expansion.")
    comparison = {"baseline_case": str(source), "baseline_report_sha256": file_sha256(source/"report.json"),
                  "frozen_measurements_archive_sha256": file_sha256(source/"measurements.npz"),
                  "frozen_proposal_fingerprint": fingerprint, "acoustic_proposals_unchanged": True,
                  "baseline_segments_retained_exactly": retained, "baseline_segments_missing_or_changed": changed,
                  "catalog_only_change_verified": not changed,
                  "selected_event_ids": [p.id for p in selected],
                  "baseline": {"candidate_flight_segments": len(baseline_tracks), **_search_summary(baseline)},
                  "expanded": {"candidate_flight_segments": len(tracks), **_search_summary(result)},
                  "settings_held_fixed": ["Acoustic proposals and review selection", "Spatial search grid", "Shared clock grid",
                                          "Receiver elevation", "Sound speed", "Mismatch and trend thresholds", "Capped event-level objective"],
                  "interpretation": "This compares candidate-pool coverage for a development case. Improved fit is not independent blind validation."}
    provenance = {**provenance,
                  "scope": "Frozen automatic acoustic proposals; source-informed region and clock; regional archive selection",
                  "candidate_pool_kind": "regional_archive", "candidate_pool": "Regional archive catalog",
                  "regional_catalog": catalog, "catalog_comparison": comparison,
                  "geoid_undulation_m_approximation": geoid_undulation_m}
    save_case(output, audio, sr, discovery, tracks, origin, provenance, review, result)
    return result, comparison, catalog
