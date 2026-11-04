"""Inspectable rejection checks, with limits stated alongside each control."""
import json
from pathlib import Path

from .archive import file_sha256
from .association import ranked_search
from .cases import load_case
from .controls import run_background_controls
from .discovery import DiscoveryConfig
from .physics import Track
from .regional import proposal_fingerprint


def run_rejection_suite(case_directory, background_directory, output):
    source = Path(case_directory)
    _, _, discovery, tracks, _, _, review, baseline = load_case(source)
    if not baseline:
        raise ValueError("Rejection checks require a searched frozen case.")
    config = DiscoveryConfig(**discovery.config)
    background = run_background_controls(background_directory, Path(output).with_name("background-controls.json"), config)
    selected = [p for p in discovery.proposals if p.id in set(review["selected_event_ids"])]
    shifted = {key: Track(track.name, track.time_s+86400., track.xyz_m.copy()) for key, track in tracks.items()}
    s = baseline.settings
    result = ranked_search(selected, shifted, s["east_m"], s["north_m"], s["clock_offsets_s"],
                           s["receiver_altitude_m"], s["sound_speed_mps"], s["max_event_rmse_hz"], backend="batched")
    timing = {"name": "Wrong-day flight catalog", "source_case_report_sha256": file_sha256(source/"report.json"),
              "frozen_proposal_fingerprint": proposal_fingerprint(discovery),
              "catalog_timestamp_shift_s": 86400., "status": result.status,
              "retained_location_hypotheses": len(result.hypotheses), "evaluated_models": result.evaluated_models,
              "expected_behavior": "Insufficient evidence because no provided flight spans the recording time.",
              "passed_timing_gate_check": result.status == "insufficient_evidence" and not result.hypotheses,
              "interpretation": "Tests missing-time-coverage rejection only; not resistance to similar but incorrect Doppler curves."}
    report = {"background": background, "wrong_day_catalog": timing,
              "interpretation": "Small development rejection suite. No calibrated false-positive rate or independent blind-geolocation accuracy claim."}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
