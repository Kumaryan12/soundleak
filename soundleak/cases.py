"""Local immutable research cases with portable inputs and review history."""
from datetime import datetime, timezone
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import uuid
import zipfile

import numpy as np

from .audio import Ridge
from .association import AssociationResult, Hypothesis
from .discovery import DiscoveryResult, Proposal
from .physics import Track, latlon


def tracks_csv(tracks, origin):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["aircraft_id", "time_s", "latitude", "longitude", "altitude_m"])
    for name, track in tracks.items():
        lat, lon = latlon(track.xyz_m[:, 0], track.xyz_m[:, 1], *origin)
        for t, a, b, z in zip(track.time_s, lat, lon, track.xyz_m[:, 2]):
            writer.writerow([name, t, a, b, z])
    return stream.getvalue()


def make_report(discovery, result, tracks, origin, provenance, review):
    runtime_code = {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ("audio.py", "physics.py", "discovery.py", "catalog.py", "association.py")}
    analysis_code = dict(runtime_code)
    # Re-exporting frozen results must not relabel them as a new analysis run.
    previous = getattr(result, "analysis_code_sha256", None) if result is not None else None
    if previous:
        analysis_code.update(previous)
    else:
        original_discovery = getattr(discovery, "analysis_code_sha256", {})
        analysis_code.update({k: v for k, v in original_discovery.items() if k in ("audio.py", "discovery.py")})
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_version": "soundleak-workbench-0.2",
            "analysis_code_sha256": analysis_code,
            "export_runtime_code_sha256": runtime_code,
            "provenance": provenance, "origin_latitude_longitude": list(origin),
            "discovery": discovery.metadata(), "review": review,
            "candidate_flight_segments": len(tracks),
            "search": None if result is None else {
                "status": result.status, "reasons": result.reasons, "settings": result.settings,
                "evaluated_models": result.evaluated_models, "elapsed_s": result.elapsed_s,
                "hypotheses": [h.metadata(origin) for h in result.hypotheses]},
            "interpretation": "Finite regional candidate fits from supplied tracks. No calibrated location probabilities or independent blind-validation claim."}


def save_case(directory, audio, sample_rate, discovery, tracks, origin, provenance, review, result=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    report = make_report(discovery, result, tracks, origin, provenance, review)
    (directory/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(directory/"audio.npz", audio=np.asarray(audio, dtype=np.float32), sample_rate=sample_rate)
    arrays = {}
    for p in discovery.proposals:
        for key in ("time_s", "frequency_hz", "prominence_db", "frequency_bins_hz", "power_db"):
            arrays[p.id+"_"+key] = getattr(p.ridge, key)
    np.savez_compressed(directory/"measurements.npz", **arrays)
    # Preserve local coordinates exactly, as well as a human-readable geodetic CSV.
    track_arrays = {}
    for i, track in enumerate(tracks.values()):
        track_arrays[f"t{i}"] = track.time_s
        track_arrays[f"xyz{i}"] = track.xyz_m
    np.savez_compressed(directory/"tracks.npz", **track_arrays)
    (directory/"track-names.json").write_text(json.dumps(list(tracks)))
    (directory/"tracks.csv").write_text(tracks_csv(tracks, origin))
    if result is not None:
        np.savez_compressed(directory/"surface.npz", east_m=result.east_m, north_m=result.north_m, surface_hz=result.surface_hz)
    (directory/"manifest.json").write_text(json.dumps({p.name: sha256(p.read_bytes()).hexdigest()
                                                       for p in directory.iterdir() if p.is_file()}, indent=2))
    return directory


def new_case_directory(root="outputs/cases"):
    return Path(root)/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8])


def load_case(directory):
    directory = Path(directory)
    if (directory/"manifest.json").exists():
        for name, expected in json.loads((directory/"manifest.json").read_text()).items():
            if Path(name).name != name or sha256((directory/name).read_bytes()).hexdigest() != expected:
                raise ValueError("Saved case failed its file-integrity check: "+name)
    report = json.loads((directory/"report.json").read_text())
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported case format.")
    with np.load(directory/"audio.npz", allow_pickle=False) as bundle:
        audio, sr = bundle["audio"], int(bundle["sample_rate"])
    with np.load(directory/"measurements.npz", allow_pickle=False) as bundle:
        proposals = []
        for meta in report["discovery"]["events"]:
            identity = meta["id"]
            ridge = Ridge(*(bundle[identity+"_"+key] for key in
                            ("time_s", "frequency_hz", "prominence_db", "frequency_bins_hz", "power_db")))
            fit = {k: meta[k] for k in ("tone_fit_rmse_hz", "trend_explained_fraction", "fitted_tone_hz", "approx_cpa_receive_s", "fit_converged")}
            proposals.append(Proposal(identity, ridge, fit, meta["grouped_proposals"]))
    meta = report["discovery"]
    discovery = DiscoveryResult(proposals, meta["duration_s"], meta["scanned_windows"], meta["raw_proposals"], meta["config"], meta["elapsed_s"])
    discovery.analysis_code_sha256 = report.get("analysis_code_sha256", {})
    names = json.loads((directory/"track-names.json").read_text())
    with np.load(directory/"tracks.npz", allow_pickle=False) as bundle:
        tracks = {name: Track(name, bundle[f"t{i}"], bundle[f"xyz{i}"]) for i, name in enumerate(names)}
    result = None
    if report["search"] is not None:
        s = report["search"]
        hypotheses = [Hypothesis(h["east_m"], h["north_m"], h["clock_offset_s"], h["capped_objective_hz"],
                                  h["matches"], h["unmatched_groups"], h["groups_without_complete_track_coverage"])
                      for h in s["hypotheses"]]
        with np.load(directory/"surface.npz", allow_pickle=False) as bundle:
            result = AssociationResult(hypotheses, bundle["east_m"], bundle["north_m"], bundle["surface_hz"],
                                       s["status"], s["reasons"], s["settings"], s["evaluated_models"], s["elapsed_s"])
            result.analysis_code_sha256 = report.get("analysis_code_sha256", {})
    return audio, sr, discovery, tracks, tuple(report["origin_latitude_longitude"]), report["provenance"], report["review"], result


def case_zip(directory):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in Path(directory).iterdir():
            if file.is_file():
                archive.write(file, arcname=file.name)
    return buffer.getvalue()
