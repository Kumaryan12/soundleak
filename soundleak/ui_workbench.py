"""Streamlit research workflow. Presentation only; analysis lives in core modules."""
from hashlib import sha256
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.io import wavfile
import streamlit as st

from .audio import decode_audio
from .association import fit_at_hypothesis, ranked_search
from .catalog import candidate_table, csv_candidates, readsb_candidates, utc_timestamp
from .cases import case_zip, load_case, make_report, new_case_directory, save_case
from .discovery import DiscoveryConfig, discover
from .real import GEOID_UNDULATION_M, ORIGIN, RECORDING_START_UNIX_S
from .report import association_figure, json_text, ridge_figure, timeline_figure
from .regional import load_regional_catalog
from .hosting import case_output_root, is_hosted, saved_case_reports

REGIONAL_CATALOG = Path("data/real/regional/catalog-75km")


def timestamp(seconds):
    return f"{int(seconds)//60:02d}:{seconds%60:04.1f}"


def proposal_label(p):
    return f"{timestamp(p.ridge.time_s[0])}–{timestamp(p.ridge.time_s[-1])} · {p.id}"


def _open_public_review():
    st.session_state['experiment_mode'] = 'Public recording review'


@st.cache_data(max_entries=2)
def read_saved(path):
    return load_case(path)


@st.cache_data(max_entries=2)
def portable_case(path):
    return case_zip(path)


@st.cache_data(max_entries=2)
def regional_tracks(origin, start_unix_s, duration_s, sound_speed, manifest_sha256):
    return load_regional_catalog(REGIONAL_CATALOG, origin, start_unix_s, duration_s,
                                 GEOID_UNDULATION_M, sound_speed)


def _set_case(values, label, directory=None):
    audio, sr, discovery, tracks, origin, provenance, review, result = values
    st.session_state["wb_case"] = {"audio": audio, "sr": sr, "discovery": discovery, "tracks": tracks,
                                   "origin": origin, "provenance": provenance, "review": review,
                                   "base_review": json.loads(json.dumps(review)), "result": result, "label": label,
                                   "directory": str(directory) if directory else None}
    for key in list(st.session_state):
        if key.startswith("wb_control_") or key.startswith("wb_editor_"):
            del st.session_state[key]
    st.session_state.pop("wb_saved_path", None)


def _reference_tracks(origin, duration, sound_speed, recording_start_unix_s):
    tracks, rejected = {}, []
    for path in sorted(Path("data/real/reference").glob("trace_full_*.json")):
        found, errors = readsb_candidates(path.read_text(), origin, recording_start_unix_s, duration,
                                         GEOID_UNDULATION_M, sound_speed=sound_speed)
        tracks.update(found)
        rejected.extend(errors)
    return tracks, rejected


def render_workbench():
    st.caption("RESEARCH WORKBENCH · Scan audio, review acoustic events, compare supplied flights, and retain competing locations.")
    saved = saved_case_reports()
    if "wb_case" not in st.session_state:
        prepared = Path("outputs/cases/reference-regional")
        if not (prepared/"manifest.json").exists():
            prepared = Path("outputs/cases/reference-automatic")
        if (prepared/"manifest.json").exists():
            try:
                _set_case(read_saved(str(prepared)), "Automatic scan · public reference recording", prepared)
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    with st.expander("Recording and saved cases", expanded="wb_case" not in st.session_state):
        if saved:
            col, button = st.columns([4, 1])
            current = st.session_state.get("wb_case", {}).get("directory")
            selected_index = next((i for i, p in enumerate(saved) if str(p.parent) == current), 0)
            choice = col.selectbox("Saved local research case", saved, index=selected_index,
                                   format_func=lambda p: p.parent.name)
            if button.button("Open case", width="stretch"):
                try:
                    _set_case(read_saved(str(choice.parent)), choice.parent.name, choice.parent)
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
        source = st.radio("Recording source", ["Bundled real recording", "Upload a recording"], horizontal=True)
        media = None
        if source == "Upload a recording":
            media = st.file_uploader("Audio or video · up to 100 MB", type=["wav", "mp3", "m4a", "mp4", "webm", "ogg", "flac", "mov"], key="wb_upload")
        else:
            st.caption("colsto’s public one-hour recording. No flight identities, coordinates, or event windows enter discovery.")
        c1, c2 = st.columns(2)
        length = c1.selectbox("Decode limit", [600, 1800, 3600, 7200], index=2, format_func=lambda x: f"{x//60} minutes")
        sensitivity = c2.selectbox("Acoustic proposal screen", ["Conservative", "Exploratory"])
        st.caption("Exploratory mode includes weaker and shorter patterns. Neither screen certifies an aircraft detection.")
        if st.button("Scan recording", type="primary", disabled=source == "Upload a recording" and media is None):
            try:
                config = DiscoveryConfig() if sensitivity == "Conservative" else DiscoveryConfig(min_duration_s=12., min_prominence_db=8., min_drop_fraction=.06)
                with st.spinner("Decoding local audio…"):
                    if source == "Bundled real recording":
                        path = Path("data/real/reference/original.webm")
                        audio, sr = decode_audio(path, max_duration_s=length)
                        provenance = json.loads(Path("data/real/reference/provenance.json").read_text())
                        provenance.update({"recording_start_unix_s": RECORDING_START_UNIX_S,
                                           "clock_source": "Inherited fitted clock from published reproduction",
                                           "scope": "Automatic discovery; source-informed region, clock and two-aircraft archive subset"})
                        name, origin = "Public reference recording", ORIGIN
                        digest = sha256(path.read_bytes()).hexdigest()
                    else:
                        contents = media.getvalue()
                        with tempfile.TemporaryDirectory() as folder:
                            path = Path(folder)/("recording"+Path(media.name).suffix)
                            path.write_bytes(contents)
                            audio, sr = decode_audio(path, max_duration_s=length)
                        name, origin = Path(media.name).name, (0., 0.)
                        digest = sha256(contents).hexdigest()
                        provenance = {"recording_name": name, "clock_source": "User must supply track alignment or recording UTC."}
                provenance.update({"input_sha256": digest, "decode_limit_s": length,
                                   "decoded_duration_s": len(audio)/sr, "sample_rate": sr,
                                   "discovery_inputs": "Audio samples and fixed acoustic screen only; no flight IDs, location or event hints."})
                bar = st.progress(0., text="Scanning the decoded recording…")
                discovery = discover(audio, sr, config, progress=bar.progress)
                bar.empty()
                _set_case((audio, sr, discovery, {}, origin, provenance,
                           {"selected_event_ids": [p.id for p in discovery.proposals], "notes": {}}, None), name)
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    case = st.session_state.get("wb_case")
    if case is None:
        st.info("Load a recording to begin. Event discovery runs before any flight data or location is supplied.")
        return
    discovery, audio, sr = case["discovery"], case["audio"], case["sr"]
    st.subheader(case["label"])
    if "source-informed" in case["provenance"].get("scope", ""):
        catalog = case["provenance"].get("regional_catalog")
        if catalog:
            st.info(f"Reference-case limits: the region and clock are source-informed. The regional archive selected {catalog['selected_aircraft_records']} aircraft records; "
                    f"{catalog['usable_aircraft']} aircraft passed the import checks. The original acoustic proposals are frozen.")
        else:
            st.info("Reference-case limits: the region and clock are source-informed, and the supplied archive contains only two aircraft. Acoustic discovery used audio alone.")
    comparison = case["provenance"].get("catalog_comparison")
    if comparison:
        with st.expander("What changed when the flight catalog expanded?", expanded=True):
            rows = []
            for label, key in [("Candidate flight segments", "candidate_flight_segments"), ("Matched acoustic groups", "matched_groups"),
                               ("Unmatched acoustic groups", "unmatched_groups"), ("Groups without complete coverage", "groups_without_complete_coverage")]:
                rows.append({"Measure": label, "Original two-aircraft pool": comparison["baseline"].get(key),
                             "Regional catalog": comparison["expanded"].get(key)})
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption("Acoustic measurements, review selection, spatial/clock grids and scoring thresholds were held fixed. More candidate flights can improve fit by chance; this comparison does not establish blind accuracy.")
    checks_path = Path("outputs/evaluations/rejection-checks.json")
    if checks_path.exists():
        with st.expander("Development rejection checks"):
            checks = json.loads(checks_path.read_text())
            st.dataframe([{"Control": c["id"], "Seconds": round(c["duration_s"]), "Acoustic proposals": c["proposal_count"]}
                          for c in checks["background"]["cases"]], hide_index=True, width="stretch")
            timing = checks["wrong_day_catalog"]
            st.write(f"Wrong-day catalog: {timing['status'].replace('_', ' ')}; {timing['retained_location_hypotheses']} retained location hypotheses.")
            st.caption("Six contiguous annotated background excerpts from AeroSonicDB (Blake Downward, CC BY-NC 4.0). "
                       "The wrong-day check tests missing-time-coverage rejection. These are small development checks, not a calibrated false-location rate.")
    public_path = Path("outputs/evaluations/public-aircraft-screen.json")
    if public_path.exists():
        with st.expander("Public aircraft recordings · acoustic screen"):
            public = json.loads(public_path.read_text())
            st.write(f"{public['screened_cases']}/{public['declared_cases']} recordings screened; "
                     f"{public['cases_with_proposals']} contain proposals; {public['cases_without_proposals']} have none; "
                     f"{public['failed_cases']} failed.")
            st.dataframe([{"Recording": c["id"], "State": c["state"], "Duration · s": round(c.get("duration_s", 0), 1),
                           "Acoustic proposals": c.get("proposal_count"), "Reason": c.get("reason", "")}
                          for c in public["cases"]], hide_index=True, width="stretch", height=260)
            st.caption("All 40 public Pathak passages (CC BY 4.0), with unchanged discovery defaults and no author-selected ridges or flight data supplied to detection. "
                       "Proposal presence is not confirmed aircraft identification. Microphone coordinates and WAV sample-zero UTC have not been established for an independent location pilot.")
            st.download_button("Download public acoustic-screen report", public_path.read_bytes(),
                               "public-aircraft-screen.json", "application/json")
            st.button("Review public recordings", on_click=_open_public_review)
    st.caption(f"Analyzed {discovery.duration_s/60:.1f} minutes · {discovery.scanned_windows} overlapping windows · "
               f"{len(discovery.proposals)} grouped acoustic proposals · {discovery.elapsed_s:.1f} s scan time")
    if not discovery.proposals:
        st.warning("Insufficient evidence: no moving-tone proposals passed this acoustic screen. This does not prove that no aircraft were present.")
    st.subheader("1 · Review acoustic events")
    rows = []
    prior = set(case["base_review"].get("selected_event_ids", [p.id for p in discovery.proposals]))
    notes = case["base_review"].get("notes", {})
    for p in discovery.proposals:
        m = p.metadata()
        rows.append({"Use": p.id in prior, "Event": p.id, "Start": timestamp(m["start_s"]), "End": timestamp(m["end_s"]),
                     "Tone fit · Hz": round(m["tone_fit_rmse_hz"], 2), "Prominence · dB": round(m["prominence_db"], 1),
                     "Grouped ridges": p.members, "Review note": notes.get(p.id, "")})
    if rows:
        reviewed = st.data_editor(rows, key="wb_editor_"+case["label"], hide_index=True, width="stretch", height=260,
                                  disabled=["Event", "Start", "End", "Tone fit · Hz", "Prominence · dB", "Grouped ridges"])
        selected = [p for p, row in zip(discovery.proposals, reviewed) if row["Use"]]
        selected_ids = {p.id for p in selected}
        review = {"selected_event_ids": [p.id for p in selected], "excluded_event_ids": [p.id for p in discovery.proposals if p.id not in selected_ids],
                  "notes": {r["Event"]: r["Review note"] for r in reviewed if r["Review note"]},
                  "selection": "Explicit workbench review; default is all automatic proposals."}
        case["review"] = review
        by_id = {p.id: p for p in discovery.proposals}
        default_id = case["result"].hypotheses[0].matches[0]["event_id"] if case["result"] and case["result"].hypotheses else next(iter(by_id))
        proposal_id = st.selectbox("Inspect acoustic group", list(by_id), index=list(by_id).index(default_id), format_func=lambda key: proposal_label(by_id[key]), key="wb_control_event")
        proposal = by_id[proposal_id]
        st.pyplot(timeline_figure(audio, sr, discovery.proposals, proposal.id), width="stretch")
    else:
        selected, proposal = [], None

    st.subheader("2 · Supply candidate flights and search bounds")
    with st.expander("Flight inputs and search settings", expanded=case["result"] is None):
        options = ["Saved candidate tracks", "Bundled archive subset (2 aircraft)", "Upload flight CSV"]
        if (REGIONAL_CATALOG/"manifest.json").exists():
            options.insert(2, "Regional archive catalog")
        pool = st.selectbox("Candidate flight source", options, index=0 if case["tracks"] else 1, key="wb_control_pool")
        if pool.startswith("Bundled"):
            st.info("This is a limited two-aircraft pool from the published example. It does not represent complete regional air traffic.")
        track_file = st.file_uploader("Flight CSV", type=["csv"], key="wb_tracks") if pool == "Upload flight CSV" else None
        settings = case["result"].settings if case["result"] is not None else {}
        a, b, c = st.columns(3)
        latitude = a.number_input("Search-center latitude", -80., 80., float(case["origin"][0]), format="%.5f", key="wb_control_lat")
        longitude = b.number_input("Search-center longitude", -179., 179., float(case["origin"][1]), format="%.5f", key="wb_control_lon")
        altitude = c.number_input("Recorder elevation · m MSL", -500., 8000., float(settings.get("receiver_altitude_m", 0.)), key="wb_control_alt")
        a, b, c = st.columns(3)
        default_radius = float(max(abs(settings.get("east_m", [-5000, 5000])[0]), abs(settings.get("east_m", [-5000, 5000])[-1]))/1000)
        radius = a.number_input("Search half-width · km", 1., 15., default_radius, step=1., key="wb_control_radius")
        n = b.select_slider("Spatial grid points per axis", [11, 21, 31, 41], value=len(settings.get("east_m", list(range(21)))), key="wb_control_n")
        c_limit = c.select_slider("Shared clock sweep · ± seconds", [0, 2, 5, 10, 20], value=5, key="wb_control_clock")
        a, b = st.columns(2)
        threshold = a.number_input("Maximum event mismatch · Hz", .25, 10., float(settings.get("max_event_rmse_hz", 3.)), step=.25, key="wb_control_threshold")
        sound_speed = b.number_input("Sound speed · m/s", 300., 370., float(settings.get("sound_speed_mps", 343.)), key="wb_control_sound")
        max_models = st.number_input("Maximum model comparisons", 300_000, 1_000_000_000, int(settings.get("max_model_evaluations", 300_000)),
                                     step=100_000, key="wb_control_models")
        start_unix = case["provenance"].get("recording_start_unix_s")
        default_clock = datetime.fromtimestamp(start_unix, timezone.utc).isoformat().replace("+00:00", "Z") if start_unix is not None else ""
        clock = st.text_input("Recording-start UTC · needed for time_utc CSV", value=default_clock, key="wb_control_utc", disabled=pool == "Saved candidate tracks")
        clock_evidence = st.text_input("Clock evidence / assumptions", value=case["provenance"].get("clock_source", ""), key="wb_control_clock_evidence")
        st.caption("CSV columns: aircraft_id, time_s, latitude, longitude, altitude_m. "
                   "Alternatively use time_utc plus recording-start UTC. Heights must be MSL meters. "
                   "Gaps over 30 s split trajectories; invalid segments are listed. The sweep tests 21 shared clock offsets.")
    origin = (latitude, longitude)
    try:
        rejected = []
        if pool == "Saved candidate tracks":
            if origin != case["origin"]:
                raise ValueError("Saved tracks use the saved coordinate origin. Restore that center or reimport the flight source for the new center.")
            tracks = case["tracks"]
        elif pool.startswith("Bundled"):
            if not clock.strip():
                raise ValueError("The native historical archive needs a recording-start UTC to align its timestamps.")
            tracks, rejected = _reference_tracks(origin, discovery.duration_s, sound_speed, utc_timestamp(clock))
        elif pool == "Regional archive catalog":
            if not clock.strip():
                raise ValueError("Regional archive tracks need a recording-start UTC.")
            manifest_hash = sha256((REGIONAL_CATALOG/"manifest.json").read_bytes()).hexdigest()
            tracks, regional_summary = regional_tracks(origin, utc_timestamp(clock), discovery.duration_s, sound_speed, manifest_hash)
            rejected = regional_summary["rejected_segments"]
            st.caption(f"Regional selection: {regional_summary['selection_radius_m']/1000:.0f} km around the catalog center; "
                       f"{regional_summary['selected_aircraft_records']} reported aircraft, {regional_summary['usable_aircraft']} usable. Provider and altitude coverage remain limited.")
        elif track_file:
            tracks, rejected = csv_candidates(track_file.getvalue().decode("utf-8-sig"), origin,
                                               utc_timestamp(clock) if clock.strip() else None, sound_speed=sound_speed)
        else:
            tracks = {}
        st.caption(f"{len(selected)} acoustic groups selected · {len(tracks)} candidate flight segments available")
        with st.expander("Inspect candidate-track coverage and rejected imports"):
            st.dataframe(candidate_table(tracks), hide_index=True, width="stretch")
            if rejected:
                st.dataframe(rejected, hide_index=True, width="stretch")
            st.write("A low score can only compare the supplied flights. Missing aircraft can leave real flyovers unmatched.")
        grid = np.linspace(-radius*1000, radius*1000, n)
        offsets = np.linspace(-c_limit, c_limit, 21) if c_limit else np.array([0.])
        signature = sha256(json.dumps({"groups": [p.id for p in selected], "origin": origin, "radius": radius, "n": n,
                                      "clock": c_limit, "threshold": threshold, "sound": sound_speed, "alt": altitude,
                                      "tracks": {k: sha256(t.xyz_m.tobytes()+t.time_s.tobytes()).hexdigest() for k, t in tracks.items()}}).encode()).hexdigest()
        if "signature" not in case and case["result"] is not None:
            case["signature"] = signature
        if st.button("Compare flight candidates & rank locations", type="primary", disabled=not selected or not tracks):
            bar = st.progress(0., text="Comparing every selected acoustic group under a shared clock…")
            result = ranked_search(selected, tracks, grid, grid, offsets, altitude, sound_speed, threshold,
                                    max_models=max_models, backend="batched", progress=bar.progress)
            bar.empty()
            case.update({"tracks": tracks, "origin": origin, "result": result, "signature": signature})
            case["provenance"].update({"candidate_pool": pool, "clock_source": clock_evidence,
                                       "entered_recording_start_utc": clock,
                                       "rejected_track_segments": rejected})
            st.session_state.pop("wb_saved_path", None)
        result = case["result"] if case.get("signature") == signature else None
        if case["result"] is not None and result is None:
            st.info("Search inputs changed. Run the comparison again to update the hypotheses.")
    except (ValueError, UnicodeError) as exc:
        st.error(str(exc))
        tracks, result = {}, None

    st.subheader("3 · Inspect location hypotheses")
    hypothesis = None
    if result is not None:
        st.warning(result.status.replace("_", " ").capitalize()+" · "+result.reasons[0])
        if result.hypotheses:
            table = []
            for i, h in enumerate(result.hypotheses, 1):
                m = h.metadata(origin)
                table.append({"Rank": i, "Latitude": round(m["latitude"], 5), "Longitude": round(m["longitude"], 5),
                              "Matched groups": len(h.matches), "Unmatched": len(h.unmatched), "Clock · s": h.clock_offset_s,
                              "Matched RMSE · Hz": round(h.rmse_hz, 3), "Capped objective · Hz": round(h.objective_hz, 3)})
            st.dataframe(table, hide_index=True, width="stretch")
            index = st.selectbox("Inspect hypothesis", range(len(result.hypotheses)), format_func=lambda i: f"Rank {i+1}", key="wb_hypothesis")
            hypothesis = result.hypotheses[index]
            left, right = st.columns([1.25, 1])
            with left:
                st.pyplot(association_figure(result, tracks, hypothesis), width="stretch")
            with right:
                st.write(f"{len(hypothesis.matches)} matched groups; {len(hypothesis.unmatched)} remain unexplained.")
                st.dataframe(hypothesis.matches, hide_index=True, width="stretch")
                st.caption("Map colors show a capped event-level objective. Each unmatched group contributes the rejection threshold. "
                           "Grid spacing and score gaps are not calibrated position uncertainty.")
                for reason in result.reasons[1:]:
                    st.caption(reason)
            st.caption(f"{result.evaluated_models:,} model evaluations · {result.elapsed_s:.1f} seconds · finite regional grid")
        else:
            st.info("No flight association passed the mismatch and trend-explanation thresholds. No location is supported by these inputs.")
    elif selected:
        st.info("Review the events and supply flight candidates, then run the regional comparison.")

    if proposal is not None:
        st.subheader("Inspect the selected acoustic event")
        prediction = None
        if hypothesis is not None:
            match = next((m for m in hypothesis.matches if m["event_id"] == proposal.id), None)
            names = list(tracks)
            default = names.index(match["candidate"]) if match else 0
            name = st.selectbox("Overlay a candidate flight at this hypothesis", names, index=default, key=f"wb_overlay_{proposal.id}_{index}")
            prediction = fit_at_hypothesis(proposal, tracks[name], hypothesis, altitude, sound_speed)
            if prediction is None:
                st.caption("This candidate does not cover the complete acoustic event at this shared clock.")
            else:
                st.caption(f"Overlay mismatch: {np.sqrt(np.mean((prediction-proposal.ridge.frequency_hz)**2)):.2f} Hz. "
                           +(f"Assigned candidate: {match['candidate']}." if match else "This event is unmatched at the selected hypothesis."))
            with st.expander("Compare all candidate flights for this event"):
                comparisons = []
                for candidate_name, track in tracks.items():
                    predicted = fit_at_hypothesis(proposal, track, hypothesis, altitude, sound_speed)
                    if predicted is None:
                        comparisons.append({"Candidate": candidate_name, "Complete coverage": False, "RMSE · Hz": None, "Trend explained": None})
                    else:
                        mse = float(np.mean((predicted-proposal.ridge.frequency_hz)**2))
                        comparisons.append({"Candidate": candidate_name, "Complete coverage": True, "RMSE · Hz": round(np.sqrt(mse), 3),
                                            "Trend explained": round(1-mse/max(float(np.var(proposal.ridge.frequency_hz)), 1e-12), 3)})
                st.dataframe(comparisons, hide_index=True, width="stretch")
        st.pyplot(ridge_figure(proposal.ridge, proposal_label(proposal), prediction, prediction_label="Candidate at inspected hypothesis"), width="stretch")
        buffer = io.BytesIO()
        start, end = proposal.ridge.time_s[[0, -1]]
        wavfile.write(buffer, sr, audio[int(max(0, start-1)*sr):int(min(len(audio)/sr, end+1)*sr)])
        st.audio(buffer.getvalue(), format="audio/wav")
        st.caption("Acoustic-fit quality screens a moving tone. It does not establish aircraft identity.")

    st.subheader("Save and inspect the evidence")
    with st.expander("Input provenance and all acoustic proposals"):
        st.json(case["provenance"])
        st.json(discovery.metadata())
    active_provenance = {**case["provenance"], "candidate_pool": pool, "clock_source": clock_evidence,
                         "entered_recording_start_utc": clock, "rejected_track_segments": rejected}
    if track_file:
        active_provenance["flight_csv_sha256"] = sha256(track_file.getvalue()).hexdigest()
    report = make_report(discovery, result, tracks, origin, active_provenance, case["review"])
    st.download_button("Download analysis report", json_text(report), "soundleak-workbench.json", "application/json", on_click="ignore")
    if is_hosted():
        st.caption("Saved cases are private to this browser session and temporary. Download a portable bundle to keep your work.")
    if st.button("Save research case locally" if not is_hosted() else "Save research case for this session"):
        with st.spinner("Saving decoded audio, measurements, tracks, review decisions and hypotheses…"):
            path = save_case(new_case_directory(case_output_root()), audio, sr, discovery, tracks, origin, active_provenance, case["review"], result)
            st.session_state["wb_saved_path"] = str(path)
            case["directory"] = str(path)
    if st.session_state.get("wb_saved_path"):
        path = st.session_state["wb_saved_path"]
        st.success(f"Saved case: {path}")
        st.download_button("Download portable case bundle", portable_case(path), Path(path).name+".zip", "application/zip", on_click="ignore")
    if case.get("directory"):
        with st.expander("Post-fit evaluation against a separate reference"):
            st.caption("The reference is used only to evaluate the saved frozen report. It is never supplied to discovery or search.")
            st.caption("Frozen snapshot: "+case["directory"])
            reference = st.file_uploader("Reference JSON · evaluation only", type=["json"], key="wb_evaluation_reference")
            st.code('{"latitude": 40.8766, "longitude": -73.5247, "kind": "public_map_marker", "independent_holdout": false, "coordinate_evidence": "source URL", "recording_clock_evidence": "timing source"}', language="json")
            if reference is not None and st.button("Evaluate frozen case"):
                try:
                    from .evaluation import evaluate
                    evaluation = evaluate(case["directory"], json.loads(reference.getvalue()))
                    st.json(evaluation)
                    st.download_button("Download evaluation", json_text(evaluation), "soundleak-evaluation.json", "application/json", on_click="ignore")
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    st.error(str(exc))
