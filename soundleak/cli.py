import argparse
from pathlib import Path
import numpy as np

from .audio import extract_ridge
from .demo import make_demo, tracks_csv, wav_bytes
from .report import event_figure, heatmap_figure, summary, png_bytes, json_text
from .search import Event, search


def run_demo(output: Path):
    demo = make_demo()
    events = [Event(demo.tracks[name], extract_ridge(demo.audio, demo.sample_rate, start, end, low, high))
              for name, start, end, low, high in demo.windows]
    result = search(events, np.arange(-3000, 3001, 100), np.arange(-3000, 3001, 100), demo.truth_m[2])
    report = summary(result, events, demo.origin, demo.truth_m[2], demo.truth_m)
    output.mkdir(parents=True, exist_ok=True)
    (output / "synthetic.wav").write_bytes(wav_bytes(demo))
    (output / "tracks.csv").write_text(tracks_csv(demo))
    (output / "report.json").write_text(json_text(report))
    (output / "heatmap.png").write_bytes(png_bytes(heatmap_figure(result, events, demo.truth_m)))
    for index, (event, prediction) in enumerate(zip(events, result.predictions_hz), 1):
        (output / f"event-{index}.png").write_bytes(png_bytes(event_figure(event, prediction)))
    print(json_text(report))
    print(f"\nAudio, flight tracks and plots written to {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Soundleak aircraft Doppler experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run the deterministic, offline synthetic experiment")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    real = commands.add_parser("real", help="Replay downloaded real audio against two historical flight tracks")
    real.add_argument("--data", type=Path, default=Path("data/real/reference"))
    real.add_argument("--output", type=Path, default=Path("outputs/real/reference"))
    workbench = commands.add_parser("workbench", help="Discover acoustic events, compare supplied flights and save a research case")
    workbench.add_argument("recording", type=Path)
    workbench.add_argument("--tracks", type=Path)
    workbench.add_argument("--center", nargs=2, type=float, metavar=("LAT", "LON"), default=(0., 0.))
    workbench.add_argument("--start-utc", help="ISO recording-start UTC; needed for time_utc CSV")
    workbench.add_argument("--clock-evidence", default="Operator supplied; independently verify timing.")
    workbench.add_argument("--max-duration", type=int, default=3600)
    workbench.add_argument("--half-width-km", type=float, default=5.)
    workbench.add_argument("--grid", type=int, default=21)
    workbench.add_argument("--clock-sweep", type=float, default=5.)
    workbench.add_argument("--elevation", type=float, default=0.)
    workbench.add_argument("--output", type=Path)
    automatic = commands.add_parser("workbench-reference", help="Prepare the whole-recording automatic reference case (source-informed region/clock; limited flight pool)")
    automatic.add_argument("--output", type=Path)
    evaluation = commands.add_parser("evaluate", help="Compare a frozen case against a separate reference JSON after fitting")
    evaluation.add_argument("case", type=Path)
    evaluation.add_argument("--reference", type=Path, required=True)
    evaluation.add_argument("--output", type=Path)
    regional = commands.add_parser("fetch-regional", help="Download and verify a dated archive, then select regional aircraft by time/position")
    regional.add_argument("--case", type=Path, default=Path("outputs/cases/reference-automatic"))
    regional.add_argument("--archive", type=Path, default=Path("data/real/regional"))
    regional.add_argument("--output", type=Path, default=Path("data/real/regional/catalog-75km"))
    regional.add_argument("--radius-km", type=float, default=75.)
    regional.add_argument("--workers", type=int, default=12)
    expand = commands.add_parser("expand-catalog", help="Rerank frozen acoustic proposals against a regional archive catalog")
    expand.add_argument("--case", type=Path, default=Path("outputs/cases/reference-automatic"))
    expand.add_argument("--catalog", type=Path, default=Path("data/real/regional/catalog-75km"))
    expand.add_argument("--output", type=Path, default=Path("outputs/cases/reference-regional"))
    expand.add_argument("--max-models", type=int, default=100_000_000)
    background = commands.add_parser("background-controls", help="Evaluate real annotated non-aircraft environmental excerpts")
    background.add_argument("--data", type=Path, default=Path("data/real/controls"))
    background.add_argument("--fetch", action="store_true")
    background.add_argument("--output", type=Path, default=Path("outputs/evaluations/background-controls.json"))
    rejection = commands.add_parser("rejection-checks", help="Run real annotated background checks and wrong-day coverage rejection")
    rejection.add_argument("--case", type=Path, default=Path("outputs/cases/reference-regional"))
    rejection.add_argument("--background", type=Path, default=Path("data/real/controls"))
    rejection.add_argument("--output", type=Path, default=Path("outputs/evaluations/rejection-checks.json"))
    pilot = commands.add_parser("pilot-evaluate", help="Evaluate multiple frozen cases against separate reference files")
    pilot.add_argument("manifest", type=Path)
    pilot.add_argument("--output", type=Path, default=Path("outputs/evaluations/pilot.json"))
    public = commands.add_parser("public-aircraft-screen", help="Screen all 40 public Pathak recordings with frozen acoustic defaults")
    public.add_argument("--data", type=Path, default=Path("data/real/pathak"))
    public.add_argument("--fetch", action="store_true")
    public.add_argument("--output", type=Path, default=Path("outputs/evaluations/public-aircraft-screen.json"))
    review = commands.add_parser("prepare-public-review", help="Prepare verified spectrograms and ridge arrays for every public recording")
    review.add_argument("--report", type=Path, default=Path("outputs/evaluations/public-aircraft-screen.json"))
    review.add_argument("--data", type=Path, default=Path("data/real/pathak"))
    review.add_argument("--output", type=Path, default=Path("outputs/reviews/public-aircraft"))
    live = commands.add_parser("prepare-live", help="Prepare cumulative fits for synchronized recorded playback")
    live.add_argument("--case", type=Path, default=Path("outputs/cases/reference-regional"))
    live.add_argument("--output", type=Path, default=Path("outputs/live/regional"))
    live.add_argument("--grid", type=int, default=11)
    live.add_argument("--clocks", type=int, default=5)
    model = commands.add_parser("fetch-sound-model", help="Fetch the pinned YAMNet ONNX sound-event model")
    model.add_argument("--output", type=Path, default=Path("data/real/models/yamnet-onnx"))
    args = parser.parse_args()
    if args.command == "demo":
        run_demo(args.output)
    elif args.command == "real":
        from .real import run_reference
        _, _, _, report = run_reference(args.data, args.output,
                                        progress=lambda value: print(f"Regional search: {value:.0%}", flush=True))
        print(json_text(report))
    elif args.command in ("workbench", "workbench-reference"):
        from .workflow import analyze
        from .catalog import utc_timestamp
        if args.command == "workbench-reference":
            from .real import ORIGIN, RECORDING_START_UNIX_S, GEOID_UNDULATION_M
            root = Path("data/real/reference")
            destination, discovery, result = analyze(root/"original.webm", origin=ORIGIN,
                readsb_files=tuple(sorted(root.glob("trace_full_*.json"))), recording_start_unix_s=RECORDING_START_UNIX_S,
                clock_evidence="Inherited fitted clock from published reproduction", geoid_undulation_m=GEOID_UNDULATION_M,
                output=args.output, provenance={"scope": "Automatic discovery; source-informed region and clock; two-aircraft archive subset",
                    "audio_source": "https://www.youtube.com/watch?v=4CxVxuMVkMc",
                    "flight_source": "https://github.com/adsblol/globe_history_2026/releases/tag/v2026.09.08-planes-readsb-staging-0"},
                progress=lambda stage, p: print(f"{stage}: {p:.0%}", flush=True))
        else:
            destination, discovery, result = analyze(args.recording, origin=tuple(args.center), flight_csv=args.tracks,
                recording_start_unix_s=utc_timestamp(args.start_utc) if args.start_utc else None, clock_evidence=args.clock_evidence,
                max_duration_s=args.max_duration, half_width_m=args.half_width_km*1000, grid_points=args.grid,
                clock_sweep_s=args.clock_sweep, receiver_altitude_m=args.elevation, output=args.output,
                progress=lambda stage, p: print(f"{stage}: {p:.0%}", flush=True))
        print(f"Saved {destination}; {len(discovery.proposals)} acoustic proposals; status: {result.status if result else 'needs_flight_data_or_events'}")
    elif args.command == "evaluate":
        import json
        from .evaluation import evaluate
        report = evaluate(args.case, json.loads(args.reference.read_text()))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json_text(report))
        print(json_text(report))
    elif args.command == "fetch-regional":
        from datetime import datetime, timezone
        import json
        import time
        from .archive import fetch_daily_archive, extract_region
        r = json.loads((args.case/"report.json").read_text())
        start = r["provenance"]["recording_start_unix_s"]
        duration = r["discovery"]["duration_s"]
        day = datetime.fromtimestamp(start, timezone.utc).date()
        if any(datetime.fromtimestamp(t, timezone.utc).date() != day for t in (start-200, start+duration+60)):
            raise ValueError("This archive command currently supports cases whose padded time window lies within one UTC day.")
        if not 1 <= args.workers <= 32 or not 1 <= args.radius_km <= 150:
            raise ValueError("Use 1–32 download workers and a 1–150 km catalog radius.")
        last = [0.]
        def update(done, total):
            if time.monotonic()-last[0] > 15 or done == total:
                print(f"Downloaded {done/1e9:.2f}/{total/1e9:.2f} GB", flush=True)
                last[0] = time.monotonic()
        parts, metadata = fetch_daily_archive(day.isoformat(), args.archive, args.workers, progress=update)
        if (args.output/"manifest.json").exists():
            cached = json.loads((args.output/"manifest.json").read_text())
            if (cached.get("complete_archive_scan") and cached.get("origin_latitude_longitude") == r["origin_latitude_longitude"]
                and cached.get("radius_m") == args.radius_km*1000
                and cached.get("time_start_unix_s") == start-200
                and cached.get("time_end_unix_s") == start+duration+60
                and cached.get("source_release") == metadata["html_url"]):
                print(f"Reusing completed regional catalog: {len(cached['selected_aircraft'])} aircraft; {args.output}")
                return
            raise ValueError("The output directory contains a different catalog selection. Choose another --output directory.")
        manifest = extract_region(parts, args.output, tuple(r["origin_latitude_longitude"]), start, duration,
                                  radius_m=args.radius_km*1000, release_metadata=metadata,
                                  progress=lambda n, k: print(f"Scanned {n:,} trace records; selected {k:,} aircraft", flush=True))
        print(f"Catalog: {len(manifest['selected_aircraft'])} aircraft; {len(manifest['errors'])} scan errors; {args.output}")
    elif args.command == "expand-catalog":
        from .regional import expand_frozen_case
        _, comparison, catalog = expand_frozen_case(args.case, args.catalog, args.output, max_models=args.max_models,
                                                    progress=lambda p: print(f"Regional comparison: {p:.0%}", flush=True))
        print(json_text(comparison))
        print(f"Catalog: {catalog['usable_aircraft']} usable aircraft, {catalog['usable_flight_segments']} flight segments")
    elif args.command == "background-controls":
        from .controls import fetch_background_controls, run_background_controls
        if args.fetch:
            fetch_background_controls(args.data, progress=lambda n: print(f"Fetched background excerpt {n}", flush=True))
        r = run_background_controls(args.data, args.output)
        print(f"{len(r['cases'])} annotated background clips; {r['total_acoustic_proposals']} acoustic proposals; report: {args.output}")
    elif args.command == "rejection-checks":
        from .rejection import run_rejection_suite
        r = run_rejection_suite(args.case, args.background, args.output)
        print(f"Background acoustic proposals: {r['background']['total_acoustic_proposals']}; wrong-day status: {r['wrong_day_catalog']['status']}")
    elif args.command == "pilot-evaluate":
        from .pilot import evaluate_pilot
        report = evaluate_pilot(args.manifest, args.output)
        print(json_text(report))
    elif args.command == "public-aircraft-screen":
        from .public_data import fetch_public_aircraft, run_public_screen
        if args.fetch:
            fetch_public_aircraft(args.data, progress=lambda name: print(f"Verified {name}", flush=True))
        report = run_public_screen(args.data, args.output,
                                   progress=lambda n, total: print(f"Screened {n}/{total}", flush=True))
        print(f"{report['screened_cases']} screened; {report['cases_with_proposals']} with proposals; "
              f"{report['failed_cases']} failures. Acoustic component check only; report: {args.output}")
    elif args.command == "prepare-public-review":
        from .public_review import prepare_review_case, read_screen
        report, _ = read_screen(args.report)
        for i, row in enumerate(report['cases'], 1):
            if row['state'] == 'screened':
                prepare_review_case(row['id'], args.report, args.data, args.output)
                print(f"Prepared {i}/{len(report['cases'])}: {row['id']}", flush=True)
            else:
                print(f"No measurements for failed screen: {row['id']}", flush=True)
    elif args.command == "prepare-live":
        from .live import prepare_case_replay
        result = prepare_case_replay(args.case, args.output, args.grid, args.clocks,
                                     progress=lambda value: print(f"Replay preparation: {value:.0%}", flush=True))
        print(f"Prepared {len(result['states'])-1} observation updates; final status: {result['states'][-1]['status']}")
    elif args.command == "fetch-sound-model":
        from .sound_events import fetch_sound_model
        fetch_sound_model(args.output)
        print(f"Verified local sound model: {args.output}")


if __name__ == "__main__":
    main()

