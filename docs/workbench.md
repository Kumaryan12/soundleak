# Research workbench

Run `.venv/bin/streamlit run app.py` and open http://127.0.0.1:8501. The default view supports an entire research case, from a recording through reviewed measurements and competing location hypotheses. The guided replay remains available in the sidebar.

## Recording and acoustic discovery

Choose the bundled public recording, upload a local audio/video file, or reopen a local saved case. Uploads are capped at 100 MB; the decoded analysis limit is selected explicitly, up to 120 minutes. The limit and actual analyzed duration are recorded. Processing is local and requires FFmpeg.

Discovery receives audio samples and a fixed configuration only. Aircraft IDs, positions, recording UTC and event windows are not inputs. The conservative defaults scan overlapping 90-second windows at 60-second intervals across fixed overlapping bands from 40 to 800 Hz. The final partial window is included. The scanner reuses each window's spectrogram to limit memory use.

A dynamic-programming tracker proposes descending tonal ridges. The conservative screen requires at least 20 seconds, median prominence of 12 dB over the band background, a sufficient frequency drop, and a reasonably fitting constant-speed acoustic Doppler curve. These are explicit heuristics rather than a calibrated aircraft classifier. A second exploratory preset includes weaker and shorter patterns. Configuration and every retained proposal are exported.

Overlapping curves and apparent low-order harmonics are grouped, with substantial overlap required to avoid merging unrelated adjacent events. One actual ridge represents each group. Harmonic relationships remain hypotheses; the reviewer must inspect them. Static tones, silence and seeded white-noise controls produce no proposals in the regression checks, but these controls do not establish a real-world false-positive rate.

## Review and flight candidates

Each proposed group has a spectrogram, audio excerpt, diagnostic tone-fit error and prominence. The full-recording timeline shows where proposals occur. Uncheck **Use** to exclude a group, and record review notes. Selection and exclusions are saved. Low acoustic-fit error alone does not establish an aircraft identity.

Supply a candidate-flight CSV, use tracks from a saved case, select the prepared regional archive catalog, or use the bundled two-aircraft subset. The regional catalog selected 628 aircraft records by time and position, and 587 aircraft passed import checks, yielding 730 flight segments. Provider coverage and geometric-altitude availability remain limited. The recording clock and regional bounds in the reference case come from the earlier reproduction; the acoustic proposals were frozen before expanding the flight catalog.

CSV columns are `aircraft_id,time_s,latitude,longitude,altitude_m`. Time is seconds relative to the recording's first decoded sample. Alternatively use an ISO `time_utc` column and supply recording-start UTC with an explicit timezone. Aircraft and recorder heights must share an MSL vertical datum. Native readsb inputs used by the reference workflow convert WGS84 geometric heights using a disclosed constant geoid approximation.

Data gaps over 30 seconds split trajectories. Segments with too few samples, missing geometric-height coverage or sonic/supersonic implied speeds are rejected explicitly. Inspect the candidate coverage table and rejected-import records before searching.

## Matching and alternatives

The workbench evaluates every selected group under each candidate listening position and shared clock offset. It includes sound propagation delay. One constant source tone is fitted per group/candidate. A proposed association needs complete sample coverage, sufficiently low frequency RMSE, and at least 60% improvement in explaining the measured frequency variation relative to a constant frequency.

Every selected group contributes equally to the objective. Unmatched groups receive a fixed penalty equal to the selected mismatch threshold squared. Matched group errors are capped at that threshold; the displayed objective is the square root of the mean capped event-level squared errors. Scores use every fourth 0.25-second frame. Spectrogram overlays use the full measurements. The objective is not a probability.

Up to eight separated spatial hypotheses are retained, each with its best tested shared clock. Inspect another rank to update the map, assignments and overlays. The flight-comparison table lets you inspect each candidate's error and data coverage for the selected event. A second overlay can disagree visibly with the measurement even when the aircraft has temporal coverage.

Statuses are review heuristics:

- **Insufficient evidence:** fewer than two distinct candidate flight segments support the acoustic groups, or nothing matches.
- **Partial fit:** at least two candidate segments contribute, but some selected groups remain unmatched.
- **Ambiguous:** the best fit is on a spatial/clock boundary, or another separated point has a similar objective (gap below 0.1 Hz).
- **Candidate fit:** none of those warnings triggered on the tested finite grid. This is still a research hypothesis.

The 0.1 Hz ambiguity rule is explicit and uncalibrated. Limited flight coverage, wind, changing engine tone, echoes and model error can create other alternatives. The search retains each spatial point's best clock, not every alternative clock or assignment. Grid spacing is not an uncertainty radius. The default computation limit is 300,000 scalar-equivalent model comparisons; the UI exposes a larger explicit budget for regional catalogs. Oversized requests return an actionable message. Batched clock evaluation uses the same acoustic equations and scores; scalar and batched results are regression-checked. The prepared regional comparison evaluated about 69.2 million models in approximately six minutes on the development machine.

## Saved cases and evaluation

**Save research case locally** creates a new directory under `outputs/cases/`. It includes decoded audio, extracted measurements, exact candidate-track coordinates, a readable track CSV, review notes, search settings, retained hypotheses, provenance, analysis-code hashes and a file-integrity manifest. Saved cases can be reopened in the app or downloaded as a portable ZIP. The ZIP contains the decoded recording, so it can be large. Downloaded bundles are not currently uploaded through the app; reopen local case directories.

For command-line analysis:

```sh
.venv/bin/python -m soundleak.cli workbench recording.wav \
  --tracks flights.csv --center 40.875 -73.53 --clock-sweep 5 \
  --start-utc 2026-09-08T20:17:30Z --output outputs/cases/new-case
```

For the fully scanned, source-informed reference example:

```sh
.venv/bin/python -m soundleak.cli workbench-reference \
  --output outputs/cases/reference-automatic
```

Ground-truth references enter only a separate post-fit evaluator. Freeze the case first, then supply a separate JSON:

```json
{
  "latitude": 40.8766,
  "longitude": -73.5247,
  "kind": "public_map_marker",
  "independent_holdout": false,
  "coordinate_evidence": "https://app.osprey-watch.org/nest/3916",
  "recording_clock_evidence": "Inherited fitted prior, not independently measured recorder UTC"
}
```

```sh
.venv/bin/python -m soundleak.cli evaluate outputs/cases/reference-automatic \
  --reference reference.json --output outputs/evaluations/reference.json
```

The UI also offers reference-JSON evaluation of a saved frozen report. Evaluation hashes that report and reports distances for every retained hypothesis, with 100/500/1,000 m diagnostic thresholds. An approximate public marker is excluded from independent accuracy summaries. Eligibility requires `kind: surveyed_recorder`, an explicit operator assertion of independent holdout, coordinate evidence, clock evidence, and no source-informed scope flag. Software cannot independently verify the operator's assertions. Ambiguous or insufficient-evidence cases are not successful location assertions even when one candidate is near a reference.

For the broader catalog experiment and real-background rejection checks, see [Regional catalog experiment](regional-catalog.md).

The **Public aircraft recordings · acoustic screen** expander shows a separate 40-passage public-audio check after running `soundleak public-aircraft-screen --fetch`. It retains every case, including misses and failures, and provides the report download. [The public dataset audit](public-datasets.md) explains why this acoustic check does not yet qualify as an independent location pilot.

A pilot evaluator accepts a manifest of frozen cases and separate reference JSON files:

```sh
.venv/bin/python -m soundleak.cli pilot-evaluate pilot-manifest.json \
  --output outputs/evaluations/my-pilot.json
```

The format is `{"schema_version": 1, "cases": [{"id": "recording-a", "case_directory": "path/to/frozen-case", "reference_file": "reference-a.json"}]}`. Paths are relative to the manifest. Missing files remain explicit `needs_input` entries; approximate references are excluded from independent summaries. [The empty template](pilot-manifest.example.json) declares no completed recordings. The existing one-hour example remains a development case. A multi-recording blind benchmark with independently documented microphone positions and sample-to-UTC timing has not been completed.
