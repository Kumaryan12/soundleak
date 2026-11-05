# Soundleak

A local sound-event and aircraft Doppler lab. **Soundleak Live** opens with synchronized recorded playback: audio, a scrolling spectrogram, reported aircraft positions, and cumulative listening-position hypotheses. Its second tab records or uploads a sample for local sound-category suggestions and a moving-tone screen.

The replay reveals precomputed analysis during playback. Microphone analysis runs after you stop recording. Continuous live microphone localization and fresh flight-data ingestion remain future work. [Live dashboard and sound inspector](docs/live.md).

The **Research workbench** opens the prepared regional-catalog comparison when available. The same 32 frozen acoustic proposals now have 31 heuristic flight associations, up from three with the original two-aircraft pool. The regional catalog contains 730 usable flight segments from 587 aircraft. The top grid point stayed the same, while other locations retained similar scores, so the outcome remains **ambiguous**. These are acoustic associations, not confirmed aircraft identifications. [Regional experiment](docs/regional-catalog.md) · [Workbench guide](docs/workbench.md).

The first real replay fits two actual flyovers to a location **179 m from a published public-camera map marker**. This is source-informed replication: the aircraft identities, event windows, rough region and recording-clock prior come from an existing reproduction. The marker is not a surveyed microphone coordinate. See [the experiment notes](docs/real-world-replay.md) for inputs, results and limits.

## Run

Python 3.11+ is required. Python 3.12 was used for development.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/streamlit run app.py
```

Open http://127.0.0.1:8501. **Soundleak Live** is the default view; choose **Research workbench** in the sidebar for the original analysis workflow. To fetch the reference inputs and prepare the automatic saved case on a new checkout:

```sh
.venv/bin/python -m pip install -e '.[real]'
.venv/bin/python scripts/fetch_reference.py
.venv/bin/python -m soundleak.cli workbench-reference --output outputs/cases/reference-automatic
.venv/bin/python -m soundleak.cli real
```

Expand the frozen automatic case using the full dated archive:

```sh
.venv/bin/python -m soundleak.cli fetch-regional --workers 12
.venv/bin/python -m soundleak.cli expand-catalog
```

The archive download is about 3.9 GB, is resumable, and verifies the publisher's SHA-256 digests. Regional selection uses the declared time window and 75 km radius, without acoustic-event or aircraft-type filtering. Completed downloads and catalog selections are reused. The app exposes the regional catalog, import audit and the before/after comparison once prepared.

Screen all 40 recordings in a separate public aircraft-audio collection using the unchanged acoustic defaults:

```sh
.venv/bin/python -m soundleak.cli public-aircraft-screen --fetch
```

This downloads approximately 132 MB of Pathak's raw-audio archives, verifies published checksums, and retains every miss and failed recording in a workbench table. The executed run screened all 40 recordings: 38 contained proposals, two had none, and none failed. It is an acoustic component check; independently documented microphone coordinates and WAV sample-zero UTC have not been established for this collection. See the [public dataset audit and results](docs/public-datasets.md).

Choose **Public recording review** in the sidebar, or click **Review public recordings** in the workbench. The review page provides original WAV playback, a zoomable spectrogram with numbered proposal ridges, a queue for the two recordings without proposals, per-proposal labels and persistent recording notes. JSON/CSV exports retain all unreviewed items. It reconstructs and verifies missing ridge arrays using the saved settings, without changing the screening report. To prepare all 40 review assets in advance:

```sh
.venv/bin/python -m soundleak.cli prepare-public-review
```

See the [recording review guide](docs/public-review.md) for labels, local storage and provenance.

FFmpeg must be installed (`brew install ffmpeg` on macOS). The downloader retrieves colsto's public one-hour recording and extracts two aircraft records from the ADSB.lol historical archive. It stores source links and SHA-256 hashes in `data/real/reference/provenance.json`; downloaded media is excluded from version control. No credentials are needed. Data sources retain their own rights and licenses.

Prepare the synchronized real-recording replay and optional sound inspector:

```sh
.venv/bin/python -m soundleak.cli prepare-live
.venv/bin/python -m pip install -e '.[classify]'
.venv/bin/python -m soundleak.cli fetch-sound-model
.venv/bin/python scripts/fetch_sound_examples.py
```

The synthetic playback works without downloaded media. The pinned sound model adds 521 sound-event labels; the optional examples include car horn, train, alarm clock and cat recordings. Drones have no dedicated label in this model and require separate validation. These labels do not locate arbitrary objects. See [setup, provenance and validation](docs/live.md).

Select **Synthetic demo** for the offline experiment, which needs no FFmpeg or internet access. Turn off **Combine two flyovers** to inspect a single flight's mirror ambiguity. The synthetic coordinate origin is near Southampton; that demo does not use Southampton recordings or ADS-B data.

Run the same experiment without a UI:

```sh
.venv/bin/python -m soundleak.cli demo --output outputs/demo
.venv/bin/python -m pytest
```

The CLI writes synthetic audio, track CSV, individual spectrogram overlays, a heatmap and a result JSON. With seed 7 and two flyovers, the estimated listener should be within one 100 m grid step of the generated ground truth. This is a consistency check of the synthetic pipeline, not a real-world accuracy claim.

## Analyze a recording

Install FFmpeg (`brew install ffmpeg` on macOS). Select **Analyze recording** and upload an audio/video file plus CSV. Processing is local. Only the first 180 seconds are decoded; trim longer recordings first. Upload size is capped at 100 MB.

CSV format:

```csv
aircraft_id,time_s,latitude,longitude,altitude_m
FLIGHT-A,-60,50.85,-1.49,1400
FLIGHT-A,-58,50.85,-1.487,1400
FLIGHT-A,-56,50.85,-1.484,1400
```

- `time_s`: aircraft position **emission** timestamp minus recording-start timestamp, in seconds. UTC epoch seconds minus UTC epoch seconds is recommended. Negative values are normal: flight data must start before audio to account for propagation delay.
- `latitude`, `longitude`: decimal degrees. Data is converted into local east/north coordinates around the search center.
- `altitude_m`: meters above sea level, using the same vertical datum as receiver elevation. Barometric altitude and ground elevation are not automatically reconciled.
- One or more `aircraft_id` values; at least three unique samples each. Dense state vectors (roughly 1–5 s intervals) are preferable. Sparse waypoints can erase the motion needed for a useful match. The three-row example shows syntax, not sufficient coverage of a normal flyover.

Select each event's time window, frequency band and associated aircraft. The ridge tracker follows one smooth spectral tone in that band. Select up to three events to combine evidence. Set a rough search center, search size and recorder elevation. Optional shared clock-offset search adds an offset to every audio event's time. It assumes all events use the same recording clock.

The **Research workbench** supports up to two hours of decoded audio, automatic acoustic proposals, manual acceptance/rejection and notes, candidate-flight import, shared-clock regional ranking, alternate-flight overlays, and immutable saved cases. Both relative `time_s` and absolute `time_utc` CSVs are supported there. The older **Analyze recording** screen retains the manual 180-second workflow described above.

The regional archive commands acquire a dated ADSB.lol archive and select all records meeting the declared time/position conditions. Arbitrary uploads still need supplied candidate tracks or a catalog prepared for their time and region. The workbench chooses among those candidates and can leave events unmatched. Live capture, global search and independent blind real-recording accuracy remain unvalidated.

Run automatic discovery and matching from the command line:

```sh
.venv/bin/python -m soundleak.cli workbench recording.wav \
  --tracks flights.csv --center 40.875 -73.53 \
  --start-utc 2026-09-08T20:17:30Z --clock-evidence 'Describe the recorder timing source' \
  --output outputs/cases/my-recording
```

Omit `--tracks` to save discovery for later review. Existing case directories are never overwritten. See the guide for post-fit evaluation against a separate reference file.

## Method

1. Compute a two-second Hann-window spectrogram with SciPy's `ShortTimeFFT` and a 0.25 s hop.
2. Suppress stationary spectral energy, then trace a smooth narrowband ridge with dynamic programming and refine frequencies between FFT bins.
3. For each receiver hypothesis, invert the sound-travel-time relation along the piecewise-linear aircraft path using its closed-form quadratic solution:

   `t_receive = t_emit + distance(position(t_emit), receiver) / c`

4. Evaluate radial aircraft velocity at **emission** time and predict:

   `f_receive = f_source * c / (c + v_radial)`

   Positive radial velocity means the source is receding. Fit one unknown, constant source frequency per event by least squares. All events share a candidate receiver and clock offset.
5. In the manual replay, minimize frequency RMSE over the grid and require complete selected-frame coverage. In the workbench, choose a candidate flight for each acoustic group under one shared location/clock, permit unmatched groups with an explicit fixed penalty, and retain separated spatial alternatives. Track extrapolation is prohibited.

The heatmap shows residual error in Hz, **not probability**. The grid spacing is not an uncertainty radius. A good score alone does not demonstrate that a tone is an aircraft, that the aircraft association is correct, or that the recording was made there. Single straight flyovers have an exact mirror ambiguity in this model; multiple passes with different headings can break it. Invalid/no-signal inputs are screened by ridge movement and prominence, but this heuristic is not a calibrated detection test.

Assumptions: a stationary recorder, still air, constant sound speed, constant source tone during each event, assumed receiver elevation, subsonic aircraft, and accurate track timestamps. Wind, terrain, temperature gradients, changing engine RPM, compression, microphone movement and reflections are not modeled. Local flat geometry is intended for search regions under 40 km across. Flight trajectories are linearly interpolated, so turns and climb rate are only as accurate as the supplied samples.

## Deploy

The **Wi-Fi lab** accepts RSSI captures from a Mac and router and compares link
variation with a declared static reference. Download the collector from the app,
or see [the Wi-Fi experiment guide](docs/wifi.md). This experiment does not detect
or locate people; it does not collect CSI.

The **Wi-Fi 3D replay** displays offline CSI2PointCloud predictions from six real
MM-Fi frames alongside measured LiDAR and zero-input controls. Coordinates are
standardized model units. The fixed first-frame baseline slightly outperforms
the changing-input predictions in this sample; this is a working inference
experiment, not independently validated scene reconstruction. See the
[reproduction guide](docs/wifi3d.md).

The **Evaluation lab** adds shuffled-CSI, zero-input and fixed-prediction controls,
sampled-frame playback, surface-error colors, shape/change plots, and CSV/JSON/PLY
exports. The current six-frame run still favors the fixed prediction. Training
overlap is unverified; the different-environment control is explicitly unavailable
while the official archive index download is quota-blocked. A fixed four-environment,
96-frame fetch/evaluation workflow is ready for a later run. See the
[evaluation guide and measured results](docs/wifi3d-evaluation.md).

See [the hosting guide](docs/deployment.md) for Streamlit Community Cloud and
Docker deployment. Use `streamlit_app.py` on Community Cloud to enable temporary,
separate storage for each visitor's saved cases and review notes.

## Sources

- [NASA: acoustic Doppler effect](https://www.grc.nasa.gov/WWW/k-12/airplane/doppler.html) — moving-source frequency relation.
- [SciPy: ShortTimeFFT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html) — spectrogram implementation.
- [OpenSky REST API](https://openskynetwork.github.io/opensky-api/rest.html) — possible external flight-data source. Access, authentication and historical coverage depend on the provider; its track endpoint returns sparse waypoints and is experimental.

The aircraft Doppler implementation was written independently in this project.
The optional Wi-Fi reconstruction runner includes the attributed MIT-licensed
CSI2PointCloud architecture in `soundleak/_vendor/`; its license is bundled there.
