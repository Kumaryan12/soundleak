# Soundleak Live

The first version provides a synchronized **recorded replay** and a separate **record / upload** sound inspector. Select Soundleak Live in the sidebar; it is the default for a new session.

## Try it

In **Recorded replay**, choose the real regional recording or the synthetic two-flyover experiment. Press **Play recording**, change playback speed, scrub the timeline or use **Next observation**. The scrolling spectrogram, flight positions, evidence feed and cumulative map share the audio player's clock. Switching to the capture tab pauses replay audio.

The initial map waits for a complete observation. A single matching flight segment leaves location support insufficient. Later observations update the map and retain competing locations. **Reveal reference** adds a reference marker for comparison after fitting. Synthetic truth and the approximate public camera marker do not enter the estimator.

In **Record / upload a sound**, record at least four seconds using the browser microphone control, then stop and select **Analyze sound**. Alternatively upload a file or choose an installed example. Up to the first two minutes are decoded. Analysis runs on this computer after recording stops; no audio is sent to a remote inference service. A browser permission prompt controls recording. Localhost works; use HTTPS for access from another device, including a phone. [Browser microphone requirements](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).

The inspector shows whole-sample mean scores, peak scores and times, labeled time intervals, and frame-level suggestions. Peaks retain brief events that would otherwise be diluted by long silent portions. Multiple labels can describe one source. The 0.25 display threshold is an uncalibrated operating choice. Intervals and model labels do not establish object counts, identity or location. The JSON download includes input hashes, model version and analysis duration.

## Which sources can it recognize?

The separate classifier is [Google YAMNet](https://www.tensorflow.org/hub/tutorials/yamnet), via a [pinned Apache-2.0 ONNX conversion](https://huggingface.co/audiomagic/yamnet-onnx/tree/f25b741c2f0bdc6d7e6db24b5fddda23347dbafd). It consumes mono 16 kHz audio and returns 521 sound-event scores. The conversion includes its own signal-processing frontend. The model, labels, README and license are all checked against pinned SHA-256 hashes before use. No upstream Python code is executed.

| Requested direction | Current support |
|---|---|
| Cars / road vehicles | Labels include Car, Motor vehicle (road), Car passing by, and car horns. |
| Trains | Labels include Rail transport, Train, Train horn and wheels squealing. |
| Alarms | Labels include Alarm, Alarm clock, Siren, Fire alarm and smoke alarm. |
| Drones | No dedicated drone label. Engine, propeller or helicopter scores do not validate drone identification. A separate dataset/model evaluation is needed. |
| Other objects | The model also covers many animals, machinery and everyday sounds. Silent objects cannot be detected by audio; unfamiliar or masked sources may not be recognized. |

Sound recognition is separate from localization. The aircraft estimator uses known, timestamped moving-source trajectories. Those inputs are not automatically available for a car, train or drone. Direction or position for arbitrary sources would require additional geometry, synchronized microphones or other sensors and new validation.

## Setup

Install the base app and optional classifier:

```sh
.venv/bin/python -m pip install -e '.[dev,real,classify]'
.venv/bin/python -m soundleak.cli fetch-sound-model
.venv/bin/python scripts/fetch_sound_examples.py
.venv/bin/streamlit run app.py
```

Soundleak Live requires Streamlit 1.66+ for its browser component. The ONNX model is approximately 16 MB; ONNX Runtime is optional for the sound inspector. No TensorFlow or PyTorch runtime is required. A checkout without downloaded reference data still has the synthetic replay.

After preparing the regional case, prepare its playback data:

```sh
.venv/bin/python -m soundleak.cli prepare-live \
  --case outputs/cases/reference-regional \
  --output outputs/live/regional --grid 11 --clocks 5
```

Preparation refuses to overwrite an existing output. The UI verifies the source report and audio archive hashes. The replay file contains precomputed per-observation states, reduced display geometry and a downsampled spectrogram. Original recordings and scientific cases are preserved.

## Incremental fitting and limitations

Proposals are ordered by their final measured audio sample. Each update includes only groups completed by that playback time. Each event's capped loss is calculated over the same finite spatial and clock grid. The losses are accumulated first, then profiled over **one common recording-clock adjustment**, rather than choosing a different clock for every event. Rejected groups continue to contribute the rejection loss. Every update retains separated spatial alternatives and the existing insufficient-evidence / partial / ambiguous interpretation.

The regional playback uses an **11 × 11 grid** over the same 10 km search box and five clock candidates from −5 to +5 seconds. These settings are coarser than the saved workbench's 21 × 21 / 21-clock search. It evaluates **4,521,165 scalar-equivalent models** once, producing **32 observation updates** in approximately **19.8 seconds** on the development machine. The final coarse playback result remains **ambiguous**, with 31 candidate matches and one unmatched group. Its top coarse grid point is at the declared center, so it should not be conflated with the finer workbench result.

The detector originally processed the complete recording. Grouping and fitted ridge summaries can therefore depend on later audio windows. Revealing those offline proposals in temporal order is not proof of a causal streaming detector. This first version does not capture continuous microphone frames, fetch a fresh live aircraft catalog or independently locate an arbitrary recording. The real recording also inherits source-informed region/clock priors. The reference is a rounded public camera marker with unknown physical microphone uncertainty.

Display flight paths are simplified independently of the acoustic scores and interpolated at playback time plus the fitted clock adjustment. They represent reported aircraft positions, while frequency predictions account for acoustic travel time. Map colors are model residuals, not location probabilities. Reference distances and grid spacing are not accuracy guarantees.

## Validation performed

- Vectorized retarded-time calculations match the original scalar physics, including out-of-coverage and nonfinite times.
- Every synthetic prefix matches a fresh existing shared-clock search; modifying a future observation leaves an earlier fit unchanged.
- The complete real coarse replay surface matches a fresh existing batched search across all 32 groups and 730 flight segments. This is recorded in `outputs/research/live-parity.json`.
- A headless Chrome check exercised playback, seeking, observation jumps, reference reveal, final map status and automatic pause when entering capture.
- Browser microphone start/stop, decoding, local inference and JSON download were exercised using an injected public test waveform in an isolated browser. No physical microphone was used in this check.
- Local model checks covered silence, a generated beep, a public cat recording, and the first filename in each requested ESC-50 category. These are component checks, not calibrated classification accuracy or an independent localization benchmark.

The three transport/alarm clips come from [ESC-50](https://github.com/karolpiczak/ESC-50), by Karol J. Piczak and original Freesound contributors. The dataset is CC BY-NC 3.0, with original source rights also applicable. `data/real/sound-examples/manifest.json` records source IDs, URLs, hashes and the deterministic selection rule. The optional fetcher pins the dataset revision. The cat example is linked by the official YAMNet tutorial. These examples are downloaded locally rather than bundled in the package.

For browser QA with the app already running, use `.venv/bin/python scripts/check_live_browser.py`. It uses an isolated headless Chrome instance and a generated beep as a fake microphone, then checks playback, seeking, reference reveal, capture-tab pause, sound inference and downloads. Install Chromium with `.venv/bin/python -m playwright install chromium` if a local Chrome executable is unavailable. Screenshots and the test capture report go to `outputs/browser-check/`.
