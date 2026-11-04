# Real-world Doppler replay

The Soundleak implementation was run on actual downloaded audio and historical aircraft positions. It estimated a listening position **179.17 m from the published Oyster Bay camera/platform marker**. This is a source-informed regional replication, not independent blind geolocation or a measured 179 m accuracy guarantee.

## Inputs

The recording is colsto's public [Unknown location – raw audio](https://www.youtube.com/watch?v=4CxVxuMVkMc), a one-hour recording accompanying the original Doppler-geolocation video. This is the Oyster Bay case from the original video, rather than the Southampton webcam in the screenshot. The downloaded source is unaltered YouTube Opus audio, format 251, 45,780,657 bytes.

Two actual aircraft records were extracted from the [ADSB.lol 8 September 2026 historical archive](https://github.com/adsblol/globe_history_2026/releases/tag/v2026.09.08-planes-readsb-staging-0). Archive byte ranges allow extracting those records without downloading the entire daily archive. The complete extracted records contain 1,472 and 1,483 samples respectively. Only the portions surrounding the selected events enter the fit.

| Event | Audio window | Aircraft | Type | Analyzed band |
|---|---|---|---|---|
| 1 | 35:00–38:00 | a1dfeb / N22NY | Pilatus PC-12 | 75–190 Hz |
| 2 | 50:40–52:05 | ace391 / N93CQ | Beechcraft Super King Air 350 | 95–185 Hz |

Event windows, bands, aircraft identities, date, approximate regional bounds, and a fitted recording-start clock prior come from the [published DOPPLERGEIST reproduction configuration](https://github.com/1337isnot1337/DOPPLERGEIST/blob/3629d9ed2a3ed1e3572bf701df7754570dafd771/evidence/configs/source/reproduce-local.json) and its [result report](https://github.com/1337isnot1337/DOPPLERGEIST/blob/3629d9ed2a3ed1e3572bf701df7754570dafd771/evidence/results/reference-local-fast/report.json). No code from that project is executed by this replay. We fit our own extracted ridges with our own acoustic model; we do not reproduce its brightness-scoring implementation or its reported numerical result.

The initial recording-start prior is **2026-09-08 20:17:29.662 UTC**. It is an inherited fitted time, not independently measured recorder UTC. Soundleak searches one shared adjustment around that prior for both events.

## Procedure

1. Decode each event directly from the downloaded original recording at 4,096 samples/s, mono. No generated tones are mixed into the recording.
2. Extract a ridge with the same spectrogram and dynamic-programming tracker used by the synthetic demo, limiting frame-to-frame slope to 4 Hz/s.
3. Parse native readsb timestamps and WGS84 geometric aircraft heights. Convert feet to meters and approximate MSL height using a constant local geoid undulation of −31.26 m. Missing geometric height can be interpolated only between samples no more than 30 s apart. Barometric height is not substituted silently.
4. Search the published region, longitude −73.59 to −73.47 and latitude 40.83 to 40.92, on a 41×41 grid. At each location, test shared clock offsets from −5 to +5 seconds in 0.5 s steps. Assume recorder elevation 0 m MSL and sound speed 343 m/s.
5. Refine within 400 m of the coarse winner on a 25 m grid, testing ±1 s around the winning coarse clock in 0.25 s steps. Each event has one freely fitted constant source frequency. Published source-frequency constraints are not used.
6. Compare the result with the rounded public reference marker after fitting. The marker is not passed to the search function.

The model includes aircraft motion during sound propagation and never extrapolates flight positions beyond the supplied samples. The score is RMS frequency residual across every selected spectrogram frame, not a location probability.

## Observed result

| Quantity | Result |
|---|---|
| Estimated listening position | 40.875225, −73.525811 |
| Published platform marker | 40.8766, −73.5247 |
| Distance between them, local-plane calculation | 179.17 m |
| Combined frequency RMSE | 2.319 Hz |
| Shared adjustment to inherited clock | +1.25 s |
| Fitted source tones | 114.91 Hz / 125.43 Hz |
| Final grid spacing | 25 m |

The reference marker is independently accessible at [OspreyWatch nest 3916](https://app.osprey-watch.org/nest/3916), labeled Oyster Bay Marine Center. It is a rounded public map coordinate with unquantified physical microphone uncertainty. The coordinate source does not establish that the microphone was surveyed at that point.

The predicted cyan curves follow the strong measured descending tones. The first ridge also includes a few brief jumps into background energy near the end; those frames are retained in the result rather than removed after inspecting the reference coordinate.

## Reproduce and inspect

```sh
.venv/bin/python -m pip install -e '.[real]'
.venv/bin/python scripts/fetch_reference.py
.venv/bin/python -m soundleak.cli real
.venv/bin/streamlit run app.py
```

FFmpeg is required. The app defaults to **Real-world replay** and provides both real audio excerpts, spectrogram overlays, the regional and refined heatmaps, and the JSON report.

Artifacts are written to `outputs/real/reference/`: `report.json`, `regional-heatmap.png`, `heatmap.png`, `event-1.png`, `event-2.png`, two event WAV files, and two ridge CSV files. Download source links, file sizes and SHA-256 hashes are retained in `data/real/reference/provenance.json`. Raw media and generated outputs are ignored by version control. The acquisition script checks input hashes before reusing files.

## Interpretation

This run demonstrates that our pipeline can fit two genuine acoustic Doppler patterns to genuine historical flight motion and produce a regional location near the public-camera reference. Its known aircraft associations, event selection, rough region and fitted clock prior make it a guided replication.

It does not establish automatic aircraft discovery, global localization, blind real-world accuracy or calibrated uncertainty. Wind, aircraft tone variation, acoustic reflections, clock error, altitude error and the flat-earth approximation can affect the fit. The 25 m grid spacing is not an uncertainty estimate. A new recording with independently documented sample-to-UTC timing and recorder position is needed for stronger validation.

Audio rights remain with colsto. Historical aircraft records are attributed to ADSB.lol under ODbL-1.0. No endorsement by either source is implied.
