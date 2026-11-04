# Public dataset audit and acoustic screening

Public sources were reviewed on 4 November 2026 before declaring any independent receiver-localization pilot. **None of the reviewed sources has yet supplied verified raw aircraft audio, independent microphone-position evidence, and a defensible mapping from WAV sample zero to trajectory time together.** This is an evidence gap in this review, not a claim that no suitable public data exists.

The machine-readable review is `outputs/evaluations/public-dataset-audit.json`. Unknown metadata stays unknown; approximate coordinates and fitted clocks are not converted into independent references.

| Public source | What can be used now | Qualification issue |
|---|---|---|
| [Pathak, 40 passages](https://zenodo.org/records/23424278) | Raw aircraft WAVs for a frozen acoustic screen; accompanying aircraft tracks for future work | UTC-stamped extracted frequencies do not independently establish WAV sample-zero UTC. A recorder-position reference has not been established in reviewed materials. |
| [AeroSonicDB v1.1.2](https://zenodo.org/records/10215080) | Existing annotated environmental controls | Publisher intentionally supplies nearby public-space coordinates instead of microphone positions. |
| [AeroSonicDB 3K](https://zenodo.org/records/12775560) | Catalog only | Files are restricted. No access request was sent. |
| [SONYC-UST-V2 paper](https://arxiv.org/html/2009.05188) | Urban sound tagging | Public coordinates are block centers, times are quantized to hours, and snippets last ten seconds. These do not meet the current screen/alignment requirements. |
| [TartanAviation](https://www.nature.com/articles/s41597-025-04775-6) | ADS-B and radio-speech research | Audio is ATC radio speech, not environmental aircraft-engine sound. |
| [TU Delft Skymaster measurements](https://research.tudelft.nl/en/datasets/data-and-code-underlying-ch-7-of-phd-thesis-phased-microphone-arr/) | Described flyover spectra and engine-runup pressure measurements | Full flyover waveforms and independent clock/receiver references were not verified. Stationary runups do not substitute for moving Doppler passages. |
| [Western Sydney UAV acoustics](https://research-data.westernsydney.edu.au/published/1f8cceb0801b11f189a547c836cf5bee/) | Promising separate drone-domain experiment | Catalog advertises WAV, telemetry and alignment files. Direct README requests returned HTTP 403 here; reference independence remains unverified. |
| [TrueNoise observations](https://truenoise.org/data.html) | Measurement/flight CSV observations | No raw waveform was established from the reviewed download. Aggregate noise metrics cannot reconstruct the needed spectrogram. |

## Forty-recording acoustic component check

All four Pathak release archives are downloaded and checked against Zenodo's published MD5 digests; local SHA-256 hashes are retained as well. Reading each complete WAV from its ZIP validates that member's CRC. The manifest declares **every one of the 40 passages before detection**, with no selection by result.

The exact existing conservative `DiscoveryConfig` is used for every raw recording. No author-picked frequency CSV, time window, flight track, location or UTC prior enters detection. The config, case list, input hashes and relevant code hashes are saved. Failed decodes and recordings without proposals remain in the output. No source code from the dataset is executed.

```sh
.venv/bin/python -m soundleak.cli public-aircraft-screen --fetch
```

The manifest is `data/real/pathak/screen-manifest.json`. The declaration and results are `outputs/evaluations/public-aircraft-screen.declaration.json` and `outputs/evaluations/public-aircraft-screen.json`. The workbench displays the complete recording-level table under **Public aircraft recordings · acoustic screen**.

The [Public recording review page](public-review.md) adds original audio playback, spectrogram/ridge inspection, filters for missed recordings, persistent human labels and JSON/CSV exports. All 40 recordings and 54 proposals have verified prepared review assets; human annotations start unreviewed.

The executed run screened **40/40 recordings**, totaling **3,605.33 seconds**, with **zero failures**. There were **54 grouped acoustic proposals** across **38 recordings**. The two recordings without proposals were `N394FX_R20` (60.53 seconds) and `N1387U_R44` (41.77 seconds). All recordings exceeded the 20-second minimum event duration; no misses were excused or removed as too short.

| Publisher's motion category | Recordings screened | Recordings with proposals | Grouped proposals |
|---|---:|---:|---:|
| Constant speed / constant height | 8 | 7 | 12 |
| Constant speed / varying height | 20 | 20 | 28 |
| Varying speed / constant height | 4 | 4 | 5 |
| Varying speed / varying height | 8 | 7 | 9 |
| **Total** | **40** | **38** | **54** |

The original proposal thresholds and frequency bands were not adjusted after observing these results. Counts indicate proposal presence, not verified matching to the author's aircraft event. The complete archives total 132,441,223 bytes, and all four published digests and all 40 WAV-member CRC checks passed.

This measures whether the current screen proposes any patterns on a separate collection of aircraft passages. Proposed patterns remain unconfirmed; clip-level proposal presence is not confirmed aircraft identification or localization accuracy. The collection was curated for a published Doppler analysis, so it is not an unbiased sample of arbitrary recordings. The independent location pilot remains pending suitable evidence.

Praveen Pathak's dataset is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), DOI [10.5281/zenodo.23424278](https://doi.org/10.5281/zenodo.23424278). This project locally decodes its WAVs and applies its own existing detector; the original recordings are not altered in their archives.
